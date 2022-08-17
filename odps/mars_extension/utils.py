#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2022 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import platform
import sys
import tokenize

import numpy as np
import pandas as pd

from ..compat import StringIO
from ..df.backends.pd.types import _np_to_df_types
from ..df.backends.odpssql.types import df_type_to_odps_type
from ..df import types
from ..errors import ODPSError


def pd_type_to_odps_type(dtype, col_name, unknown_as_string=None, project=None):
    import numpy as np

    if dtype in _np_to_df_types:
        df_type = _np_to_df_types[dtype]
    elif dtype == np.datetime64(0, "ns"):
        df_type = types.timestamp
    elif unknown_as_string:
        df_type = types.string
    else:
        raise ValueError(
            "Unknown type {}, column name is {},"
            "specify `unknown_as_string=True` "
            "or `as_type` to set column dtype".format(dtype, col_name)
        )

    return df_type_to_odps_type(df_type, project=project)


def use_odps2_type(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        from odps import options

        old_value = options.sql.use_odps2_extension
        options.sql.use_odps2_extension = True

        old_settings = options.sql.settings
        options.sql.settings = old_settings or {}
        options.sql.settings.update({"odps.sql.hive.compatible": True})
        options.sql.settings.update({"odps.sql.decimal.odps2": True})
        try:
            func(*args, **kwargs)
        finally:
            options.sql.use_odps2_extension = old_value
            options.sql.settings = old_settings

    return wrapped


def convert_pandas_object_to_string(df):
    import pandas as pd
    import numpy as np

    def convert_to_string(v):
        if v is None or pd.isna(v):
            return None
        else:
            return str(v)

    object_columns = [c for c, t in df.dtypes.iteritems() if t == np.dtype("O")]
    for col in object_columns:
        df[col] = df[col].map(convert_to_string)
    return df


def check_partition_exist(table, partition_spec):
    try:
        return table.exist_partition(partition_spec)
    except (ValueError, ODPSError):
        return False


def rewrite_partition_predicate(predicate, cols):
    cols = [c.lower() for c in cols]

    def _tokenize_predicate():
        reader = StringIO(predicate).readline
        last_num = ""
        last_bool_cmp = ""
        last_toknum = None

        for toknum, tokval, _, _, _ in tokenize.generate_tokens(reader):
            if last_toknum != toknum:
                # handle consecutive operands / numbers
                if last_num:
                    yield tokenize.STRING, '"%s"' % last_num
                    last_num = ""
                if last_bool_cmp:
                    if last_bool_cmp not in ("&", "&&", "|", "||"):
                        raise SyntaxError("Operand %s not recognized" % last_bool_cmp)

                    yield tokenize.OP, last_bool_cmp[-1]
                    last_bool_cmp = ""

            last_toknum = toknum

            if toknum == tokenize.NUMBER:
                last_num += tokval
            elif toknum == tokenize.OP and tokval in ("&", "|"):
                last_bool_cmp += tokval
            else:
                if toknum == tokenize.NAME:
                    lower_tokval = tokval.lower()
                    if lower_tokval == "max_pt":
                        tokval = "@" + lower_tokval
                    elif lower_tokval not in cols:
                        toknum = tokenize.STRING
                        tokval = '"%s"' % tokval
                elif toknum == tokenize.OP:
                    if tokval == "=":
                        tokval = "=="
                    elif tokval == ",":
                        tokval = "&"
                yield toknum, tokval

    return tokenize.untokenize(list(_tokenize_predicate()))


def calc_max_partition(
    table_name=None, odps=None, query_table_name=None, result_cache=None
):
    result_cache = result_cache or dict()
    table_name = table_name or query_table_name
    if table_name in result_cache:
        return result_cache[table_name]

    table = odps.get_table(table_name)
    if not table.schema.partitions:
        raise ValueError("Table %r not partitioned" % table_name)
    first_part_name = table.schema.partitions[0].name

    reversed_table_parts = sorted(
        list(table.partitions),
        key=lambda part: str(part.partition_spec.kv[first_part_name]),
        reverse=True,  # make larger partition come first
    )
    result = next(
        (
            part.partition_spec.kv[first_part_name]
            for part in reversed_table_parts
            if part.physical_size > 0
        ),
        None,
    )

    if result is None:
        raise ValueError(
            "Table %r has no partitions or none of "
            "the partitions have any data" % table_name
        )
    result_cache[table_name] = result
    return result


def filter_partitions(odps, partitions, predicate):
    cols = partitions[0].partition_spec.keys
    part_df = pd.DataFrame(
        [part.partition_spec.kv.values() for part in partitions], columns=cols
    )
    part_df = part_df.astype(str)
    part_df["__pt_obj__"] = pd.Series(partitions, dtype=np.dtype("O"))

    predicate = rewrite_partition_predicate(predicate, cols)

    global_dict = globals().copy()
    query_table = partitions[0].parent.parent
    query_table_name = query_table.project.name + "." + query_table.name
    result_cache = dict()
    global_dict["max_pt"] = functools.partial(
        calc_max_partition,
        odps=odps,
        query_table_name=query_table_name,
        result_cache=result_cache,
    )

    part_df.query(predicate, engine="python", global_dict=global_dict, inplace=True)

    return part_df["__pt_obj__"].to_list()


def get_default_resource_files(names, project="public"):
    """
    Use Python version plus architecture to name packages.
    For instance, public.pymars-0.8.6-cp37-x86_64.zip
    """
    suffix = (
        "-cp"
        + "".join(str(x) for x in sys.version_info[:2])
        + "-x86_64.zip"  # todo fix for other architectures
    )
    return [project + "." + name + suffix for name in names]


def build_mars_image_name(mars_image=None, with_dist_suffix=True):
    from cupid.config import options
    from cupid.utils import build_image_name

    prefix = options.cupid.image_prefix
    if prefix is not None:
        dockerhub_address = prefix.split("/")[0]
    else:
        dockerhub_address = None

    mars_image = mars_image or "mars"
    if ":" not in mars_image:
        mars_image = build_image_name(mars_image, with_dist_suffix=with_dist_suffix)
    if dockerhub_address is not None and dockerhub_address not in mars_image:
        mars_image = prefix + mars_image
    return mars_image
