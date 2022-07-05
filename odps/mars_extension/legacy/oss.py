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

import logging
import uuid
import warnings
from itertools import chain

import numpy as np
import oss2
from mars.tensor.utils import decide_chunk_sizes

from ...df import CollectionExpr
from ...compat import six, OrderedDict
from ...df.backends.engine import get_default_engine
from ...df.backends.odpssql.types import df_type_to_odps_type
from ...df.backends.pd.types import df_type_to_np_type
from ...models import Schema, Table
from ...df import Scalar
from ...utils import to_str
from ...errors import ODPSError
from ...types import PartitionSpec
from ... import types as odps_types
from .tensor import read_coo
from .io import open as fs_open


np_int_types = map(np.dtype, [np.int_, np.int8, np.int16, np.int32, np.int64])
np_float_types = map(np.dtype, [np.float_, np.float16, np.float32, np.float64])
np_to_odps_types = dict(
    [(t, odps_types.bigint) for t in np_int_types]
    + [(t, odps_types.double) for t in np_float_types]
)


logger = logging.getLogger(__name__)


def _clean_oss_object(
    path, access_id=None, secret_access_key=None, endpoint=None, bucket_name=None
):
    bucket = oss2.Bucket(oss2.Auth(access_id, secret_access_key), endpoint, bucket_name)
    while True:
        keys = [l.key for l in oss2.ObjectIterator(bucket, prefix=path + "/")]
        if not keys:
            return
        bucket.batch_delete_objects(keys)


def _copy_to_ext_table(
    expr,
    odps,
    dim_cols,
    val_col,
    chunks,
    partitions=None,
    oss_path=None,
    handler=None,
    hints=None,
    **oss_opts
):
    column_names = list(dim_cols) + [val_col]
    column_types = [df_type_to_odps_type(expr.schema[n].type) for n in column_names]

    if partitions is not None:
        partitions = (
            [partitions]
            if not isinstance(partitions, (list, tuple, set))
            else partitions
        )
        partition_types = ["string"] * len(partitions)
        target_schema = Schema.from_lists(
            column_names, column_types, partitions, partition_types
        )
        column_names.extend(list(partitions))
    else:
        target_schema = Schema.from_lists(column_names, column_types)

    _clean_oss_object(oss_path + "/data", **oss_opts)

    if not isinstance(chunks, (list, tuple)):
        chunks = [chunks] * len(dim_cols)
    ext_table_name = "mars_ext_%s" % str(uuid.uuid4()).replace("-", "_")
    ext_table = odps.create_table(
        ext_table_name,
        target_schema,
        storage_handler="com.aliyun.odps.mars.ChunkStorageHandler",
        serde_properties={
            "mars.chunk_sizes": ",".join("%s" % ch for ch in chunks),
            "mars.parquet.page_size": "%s" % (128 * 1024),
            "mars.parquet.row_group_size": "%s" % (128 * 1024),
        },
        location="oss://%s:%s@%s/%s/%s/data"
        % (
            oss_opts.get("access_id"),
            oss_opts.get("secret_access_key"),
            oss_opts.get("endpoint").split("://")[1],
            oss_opts.get("bucket_name"),
            oss_path,
        ),
        resources=handler,
        lifecycle=14,
    )
    if "/" in handler:
        assert handler in ext_table.resources
    if partitions:
        shuffled = expr.select(column_names).reshuffle(
            partitions, sort=partitions + dim_cols
        )
    else:
        shuffled = expr.select(column_names).reshuffle(dim_cols, sort=dim_cols)
    logger.debug("writing to external table %s" % ext_table_name)

    odps_hints = {
        "odps.sql.reshuffle.dynamicpt": False,
        "odps.isolation.session.enable": True,
    }
    if hints:
        odps_hints.update(hints)
    shuffled.persist(
        ext_table.name, overwrite=False, partitions=partitions, hints=odps_hints
    )


def _write_shape_to_oss(shape, path, **oss_opts):
    shapes_string = ",".join("%s" % s for s in shape)
    with fs_open(path, "wb", **oss_opts) as out_file:
        out_file.write(shapes_string)


def _read_shape_from_oss(path, **oss_opts):
    with fs_open(path, "rb", **oss_opts) as file:
        shape_string = to_str(file.read())
    return tuple(int(i) for i in shape_string.split(","))


def _as_scalar(s):
    return s.item() if hasattr(s, "item") else s


def _get_shapes(expr, dim_columns, partitions=None, hints=None):
    if partitions is None:
        shapes = expr[
            list((expr[col].max() + 1).rename(col) for col in dim_columns)
        ].execute(hints=hints)
        return tuple(_as_scalar(s) for s in shapes[0])
    else:
        odps_hints = {"odps.task.merge.enabled": False}
        if hints:
            odps_hints.update(odps_hints)
        shapes = (
            expr.groupby(partitions)
            .agg(list((expr[col].max() + 1).rename(col) for col in dim_columns))
            .execute(hints=odps_hints)
        )
        return dict(
            (
                ",".join(tuple(record[partitions])),
                tuple(_as_scalar(s) for s in record[dim_columns]),
            )
            for record in shapes
        )


def _preprocess(expr, table_name, columns, partitions=None, hints=None):
    if partitions:
        expr_grouped = expr.groupby(partitions)
        expr_ranked = expr[
            [
                (expr_grouped[col].dense_rank() - 1).rename(col + "_continuous")
                for col in columns
            ]
        ]
    else:
        expr_ranked = expr[
            [
                (expr.groupby(Scalar(1))[col].dense_rank() - 1).rename(
                    col + "_continuous"
                )
                for col in columns
            ]
        ]
    logger.debug("creating new table %s")
    expr[expr, expr_ranked].persist(table_name, partitions=partitions, hints=hints)


def _get_df_partitions(df, partitions):
    if (
        df._source_data is not None
        and isinstance(df._source_data, Table)
        and set([p.name for p in df._source_data.schema.partitions]) == set(partitions)
    ):
        return list(p.partition_spec for p in df._source_data.partitions)
    else:
        return df[partitions].distinct().execute()


def _decide_chunk_sizes(shape, chunk_size, dtype):
    if not isinstance(shape, dict):
        nsplits = decide_chunk_sizes(shape, chunk_size, dtype.itemsize)
        return [int(ns[0]) for ns in nsplits]
    else:
        result_chunks = [None] * len(shape)
        for s in shape.values():
            nsplits = decide_chunk_sizes(s, chunk_size, dtype.itemsize)
            ns = [int(n[0]) for n in nsplits]
            for i, n in enumerate(ns):
                if result_chunks[i] is None or result_chunks[i] > n:
                    result_chunks[i] = n
        return result_chunks


def to_mars_tensor_via_oss(
    expr,
    dim_columns,
    value_column,
    chunks=None,
    chunk_size=None,
    oss_access_id=None,
    oss_access_key=None,
    oss_endpoint=None,
    oss_bucket_name=None,
    oss_path=None,
    odps=None,
    discontinuous_columns=None,
    new_table_name=None,
    partitions=None,
    shape=None,
    sparse=False,
    oss_file_exist=False,
    handler="public/resources/chunkwriter.jar",
    hints=None,
):
    if (discontinuous_columns is not None) != (new_table_name is not None):
        raise ValueError(
            "if some dimension columns are discontinuous,"
            "preprocessed table name should be provided"
        )
    if oss_path is None:
        raise ValueError("oss_path should be provided")

    oss_prefix = "oss://%s/" % oss_bucket_name
    if oss_path.startswith(oss_prefix):
        oss_path = oss_path[len(oss_prefix) :]

    if sparse and (shape is None):
        raise ValueError("shape is necessary if target tensor is sparse")
    if odps is None:
        odps = get_default_engine(expr)._odps
        if odps is None:
            raise ODPSError("ODPS entrance should be provided")
    if isinstance(partitions, six.string_types):
        partitions = [partitions]
    if discontinuous_columns is not None:
        if not odps.exist_table(new_table_name):
            _preprocess(
                expr, new_table_name, discontinuous_columns, partitions, hints=hints
            )
        expr = odps.get_table(new_table_name).to_df()
        disordered_cols = discontinuous_columns or []
        dim_columns = [
            col + "_continuous" if col in disordered_cols else col
            for col in dim_columns
        ]
    oss_opts = dict(
        endpoint=oss_endpoint,
        bucket_name=oss_bucket_name,
        access_id=oss_access_id,
        secret_access_key=oss_access_key,
    )
    dtype = df_type_to_np_type(expr[value_column].dtype)

    if shape is not None and isinstance(shape, dict):
        new_shape = dict()
        for k, v in shape.items():
            p_spec = PartitionSpec(k)
            p_key = ",".join([p_spec[p] for p in partitions])
            new_shape[p_key] = v
        shape = new_shape

    # get shape
    partition_values = None
    if not oss_file_exist:
        shape = shape or _get_shapes(expr, dim_columns, partitions, hints=hints)
    if partitions is None:
        shape_path = "oss://%s/meta/shape" % oss_path
        if shape is None:
            shape = _read_shape_from_oss(shape_path, **oss_opts)
        if not oss_file_exist:
            _write_shape_to_oss(shape, shape_path, **oss_opts)
    else:
        partition_values = _get_df_partitions(expr, partitions)
        new_shape = dict()
        for partition in partition_values:
            partition_path = "/".join("%s=%s" % (p, partition[p]) for p in partitions)
            shape_path = "oss://%s/meta/%s/shape" % (oss_path, partition_path)
            partition_key = ",".join(partition[p] for p in partitions)
            if shape is None:
                partition_shape = _read_shape_from_oss(shape_path, **oss_opts)
                new_shape[partition_key] = partition_shape
            else:
                partition_shape = shape[partition_key]
            if not oss_file_exist:
                _write_shape_to_oss(partition_shape, shape_path, **oss_opts)
        if shape is None:
            shape = new_shape

    if chunks is not None:
        warnings.warn(
            "chunks is deprecated, use chunk_size instead", DeprecationWarning
        )

    chunk_size = chunk_size if chunk_size is not None else chunks
    if chunk_size is None:
        chunk_size = _decide_chunk_sizes(shape, chunk_size, dtype)

    if not oss_file_exist:
        _copy_to_ext_table(
            expr,
            odps,
            dim_columns,
            value_column,
            chunk_size,
            oss_path=oss_path,
            partitions=partitions,
            handler=handler,
            hints=hints,
            **oss_opts
        )

    if partitions is None:
        # for non-partitioned table
        path = "oss://%s/data/*.parquet" % oss_path
        tensor = read_coo(
            path,
            dim_columns,
            value_column,
            shape=shape,
            chunk_size=chunk_size,
            dtype=dtype,
            sparse=sparse,
            **oss_opts
        )
        return tensor
    else:
        # for partitioned table
        tensors = OrderedDict()
        for partition in partition_values:
            partition_path = "/".join("%s=%s" % (p, partition[p]) for p in partitions)
            path = "oss://%s/data/%s/*.parquet" % (oss_path, partition_path)
            partition_key = ",".join(partition[p] for p in partitions)
            tensor = read_coo(
                path,
                dim_columns,
                value_column,
                shape=shape[partition_key],
                chunk_size=chunk_size,
                dtype=dtype,
                sparse=sparse,
                **oss_opts
            )
            partition_spec = ",".join("%s=%s" % (p, partition[p]) for p in partitions)
            tensors[partition_spec] = tensor
        return tensors


def persist_tensor_via_oss(odps, *args, **kwargs):
    from mars.session import Session
    from .tensor.datastore import write_coo

    session = kwargs.pop("session", Session.default_or_local())
    oss_endpoint = kwargs.pop("oss_endpoint")
    oss_access_id = kwargs.pop("oss_access_id")
    oss_access_key = kwargs.pop("oss_access_key")
    oss_bucket_name = kwargs.pop("oss_bucket_name")
    oss_path = kwargs.pop("oss_path")

    oss_prefix = "oss://%s/" % oss_bucket_name
    if oss_path.startswith(oss_prefix):
        oss_path = oss_path[len(oss_prefix) :]

    oss_opts = dict(
        endpoint=oss_endpoint,
        bucket_name=oss_bucket_name,
        access_id=oss_access_id,
        secret_access_key=oss_access_key,
    )

    tensor, table_name, dim_columns, value_column = args
    oss_dir = "oss://%s" % oss_path
    _clean_oss_object(oss_path, **oss_opts)

    t_type = None
    partitions = None

    # submit tensor to mars cluster
    tensors = []
    if isinstance(tensor, dict):
        for p, t in tensor.items():
            if t_type is None:
                t_type = t.dtype
            p_spec = PartitionSpec(p)
            if partitions is None:
                partitions = p_spec.keys
            else:
                if set(partitions) != set(p_spec.keys):
                    raise TypeError("all tensors partitions name must be the same.")

            if t.ndim > len(dim_columns):
                raise TypeError("tensor dimensions cannot more than dim_columns length")

            # write shape to oss
            shape_path = "%s/meta/%s/shape" % (oss_dir, p.replace(",", "/"))
            _write_shape_to_oss(t.shape, shape_path, **oss_opts)

            # write data to oss
            data_path = "%s/data/%s" % (oss_dir, p.replace(",", "/"))
            writer_tensor = write_coo(
                t, data_path, dim_columns, value_column, global_index=True, **oss_opts
            )
            tensors.append(writer_tensor)

        session.run(tensors)
    else:
        shape_path = oss_dir + "/meta/shape"
        _write_shape_to_oss(tensor.shape, shape_path, **oss_opts)

        t_type = tensor.dtype
        data_path = oss_dir + "/data"
        writer_tensor = write_coo(
            tensor, data_path, dim_columns, value_column, global_index=True, **oss_opts
        )
        session.run(writer_tensor)

    # persist to odps table
    ext_table_name = "mars_persist_ext_%s" % str(uuid.uuid4()).replace("-", "_")
    column_types = ["bigint"] * len(dim_columns) + [np_to_odps_types[t_type]]
    ext_column_types = ["bigint"] * (2 * len(dim_columns)) + [np_to_odps_types[t_type]]
    column_names = dim_columns + [value_column]
    ext_column_names = list(chain(*([c, "global_" + c] for c in dim_columns))) + [
        value_column
    ]
    if partitions:
        if isinstance(partitions, six.string_types):
            partitions = [partitions]
        target_schema = Schema.from_lists(
            column_names, column_types, partitions, ["string"] * len(partitions)
        )
        ext_schema = Schema.from_lists(
            ext_column_names, ext_column_types, partitions, ["string"] * len(partitions)
        )
    else:
        target_schema = Schema.from_lists(column_names, column_types)
        ext_schema = Schema.from_lists(ext_column_names, ext_column_types)

    ext_table = odps.create_table(
        ext_table_name,
        ext_schema,
        external_stored_as="PARQUET",
        location="oss://%s:%s@%s/%s/%s/data"
        % (
            oss_opts["access_id"],
            oss_opts["secret_access_key"],
            oss_opts["endpoint"].split("://")[1],
            oss_opts["bucket_name"],
            oss_path,
        ),
    )
    if partitions:
        for partition in tensor.keys():
            ext_table.create_partition(partition)
    odps.create_table(table_name, target_schema, if_not_exists=True)
    ext_df = ext_table.to_df()
    fields = [
        ext_df["global_" + f].rename(f) for f in target_schema.names[:-1]
    ] + target_schema.names[-1:]
    if partitions:
        fields = fields + partitions
        ext_df[fields].persist(table_name, partitions=partitions)
    else:
        ext_df[fields].persist(table_name)


CollectionExpr.to_mars_tensor_via_oss = to_mars_tensor_via_oss
