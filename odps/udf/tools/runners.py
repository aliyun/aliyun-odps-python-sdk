# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

"""UDF runners implementing the local run framework."""

import csv
import re
import sys

from ... import distcache
from ... import types as odps_types
from ... import udf
from ...utils import to_date, to_milliseconds
from . import utils

__all__ = ["get_csv_runner", "get_table_runner"]

PY2 = sys.version_info[0] == 2
_table_bracket_re = re.compile(r"[^\(]+\([^\)]+\)")


def get_csv_runner(
    udf_class,
    input_col_delim=",",
    null_indicator="NULL",
    stdin=None,
    collector_cls=None,
):
    """Create a runner to read csv with specified udf class."""
    proto = udf.get_annotation(udf_class)
    in_types, out_types = parse_proto(proto)
    stdin = stdin or sys.stdin
    arg_parser = ArgParser(in_types, stdin, input_col_delim, null_indicator)
    stdin_feed = arg_parser.parse()

    collector_cls = collector_cls or StdoutCollector
    collector = collector_cls(out_types)
    ctor = _get_runner_class(udf_class)
    return ctor(udf_class, stdin_feed, collector)


def get_table_runner(
    udf_class, odps_entry, table_desc, record_limit=None, collector_cls=None
):
    """Create a runner to read table with specified udf class."""
    proto = udf.get_annotation(udf_class)
    in_types, out_types = parse_proto(proto)
    tb_feed = table_feed(odps_entry, table_desc, in_types, record_limit)

    collector_cls = collector_cls or StdoutCollector
    collector = collector_cls(out_types)
    ctor = _get_runner_class(udf_class)
    return ctor(udf_class, tb_feed, collector)


def simple_run(udf_class, args):
    proto = udf.get_annotation(udf_class)
    in_types, out_types = parse_proto(proto)
    feed = direct_feed(args)
    collector = DirectCollector(out_types)
    ctor = _get_runner_class(udf_class)
    runner = ctor(udf_class, feed, collector)
    runner.run()
    return collector.results


def initialize():
    """Initialize the local run environment."""
    distcache.get_cache_table = utils.get_cache_table


def _split_data_types(types_str):
    bracket_level = 0
    ret_types = [""]
    for ch in types_str:
        if bracket_level == 0 and ch == ",":
            ret_types[-1] = ret_types[-1].strip()
            ret_types.append("")
        else:
            ret_types[-1] += ch
            if ch in ("<", "("):
                bracket_level += 1
            elif ch in (">", ")"):
                bracket_level -= 1
    return [s for s in ret_types if s]


def _get_types(types_str):
    entries = []
    for t in _split_data_types(types_str):
        t = t.strip()
        entries.append(odps_types.validate_data_type(t))
    return entries


def _get_in_types(types):
    if types == "":
        return []
    return _get_types(types) if types != "*" else ["*"]


def _get_runner_class(udf_class):
    if udf.BaseUDAF in udf_class.__mro__:
        ctor = UDAFRunner
    elif udf.BaseUDTF in udf_class.__mro__:
        ctor = UDTFRunner
    else:
        ctor = UDFRunner
    return ctor


def parse_proto(proto):
    tokens = proto.lower().split("->")
    if len(tokens) != 2:
        raise ValueError("Illegal format of @annotate(%s)" % proto)
    return _get_in_types(tokens[0].strip()), _get_types(tokens[1].strip())


def direct_feed(args):
    for a in args:
        yield a


def _convert_value(value, tp):
    try:
        odps_types._date_allow_int_conversion = True
        value = odps_types.validate_value(value, tp)
    finally:
        odps_types._date_allow_int_conversion = False

    if not PY2:
        return value

    if isinstance(tp, odps_types.Datetime):
        return to_milliseconds(value)
    elif isinstance(tp, odps_types.Date):
        return to_date(value)
    elif isinstance(tp, odps_types.Array):
        return [_convert_value(v, tp.value_type) for v in value]
    elif isinstance(tp, odps_types.Map):
        return {
            _convert_value(k, tp.key_type): _convert_value(v, tp.value_type)
            for k, v in value.items()
        }
    elif isinstance(tp, odps_types.Struct):
        if isinstance(value, dict):
            vals = {
                k: _convert_value(value[k], ftp) for k, ftp in tp.field_types.items()
            }
        else:
            vals = {
                k: _convert_value(getattr(value, k), ftp)
                for k, ftp in tp.field_types.items()
            }
        return tp.namedtuple_type(**vals)
    else:
        return value


def _validate_values(values, types):
    if types == ["*"]:
        return values
    if len(values) != len(types):
        raise ValueError(
            "Input length mismatch: %d expected, %d provided"
            % (len(types), len(values))
        )
    ret_vals = [None] * len(values)
    for idx, (tp, d) in enumerate(zip(types, values)):
        if d is None:
            continue
        try:
            ret_vals[idx] = _convert_value(d, tp)
        except:
            raise ValueError("Input type mismatch: expected %s, received %r" % (tp, d))
    return ret_vals


class ArgParser(object):
    NULL_INDICATOR = "NULL"

    def __init__(self, types, fileobj, delim=",", null_indicator="NULL"):
        self.types = types
        self.delim = delim
        self.null_indicator = null_indicator

        self.reader = csv.reader(fileobj, delimiter=delim)

    def parse(self):
        for record in self.reader:
            tokens = []
            for token in record:
                if token == self.null_indicator:
                    tokens.append(None)
                else:
                    tokens.append(token)

            if len(self.types) == 0 and len(tokens) == 0:
                yield ""
                continue
            yield _validate_values(tokens, self.types)


def _get_table_or_partition(odps_entry, table_desc):
    table_names = []
    table_part = None
    table_cols = None
    for part in table_desc.split("."):
        part = part.strip()
        if not _table_bracket_re.match(part):
            table_names.append(part)
        elif part.startswith("p("):
            table_part = part[2:-1]
        elif part.startswith("c("):
            table_cols = [s.strip() for s in part[2:-1].split(",")]
    data_obj = odps_entry.get_table(".".join(table_names))
    if table_part is not None:
        data_obj = data_obj.get_partition(table_part)
    return data_obj, table_cols


def table_feed(odps_entry, table_desc, in_types, record_limit):
    data_obj, cols = _get_table_or_partition(odps_entry, table_desc)
    with data_obj.open_reader(columns=cols) as reader:
        if record_limit is not None:
            data_src = reader[:record_limit]
        else:
            data_src = reader

        for row in data_src:
            yield _validate_values(row.values, in_types)


class ArgFormatter(object):
    DELIM = "\t"
    NULL_INDICATOR = "NULL"

    def __init__(self, types):
        self.types = types

    def format(self, *args):
        ret = self.DELIM.join([str(a) for a in args])
        return ret


class BaseCollector(object):
    """Basic common logic of collector."""

    def __init__(self, schema):
        self.schema = schema

    def collect(self, *args):
        _validate_values(args, self.schema)
        self.handle_collect(*args)

    def handle_collect(self, *args):
        raise NotImplementedError


class StdoutCollector(BaseCollector):
    """Collect the results to stdout."""

    def __init__(self, schema):
        super(StdoutCollector, self).__init__(schema)
        self.formatter = ArgFormatter(schema)

    def handle_collect(self, *args):
        print(self.formatter.format(*args))


class DirectCollector(BaseCollector):
    """Collect results which can be fetched via self.results into memory."""

    def __init__(self, schema):
        super(DirectCollector, self).__init__(schema)
        self.results = []

    def handle_collect(self, *args):
        if len(self.schema) == 1:
            self.results.append(args[0])
        else:
            self.results.append(args)


class BaseRunner(object):
    def __init__(self, udf_class, feed, collector):
        self.udf_class = udf_class
        self.feed = feed
        self.collector = collector
        # check signature
        self.obj = udf_class()


class UDFRunner(BaseRunner):
    def run(self):
        obj = self.obj
        collector = self.collector
        for args in self.feed:
            r = obj.evaluate(*args)
            collector.collect(r)


class UDTFRunner(BaseRunner):
    def run(self):
        obj = self.obj
        collector = self.collector

        def local_forward(*r):
            collector.collect(*r)

        obj.forward = local_forward
        for args in self.feed:
            obj.process(*args)
        obj.close()


class UDAFRunner(BaseRunner):
    def run(self):
        obj = self.obj
        collector = self.collector
        buf0 = obj.new_buffer()
        buf1 = obj.new_buffer()
        turn = True
        for args in self.feed:
            if turn:
                buf = buf0
                turn = False
            else:
                buf = buf1
                turn = True
            obj.iterate(buf, *args)
        merge_buf = obj.new_buffer()
        obj.merge(merge_buf, buf0)
        obj.merge(merge_buf, buf1)
        collector.collect(obj.terminate(merge_buf))


initialize()
