#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

import copy
import csv
import io
import itertools
import math
from collections import OrderedDict
from collections.abc import Iterator

from requests import Response

from . import options, types, utils
from .models.record import Record


def _split_at_depth(s, delimiter, max_splits=None):
    """
    Split a string at the given delimiter, but only when at bracket depth 0
    and not inside quotes.
    """
    depth = 0
    in_quote = None  # None, '"', or "'"
    escape = False
    start = 0
    splits = 0

    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_quote:
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ('"', "'"):
            in_quote = ch
            continue
        if ch in ("{", "["):
            depth += 1
            continue
        if ch in ("}", "]"):
            depth -= 1
            continue
        if ch == delimiter and depth == 0:
            yield s[start:i]
            start = i + 1
            splits += 1
            if max_splits is not None and splits >= max_splits:
                break
    yield s[start:]


_COMPOSITE_TYPES = (types.Array, types.Map, types.Struct)


def _is_composite_type(data_type):
    return isinstance(data_type, _COMPOSITE_TYPES)


def _unquote_value(s):
    """Strip surrounding quotes from a value and unescape inner content."""
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        inner = s[1:-1]
    else:
        inner = s
    # unescape: \" -> " and \' -> '
    return inner.replace('\\"', '"').replace("\\'", "'")


def parse_complex_value(value_str, data_type, keep_strings=False):
    """
    Parse a string representation of a complex ODPS type value into a Python object.

    Supports nested Map, Array, and Struct types with correct handling of
    bracket depth and quoting.
    """
    if isinstance(data_type, types.Array):
        return _parse_array(value_str, data_type, keep_strings=keep_strings)
    elif isinstance(data_type, types.Map):
        return _parse_map(value_str, data_type, keep_strings=keep_strings)
    elif isinstance(data_type, types.Struct):
        return _parse_struct(value_str, data_type, keep_strings=keep_strings)
    else:
        raise ValueError(f"Unsupported complex type: {data_type}")


def _cast_primitive(part, data_type, keep_strings):
    """Cast a primitive value string based on the keep_strings mode."""
    part = _unquote_value(part)
    if keep_strings:
        return data_type.cast_value(part, types.string)
    else:
        return types.validate_value(part, data_type)


def _parse_array(value_str, data_type, keep_strings=False):
    if not (value_str.startswith("[") and value_str.endswith("]")):
        raise ValueError(f"Array format error: {value_str}")
    inner = value_str[1:-1].strip()
    if not inner:
        return []
    element_type = data_type.value_type
    items = []
    for part in _split_at_depth(inner, ","):
        part = part.strip()
        if part == CsvRecordReader.NULL_TOKEN:
            items.append(None)
        elif _is_composite_type(element_type):
            items.append(
                parse_complex_value(part, element_type, keep_strings=keep_strings)
            )
        else:
            items.append(_cast_primitive(part, element_type, keep_strings))
    return items


def _parse_map(value_str, data_type, keep_strings=False):
    if not (value_str.startswith("{") and value_str.endswith("}")):
        raise ValueError(f"Map format error: {value_str}")
    inner = value_str[1:-1].strip()
    if not inner:
        return OrderedDict()
    key_type = data_type.key_type
    value_type = data_type.value_type
    items = []
    for entry in _split_at_depth(inner, ","):
        entry = entry.strip()
        kv_parts = list(_split_at_depth(entry, ":", max_splits=1))
        if len(kv_parts) != 2:
            raise ValueError(f"Map entry format error: {entry}")
        k_str, v_str = kv_parts[0].strip(), kv_parts[1].strip()

        if _is_composite_type(key_type):
            k = parse_complex_value(k_str, key_type, keep_strings=keep_strings)
        else:
            k = _cast_primitive(k_str, key_type, keep_strings)

        if v_str == CsvRecordReader.NULL_TOKEN:
            v = None
        elif _is_composite_type(value_type):
            v = parse_complex_value(v_str, value_type, keep_strings=keep_strings)
        else:
            v = _cast_primitive(v_str, value_type, keep_strings)

        items.append((k, v))
    return OrderedDict(items)


def _parse_struct(value_str, data_type, keep_strings=False):
    if not (value_str.startswith("{") and value_str.endswith("}")):
        raise ValueError(f"Struct format error: {value_str}")
    inner = value_str[1:-1].strip()
    if not inner:
        values = [None] * len(data_type.field_types)
    else:
        values = []
        field_type_iter = iter(data_type.field_types.values())
        for part in _split_at_depth(inner, ","):
            part = part.strip()
            # split field_name:value on first colon at depth 0
            fv_parts = list(_split_at_depth(part, ":", max_splits=1))
            if len(fv_parts) == 2:
                v_str = fv_parts[1].strip()
            else:
                # no colon found, treat entire part as value
                v_str = part

            field_type = next(field_type_iter)

            if v_str == CsvRecordReader.NULL_TOKEN:
                values.append(None)
            elif _is_composite_type(field_type):
                values.append(
                    parse_complex_value(v_str, field_type, keep_strings=keep_strings)
                )
            else:
                values.append(_cast_primitive(v_str, field_type, keep_strings))
    if data_type._struct_as_dict:
        dict_hook = OrderedDict if data_type._use_ordered_dict else dict
        return dict_hook(zip(data_type.field_types.keys(), values))
    return data_type.namedtuple_type(*values)


class AbstractRecordReader(object):
    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def __del__(self):
        if hasattr(self, "close"):
            self.close()

    next = __next__

    @classmethod
    def _calc_count(cls, start, end, step):
        if end is None:
            return end
        step = step or 1
        return int(math.ceil(float(end - start) / step))

    @classmethod
    def _get_slice(cls, item):
        if isinstance(item, int):
            start = item
            end = start + 1
            step = 1
        elif isinstance(item, slice):
            start = item.start or 0
            end = item.stop
            step = item.step or 1
        else:
            raise ValueError("Reader only supports index and slice operation.")

        return start, end, step

    def __getitem__(self, item):
        start, end, step = self._get_slice(item)
        count = self._calc_count(start, end, step)

        if start < 0 or (count is not None and count <= 0) or step < 0:
            raise ValueError("start, count, or step cannot be negative")

        it = self._get_slice_iter(start=start, end=end, step=step)
        if isinstance(item, int):
            try:
                return next(it)
            except StopIteration:
                raise IndexError(f"Index out of range: {item}")
        return it

    def _get_slice_iter(self, start=None, end=None, step=None):
        class SliceIterator(Iterator):
            def __init__(self, it):
                self.it = it

            def __iter__(self):
                return self.it

            def __next__(self):
                return next(self.it)

            @staticmethod
            def to_pandas():
                if end is not None:
                    count = (end - (start or 0)) // (step or 1)
                else:
                    count = None
                pstep = None if step == 1 else step
                kw = dict(start=start, count=count, step=pstep)
                kw = {k: v for k, v in kw.items() if v is not None}
                return parent.to_pandas(**kw)

        parent = self
        return SliceIterator(self._iter(start=start, end=end, step=step))

    def _iter(self, start=None, end=None, step=None):
        start = start or 0
        step = step or 1
        curr = start

        for _ in range(start):
            try:
                next(self)
            except StopIteration:
                return

        while True:
            for i in range(step):
                try:
                    record = next(self)
                except StopIteration:
                    return
                if i == 0:
                    yield record
                curr += 1
                if end is not None and curr >= end:
                    return

    def _data_to_result_frame(
        self, data, unknown_as_string=True, as_type=None, columns=None
    ):
        from .df.backends.frame import ResultFrame
        from .df.backends.odpssql.types import (
            odps_schema_to_df_schema,
            odps_type_to_df_type,
        )

        kw = dict()
        if getattr(self, "schema", None) is not None:
            kw["schema"] = odps_schema_to_df_schema(self.schema)
        elif getattr(self, "_schema", None) is not None:
            # do not remove as there might be coverage missing
            kw["schema"] = odps_schema_to_df_schema(self._schema)

        column_names = columns or getattr(self, "_column_names", None)
        if column_names is not None:
            self._columns = [self.schema[c] for c in column_names]
        if getattr(self, "_columns", None) is not None:
            cols = []
            for col in self._columns:
                col = copy.copy(col)
                col.type = odps_type_to_df_type(col.type)
                cols.append(col)
            kw["columns"] = cols

        if hasattr(self, "raw"):
            try:
                import pandas as pd

                from .df.backends.pd.types import pd_to_df_schema

                data = pd.read_csv(io.StringIO(self.raw))
                schema = kw["schema"] = pd_to_df_schema(
                    data, unknown_as_string=unknown_as_string, as_type=as_type
                )
                columns = kw.pop("columns", None)
                if columns and len(columns) < len(schema):
                    sel_cols = [c.name for c in self._columns]
                    data = data[sel_cols]
                    kw["schema"] = types.OdpsSchema(columns)
            except (ImportError, ValueError):
                pass

        if not kw:
            raise ValueError(
                f"Cannot convert to ResultFrame from {type(self).__name__}."
            )

        return ResultFrame(data, **kw)

    def to_result_frame(
        self,
        unknown_as_string=True,
        as_type=None,
        start=None,
        count=None,
        columns=None,
        **iter_kw
    ):
        read_row_batch_size = options.tunnel.read_row_batch_size
        if "end" in iter_kw:
            end = iter_kw["end"]
        else:
            end = (
                None
                if count is None
                else (start or 0) + count * (iter_kw.get("step") or 1)
            )

        frames = []
        if hasattr(self, "raw"):
            # data represented as raw csv: just skip iteration
            data = [r for r in self._iter(start=start, end=end, **iter_kw)]
        else:
            offset_iter = itertools.cycle(range(read_row_batch_size))
            data = [None] * read_row_batch_size
            for offset, rec in zip(
                offset_iter, self._iter(start=start, end=end, **iter_kw)
            ):
                data[offset] = rec
                if offset != read_row_batch_size - 1:
                    continue

                frames.append(
                    self._data_to_result_frame(
                        data, unknown_as_string=unknown_as_string, as_type=as_type
                    )
                )
                data = [None] * read_row_batch_size
                if len(frames) > options.tunnel.batch_merge_threshold:
                    frames = [frames[0].concat(*frames[1:])]

        if not frames or data[0] is not None:
            data = list(itertools.takewhile(lambda x: x is not None, data))
            frames.append(
                self._data_to_result_frame(
                    data,
                    unknown_as_string=unknown_as_string,
                    as_type=as_type,
                    columns=columns,
                )
            )
        return frames[0].concat(*frames[1:])

    def to_pandas(self, start=None, count=None, **kw):
        import pandas  # noqa: F401

        return self.to_result_frame(start=start, count=count, **kw).values


class CsvRecordReader(AbstractRecordReader):
    NULL_TOKEN = "\\N"
    BACK_SLASH_ESCAPE = "\\x%02x" % ord("\\")
    UNSUPPORTED_TYPES = (
        types.Binary,
        types.Blob,
        types.Vector,
    )

    def __init__(self, schema, stream, **kwargs):
        # shift csv field limit size to match table field size
        max_field_size = kwargs.pop("max_field_size", 0) or types.String._max_length
        if csv.field_size_limit() < max_field_size:
            csv.field_size_limit(max_field_size)

        self._schema = schema
        self._csv_columns = None
        self._fp = stream
        if isinstance(self._fp, Response):
            self.raw = self._fp.text
        else:
            self.raw = self._fp

        if options.tunnel.string_as_binary:
            self._csv = csv.reader(io.StringIO(self._escape_csv_bin(self.raw)))
        else:
            self._csv = csv.reader(io.StringIO(self._escape_csv(self.raw)))

        self._filtered_col_names = (
            set(x.lower() for x in kwargs["columns"]) if "columns" in kwargs else None
        )
        self._columns = None
        self._filtered_col_idxes = None

    @classmethod
    def _escape_csv(cls, s):
        escaped = utils.to_text(s).encode("unicode_escape")
        # Make invisible chars available to `csv` library.
        # Note that '\n' and '\r' should be unescaped.
        # '\\' should be replaced with '\x5c' before unescaping
        # to avoid mis-escaped strings like '\\n'.
        return (
            utils.to_text(escaped)
            .replace("\\\\", cls.BACK_SLASH_ESCAPE)
            .replace("\\n", "\n")
            .replace("\\r", "\r")
        )

    @classmethod
    def _escape_csv_bin(cls, s):
        escaped = utils.to_binary(s).decode("latin1").encode("unicode_escape")
        # Make invisible chars available to `csv` library.
        # Note that '\n' and '\r' should be unescaped.
        # '\\' should be replaced with '\x5c' before unescaping
        # to avoid mis-escaped strings like '\\n'.
        return (
            utils.to_text(escaped)
            .replace("\\\\", cls.BACK_SLASH_ESCAPE)
            .replace("\\n", "\n")
            .replace("\\r", "\r")
        )

    @staticmethod
    def _unescape_csv(s):
        return s.encode("utf-8").decode("unicode_escape")

    @staticmethod
    def _unescape_csv_bin(s):
        return s.encode("utf-8").decode("unicode_escape").encode("latin1")

    def _readline(self):
        try:
            values = next(self._csv)
        except StopIteration:
            return

        read_binary = options.tunnel.string_as_binary
        unescape_csv = self._unescape_csv_bin if read_binary else self._unescape_csv
        cast_value = (
            self._cast_value_legacy
            if options.legacy_cast_csv_result
            else self._cast_value
        )
        return [cast_value(i, unescape_csv(v)) for i, v in enumerate(values)]

    @classmethod
    def _contains_unsupported_type(cls, col_type):
        """Check whether a type (or any of its nested element types) is unsupported."""
        if isinstance(col_type, cls.UNSUPPORTED_TYPES):
            return True
        if isinstance(col_type, types.Array):
            return cls._contains_unsupported_type(col_type.value_type)
        if isinstance(col_type, types.Map):
            return cls._contains_unsupported_type(
                col_type.key_type
            ) or cls._contains_unsupported_type(col_type.value_type)
        if isinstance(col_type, types.Struct):
            return any(
                cls._contains_unsupported_type(ft)
                for ft in col_type.field_types.values()
            )
        return False

    @classmethod
    def _get_caster(cls, col_type):
        if col_type == types.boolean:
            return utils.str_to_bool
        elif cls._contains_unsupported_type(col_type):
            return None
        elif isinstance(col_type, (types.Array, types.Map, types.Struct)):
            return lambda v: parse_complex_value(v, col_type, keep_strings=False)
        return lambda v: types.validate_value(v, col_type)

    def _cast_value(self, idx, value):
        if value == self.NULL_TOKEN:
            return None
        if self._csv_columns:
            caster = self._get_caster(self._csv_columns[idx].type)
            if caster is not None:
                return caster(value)
        return value

    def _cast_value_legacy(self, idx, value):
        if value == self.NULL_TOKEN:
            return None
        col = self._csv_columns[idx] if self._csv_columns else None
        if col and col.type == types.boolean:
            if value == "true":
                return True
            if value == "false":
                return False
            return value
        if col and isinstance(col.type, (types.Array, types.Map, types.Struct)):
            return parse_complex_value(value, col.type, keep_strings=True)
        return value

    def __next__(self):
        self._load_columns()

        values = self._readline()
        if not values:
            raise StopIteration

        if self._filtered_col_idxes:
            values = [values[idx] for idx in self._filtered_col_idxes]
        return Record(self._columns, values=values)

    next = __next__

    def read(self, start=None, count=None, step=None):
        if count is None:
            end = None
        else:
            start = start or 0
            step = step or 1
            end = start + count * step
        return self._iter(start=start, end=end, step=step)

    def _load_columns(self):
        if self._csv_columns is not None:
            return

        values = self._readline()
        self._csv_columns = []
        for value in values:
            if self._schema is None:
                self._csv_columns.append(types.Column(name=value, typo="string"))
            else:
                if self._schema.is_partition(value):
                    self._csv_columns.append(self._schema.get_partition(value))
                else:
                    self._csv_columns.append(self._schema.get_column(value))

        if self._csv_columns is not None and self._filtered_col_names:
            self._filtered_col_idxes = []
            self._columns = []
            for idx, col in enumerate(self._csv_columns):
                if col.name.lower() in self._filtered_col_names:
                    self._filtered_col_idxes.append(idx)
                    self._columns.append(col)
        else:
            self._columns = self._csv_columns

    def to_pandas(self, start=None, count=None, **kw):
        kw.pop("n_process", None)
        return super(CsvRecordReader, self).to_pandas(start=start, count=count, **kw)

    def close(self):
        if hasattr(self._fp, "close"):
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# make class name compatible
RecordReader = CsvRecordReader
