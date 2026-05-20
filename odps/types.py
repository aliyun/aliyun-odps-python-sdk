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

import decimal
import io
import json as _json
import math
import warnings
from collections import OrderedDict
from datetime import date as _date
from datetime import datetime as _datetime
from datetime import timedelta as _timedelta

from . import utils
from .compat import Monthdelta, east_asian_len
from .config import options
from .lib.xnamedtuple import xnamedtuple

try:
    import numpy as np
except ImportError:
    np = None
try:
    from pandas import NA as _pd_na

    pd_na_type = type(_pd_na)
except (ImportError, ValueError):
    pd_na_type = None

force_py = options.force_py
force_c = options.force_c

_date_allow_int_conversion = False


class Column:
    """
    Represents a column in a table schema.

    :param str name: column name
    :param str typo: column type. Can also use `type` as keyword.
    :param str comment: comment of the column, None by default
    :param bool nullable: is column nullable, True by default

    :Example:

    >>> col = Column("col1", "bigint")
    >>> print(col.name)
    col1
    >>> print(col.type)
    bigint
    """

    def __init__(
        self,
        name=None,
        typo=None,
        comment=None,
        label=None,
        nullable=True,
        generate_expression=None,
        **kw
    ):
        self.name = utils.to_str(name)
        self.type = validate_data_type(
            typo if typo is not None else kw.pop("type", None)
        )
        self.comment = comment
        if label:
            warnings.warn("label is deprecated.", DeprecationWarning)
        self.label = label
        self.nullable = nullable

        self._generate_expression = generate_expression
        self._parsed_generate_expression = None

        if kw:
            raise TypeError(f"Arguments not supported for Column: {list(kw)}")

    def __repr__(self):
        not_null_str = ", not null" if not self.nullable else ""
        return f"<column {utils.to_str(self.name)}, type {self.type.name.lower()}{not_null_str}>"

    def __hash__(self):
        return hash(
            (
                type(self),
                self.name,
                self.type,
                self.comment,
                self.label,
                self.nullable,
                self._generate_expression,
            )
        )

    def __eq__(self, other):
        return self is other or all(
            getattr(self, attr, None) == getattr(other, attr, None)
            for attr in (
                "name",
                "type",
                "comment",
                "label",
                "nullable",
                "_generate_expression",
            )
        )

    @property
    def generate_expression(self):
        from .expressions import parse as parse_expression

        if not self._generate_expression:
            return None
        if not self._parsed_generate_expression:
            try:
                self._parsed_generate_expression = parse_expression(
                    self._generate_expression
                )
            except (SyntaxError, ValueError):
                self._parsed_generate_expression = self._generate_expression
        return self._parsed_generate_expression

    def to_sql_clause(self, with_column_comments=True):
        sio = io.StringIO()
        if self.generate_expression:
            sio.write(
                f"  {utils.to_text(self.generate_expression)} AS {utils.backquote_string(self.name)}"
            )
        else:
            sio.write(
                f"  {utils.to_text(utils.backquote_string(self.name))} {utils.to_text(self.type)}"
            )
            if not self.nullable and not options.sql.ignore_fields_not_null:
                sio.write(" NOT NULL")
        if with_column_comments and self.comment:
            comment_str = utils.escape_odps_string(utils.to_text(self.comment))
            sio.write(f" COMMENT '{comment_str}'")
        return sio.getvalue()

    def replace(
        self,
        name=None,
        type=None,
        comment=None,
        label=None,
        nullable=None,
        generate_expression=None,
    ):
        return Column(
            name=name or self.name,
            typo=type or self.type,
            comment=comment or self.comment,
            label=label or self.label,
            nullable=nullable or self.nullable,
            generate_expression=generate_expression or self._generate_expression,
        )


class Partition(Column):
    """
    Represents a partition column in a table schema.

    :param str name: column name
    :param str typo: column type. Can also use `type` as keyword.
    :param str comment: comment of the column, None by default
    :param bool nullable: is column nullable, True by default

    :Example:

    >>> col = Partition("col1", "bigint")
    >>> print(col.name)
    col1
    >>> print(col.type)
    bigint
    """

    def __repr__(self):
        return f"<partition {utils.to_str(self.name)}, type {self.type.name.lower()}>"


class _CallableList(list):
    """Make sure keys and values properties also callable"""

    def __call__(self):
        return self


class PartitionSpec:
    def __init__(self, spec=None):
        self.kv = OrderedDict()

        if isinstance(spec, PartitionSpec):
            self.kv = spec.kv.copy()
        elif isinstance(spec, dict):
            self.kv = OrderedDict(spec)
        elif isinstance(spec, str):
            splits = spec.split(",")
            for sp in splits:
                kv = sp.split("=")
                if len(kv) != 2:
                    raise ValueError(
                        "Invalid partition spec: a partition spec should "
                        'look like "part1=v1,part2=v2"'
                    )

                k, v = kv[0].strip(), kv[1].strip().strip("'\"")

                if len(k) == 0 or len(v) == 0:
                    raise ValueError("Invalid partition spec")
                if k in self.kv:
                    raise ValueError(
                        "Invalid partition spec: found duplicate partition key " + k
                    )

                self.kv[k] = v
        elif spec is not None:
            raise TypeError(f"Cannot accept spec {spec!r}")

    def __setitem__(self, key, value):
        self.kv[key] = value

    def __getitem__(self, key):
        return self.kv[key]

    def __len__(self):
        return len(self.kv)

    @property
    def is_empty(self):
        return len(self) == 0

    @property
    def keys(self):
        return _CallableList(self.kv.keys())

    @property
    def values(self):
        return _CallableList(self.kv.values())

    def items(self):
        for k, v in self.kv.items():
            yield k, v

    def __contains__(self, key):
        return key in self.kv

    def __str__(self):
        return ",".join(f"{k}='{v}'" for k, v in self.kv.items())

    def __repr__(self):
        return f"<PartitionSpec {self}>"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, PartitionSpec):
            other = PartitionSpec(other)

        return str(self) == str(other)


class Schema:
    def __init__(self, names, types):
        self._init(names, types)

    def _init(self, names, types):
        if not isinstance(names, list):
            names = list(names)
        self.names = names
        self.types = [validate_data_type(t) for t in types]

        lower_names = [utils.to_lower_str(n) for n in self.names]
        self._name_indexes = {n: i for i, n in enumerate(lower_names)}

        if len(self._name_indexes) < len(self.names):
            duplicates = [n for n in self._name_indexes if lower_names.count(n) > 1]
            raise ValueError(f"Duplicate column names: {', '.join(duplicates)}")

        self._snapshot = None

    def __repr__(self):
        return self._repr()

    def __len__(self):
        return len(self.names)

    def __contains__(self, name):
        return utils.to_lower_str(name) in self._name_indexes

    def _repr(self):
        buf = io.StringIO()
        names = [self._to_printable(n) for n in self.names]
        space = 2 * max(len(it) for it in names)
        for name, tp in zip(names, self.types):
            buf.write(f"\n{name.ljust(space)}{repr(tp)}")

        return f"Schema {{{utils.indent(buf.getvalue(), 2)}\n}}"

    def __hash__(self):
        return hash((type(self), tuple(self.names), tuple(self.types)))

    def __eq__(self, other):
        if not isinstance(other, Schema):
            return False
        return self.names == other.names and self.types == other.types

    def get_type(self, name):
        return self.types[self._name_indexes[utils.to_lower_str(name)]]

    def append(self, name, typo):
        names = self.names + [name]
        types = self.types + [validate_data_type(typo)]
        return Schema(names, types)

    def extend(self, schema):
        names = self.names + schema.names
        types = self.types + schema.types
        return Schema(names, types)


class OdpsSchema(Schema):
    def __init__(self, columns=None, partitions=None):
        self._columns = columns
        self._partitions = partitions

        if self._columns:
            super(OdpsSchema, self).__init__(
                *list(zip(*[(c.name, c.type) for c in self._columns]))
            )
        else:
            super(OdpsSchema, self).__init__([], [])

        if self._partitions:
            self._partition_schema = Schema(
                *list(zip(*[(c.name, c.type) for c in self._partitions]))
            )
        else:
            self._partition_schema = Schema([], [])

    def __len__(self):
        return super(OdpsSchema, self).__len__() + len(self._partition_schema)

    def __setattr__(self, key, value):
        if (
            key == "_columns"
            and value
            and not getattr(self, "names", None)
            and not getattr(self, "types", None)
        ):
            names = [c.name for c in value]
            types = [c.type for c in value]
            self._init(names, types)
        elif key == "_partitions" and value:
            self._partition_schema = Schema(
                *list(zip(*[(c.name, c.type) for c in value]))
            )
        object.__setattr__(self, key, value)

    def __contains__(self, name):
        return (
            super(OdpsSchema, self).__contains__(name)
            or utils.to_str(name) in self._partition_schema
        )

    def __eq__(self, other):
        if not isinstance(other, OdpsSchema):
            return False

        return (
            super(OdpsSchema, self).__eq__(other)
            and self._partition_schema == other._partition_schema
        )

    def __hash__(self):
        return hash(
            (type(self), tuple(self.names), tuple(self.types), self._partition_schema)
        )

    def __getitem__(self, item):
        if isinstance(item, int):
            n_columns = len(self._name_indexes)
            if item < n_columns:
                return self._columns[item]
            elif item < len(self):
                return self._partitions[item - n_columns]
            else:
                raise IndexError("Index out of range")
        elif isinstance(item, str):
            lower_item = utils.to_lower_str(item)
            if lower_item in self._name_indexes:
                idx = self._name_indexes[lower_item]
                return self[idx]
            elif item in self._partition_schema:
                idx = self._partition_schema._name_indexes[lower_item]
                n_columns = len(self._name_indexes)
                return self[n_columns + idx]
            else:
                raise ValueError(f"Unknown column name: {item}")
        elif isinstance(item, (list, tuple)):
            return [self[it] for it in item]
        else:
            return self.columns[item]

    def _repr(self, strip=True):
        def _strip(line):
            return line.rstrip() if strip else line

        buf = io.StringIO()

        name_dict = dict(
            [(col.name, utils.str_to_printable(col.name)) for col in self.columns]
        )
        name_display_lens = dict(
            [
                (k, east_asian_len(utils.to_text(v), encoding=options.display.encoding))
                for k, v in name_dict.items()
            ]
        )
        max_name_len = max(name_display_lens.values())
        name_space = max_name_len + min(16, max_name_len)
        max_type_len = max(len(repr(col.type)) for col in self.columns)
        type_space = max_type_len + min(16, max_type_len)
        has_not_null = any(not col.nullable for col in self.columns)

        not_empty = lambda field: field is not None and len(field.strip()) > 0

        buf.write("odps.Schema {\n")
        cols_strs = []
        for col in self._columns:
            pad_spaces = name_space - name_display_lens[col.name]
            not_null = "not null" if not col.nullable else " " * 8
            row = (
                f"{utils.to_str(name_dict[col.name] + ' ' * pad_spaces)}"
                f"{repr(col.type).ljust(type_space)}"
                f"{not_null + ' ' * 4 if has_not_null else ''}"
                f"{'# ' + utils.to_str(col.comment) if not_empty(col.comment) else ''}"
            )
            cols_strs.append(_strip(row))
        buf.write(utils.indent("\n".join(cols_strs), 2))
        buf.write("\n")
        buf.write("}\n")

        if self._partitions:
            buf.write("Partitions {\n")

            partition_strs = []
            for partition in self._partitions:
                row = (
                    f"{utils.to_str(name_dict[partition.name].ljust(name_space))}"
                    f"{repr(partition.type).ljust(type_space)}"
                    f"{'# ' + utils.to_str(partition.comment) if not_empty(partition.comment) else ''}"
                )
                partition_strs.append(_strip(row))
            buf.write(utils.indent("\n".join(partition_strs), 2))
            buf.write("\n")
            buf.write("}\n")

        return buf.getvalue()

    def build_snapshot(self):
        if not options.force_py:
            if not self._columns:
                return None

            try:
                from .src.types_c import SchemaSnapshot

                self._snapshot = SchemaSnapshot(self)
            except ImportError:
                pass
        return self._snapshot

    @property
    def simple_columns(self):
        """
        List of columns as a list of :class:`~odps.types.Column`.
        Partition columns are excluded.
        """
        return self._columns

    @property
    def columns(self):
        """List of columns and partition columns as a list of :class:`~odps.types.Column`."""
        partitions = self._partitions or []
        return self._columns + partitions

    @property
    def partitions(self):
        """List of partition columns as a list of :class:`~odps.types.Partition`."""
        try:
            return self._partitions
        except AttributeError:
            return []

    @utils.deprecated("use simple_columns property instead")
    def get_columns(self):
        return self._columns

    @utils.deprecated("use partitions property instead")
    def get_partitions(self):
        return self._partitions

    def get_column(self, name):
        index = self._name_indexes.get(utils.to_lower_str(name))
        if index is None:
            raise ValueError(f"Column {name} does not exists")
        return self._columns[index]

    def get_partition(self, name):
        index = self._partition_schema._name_indexes.get(utils.to_lower_str(name))
        if index is None:
            raise ValueError(f"Partition {name} does not exists")
        return self._partitions[index]

    def is_partition(self, name):
        try:
            name = name.name
        except AttributeError:
            pass
        return utils.to_lower_str(name) in self._partition_schema._name_indexes

    def get_type(self, name):
        lower_name = utils.to_lower_str(name)
        if lower_name in self._name_indexes:
            return super(OdpsSchema, self).get_type(name)
        elif lower_name in self._partition_schema:
            return self._partition_schema.get_type(name)
        raise ValueError(f"Column does not exist: {name}")

    def update(self, columns, partitions):
        self._columns = columns
        self._partitions = partitions

        names = map(lambda c: c.name, self._columns)
        types = map(lambda c: c.type, self._columns)

        self._init(names, types)
        if self._partitions:
            self._partition_schema = Schema(
                *list(zip(*[(c.name, c.type) for c in self._partitions]))
            )
        else:
            self._partition_schema = Schema([], [])

    def extend(self, schema):
        if isinstance(schema, Schema):
            ext_cols = [Column(n, tp) for n, tp in zip(schema.names, schema.types)]
            ext_parts = []
        else:
            ext_cols = schema.simple_columns
            ext_parts = schema.partitions
        return type(self)(
            columns=self.simple_columns + ext_cols,
            partitions=self.partitions + ext_parts,
        )

    def to_ignorecase_schema(self):
        cols = [
            Column(col.name.lower(), col.type, col.comment, col.label)
            for col in self._columns
        ]
        parts = None
        if self._partitions:
            parts = [
                Partition(part.name.lower(), part.type, part.comment, part.label)
                for part in self._partitions
            ]

        return type(self)(columns=cols, partitions=parts)

    @classmethod
    def from_lists(cls, names, types, partition_names=None, partition_types=None):
        """
        Create a schema from lists of column names and types.

        :param names: List of column names.
        :param types: List of column types.
        :param partition_names: List of partition names.
        :param partition_types: List of partition types.

        :Example:

        >>> schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'])
        >>> print(schema.columns)
        [<column id, type bigint>, <column name, type string>]
        """
        columns = [Column(name=name, typo=typo) for name, typo in zip(names, types)]
        if partition_names is not None and partition_types is not None:
            partitions = [
                Partition(name=name, typo=typo)
                for name, typo in zip(partition_names, partition_types)
            ]
        else:
            partitions = None
        return cls(columns=columns, partitions=partitions)

    @classmethod
    def from_dict(cls, fields_dict, partitions_dict=None):
        fields = list(fields_dict.keys())
        fields_types = list(fields_dict.values())
        partitions = (
            list(partitions_dict.keys()) if partitions_dict is not None else None
        )
        partitions_types = (
            list(partitions_dict.values()) if partitions_dict is not None else None
        )

        return cls.from_lists(
            fields,
            fields_types,
            partition_names=partitions,
            partition_types=partitions_types,
        )

    def get_table_ddl(self, table_name="table_name", with_comments=True):
        from .models.table import Table

        return Table.gen_create_table_sql(
            table_name, self, with_column_comments=with_comments
        )


class RecordMeta(type):
    record_types = set()

    def __new__(mcs, name, bases, dct):
        inst = super(RecordMeta, mcs).__new__(mcs, name, bases, dct)
        mcs.record_types.add(inst)
        return inst

    def __instancecheck__(cls, instance):
        return isinstance(instance, RecordReprMixin)


def is_record(obj):
    return type(obj) in RecordMeta.record_types


class BaseRecord:
    # set __slots__ to save memory in the situation that records' size may be quite large
    __slots__ = "_values", "_columns", "_name_indexes", "_max_field_size"

    def __init__(self, columns=None, schema=None, values=None, max_field_size=None):
        if isinstance(columns, Schema):
            schema, columns = columns, None
        if columns is not None:
            self._columns = columns
            self._name_indexes = {
                col.name.lower(): i for i, col in enumerate(self._columns)
            }
        else:
            self._columns = schema.columns
            self._name_indexes = schema._name_indexes

        self._max_field_size = max_field_size

        if self._columns is None:
            raise ValueError("Either columns or schema should not be provided")

        self._values = [None] * len(self._columns)
        if values is not None:
            self._sets(values)

    def _mode(self):
        return "py"

    def _exclude_partition_columns(self):
        return [col for col in self._columns if not isinstance(col, Partition)]

    def _get(self, i):
        return self._values[i]

    def _set(self, i, value):
        data_type = self._columns[i].type
        val = validate_value(value, data_type, max_field_size=self._max_field_size)
        self._values[i] = val

    get = _get  # to keep compatible
    set = _set  # to keep compatible

    def _sets(self, values):
        if len(values) != len(self._columns) and len(values) != len(
            self._exclude_partition_columns()
        ):
            raise ValueError(
                "The values set to records are against the schema, "
                f"expect len {len(self._columns)}, got len {len(values)}"
            )
        [self._set(i, value) for i, value in enumerate(values)]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.get_by_name(item)
        elif isinstance(item, (list, tuple)):
            return [self[it] for it in item]
        return self._values[item]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.set_by_name(key, value)
        else:
            self._set(key, value)

    def __getattr__(self, item):
        if item == "_name_indexes":
            return object.__getattribute__(self, item)
        if hasattr(self, "_name_indexes") and item in self._name_indexes:
            return self.get_by_name(item)
        return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        if hasattr(self, "_name_indexes") and key in self._name_indexes:
            self.set_by_name(key, value)
        else:
            object.__setattr__(self, key, value)

    def get_by_name(self, name):
        i = self._name_indexes[utils.to_lower_str(name)]
        return self._values[i]

    def set_by_name(self, name, value):
        i = self._name_indexes[utils.to_lower_str(name)]
        self._set(i, value)

    def __len__(self):
        return len(self._columns)

    def __contains__(self, item):
        return utils.to_lower_str(item) in self._name_indexes

    def __iter__(self):
        for i, col in enumerate(self._columns):
            yield (col.name, self[i])

    @property
    def values(self):
        return self._values

    @property
    def n_columns(self):
        return len(self._columns)

    def get_columns_count(self):  # compatible
        return self.n_columns


class RecordReprMixin:
    def __repr__(self):
        buf = io.StringIO()

        buf.write("odps.Record {\n")

        space = 2 * max(len(it.name) for it in self._columns)
        content = "\n".join(
            [
                f"{col.name.ljust(space)}{value!r}"
                for col, value in zip(self._columns, self._values)
            ]
        )
        buf.write(utils.indent(content, 2))

        buf.write("\n}")

        return buf.getvalue()

    def __hash__(self):
        return hash((type(self), tuple(self._columns), tuple(self._values)))

    def __eq__(self, other):
        if not is_record(other):
            return False

        return self._columns == other._columns and self._values == other._values


class Record(RecordReprMixin, BaseRecord, metaclass=RecordMeta):
    """
    A record generally means the data of a single line in a table. It can be
    created from a schema, or by :meth:`odps.models.Table.new_record` or by
    :meth:`odps.tunnel.TableUploadSession.new_record`.

    Hints on getting or setting different types of data can be
    seen :ref:`here <record-type>`.

    :Example:

    >>> schema = TableSchema.from_lists(['name', 'id'], ['string', 'string'])
    >>> record = Record(schema=schema, values=['test', 'test2'])
    >>> record[0] = 'test'
    >>> record[0]
    >>> 'test'
    >>> record['name']
    >>> 'test'
    >>> record[0:2]
    >>> ('test', 'test2')
    >>> record[0, 1]
    >>> ('test', 'test2')
    >>> record['name', 'id']
    >>> for field in record:
    >>>     print(field)
    ('name', 'test')
    ('id', 'test2')
    >>> len(record)
    2
    >>> 'name' in record
    True
    """


class DataType:
    """
    Base class of all data types in MaxCompute.
    """

    _singleton = True
    _type_id = -1
    __slots__ = ("nullable",)

    def __new__(cls, *args, **kwargs):
        if cls._singleton:
            if not hasattr(cls, "_instance"):
                cls._instance = object.__new__(cls)
                cls._hash = hash(cls)
            return cls._instance
        else:
            return object.__new__(cls)

    def __init__(self, nullable=True):
        self.nullable = nullable

    def __call__(self, nullable=True):
        return self._factory(nullable=nullable)

    def _factory(self, nullable=True):
        return type(self)(nullable=nullable)

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        try:
            return self._equals(other)
        except (TypeError, ValueError):
            return False

    def _equals(self, other):
        if self is other:
            return True

        other = validate_data_type(other)

        if self.nullable != other.nullable:
            return False
        if type(self) == type(other):
            return True
        return isinstance(other, type(self))

    def __hash__(self):
        return self._hash

    @property
    def name(self):
        return type(self).__name__.lower()

    def __repr__(self):
        if self.nullable:
            return self.name
        return f"{self.name}[non-nullable]"

    def __str__(self):
        return self.name.upper()

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return isinstance(self, type(other))

    def can_explicit_cast(self, other):
        return self.can_implicit_cast(other)

    def __reduce__(self):
        return validate_data_type, (str(self),)

    def validate_value(self, val, max_field_size=None):
        # directly return True means without checking
        return True

    def _can_cast_or_throw(self, value, data_type):
        if not self.can_implicit_cast(data_type):
            raise ValueError(
                f"Cannot cast value({value}) from type({data_type}) to type({self})"
            )

    def cast_value(self, value, data_type):
        raise NotImplementedError


class OdpsPrimitive(DataType):
    __slots__ = ()


class BaseInteger(OdpsPrimitive):
    __slots__ = ()

    _type_id = 0
    _bounds = None
    _store_bytes = None

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(other, (BaseFloat, String, Decimal)):
            return True
        if isinstance(other, BaseInteger):
            return self._store_bytes >= other._store_bytes
        return super(BaseInteger, self).can_implicit_cast(other)

    def validate_value(self, val, max_field_size=None):
        if val is None and self.nullable:
            return True
        smallest, largest = self._bounds
        if smallest <= val <= largest:
            return True
        raise ValueError(f"InvalidData: Bigint({val}) out of range")

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        return int(value)


_primitive_doc_template = """
Represents {cls_name} type in MaxCompute.

:Note:
This class may not be used directly. Use its singleton instance (``odps.types.{cls_attr}``) instead.
{odps2_note}
"""
_primitive_odps2_note = """
Need to set ``options.sql.use_odps2_extension = True`` to enable full functionality.
"""


def _primitive_doc(cls=None, is_odps2=True):
    def wrapper(cls_internal):
        cls_name = cls_attr = cls_internal().name
        if cls_name in ("int", "float"):
            cls_attr += "_"
        odps2_note = _primitive_odps2_note if is_odps2 else ""
        docstr = _primitive_doc_template.format(
            cls_name=cls_name, cls_attr=cls_attr, odps2_note=odps2_note
        )
        try:
            cls_internal.__doc__ = docstr
        except AttributeError:
            pass
        return cls_internal

    if cls is None:
        return wrapper
    return wrapper(cls)


@_primitive_doc
class Tinyint(BaseInteger):
    _bounds = (-128, 127)
    _store_bytes = 1


@_primitive_doc
class Smallint(BaseInteger):
    _bounds = (-32768, 32767)
    _store_bytes = 2


@_primitive_doc
class Int(BaseInteger):
    _bounds = (-2147483648, 2147483647)
    _store_bytes = 4


@_primitive_doc(is_odps2=False)
class Bigint(BaseInteger):
    _bounds = (-9223372036854775808, 9223372036854775807)
    _store_bytes = 8


class BaseFloat(OdpsPrimitive):
    __slots__ = ()
    _store_bytes = None

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(other, (BaseInteger, String, Decimal)):
            return True
        if isinstance(other, BaseFloat):
            return self._store_bytes >= other._store_bytes
        return super(BaseFloat, self).can_implicit_cast(other)

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        return float(value)


@_primitive_doc
class Float(BaseFloat):
    _store_bytes = 4
    _type_id = 6


@_primitive_doc(is_odps2=False)
class Double(BaseFloat):
    _store_bytes = 8
    _type_id = 1


def _check_string_byte_size(val, max_size):
    if isinstance(val, bytes):
        byt_len = len(val)
    else:
        byt_len = 4 * len(val)
        if byt_len > max_size:
            # encode only when necessary
            byt_len = len(utils.to_binary(val))
    return byt_len <= max_size, byt_len


@_primitive_doc(is_odps2=False)
class String(OdpsPrimitive):
    __slots__ = ()

    _type_id = 2
    _max_length = 8 * 1024 * 1024  # 8M

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(other, (BaseInteger, BaseFloat, Datetime, Decimal, Binary, Json)):
            return True
        return super(String, self).can_implicit_cast(other)

    def validate_value(self, val, max_field_size=None):
        if val is None and self.nullable:
            return True
        max_field_size = max_field_size or self._max_length
        valid, byt_len = _check_string_byte_size(val, max_field_size)
        if valid:
            return True
        raise ValueError(
            f"InvalidData: Byte length of string({byt_len}) is more than {max_field_size / (1024**2)}M.'"
        )

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(data_type, Datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        if options.tunnel.string_as_binary:
            val = utils.to_binary(value)
        else:
            val = utils.to_text(value)
        return val


@_primitive_doc(is_odps2=False)
class Datetime(OdpsPrimitive):
    __slots__ = ()
    _type_id = 3

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        from_types = (BaseTimestamp, Datetime, Date, String)
        if _date_allow_int_conversion:
            from_types += (Bigint,)
        if isinstance(other, from_types):
            return True
        return super(Datetime, self).can_implicit_cast(other)

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(data_type, String):
            return _datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        elif isinstance(data_type, Date):
            return _datetime(value.year, value.month, value.day)
        elif isinstance(data_type, BaseTimestamp):
            return value.to_pydatetime()
        elif _date_allow_int_conversion and isinstance(data_type, Bigint):
            return utils.to_datetime(value)
        return value


@_primitive_doc
class Date(OdpsPrimitive):
    __slots__ = ()
    _type_id = 11

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        from_types = (BaseTimestamp, Datetime, String)
        if _date_allow_int_conversion:
            from_types += (Bigint,)
        if isinstance(other, from_types):
            return True
        return super(Date, self).can_implicit_cast(other)

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(data_type, String):
            return _datetime.strptime(value, "%Y-%m-%d").date()
        elif isinstance(data_type, Datetime):
            return value.date()
        elif isinstance(data_type, BaseTimestamp):
            return value.to_pydatetime().date()
        elif _date_allow_int_conversion and isinstance(data_type, Bigint):
            return utils.to_date(value)
        return value


@_primitive_doc(is_odps2=False)
class Boolean(OdpsPrimitive):
    __slots__ = ()
    _type_id = 4

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        return value


@_primitive_doc
class Binary(OdpsPrimitive):
    __slots__ = ()
    _type_id = 7
    _max_length = 8 * 1024 * 1024  # 8M

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(other, (BaseInteger, BaseFloat, Datetime, Decimal, String)):
            return True
        return super(Binary, self).can_implicit_cast(other)

    def validate_value(self, val, max_field_size=None):
        if val is None and self.nullable:
            return True
        max_field_size = max_field_size or self._max_length
        valid, byt_len = _check_string_byte_size(val, max_field_size)
        if valid:
            return True
        raise ValueError(
            f"InvalidData: Byte length of binary({byt_len}) is more than {max_field_size / (1024**2)}M.'"
        )

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(data_type, Datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return utils.to_binary(value)


class BaseTimestamp(OdpsPrimitive):
    __slots__ = ()
    _type_id = 8

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(other, (BaseTimestamp, Datetime, String)):
            return True
        return super(BaseTimestamp, self).can_implicit_cast(other)

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        try:
            import pandas as pd
        except (ImportError, ValueError):
            raise ImportError("To use TIMESTAMP in pyodps, you need to install pandas.")

        if isinstance(data_type, String):
            return pd.to_datetime(value)
        elif isinstance(data_type, Datetime):
            return pd.Timestamp(value)
        return value


@_primitive_doc
class Timestamp(BaseTimestamp):
    _type_id = 8


@_primitive_doc
class TimestampNTZ(BaseTimestamp):
    _type_id = 13

    @property
    def name(self):
        return "timestamp_ntz"


@_primitive_doc
class IntervalDayTime(OdpsPrimitive):
    __slots__ = ()
    _type_id = 9

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(other, (BaseTimestamp, Datetime, String)):
            return True
        return super(IntervalDayTime, self).can_implicit_cast(other)

    @property
    def name(self):
        return "interval_day_time"

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        try:
            import pandas as pd
        except (ImportError, ValueError):
            raise ImportError(
                "To use INTERVAL_DAY_TIME in pyodps, you need to install pandas."
            )

        if isinstance(value, float):
            return pd.Timedelta(seconds=value)
        elif isinstance(value, _timedelta):
            return pd.Timedelta(value)
        return value


@_primitive_doc
class IntervalYearMonth(OdpsPrimitive):
    __slots__ = ()
    _type_id = 10

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(other, (String, BaseInteger)):
            return True
        return super(IntervalYearMonth, self).can_implicit_cast(other)

    @property
    def name(self):
        return "interval_year_month"

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(value, (int, str)):
            return Monthdelta(value)
        return value


class CompositeDataType(DataType):
    _singleton = False

    @classmethod
    def parse_composite(cls, args):
        raise NotImplementedError

    def cast_composite_values(self, value):
        raise NotImplementedError


class SizeLimitedString(String, CompositeDataType):
    _singleton = False
    __slots__ = "nullable", "size_limit", "_hash"
    _max_length = 65535

    def __init__(self, size_limit, nullable=True):
        super(SizeLimitedString, self).__init__(nullable=nullable)
        if size_limit > self._max_length:
            raise ValueError(
                f"InvalidData: Length of varchar({size_limit}) is larger than {self._max_length}."
            )
        self.size_limit = size_limit

    @property
    def name(self):
        return f"{type(self).__name__.lower()}({self.size_limit})"

    def _equals(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return DataType._equals(self, other) and self.size_limit == other.size_limit

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash((type(self), self.nullable, self.size_limit))
        return self._hash

    def validate_value(self, val, max_field_size=None):
        if val is None and self.nullable:
            return True
        if len(val) <= self.size_limit:
            # binary size >= unicode size
            return True
        elif isinstance(val, bytes):
            val = val.decode("utf-8")
            if len(val) <= self.size_limit:
                return True
        raise ValueError(
            f"InvalidData: Length of string({len(val)}) is more "
            f"than {self.size_limit}.'"
        )

    @classmethod
    def parse_composite(cls, args):
        if len(args) != 1:
            raise ValueError(
                f"{cls.__name__.upper()}() only accept one length argument."
            )
        try:
            return cls(int(args[0]))
        except TypeError:
            raise ValueError(
                f"{cls.__name__.upper()}() only accept an integer length argument."
            )

    def cast_composite_values(self, value):
        self.validate_value(value)
        return self.cast_value(value, self)


class Varchar(SizeLimitedString):
    """
    Represents varchar type with size limit in MaxCompute.

    :param int size_limit: The size limit of varchar type.

    :Example:

    >>> varchar_type = Varchar(65535)
    >>> print(varchar_type)
    varchar(65535)
    >>> print(varchar_type.size_limit)
    65535

    :Note:

    Need to set ``options.sql.use_odps2_extension = True`` to enable full functionality.
    """

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(
            other, (BaseInteger, BaseFloat, Datetime, Decimal, String, Binary)
        ):
            return True
        return (
            isinstance(other, (Char, Varchar))
            and self.size_limit >= other.size_limit
            and self.nullable == other.nullable
        )


class Char(SizeLimitedString):
    """
    Represents char type with size limit in MaxCompute.

    :param int size_limit: The size limit of char type.

    :Example:

    >>> char_type = Char(65535)
    >>> print(char_type)
    char(65535)
    >>> print(char_type.size_limit)
    65535

    :Note:

    Need to set ``options.sql.use_odps2_extension = True`` to enable full functionality.
    """

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(
            other, (BaseInteger, BaseFloat, Datetime, Decimal, String, Binary)
        ):
            return True
        return (
            isinstance(other, (Char, Varchar))
            and self.size_limit >= other.size_limit
            and self.nullable == other.nullable
        )


class Decimal(CompositeDataType):
    """
    Represents decimal type with size limit in MaxCompute.

    :param int precision: The precision (or total digits) of decimal type.
    :param int scale: The decimal scale (or decimal digits) of decimal type.

    :Example:

    >>> decimal_type = Decimal(18, 6)
    >>> print(decimal_type)
    decimal(18, 6)
    >>> print(decimal_type.precision, decimal_type.scale)
    18 6

    :Note:

    Need to set ``options.sql.use_odps2_extension = True`` to enable full functionality
    when you are setting precision or scale.
    """

    __slots__ = "nullable", "precision", "scale", "_hash"
    _type_id = 5

    _default_precision = 54
    _default_scale = 18
    _decimal_ctx = decimal.Context(prec=_default_precision)

    def __init__(self, precision=None, scale=None, nullable=True):
        super(Decimal, self).__init__(nullable=nullable)
        if precision is None and scale is not None:
            raise ValueError(
                "InvalidData: Scale should be provided along with precision."
            )
        if precision is not None and precision < 1:
            raise ValueError("InvalidData: Decimal precision < 1")
        if precision is not None and scale is not None and scale > precision:
            raise ValueError(
                "InvalidData: Decimal precision must be larger than or equal to scale"
            )
        self.precision = precision
        self.scale = scale
        self._scale_decimal = decimal.Decimal(
            f"1e{-(scale if scale is not None else self._default_scale)}"
        )
        self._no_decimal_check = options.tunnel.no_decimal_check

    @property
    def name(self):
        type_name = type(self).__name__.lower()
        if self.precision is None:
            return type_name
        elif self.scale is None:
            return f"{type_name}({self.precision})"
        else:
            return f"{type_name}({self.precision},{self.scale})"

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash((type(self), self.nullable, self.precision, self.scale))
        return self._hash

    def _equals(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return (
            DataType._equals(self, other)
            and self.precision == other.precision
            and self.scale == other.scale
        )

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(other, (BaseInteger, BaseFloat, String)):
            return True
        return super(Decimal, self).can_implicit_cast(other)

    def validate_value(self, val, max_field_size=None):
        if val is None and self.nullable:
            return True
        if self._no_decimal_check:
            return True

        precision = (
            self.precision if self.precision is not None else self._default_precision
        )
        scale = self.scale if self.scale is not None else self._default_scale
        scaled_val = val.quantize(
            self._scale_decimal, decimal.ROUND_HALF_UP, self._decimal_ctx
        )
        if scaled_val < 0:
            scaled_val = -scaled_val
        int_len = len(str(scaled_val).lstrip("0")) - 1
        if int_len > precision:
            raise ValueError(
                f"decimal value {val} overflow, max integer digit number is {precision - scale}."
            )
        return True

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return decimal.Decimal(value)

    @classmethod
    def parse_composite(cls, args):
        if len(args) > 2:
            raise ValueError(
                f"{cls.__name__.upper()}() accepts no more than two arguments."
            )
        try:
            return cls(*[int(v) for v in args])
        except TypeError:
            raise ValueError(
                f"{cls.__name__.upper()}() only accept integers as arguments."
            )

    def cast_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if type(value) is not decimal.Decimal and not isinstance(
            value, decimal.Decimal
        ):
            value = self.cast_value(value, infer_primitive_data_type(value))
        return value


class Array(CompositeDataType):
    """
    Represents array type in MaxCompute.

    :param value_type: type of elements in the array

    :Example:

    >>> from odps import types as odps_types
    >>>
    >>> array_type = odps_types.Array(odps_types.bigint)
    >>> print(array_type)
    array<bigint>
    >>> print(array_type.value_type)
    bigint

    :Note:

    Need to set ``options.sql.use_odps2_extension = True`` to enable full functionality.
    """

    __slots__ = "nullable", "value_type", "_hash"
    _type_id = 101

    def __init__(self, value_type, nullable=True):
        super(Array, self).__init__(nullable=nullable)
        value_type = validate_data_type(value_type)
        self.value_type = value_type

    @property
    def name(self):
        return f"{type(self).__name__.lower()}<{self.value_type.name}>"

    def _equals(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return DataType._equals(self, other) and self.value_type == other.value_type

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash((type(self), self.nullable, hash(self.value_type)))
        return self._hash

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return (
            isinstance(other, Array)
            and self.value_type == other.value_type
            and self.nullable == other.nullable
        )

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        return value

    @classmethod
    def parse_composite(cls, args):
        if len(args) != 1:
            raise ValueError(
                f"{cls.__name__.upper()}<> should be supplied with exactly one type."
            )
        return cls(args[0])

    def cast_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if not isinstance(value, list):
            raise ValueError(f"Array data type requires `list`, instead of {value}")
        element_data_type = self.value_type
        return [validate_value(element, element_data_type) for element in value]


class Map(CompositeDataType):
    """
    Represents map type in MaxCompute.

    :param key_type: type of keys in the array
    :param value_type: type of values in the array

    :Example:

    >>> from odps import types as odps_types
    >>>
    >>> map_type = odps_types.Map(odps_types.string, odps_types.Array(odps_types.bigint))
    >>> print(map_type)
    map<string, array<bigint>>
    >>> print(map_type.key_type)
    string
    >>> print(map_type.value_type)
    array<bigint>

    :Note:

    Need to set ``options.sql.use_odps2_extension = True`` to enable full functionality.
    """

    __slots__ = "nullable", "key_type", "value_type", "_hash", "_use_ordered_dict"
    _type_id = 102

    def __init__(self, key_type, value_type, nullable=True):
        super(Map, self).__init__(nullable=nullable)
        key_type = validate_data_type(key_type)
        value_type = validate_data_type(value_type)
        self.key_type = key_type
        self.value_type = value_type
        self._use_ordered_dict = options.map_as_ordered_dict
        if self._use_ordered_dict is None:
            self._use_ordered_dict = False

    @property
    def name(self):
        return f"{type(self).__name__.lower()}<{self.key_type.name},{self.value_type.name}>"

    def _equals(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return (
            DataType._equals(self, other)
            and self.key_type == other.key_type
            and self.value_type == other.value_type
        )

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash(
                (type(self), self.nullable, hash(self.key_type), hash(self.value_type))
            )
        return self._hash

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return (
            isinstance(other, Map)
            and self.key_type == other.key_type
            and self.value_type == other.value_type
            and self.nullable == other.nullable
        )

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        return value

    @classmethod
    def parse_composite(cls, args):
        if len(args) != 2:
            raise ValueError(
                f"{cls.__name__.upper()}<> should be supplied with exactly two types."
            )
        return cls(*args)

    def cast_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if not isinstance(value, dict):
            raise ValueError(f"Map data type requires `dict`, instead of {value}")
        key_data_type = self.key_type
        value_data_type = self.value_type

        convert = lambda k, v: (
            validate_value(k, key_data_type),
            validate_value(v, value_data_type),
        )
        dict_type = OrderedDict if self._use_ordered_dict else dict
        return dict_type(convert(k, v) for k, v in value.items())


class Struct(CompositeDataType):
    """
    Represents struct type in MaxCompute.

    :param field_types: types of every field, can be a list of (field_name, field_type) tuples
        or a dict with field names as keys and field types as values.

    :Example:

    >>> from odps import types as odps_types
    >>>
    >>> struct_type = odps_types.Struct([("a", "bigint"), ("b", "array<string>")])
    >>> print(struct_type)
    struct<`a`:bigint, `b`:array<string>>
    >>> print(struct_type.field_types)
    OrderedDict([("a", "bigint"), ("b", "array<string>")])
    >>> print(struct_type.field_types["b"])
    array<string>

    :Note:

    Need to set ``options.sql.use_odps2_extension = True`` to enable full functionality.
    """

    __slots__ = "nullable", "field_types", "_hash"
    _type_id = 103

    def __init__(self, field_types, nullable=True):
        super(Struct, self).__init__(nullable=nullable)
        self.field_types = OrderedDict()
        if isinstance(field_types, dict):
            field_types = field_types.items()
        for k, v in field_types:
            self.field_types[k] = validate_data_type(v)
        self.namedtuple_type = xnamedtuple(
            "StructNamedTuple", list(self.field_types.keys())
        )

        self._struct_as_dict = options.struct_as_dict
        if self._struct_as_dict:
            self._use_ordered_dict = options.struct_as_ordered_dict
            if self._use_ordered_dict is None:
                self._use_ordered_dict = False
            warnings.warn(
                "Representing struct values as dicts is now deprecated. Try config "
                "`options.struct_as_dict=False` and return structs as named tuples "
                "instead.",
                DeprecationWarning,
            )
        else:
            self._use_ordered_dict = False

    @property
    def name(self):
        parts = ",".join(
            f"{utils.backquote_string(k)}:{v.name}" for k, v in self.field_types.items()
        )
        return f"{type(self).__name__.lower()}<{parts}>"

    def _equals(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return (
            isinstance(other, Struct)
            and len(self.field_types) == len(other.field_types)
            and all(
                self.field_types[k] == other.field_types.get(k)
                for k in self.field_types.keys()
            )
        )

    def __hash__(self):
        if not hasattr(self, "_hash"):
            fields_hash = hash(
                tuple((hash(k), hash(v)) for k, v in self.field_types.items())
            )
            self._hash = hash((type(self), self.nullable, fields_hash))
        return self._hash

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return (
            isinstance(other, Struct)
            and self == other
            and self.nullable == other.nullable
        )

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        return value

    @classmethod
    def parse_composite(cls, args):
        if any(not isinstance(a, tuple) and ":" not in a for a in args):
            raise ValueError("Every field defined in STRUCT should be given a name.")

        def conv_type_tuple(type_tuple):
            if isinstance(type_tuple, tuple):
                return type_tuple
            else:
                return tuple(_split_struct_kv(type_tuple))

        return cls(conv_type_tuple(a) for a in args)

    def cast_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if self._struct_as_dict:
            dict_hook = OrderedDict if self._use_ordered_dict else dict
            if isinstance(value, tuple):
                fields = getattr(value, "_fields", None) or self.field_types.keys()
                value = dict_hook(zip(fields, value))
            if isinstance(value, dict):
                return dict_hook(
                    (validate_value(k, string), validate_value(value[k], tp))
                    for k, tp in self.field_types.items()
                )
        else:
            if isinstance(value, tuple):
                return self.namedtuple_type(
                    *(
                        validate_value(v, t)
                        for v, t in zip(value, self.field_types.values())
                    )
                )
            elif isinstance(value, dict):
                list_val = [
                    validate_value(value.get(key), field_type)
                    for key, field_type in self.field_types.items()
                ]
                return self.namedtuple_type(*list_val)
        raise ValueError(
            f"Struct data type requires `tuple` or `dict`, instead of {type(value)}"
        )


class Vector(CompositeDataType):
    """
    Represents vector type in MaxCompute.

    :param element_type: type of elements in the vector (float or double)
    :param dimension: dimension of the vector (positive integer)

    :Example:

    >>> from odps import types as odps_types
    >>>
    >>> vector_type = odps_types.Vector(odps_types.float_, 1536)
    >>> print(vector_type)
    vector(float,1536)

    :Note:

    Need to set ``options.sql.use_odps2_extension = True`` to enable full functionality.
    """

    __slots__ = "nullable", "element_type", "dimension", "_hash"
    _type_id = 104  # Following Array=101, Map=102, Struct=103
    _singleton = False

    def __init__(self, element_type, dimension, nullable=True):
        super(Vector, self).__init__(nullable=nullable)
        element_type = validate_data_type(element_type)

        # Validate element type is float or double
        if element_type not in (float_, double):
            raise ValueError(
                f"Vector element type must be float or double, got: {element_type.name}"
            )

        # Validate dimension is positive integer and multiple of 32
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(
                f"Vector dimension must be a positive integer, got: {dimension}"
            )

        if dimension % 32 != 0:
            raise ValueError(
                f"Vector dimension must be a multiple of 32, got: {dimension}"
            )

        self.element_type = element_type
        self.dimension = dimension

    @property
    def name(self):
        return (
            f"{type(self).__name__.lower()}({self.element_type.name},{self.dimension})"
        )

    def _equals(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return (
            DataType._equals(self, other)
            and self.element_type == other.element_type
            and self.dimension == other.dimension
        )

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash(
                (type(self), self.nullable, hash(self.element_type), self.dimension)
            )
        return self._hash

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        return (
            isinstance(other, Vector)
            and self.element_type == other.element_type
            and self.dimension == other.dimension
            and self.nullable == other.nullable
        )

    def validate_value(self, val, max_field_size=None):
        if val is None:
            if self.nullable:
                return True
            else:
                raise ValueError("InvalidData: value cannot be null")

        # Validate it's a list/tuple/array with correct dimension
        if np is not None:
            if not isinstance(val, (list, tuple, np.ndarray)):
                raise ValueError("Vector value must be a list, tuple, or numpy array")
        elif not isinstance(val, (list, tuple)):
            raise ValueError("Vector value must be a list or tuple")

        if len(val) != self.dimension:
            raise ValueError(
                f"InvalidData: Vector dimension mismatch, expected {self.dimension}, got {len(val)}"
            )

        # Validate no NaN or Infinity
        for i, v in enumerate(val):
            # Check for NaN/Infinity
            has_nan_or_inf = False
            try:
                if np is not None:
                    # Check if value is scalar first
                    if not np.isscalar(v):
                        raise ValueError(
                            f"Vector value {v} must be numeric, got: {type(v).__name__}"
                        )
                    # Use np.isnan which handles both numpy types and Python floats
                    has_nan_or_inf = np.isnan(v) or np.isinf(v)
                else:
                    # Fallback to math for Python floats
                    if isinstance(v, float):
                        has_nan_or_inf = math.isnan(v) or math.isinf(v)
                    elif not isinstance(v, (int, float)):
                        raise ValueError(
                            f"Vector value {v} must be numeric, got: {type(v).__name__}"
                        )
            except TypeError:
                # Not a numeric type that can be NaN/Inf
                raise ValueError(
                    f"Vector value {v} must be numeric, got: {type(v).__name__}"
                ) from None

            if has_nan_or_inf:
                raise ValueError(
                    f"InvalidData: Vector contains NaN or Infinity at index {i}"
                )

        return True

    def cast_composite_values(self, value):
        if value is None and self.nullable:
            return value

        if np is not None:
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise ValueError(
                    f"Vector data type requires list/tuple/numpy.ndarray, instead of {type(value)}"
                )
        elif not isinstance(value, (list, tuple)):
            raise ValueError(
                f"Vector data type requires list/tuple, instead of {type(value)}"
            )

        # Cast each element to the element type
        return [validate_value(element, self.element_type) for element in value]

    @classmethod
    def parse_composite(cls, args):
        """Parse from type string like 'VECTOR(FLOAT,1536)'"""
        if len(args) != 2:
            raise ValueError(
                f"{cls.__name__.upper()}() should be supplied with exactly two arguments: element_type and dimension."
            )

        element_type_str, dimension_str = args

        # Parse element type
        element_type = validate_data_type(element_type_str)

        # Parse dimension (may be string or int)
        try:
            dimension = int(dimension_str)
        except (ValueError, TypeError):
            raise ValueError(
                f"Vector dimension must be an integer, got: {dimension_str}"
            ) from None

        return cls(element_type, dimension)


@_primitive_doc
class Json(DataType):
    _type_id = 12

    _max_length = 8 * 1024 * 1024  # 8M

    def can_implicit_cast(self, other):
        if isinstance(other, str):
            other = validate_data_type(other)

        if isinstance(other, (String, Binary)):
            return True
        return super(Json, self).can_implicit_cast(other)

    def validate_value(self, val, max_field_size=None):
        if val is None and self.nullable:
            return True
        max_field_size = max_field_size or self._max_length
        if len(val) > max_field_size:
            raise ValueError(
                f"InvalidData: Length of string({val}) is more than {max_field_size / (1024**2)}M.'"
            )
        if not isinstance(val, (str, list, dict, int, float)):
            raise ValueError("InvalidData: cannot accept %r as json", val)
        return True

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(data_type, String):
            return _json.loads(utils.to_text(value))
        return value


@_primitive_doc
class Geography(OdpsPrimitive):
    __slots__ = ()
    _type_id = 14
    _max_length = 8 * 1024 * 1024  # 8M


@_primitive_doc
class Blob(OdpsPrimitive):
    _type_id = 15


tinyint = Tinyint()
smallint = Smallint()
int_ = Int()
bigint = Bigint()
float_ = Float()
double = Double()
string = String()
datetime = Datetime()
boolean = Boolean()
binary = Binary()
timestamp = Timestamp()
timestamp_ntz = TimestampNTZ()
interval_day_time = IntervalDayTime()
interval_year_month = IntervalYearMonth()
date = Date()
json = Json()
geography = Geography()
blob = Blob()

_odps_primitive_data_types = dict(
    [
        (t.name, t)
        for t in (
            tinyint,
            smallint,
            int_,
            bigint,
            float_,
            double,
            string,
            datetime,
            date,
            boolean,
            binary,
            timestamp,
            timestamp_ntz,
            interval_day_time,
            interval_year_month,
            json,
            geography,
            blob,
        )
    ]
)


_composite_handlers = dict(
    varchar=Varchar,
    char=Char,
    decimal=Decimal,
    array=Array,
    map=Map,
    struct=Struct,
    vector=Vector,
)


def _split_struct_kv(kv_str):
    parts = utils.split_backquoted(kv_str, ":", 1)
    if len(parts) > 2 or len(parts) <= 0:
        raise ValueError(f"Invalid type string: {kv_str}")

    parts[-1] = parts[-1].strip()
    if len(parts) > 1:
        parts[0] = utils.strip_backquotes(parts[0])
    return parts


def parse_composite_types(type_str, handlers=None):
    handlers = handlers or _composite_handlers

    def _create_composite_type(typ, *args):
        parts = _split_struct_kv(typ)
        typ = parts[-1]
        if typ not in handlers:
            raise ValueError(f"Composite type {typ.upper()} not supported.")
        ctype = handlers[typ].parse_composite(args)

        if len(parts) == 1:
            return ctype
        else:
            return parts[0], ctype

    token_stack = []
    bracket_stack = []
    token_start = 0
    type_str = type_str.strip()
    quoted = False

    for idx, ch in enumerate(type_str):
        if ch == "`":
            quoted = not quoted
        elif not quoted:
            if ch == "<" or ch == "(":
                bracket_stack.append(len(token_stack))
                token = type_str[token_start:idx].strip()
                token_stack.append(token)
                token_start = idx + 1
            elif ch == ">" or ch == ")":
                token = type_str[token_start:idx].strip()
                if token:
                    token_stack.append(token)
                bracket_pos = bracket_stack.pop()
                ctype = _create_composite_type(*token_stack[bracket_pos:])
                token_stack = token_stack[:bracket_pos]
                token_stack.append(ctype)
                token_start = idx + 1
            elif ch == ",":
                token = type_str[token_start:idx].strip()
                if token:
                    token_stack.append(token)
                token_start = idx + 1
    if len(token_stack) != 1:
        return _create_composite_type(type_str)
    return token_stack[0]


def validate_data_type(data_type):
    """
    Parse data type instance from string in MaxCompute DDL.

    :Example:

    >>> field_type = validate_data_type("array<int>")
    >>> print(field_type)
    array<int>
    >>> print(field_type.value_type)
    int
    """
    if isinstance(data_type, DataType):
        return data_type

    composite_err_msg = None
    if isinstance(data_type, str):
        data_type = data_type.strip().lower()
        if data_type in _odps_primitive_data_types:
            return _odps_primitive_data_types[data_type]

        try:
            return parse_composite_types(data_type)
        except ValueError as ex:
            composite_err_msg = str(ex)

    if composite_err_msg is not None:
        raise ValueError(f"Invalid data type: {data_type!r}. {composite_err_msg}")
    raise ValueError(f"Invalid data type: {data_type!r}")


integer_builtins = (int,)
float_builtins = (float,)
try:
    import numpy as np

    integer_builtins += (np.integer,)
    # Add all numpy floating point types
    float_builtins += (np.floating,)
except ImportError:
    pass

_odps_primitive_to_builtin_types = OrderedDict(
    (
        (bigint, integer_builtins),
        (tinyint, integer_builtins),
        (smallint, integer_builtins),
        (int_, integer_builtins),
        (double, float_builtins),
        (float_, float_builtins),
        (string, (str, bytes)),
        (binary, bytes),
        (datetime, _datetime),
        (boolean, bool),
        (interval_year_month, Monthdelta),
        (date, _date),
        (json, (list, dict, str, int, float)),
    )
)
_odps_primitive_clses = set(type(dt) for dt in _odps_primitive_to_builtin_types.keys())


integer_types = (tinyint, smallint, int_, bigint)


def infer_primitive_data_type(value):
    for data_type, builtin_types in _odps_primitive_to_builtin_types.items():
        if isinstance(value, builtin_types):
            return data_type


_pd_type_patched = False


def _patch_pd_types():
    if (
        timestamp not in _odps_primitive_to_builtin_types
        or timestamp_ntz not in _odps_primitive_to_builtin_types
        or interval_day_time not in _odps_primitive_to_builtin_types
    ):
        try:
            import pandas as pd

            new_type_map = {
                timestamp: pd.Timestamp,
                timestamp_ntz: pd.Timestamp,
                interval_day_time: pd.Timedelta,
            }
            _odps_primitive_to_builtin_types.update(new_type_map)
            _odps_primitive_clses.update({type(tp) for tp in new_type_map})
        except (ImportError, ValueError):
            pass


def _cast_primitive_value(value, data_type):
    if value is None or type(value) is pd_na_type:
        return None

    if options.tunnel.string_as_binary:
        if isinstance(value, str):
            value = value.encode("utf-8")
    else:
        if isinstance(value, (bytearray, bytes)):
            value = value.decode("utf-8")

    builtin_types = _odps_primitive_to_builtin_types[data_type]
    if isinstance(value, builtin_types):
        return value

    inferred_data_type = infer_primitive_data_type(value)
    if inferred_data_type is None:
        raise ValueError(
            f"Unknown value type, cannot infer from value: {value}, type: {type(value)}"
        )

    return data_type.cast_value(value, inferred_data_type)


def validate_value(value, data_type, max_field_size=None):
    global _pd_type_patched

    if not _pd_type_patched:
        _patch_pd_types()
        _pd_type_patched = True

    if type(data_type) in _odps_primitive_clses:
        res = _cast_primitive_value(value, data_type)
    else:
        if isinstance(data_type, (BaseTimestamp, IntervalDayTime)):
            raise ImportError(
                "To use %s in pyodps, you need to install pandas.",
                data_type.name.upper(),
            )

        failed = False
        try:
            res = data_type.cast_composite_values(value)
        except AttributeError:
            failed = True
        if failed:
            raise ValueError(f"Unknown data type: {data_type}")

    data_type.validate_value(res, max_field_size=max_field_size)
    return res
