#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import decimal as _builtin_decimal
import json as _json
import sys
import warnings
from collections import OrderedDict
from datetime import date as _date
from datetime import datetime as _datetime
from datetime import timedelta as _timedelta

from . import compat, utils
from .compat import DECIMAL_TYPES, Monthdelta
from .compat import decimal as _decimal
from .compat import east_asian_len, six
from .config import options
from .lib.xnamedtuple import xnamedtuple

try:
    from pandas import NA as _pd_na

    pd_na_type = type(_pd_na)
except (ImportError, ValueError):
    pd_na_type = None

force_py = options.force_py
force_c = options.force_c

_date_allow_int_conversion = False


class Column(object):
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
            raise TypeError("Arguments not supported for Column: %s" % list(kw))

    def __repr__(self):
        not_null_str = ", not null" if not self.nullable else ""
        return "<column {0}, type {1}{2}>".format(
            utils.to_str(self.name), self.type.name.lower(), not_null_str
        )

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
        sio = six.StringIO()
        if self.generate_expression:
            sio.write(
                u"  %s AS %s"
                % (
                    utils.to_text(self.generate_expression),
                    utils.backquote_string(self.name),
                )
            )
        else:
            sio.write(
                u"  %s %s"
                % (
                    utils.to_text(utils.backquote_string(self.name)),
                    utils.to_text(self.type),
                )
            )
            if not self.nullable and not options.sql.ignore_fields_not_null:
                sio.write(u" NOT NULL")
        if with_column_comments and self.comment:
            comment_str = utils.escape_odps_string(utils.to_text(self.comment))
            sio.write(u" COMMENT '%s'" % comment_str)
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
        return "<partition {0}, type {1}>".format(
            utils.to_str(self.name), self.type.name.lower()
        )


class _CallableList(list):
    """Make sure keys and values properties also callable"""

    def __call__(self):
        return self


class PartitionSpec(object):
    def __init__(self, spec=None):
        self.kv = OrderedDict()

        if isinstance(spec, PartitionSpec):
            self.kv = spec.kv.copy()
        elif isinstance(spec, dict):
            self.kv = OrderedDict(spec)
        elif isinstance(spec, six.string_types):
            splits = spec.split(",")
            for sp in splits:
                kv = sp.split("=")
                if len(kv) != 2:
                    raise ValueError(
                        "Invalid partition spec: a partition spec should "
                        'look like "part1=v1,part2=v2"'
                    )

                k, v = kv[0].strip(), kv[1].strip().strip('\'"')

                if len(k) == 0 or len(v) == 0:
                    raise ValueError("Invalid partition spec")
                if k in self.kv:
                    raise ValueError(
                        "Invalid partition spec: found duplicate partition key " + k
                    )

                self.kv[k] = v
        elif spec is not None:
            raise TypeError("Cannot accept spec %r" % spec)

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
        return ",".join("%s='%s'" % (k, v) for k, v in six.iteritems(self.kv))

    def __repr__(self):
        return "<PartitionSpec %s>" % str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, PartitionSpec):
            other = PartitionSpec(other)

        return str(self) == str(other)


class Schema(object):
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
            raise ValueError("Duplicate column names: %s" % ", ".join(duplicates))

        self._snapshot = None

    def __repr__(self):
        return self._repr()

    def __len__(self):
        return len(self.names)

    def __contains__(self, name):
        return utils.to_lower_str(name) in self._name_indexes

    def _repr(self):
        buf = six.StringIO()
        names = [self._to_printable(n) for n in self.names]
        space = 2 * max(len(it) for it in names)
        for name, tp in zip(names, self.types):
            buf.write("\n{0}{1}".format(name.ljust(space), repr(tp)))

        return "Schema {{{0}\n}}".format(utils.indent(buf.getvalue(), 2))

    def __hash__(self):
        return hash((type(self), tuple(self.names), tuple(self.types)))

    def __eq__(self, other):
        if not isinstance(other, Schema):
            return False
        return self.names == other.names and self.types == self.types

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
                *compat.lzip(*[(c.name, c.type) for c in self._columns])
            )
        else:
            super(OdpsSchema, self).__init__([], [])

        if self._partitions:
            self._partition_schema = Schema(
                *compat.lzip(*[(c.name, c.type) for c in self._partitions])
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
                *compat.lzip(*[(c.name, c.type) for c in value])
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
        if isinstance(item, six.integer_types):
            n_columns = len(self._name_indexes)
            if item < n_columns:
                return self._columns[item]
            elif item < len(self):
                return self._partitions[item - n_columns]
            else:
                raise IndexError("Index out of range")
        elif isinstance(item, six.string_types):
            lower_item = utils.to_lower_str(item)
            if lower_item in self._name_indexes:
                idx = self._name_indexes[lower_item]
                return self[idx]
            elif item in self._partition_schema:
                idx = self._partition_schema._name_indexes[lower_item]
                n_columns = len(self._name_indexes)
                return self[n_columns + idx]
            else:
                raise ValueError("Unknown column name: %s" % item)
        elif isinstance(item, (list, tuple)):
            return [self[it] for it in item]
        else:
            return self.columns[item]

    def _repr(self, strip=True):
        def _strip(line):
            return line.rstrip() if strip else line

        buf = six.StringIO()

        name_dict = dict(
            [(col.name, utils.str_to_printable(col.name)) for col in self.columns]
        )
        name_display_lens = dict(
            [
                (k, east_asian_len(utils.to_text(v), encoding=options.display.encoding))
                for k, v in six.iteritems(name_dict)
            ]
        )
        max_name_len = max(six.itervalues(name_display_lens))
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
            row = "{0}{1}{2}{3}".format(
                utils.to_str(name_dict[col.name] + " " * pad_spaces),
                repr(col.type).ljust(type_space),
                not_null + " " * 4 if has_not_null else "",
                "# {0}".format(utils.to_str(col.comment))
                if not_empty(col.comment)
                else "",
            )
            cols_strs.append(_strip(row))
        buf.write(utils.indent("\n".join(cols_strs), 2))
        buf.write("\n")
        buf.write("}\n")

        if self._partitions:
            buf.write("Partitions {\n")

            partition_strs = []
            for partition in self._partitions:
                row = "{0}{1}{2}".format(
                    utils.to_str(name_dict[partition.name].ljust(name_space)),
                    repr(partition.type).ljust(type_space),
                    "# {0}".format(utils.to_str(partition.comment))
                    if not_empty(partition.comment)
                    else "",
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
            raise ValueError("Column %s does not exists" % name)
        return self._columns[index]

    def get_partition(self, name):
        index = self._partition_schema._name_indexes.get(utils.to_lower_str(name))
        if index is None:
            raise ValueError("Partition %s does not exists" % name)
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
        raise ValueError("Column does not exist: %s" % name)

    def update(self, columns, partitions):
        self._columns = columns
        self._partitions = partitions

        names = map(lambda c: c.name, self._columns)
        types = map(lambda c: c.type, self._columns)

        self._init(names, types)
        if self._partitions:
            self._partition_schema = Schema(
                *compat.lzip(*[(c.name, c.type) for c in self._partitions])
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
        fields = compat.lkeys(fields_dict)
        fields_types = compat.lvalues(fields_dict)
        partitions = (
            compat.lkeys(partitions_dict) if partitions_dict is not None else None
        )
        partitions_types = (
            compat.lvalues(partitions_dict) if partitions_dict is not None else None
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


class BaseRecord(object):
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
                "expect len %s, got len %s" % (len(self._columns), len(values))
            )
        [self._set(i, value) for i, value in enumerate(values)]

    def __getitem__(self, item):
        if isinstance(item, six.string_types):
            return self.get_by_name(item)
        elif isinstance(item, (list, tuple)):
            return [self[it] for it in item]
        return self._values[item]

    def __setitem__(self, key, value):
        if isinstance(key, six.string_types):
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


class RecordReprMixin(object):
    def __repr__(self):
        buf = six.StringIO()

        buf.write("odps.Record {\n")

        space = 2 * max(len(it.name) for it in self._columns)
        content = "\n".join(
            [
                "{0}{1}".format(col.name.ljust(space), repr(value))
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


class Record(six.with_metaclass(RecordMeta, RecordReprMixin, BaseRecord)):
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
    ('name', u'test')
    ('id', u'test2')
    >>> len(record)
    2
    >>> 'name' in record
    True
    """


class DataType(object):
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
        return "{0}[non-nullable]".format(self.name)

    def __str__(self):
        return self.name.upper()

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return isinstance(self, type(other))

    def can_explicit_cast(self, other):
        return self.can_implicit_cast(other)

    def validate_value(self, val, max_field_size=None):
        # directly return True means without checking
        return True

    def _can_cast_or_throw(self, value, data_type):
        if not self.can_implicit_cast(data_type):
            raise ValueError(
                "Cannot cast value(%s) from type(%s) to type(%s)"
                % (value, data_type, self)
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
        if isinstance(other, six.string_types):
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
        raise ValueError("InvalidData: Bigint(%s) out of range" % val)

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
        if isinstance(other, six.string_types):
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
    if isinstance(val, six.binary_type):
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
        if isinstance(other, six.string_types):
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
            "InvalidData: Byte length of string(%s) is more than %sM.'"
            % (byt_len, max_field_size / (1024**2))
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
        if isinstance(other, six.string_types):
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
        if isinstance(other, six.string_types):
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
        if isinstance(other, six.string_types):
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
            "InvalidData: Byte length of binary(%s) is more than %sM.'"
            % (byt_len, max_field_size / (1024**2))
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
        if isinstance(other, six.string_types):
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
        if isinstance(other, six.string_types):
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
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (String, BaseInteger)):
            return True
        return super(IntervalYearMonth, self).can_implicit_cast(other)

    @property
    def name(self):
        return "interval_year_month"

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(value, (int, compat.long_type, six.string_types)):
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
                "InvalidData: Length of varchar(%d) is larger than %d."
                % (size_limit, self._max_length)
            )
        self.size_limit = size_limit

    @property
    def name(self):
        return "{0}({1})".format(type(self).__name__.lower(), self.size_limit)

    def _equals(self, other):
        if isinstance(other, six.string_types):
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
        elif isinstance(val, six.binary_type):
            val = val.decode("utf-8")
            if len(val) <= self.size_limit:
                return True
        raise ValueError(
            "InvalidData: Length of string(%d) is more than %s.'"
            % (len(val), self.size_limit)
        )

    @classmethod
    def parse_composite(cls, args):
        if len(args) != 1:
            raise ValueError(
                "%s() only accept one length argument." % cls.__name__.upper()
            )
        try:
            return cls(int(args[0]))
        except TypeError:
            raise ValueError(
                "%s() only accept an integer length argument." % cls.__name__.upper()
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
        if isinstance(other, six.string_types):
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
        if isinstance(other, six.string_types):
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

    _has_other_decimal_type = len(DECIMAL_TYPES) > 1

    _default_precision = 54
    _default_scale = 18
    _decimal_ctx = _decimal.Context(prec=_default_precision)

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
        self._scale_decimal = _decimal.Decimal(
            "1e%d" % -(scale if scale is not None else self._default_scale)
        )
        self._no_decimal_check = options.tunnel.no_decimal_check

    @property
    def name(self):
        if self.precision is None:
            return type(self).__name__.lower()
        elif self.scale is None:
            return "{0}({1})".format(type(self).__name__.lower(), self.precision)
        else:
            return "{0}({1},{2})".format(
                type(self).__name__.lower(), self.precision, self.scale
            )

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash((type(self), self.nullable, self.precision, self.scale))
        return self._hash

    def _equals(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return (
            DataType._equals(self, other)
            and self.precision == other.precision
            and self.scale == other.scale
        )

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (BaseInteger, BaseFloat, String)):
            return True
        return super(Decimal, self).can_implicit_cast(other)

    def validate_value(self, val, max_field_size=None):
        if val is None and self.nullable:
            return True
        if self._no_decimal_check:
            return True

        if (
            self._has_other_decimal_type
            and not isinstance(val, _decimal.Decimal)
            and isinstance(val, DECIMAL_TYPES)
        ):
            val = _decimal.Decimal(str(val))

        precision = (
            self.precision if self.precision is not None else self._default_precision
        )
        scale = self.scale if self.scale is not None else self._default_scale
        scaled_val = val.quantize(
            self._scale_decimal, _decimal.ROUND_HALF_UP, self._decimal_ctx
        )
        if scaled_val < 0:
            scaled_val = -scaled_val
        int_len = len(str(scaled_val).lstrip("0")) - 1
        if int_len > precision:
            raise ValueError(
                "decimal value %s overflow, max integer digit number is %s."
                % (val, precision - scale)
            )
        return True

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if (
            self._has_other_decimal_type
            and not isinstance(value, _decimal.Decimal)
            and isinstance(value, DECIMAL_TYPES)
        ):
            value = _decimal.Decimal(str(value))
        if six.PY3 and isinstance(value, six.binary_type):
            value = value.decode("utf-8")
        return _builtin_decimal.Decimal(value)

    @classmethod
    def parse_composite(cls, args):
        if len(args) > 2:
            raise ValueError(
                "%s() accepts no more than two arguments." % cls.__name__.upper()
            )
        try:
            return cls(*[int(v) for v in args])
        except TypeError:
            raise ValueError(
                "%s() only accept integers as arguments." % cls.__name__.upper()
            )

    def cast_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if type(value) is not _builtin_decimal.Decimal and not isinstance(
            value, _builtin_decimal.Decimal
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
        return "{0}<{1}>".format(type(self).__name__.lower(), self.value_type.name)

    def _equals(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return DataType._equals(self, other) and self.value_type == other.value_type

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash((type(self), self.nullable, hash(self.value_type)))
        return self._hash

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
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
                "%s<> should be supplied with exactly one type." % cls.__name__.upper()
            )
        return cls(args[0])

    def cast_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if not isinstance(value, list):
            raise ValueError("Array data type requires `list`, instead of %s" % value)
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
            self._use_ordered_dict = sys.version_info[:2] <= (3, 6)

    @property
    def name(self):
        return "{0}<{1},{2}>".format(
            type(self).__name__.lower(), self.key_type.name, self.value_type.name
        )

    def _equals(self, other):
        if isinstance(other, six.string_types):
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
        if isinstance(other, six.string_types):
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
                "%s<> should be supplied with exactly two types." % cls.__name__.upper()
            )
        return cls(*args)

    def cast_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if not isinstance(value, dict):
            raise ValueError("Map data type requires `dict`, instead of %s" % value)
        key_data_type = self.key_type
        value_data_type = self.value_type

        convert = lambda k, v: (
            validate_value(k, key_data_type),
            validate_value(v, value_data_type),
        )
        dict_type = OrderedDict if self._use_ordered_dict else dict
        return dict_type(convert(k, v) for k, v in six.iteritems(value))


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
            field_types = six.iteritems(field_types)
        for k, v in field_types:
            self.field_types[k] = validate_data_type(v)
        self.namedtuple_type = xnamedtuple(
            "StructNamedTuple", list(self.field_types.keys())
        )

        self._struct_as_dict = options.struct_as_dict
        if self._struct_as_dict:
            self._use_ordered_dict = options.struct_as_ordered_dict
            if self._use_ordered_dict is None:
                self._use_ordered_dict = sys.version_info[:2] <= (3, 6)
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
            "%s:%s" % (utils.backquote_string(k), v.name)
            for k, v in six.iteritems(self.field_types)
        )
        return "{0}<{1}>".format(type(self).__name__.lower(), parts)

    def _equals(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return (
            isinstance(other, Struct)
            and len(self.field_types) == len(other.field_types)
            and all(
                self.field_types[k] == other.field_types.get(k)
                for k in six.iterkeys(self.field_types)
            )
        )

    def __hash__(self):
        if not hasattr(self, "_hash"):
            fields_hash = hash(
                tuple((hash(k), hash(v)) for k, v in six.iteritems(self.field_types))
            )
            self._hash = hash((type(self), self.nullable, fields_hash))
        return self._hash

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
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
                value = dict_hook(compat.izip(fields, value))
            if isinstance(value, dict):
                return dict_hook(
                    (validate_value(k, string), validate_value(value[k], tp))
                    for k, tp in six.iteritems(self.field_types)
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
            "Struct data type requires `tuple` or `dict`, instead of %s" % type(value)
        )


@_primitive_doc
class Json(DataType):
    _type_id = 12

    _max_length = 8 * 1024 * 1024  # 8M

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
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
                "InvalidData: Length of string(%s) is more than %sM.'"
                % (val, max_field_size / (1024**2))
            )
        if not isinstance(
            val, (six.string_types, list, dict, six.integer_types, float)
        ):
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
)


def _split_struct_kv(kv_str):
    parts = utils.split_backquoted(kv_str, ":", 1)
    if len(parts) > 2 or len(parts) <= 0:
        raise ValueError("Invalid type string: %s" % kv_str)

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
            raise ValueError("Composite type %s not supported." % typ.upper())
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
    if isinstance(data_type, six.string_types):
        data_type = data_type.strip().lower()
        if data_type in _odps_primitive_data_types:
            return _odps_primitive_data_types[data_type]

        try:
            return parse_composite_types(data_type)
        except ValueError as ex:
            composite_err_msg = str(ex)

    if composite_err_msg is not None:
        raise ValueError(
            "Invalid data type: %s. %s" % (repr(data_type), composite_err_msg)
        )
    raise ValueError("Invalid data type: %s" % repr(data_type))


integer_builtins = six.integer_types
float_builtins = (float,)
try:
    import numpy as np

    integer_builtins += (np.integer,)
    float_builtins += (np.float_,)
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
        (string, (six.text_type, six.binary_type)),
        (binary, six.binary_type),
        (datetime, _datetime),
        (boolean, bool),
        (interval_year_month, Monthdelta),
        (date, _date),
        (json, (list, dict, six.string_types, six.integer_types, float)),
    )
)
_odps_primitive_clses = set(type(dt) for dt in _odps_primitive_to_builtin_types.keys())


integer_types = (tinyint, smallint, int_, bigint)


def infer_primitive_data_type(value):
    for data_type, builtin_types in six.iteritems(_odps_primitive_to_builtin_types):
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
        if isinstance(value, six.text_type):
            value = value.encode("utf-8")
    else:
        if isinstance(value, (bytearray, six.binary_type)):
            value = value.decode("utf-8")

    builtin_types = _odps_primitive_to_builtin_types[data_type]
    if isinstance(value, builtin_types):
        return value

    inferred_data_type = infer_primitive_data_type(value)
    if inferred_data_type is None:
        raise ValueError(
            "Unknown value type, cannot infer from value: %s, type: %s"
            % (value, type(value))
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
            raise ValueError("Unknown data type: %s" % data_type)

    data_type.validate_value(res, max_field_size=max_field_size)
    return res
