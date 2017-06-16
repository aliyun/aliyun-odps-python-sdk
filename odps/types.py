#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

import re
from datetime import datetime as _datetime
import decimal as _builtin_decimal

from . import utils
from . import compat
from .compat import six, DECIMAL_TYPES, decimal as _decimal, east_asian_len
from .config import options

force_py = options.force_py
force_c = options.force_c


class Column(object):
    def __init__(self, name=None, typo=None, comment=None, label=None):
        self.name = utils.to_str(name)
        self.type = validate_data_type(typo)
        self.comment = comment
        self.label = label

    def __repr__(self):
        return '<column {0}, type {1}>'.format(utils.to_str(self.name), self.type.name.lower())

    def __hash__(self):
        return hash((type(self), self.name, self.type, self.comment, self.label))


class Partition(Column):
    def __repr__(self):
        return '<partition {0}, type {1}>'.format(utils.to_str(self.name), self.type.name.lower())


class PartitionSpec(object):
    def __init__(self, spec=None):
        self.kv = compat.OrderedDict()

        if spec is not None:
            splits = spec.split(',')
            for sp in splits:
                kv = sp.split('=')
                if len(kv) != 2:
                    raise ValueError('Invalid partition spec')

                k, v = kv[0].strip(), kv[1].strip()\
                    .replace('"', '').replace("'", '')

                if len(k) == 0 or len(v) == 0:
                    raise ValueError('Invalid partition spec')

                self.kv[k] = v

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
        return compat.lkeys(self.kv)

    def __contains__(self, key):
        return key in self.kv

    def __str__(self):
        return ','.join("%s='%s'" % (k, v) for k, v in six.iteritems(self.kv))

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

        self._name_indexes = dict((n, i) for i, n in enumerate(self.names))

        if len(self._name_indexes) < len(self.names):
            duplicates = [n for n in self._name_indexes if self.names.count(n) > 1]
            raise ValueError('Duplicate column names: %s' % ', '.join(duplicates))

        self._snapshot = None

    def __repr__(self):
        return self._repr()

    def __len__(self):
        return len(self.names)

    def __contains__(self, name):
        return utils.to_str(name) in self._name_indexes

    def _repr(self):
        buf = six.StringIO()
        names = [self._to_printable(n) for n in self.names]
        space = 2 * max(len(it) for it in names)
        for name, tp in zip(names, self.types):
            buf.write('\n{0}{1}'.format(name.ljust(space), repr(tp)))

        return 'Schema {{{0}\n}}'.format(utils.indent(buf.getvalue(), 2))

    def __hash__(self):
        return hash((type(self), tuple(self.names), tuple(self.types)))

    def __eq__(self, other):
        if not isinstance(other, Schema):
            return False
        return self.names == other.names and self.types == self.types

    def get_type(self, name):
        return self.types[self._name_indexes[utils.to_str(name)]]

    def append(self, name, typo):
        names = self.names + [name, ]
        types = self.types + [typo, ]
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
                *compat.lzip(*[(c.name, c.type) for c in self._columns]))
        else:
            super(OdpsSchema, self).__init__([], [])

        if self._partitions:
            self._partition_schema = Schema(
                *compat.lzip(*[(c.name, c.type) for c in self._partitions]))
        else:
            self._partition_schema = Schema([], [])

    def __len__(self):
        return super(OdpsSchema, self).__len__() + len(self._partition_schema)

    def __setattr__(self, key, value):
        if key == '_columns' and value and not getattr(self, 'names', None) and \
                not getattr(self, 'types', None):
            names = [c.name for c in value]
            types = [c.type for c in value]
            self._init(names, types)
        elif key == '_partitions' and value:
            self._partition_schema = Schema(
                *compat.lzip(*[(c.name, c.type) for c in value]))
        object.__setattr__(self, key, value)

    def __contains__(self, name):
        return super(OdpsSchema, self).__contains__(name) or \
               utils.to_str(name) in self._partition_schema

    def __eq__(self, other):
        if not isinstance(other, OdpsSchema):
            return False

        return super(OdpsSchema, self).__eq__(other) and \
            self._partition_schema == other._partition_schema

    def __hash__(self):
        return hash((type(self), tuple(self.names), tuple(self.types),
                     self._partition_schema))

    def __getitem__(self, item):
        if isinstance(item, six.integer_types):
            n_columns = len(self._name_indexes)
            if item < n_columns:
                return self._columns[item]
            elif item < len(self):
                return self._partitions[item-n_columns]
            else:
                raise IndexError('Index out of range')
        elif isinstance(item, six.string_types):
            item = utils.to_str(item)
            if item in self._name_indexes:
                idx = self._name_indexes[item]
                return self[idx]
            elif item in self._partition_schema:
                idx = self._partition_schema._name_indexes[item]
                n_columns = len(self._name_indexes)
                return self[n_columns+idx]
            else:
                raise ValueError('Unknown column name: %s' % item)
        elif isinstance(item, (list, tuple)):
            return [self[it] for it in item]
        else:
            return self.columns[item]

    def _repr(self):
        buf = six.StringIO()

        name_dict = dict([(col.name, utils.str_to_printable(col.name)) for col in self.columns])
        name_display_lens = dict([(k, east_asian_len(utils.to_text(v), encoding=options.display.encoding))
                                  for k, v in six.iteritems(name_dict)])
        name_space = 2 * max(six.itervalues(name_display_lens))
        type_space = 2 * max(len(repr(col.type)) for col in self.columns)

        not_empty = lambda field: field is not None and len(field.strip()) > 0

        buf.write('odps.Schema {\n')
        cols_strs = []
        for col in self._columns:
            pad_spaces = name_space - name_display_lens[col.name]
            cols_strs.append('{0}{1}{2}'.format(
                utils.to_str(name_dict[col.name] + ' ' * pad_spaces),
                repr(col.type).ljust(type_space),
                '# {0}'.format(utils.to_str(col.comment)) if not_empty(col.comment) else ''
            ))
        buf.write(utils.indent('\n'.join(cols_strs), 2))
        buf.write('\n')
        buf.write('}\n')

        if self._partitions:
            buf.write('Partitions {\n')

            partition_strs = []
            for partition in self._partitions:
                partition_strs.append('{0}{1}{2}'.format(
                    utils.to_str(name_dict[partition.name].ljust(name_space)),
                    repr(partition.type).ljust(type_space),
                    '# {0}'.format(utils.to_str(partition.comment)) if not_empty(partition.comment) else ''
                ))
            buf.write(utils.indent('\n'.join(partition_strs), 2))
            buf.write('\n')
            buf.write('}\n')

        return buf.getvalue()

    def build_snapshot(self):
        if self._snapshot is None and not options.force_py:
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
        return self._columns

    @property
    def columns(self):
        partitions = self._partitions or []
        return self._columns + partitions

    @property
    def partitions(self):
        return self._partitions

    @utils.deprecated('use simple_columns property instead')
    def get_columns(self):
        return self._columns

    @utils.deprecated('use partitions property instead')
    def get_partitions(self):
        return self._partitions

    def get_column(self, name):
        index = self._name_indexes.get(utils.to_str(name))
        if index is None:
            raise ValueError('Column %s does not exists' % name)
        return self._columns[index]

    def get_partition(self, name):
        index = self._partition_schema._name_indexes.get(utils.to_str(name))
        if index is None:
            raise ValueError('Partition %s does not exists' % name)
        return self._partitions[index]

    def is_partition(self, name):
        try:
            name = name.name
        except AttributeError:
            pass
        return name in self._partition_schema._name_indexes

    def get_type(self, name):
        if name in self._name_indexes:
            return super(OdpsSchema, self).get_type(name)
        elif name in self._partition_schema:
            return self._partition_schema.get_type(name)
        raise ValueError('Column does not exist: %s' % name)

    def update(self, columns, partitions):
        self._columns = columns
        self._partitions = partitions

        names = map(lambda c: c.name, self._columns)
        types = map(lambda c: c.type, self._columns)

        self._init(names, types)
        if self._partitions:
            self._partition_schema = Schema(
                *compat.lzip(*[(c.name, c.type) for c in self._partitions]))
        else:
            self._partition_schema = Schema([], [])

    def to_ignorecase_schema(self):
        cols = [Column(col.name.lower(), col.type, col.comment, col.label)
                for col in self._columns]
        parts = None
        if self._partitions:
            parts = [Partition(part.name.lower(), part.type, part.comment, part.label)
                     for part in self._partitions]

        return type(self)(columns=cols, partitions=parts)

    @classmethod
    def from_lists(cls, names, types, partition_names=None, partition_types=None):
        columns = [Column(name=name, typo=typo) for name, typo in zip(names, types)]
        if partition_names is not None and partition_types is not None:
            partitions = [Partition(name=name, typo=typo)
                          for name, typo in zip(partition_names, partition_types)]
        else:
            partitions = None
        return cls(columns=columns, partitions=partitions)

    @classmethod
    def from_dict(cls, fields_dict, partitions_dict=None):
        fields = compat.lkeys(fields_dict)
        fields_types = compat.lvalues(fields_dict)
        partitions = compat.lkeys(partitions_dict) \
            if partitions_dict is not None else None
        partitions_types = compat.lvalues(partitions_dict) \
            if partitions_dict is not None else None

        return cls.from_lists(fields, fields_types,
                              partition_names=partitions,
                              partition_types=partitions_types)

    def get_table_ddl(self, table_name='table_name', with_comments=True):
        def _format_col(col):
            col_str = u'`%s` %s' % (utils.to_text(col.name), str(col.type))
            if with_comments and col.comment:
                col_str += u' COMMENT \'%s\'' % utils.to_text(col.comment)
            return col_str

        def _format_cols(cols):
            col_text = u',\n'.join((_format_col(col) for col in cols))
            return '\n'.join(('  ' + col_text for col_text in col_text.splitlines()))

        create_str = u'CREATE TABLE %s (\n' % utils.to_text(table_name) + _format_cols(self._columns) + u'\n)'
        if self._partitions:
            create_str += u' PARTITIONED BY (\n' + _format_cols(self._partitions) + u'\n)'
        return create_str


class RecordMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, RecordReprMixin)


class BaseRecord(object):

    # set __slots__ to save memory in the situation that records' size may be quite large
    __slots__ = '_values', '_columns', '_name_indexes'

    def __init__(self, columns=None, schema=None, values=None):
        if columns is not None:
            self._columns = columns
            self._name_indexes = dict((col.name, i) for i, col in enumerate(self._columns))
        else:
            self._columns = schema.columns
            self._name_indexes = schema._name_indexes

        if self._columns is None:
            raise ValueError('Either columns or schema should not be provided')

        self._values = [None, ] * len(self._columns)
        if values is not None:
            self._sets(values)

    def _mode(self):
        return 'py'

    def _exclude_partition_columns(self):
        return [col for col in self._columns if not isinstance(col, Partition)]

    def _get(self, i):
        return self._values[i]

    def _set(self, i, value):
        data_type = self._columns[i].type
        val = validate_value(value, data_type)
        self._values[i] = val

    get = _get  # to keep compatible
    set = _set  # to keep compatible

    def _sets(self, values):
        if len(values) != len(self._columns) and \
                        len(values) != len(self._exclude_partition_columns()):
            raise ValueError('The values set to records are against the schema, '
                             'expect len %s, got len %s' % (len(self._columns), len(values)))
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
        if item == '_name_indexes':
            return object.__getattribute__(self, item)
        if hasattr(self, '_name_indexes') and item in self._name_indexes:
            return self.get_by_name(item)
        return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        if hasattr(self, '_name_indexes') and key in self._name_indexes:
            self.set_by_name(key, value)
        else:
            object.__setattr__(self, key, value)

    def get_by_name(self, name):
        i = self._name_indexes[name]
        return self._values[i]

    def set_by_name(self, name, value):
        i = self._name_indexes[name]
        self._set(i, value)

    def __len__(self):
        return len(self._columns)

    def __contains__(self, item):
        return item in self._name_indexes

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

        buf.write('odps.Record {\n')

        space = 2 * max(len(it.name) for it in self._columns)
        content = '\n'.join(
            ['{0}{1}'.format(col.name.ljust(space), repr(value))
             for col, value in zip(self._columns, self._values)])
        buf.write(utils.indent(content, 2))

        buf.write('\n}')

        return buf.getvalue()

    def __hash__(self):
        return hash((type(self), tuple(self._columns), tuple(self._values)))

    def __eq__(self, other):
        if not isinstance(other, Record):
            return False

        return self._columns == other._columns and self._values == other._values


class Record(six.with_metaclass(RecordMeta, RecordReprMixin, BaseRecord)):
    """
    A record generally means the data of a single line in a table.

    :Example:

    >>> schema = Schema.from_lists(['name', 'id'], ['string', 'string'])
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
    Abstract data type
    """
    _singleton = True
    _type_id = None
    __slots__ = 'nullable',

    def __new__(cls, *args, **kwargs):
        if cls._singleton:
            if not hasattr(cls, '_instance'):
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
        return self._equals(other)

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
        return '{0}[non-nullable]'.format(self.name)

    def __str__(self):
        return self.name.upper()

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return isinstance(self, type(other))

    def can_explicit_cast(self, other):
        return self.can_implicit_cast(other)

    def validate_value(self, val):
        # directly return True means without checking
        return True

    def _can_cast_or_throw(self, value, data_type):
        if not self.can_implicit_cast(data_type):
            raise ValueError('Cannot cast value(%s) from type(%s) to type(%s)' % (
                value, data_type, self))

    def cast_value(self, value, data_type):
        raise NotImplementedError


class OdpsPrimitive(DataType):
    __slots__ = ()


class Bigint(OdpsPrimitive):
    __slots__ = ()

    _type_id = 0
    _bounds = (-9223372036854775808, 9223372036854775807)

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (Double, String, Decimal)):
            return True
        return super(Bigint, self).can_implicit_cast(other)

    def validate_value(self, val):
        if val is None and self.nullable:
            return True
        smallest, largest = self._bounds
        if smallest <= val <= largest:
            return True
        raise ValueError('InvalidData: Bigint(%s) out of range' % val)

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        return int(value)


class Double(OdpsPrimitive):
    __slots__ = ()
    _type_id = 1

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (Bigint, String, Decimal)):
            return True
        return super(Double, self).can_implicit_cast(other)

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        return float(value)


class String(OdpsPrimitive):
    __slots__ = ()

    _type_id = 2
    _max_length = 8 * 1024 * 1024  # 8M

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (Bigint, Double, Datetime, Decimal)):
            return True
        return super(String, self).can_implicit_cast(other)

    def validate_value(self, val):
        if val is None and self.nullable:
            return True
        if len(val) <= self._max_length:
            return True
        raise ValueError(
            "InvalidData: Length of string(%s) is more than %sM.'" %
            (val, self._max_length / (1024 ** 2)))

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(data_type, Datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        if options.tunnel.string_as_binary:
            val = utils.to_binary(value)
        else:
            val = utils.to_text(value)
        return val


class Datetime(OdpsPrimitive):
    __slots__ = ()
    _type_id = 3

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (Datetime, String)):
            return True
        return super(Datetime, self).can_implicit_cast(other)

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(data_type, String):
            return _datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        return value


class Boolean(OdpsPrimitive):
    __slots__ = ()
    _type_id = 4

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        return value


class Decimal(OdpsPrimitive):
    __slots__ = ()
    _type_id = 5

    _max_int_len = 36
    _max_scale = 18

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (Bigint, Double, String)):
            return True
        return super(Decimal, self).can_implicit_cast(other)

    def validate_value(self, val):
        if val is None and self.nullable:
            return True

        if not isinstance(val, _decimal.Decimal) and isinstance(val, DECIMAL_TYPES):
            val = _decimal.Decimal(str(val))
        to_scale = _decimal.Decimal(str(10 ** -self._max_scale))
        scaled_val = val.quantize(to_scale, _decimal.ROUND_HALF_UP)
        int_len = len(str(scaled_val)) - self._max_scale - 1
        if int_len > self._max_int_len:
            raise ValueError(
                'decimal value %s overflow, max integer digit number is %s.' %
                (val, self._max_int_len))
        return True

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if six.PY3 and isinstance(value, six.binary_type):
            value = value.decode('utf-8')
        return _builtin_decimal.Decimal(value)


class Array(DataType):
    _singleton = False
    __slots__ = 'nullable', 'value_type', '_hash'

    def __init__(self, value_type, nullable=True):
        super(Array, self).__init__(nullable=nullable)
        value_type = validate_data_type(value_type)
        if value_type not in (bigint, double, string, boolean, decimal):
            raise ValueError('Invalid value type: %s' % repr(value_type))
        self.value_type = value_type

    @property
    def name(self):
        return '{0}<{1}>'.format(type(self).__name__.lower(),
                                 self.value_type.name)

    def _equals(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return DataType._equals(self, other) and \
            self.value_type == other.value_type

    def __hash__(self):
        if not hasattr(self, '_hash'):
            self._hash = hash(hash(type(self)) * hash(type(self.value_type)))
        return self._hash

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return isinstance(other, Array) and \
            self.value_type == other.value_type and \
            self.nullable == other.nullable

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        return value


class Map(DataType):
    _singleton = False
    __slots__ = 'nullable', 'key_type', 'value_type', '_hash'

    def __init__(self, key_type, value_type, nullable=True):
        super(Map, self).__init__(nullable=nullable)
        key_type = validate_data_type(key_type)
        if key_type not in (bigint, string):
            raise ValueError('Invalid key type: %s' % repr(key_type))
        value_type = validate_data_type(value_type)
        if value_type not in (bigint, double, string):
            raise ValueError('Invalid value type: %s' % repr(value_type))
        self.key_type = key_type
        self.value_type = value_type

    @property
    def name(self):
        return '{0}<{1},{2}>'.format(type(self).__name__.lower(),
                                     self.key_type.name,
                                     self.value_type.name)

    def _equals(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return DataType._equals(self, other) and \
            self.key_type == other.key_type and \
            self.value_type == other.value_type

    def __hash__(self):
        if not hasattr(self, '_hash'):
            self._hash = hash(
                hash(type(self)) *
                hash(type(self.key_type)) *
                hash(type(self.value_type))
            )
        return self._hash

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return isinstance(other, Map) and \
            self.key_type == other.key_type and \
            self.value_type == other.value_type and \
            self.nullable == other.nullable

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        return value


bigint = Bigint()
double = Double()
string = String()
datetime = Datetime()
boolean = Boolean()
decimal = Decimal()

_odps_primitive_data_types = dict(
    [(t.name, t) for t in (
        bigint, double, string, datetime, boolean, decimal
    )]
)

ARRAY_RE = re.compile(r'^array<([^>]+)>$', re.IGNORECASE)
MAP_RE = re.compile(r'^map<([^,]+),([^>]+)>$', re.IGNORECASE)


def validate_data_type(data_type):
    if isinstance(data_type, DataType):
        return data_type

    if isinstance(data_type, six.string_types):
        data_type = data_type.lower()
        if data_type in _odps_primitive_data_types:
            return _odps_primitive_data_types[data_type]

        array_match = ARRAY_RE.match(data_type)
        if array_match:
            value_type = array_match.group(1)
            return Array(value_type)

        map_match = MAP_RE.match(data_type)
        if map_match:
            key_type = map_match.group(1).strip()
            value_type = map_match.group(2).strip()
            return Map(key_type, value_type)

    raise ValueError('Invalid data type: %s' % repr(data_type))


integer_builtins = six.integer_types
float_builtins = (float,)
try:
    import numpy as np
    integer_builtins += (np.integer,)
    float_builtins += (np.float,)
except ImportError:
    pass

_odps_primitive_to_builtin_types = {
    bigint: integer_builtins,
    double: float_builtins,
    string: (six.text_type, six.binary_type),
    datetime: _datetime,
    boolean: bool,
    decimal: DECIMAL_TYPES
}


def infer_primitive_data_type(value):
    for data_type, builtin_types in six.iteritems(_odps_primitive_to_builtin_types):
        if isinstance(value, builtin_types):
            return data_type


def _validate_primitive_value(value, data_type):
    if value is None:
        return None
    if options.tunnel.string_as_binary:
        if isinstance(value, six.text_type):
            value = value.encode('utf-8')
    else:
        if isinstance(value, (bytearray, six.binary_type)):
            value = value.decode('utf-8')

    builtin_types = _odps_primitive_to_builtin_types[data_type]
    if isinstance(value, builtin_types):
        return value

    inferred_data_type = infer_primitive_data_type(value)
    if inferred_data_type is None:
        raise ValueError(
            'Unknown value type, cannot infer from value: %s, type: %s' % (value, type(value)))

    return data_type.cast_value(value, inferred_data_type)


def validate_value(value, data_type):
    if data_type in _odps_primitive_to_builtin_types:
        res = _validate_primitive_value(value, data_type)
    elif isinstance(data_type, Array):
        if value is None and data_type.nullable:
            return value
        if not isinstance(value, list):
            raise ValueError('Array data type requires `list`, instead of %s' % value)
        element_data_type = data_type.value_type
        res = [_validate_primitive_value(element, element_data_type)
               for element in value]
    elif isinstance(data_type, Map):
        if value is None and data_type.nullable:
            return value
        if not isinstance(value, dict):
            raise ValueError('Map data type requires `dict`, instead of %s' % value)
        key_data_type = data_type.key_type
        value_data_type = data_type.value_type

        convert = lambda k, v: (_validate_primitive_value(k, key_data_type),
                                _validate_primitive_value(v, value_data_type))
        res = compat.OrderedDict(convert(k, v) for k, v in six.iteritems(value))
    else:
        raise ValueError('Unknown data type: %s' % data_type)

    data_type.validate_value(res)
    return res
