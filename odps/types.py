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

from datetime import datetime as _datetime, timedelta as _timedelta, date as _date
import decimal as _builtin_decimal

from . import utils
from . import compat
from .compat import six, DECIMAL_TYPES, decimal as _decimal, east_asian_len, OrderedDict, Monthdelta
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

    def items(self):
        for k, v in self.kv.items():
            yield k, v

    def __contains__(self, key):
        return key in self.kv

    def __str__(self):
        return ','.join("%s='%s'" % (k, v) for k, v in six.iteritems(self.kv))

    def __repr__(self):
        return '<PartitionSpec %s>' % str(self)

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
        return self._columns

    @property
    def columns(self):
        partitions = self._partitions or []
        return self._columns + partitions

    @property
    def partitions(self):
        try:
            return self._partitions
        except AttributeError:
            return []

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
        from .models.table import Table
        return Table.gen_create_table_sql(table_name, self, with_column_comments=with_comments)


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
    _type_id = -1
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


class Tinyint(BaseInteger):
    _bounds = (-128, 127)
    _store_bytes = 1


class Smallint(BaseInteger):
    _bounds = (-32768, 32767)
    _store_bytes = 2


class Int(BaseInteger):
    _bounds = (-2147483648, 2147483647)
    _store_bytes = 4


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


class Float(BaseFloat):
    _store_bytes = 4
    _type_id = 6


class Double(BaseFloat):
    _store_bytes = 8
    _type_id = 1


class String(OdpsPrimitive):
    __slots__ = ()

    _type_id = 2
    _max_length = 8 * 1024 * 1024  # 8M

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (BaseInteger, BaseFloat, Datetime, Decimal, Binary)):
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

        if isinstance(other, (Timestamp, Datetime, String)):
            return True
        return super(Datetime, self).can_implicit_cast(other)

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(data_type, String):
            return _datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        elif isinstance(data_type, Timestamp):
            return value.to_pydatetime()
        return value


class Date(OdpsPrimitive):
    __slots__ = ()
    _type_id = 11

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (Timestamp, Datetime, String)):
            return True
        return super(Date, self).can_implicit_cast(other)

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(data_type, String):
            datetime = _datetime.strptime(value, '%Y-%m-%d')
            return _date(datetime.year, datetime.month, datetime.day)
        elif isinstance(data_type, Timestamp):
            return value.to_pydatetime().date()
        return value


class Boolean(OdpsPrimitive):
    __slots__ = ()
    _type_id = 4

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        return value


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

    def validate_value(self, val):
        if val is None and self.nullable:
            return True
        if len(val) <= self._max_length:
            return True
        raise ValueError(
            "InvalidData: Length of binary(%s) is more than %sM.'" %
            (val, self._max_length / (1024 ** 2)))

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(data_type, Datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        return utils.to_binary(value)


class Timestamp(OdpsPrimitive):
    __slots__ = ()
    _type_id = 8

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (Timestamp, Datetime, String)):
            return True
        return super(Timestamp, self).can_implicit_cast(other)

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('To use TIMESTAMP in pyodps, you need to install pandas.')

        if isinstance(data_type, String):
            return pd.Timestamp.strptime(value, '%Y-%m-%d %H:%M:%S')
        elif isinstance(data_type, Datetime):
            return pd.Timestamp(value)
        return value


class IntervalDayTime(OdpsPrimitive):
    __slots__ = ()
    _type_id = 9

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (Timestamp, Datetime, String)):
            return True
        return super(IntervalDayTime, self).can_implicit_cast(other)

    @property
    def name(self):
        return 'interval_day_time'

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('To use INTERVAL_DAY_TIME in pyodps, you need to install pandas.')

        if isinstance(value, float):
            return pd.Timedelta(seconds=value)
        elif isinstance(value, _timedelta):
            return pd.Timedelta(value)
        return value


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
        return 'interval_year_month'

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if isinstance(value, (int, compat.long_type, six.string_types)):
            return Monthdelta(value)
        return value


class CompositeMixin(DataType):
    _singleton = False

    @classmethod
    def parse_composite(cls, args):
        raise NotImplementedError

    def validate_composite_values(self, value):
        raise NotImplementedError


class SizeLimitedString(String, CompositeMixin):
    _singleton = False
    __slots__ = 'nullable', 'size_limit', '_hash'
    _max_length = 65535

    def __init__(self, size_limit, nullable=True):
        super(SizeLimitedString, self).__init__(nullable=nullable)
        if size_limit > self._max_length:
            raise ValueError("InvalidData: Length of varchar(%d) is larger than %d." %
                             (size_limit, self._max_length))
        self.size_limit = size_limit

    @property
    def name(self):
        return '{0}({1})'.format(type(self).__name__.lower(),
                                 self.size_limit)

    def _equals(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return DataType._equals(self, other) and \
               self.size_limit == other.size_limit

    def __hash__(self):
        if not hasattr(self, '_hash'):
            self._hash = hash((type(self), self.nullable, self.size_limit))
        return self._hash

    def validate_value(self, val):
        if val is None and self.nullable:
            return True
        if len(val) <= self.size_limit:
            return True
        raise ValueError(
            "InvalidData: Length of string(%d) is more than %sM.'" %
            (len(val), self.size_limit))

    @classmethod
    def parse_composite(cls, args):
        if len(args) != 1:
            raise ValueError('%s() only accept one length argument.' % cls.__name__.upper())
        try:
            return cls(int(args[0]))
        except TypeError:
            raise ValueError('%s() only accept an integer length argument.' % cls.__name__.upper())

    def validate_composite_values(self, value):
        self.validate_value(value)
        return self.cast_value(value, self)


class Varchar(SizeLimitedString):
    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (BaseInteger, BaseFloat, Datetime, Decimal, String, Binary)):
            return True
        return isinstance(other, (Char, Varchar)) and \
               self.size_limit >= other.size_limit and \
               self.nullable == other.nullable


class Char(SizeLimitedString):
    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (BaseInteger, BaseFloat, Datetime, Decimal, String, Binary)):
            return True
        return isinstance(other, (Char, Varchar)) and \
               self.size_limit >= other.size_limit and \
               self.nullable == other.nullable


class Decimal(CompositeMixin):
    __slots__ = 'nullable', 'precision', 'scale', '_hash'
    _type_id = 5

    _max_precision = 38
    _max_scale = 18
    _decimal_ctx = _decimal.Context(prec=_max_precision)

    def __init__(self, precision=None, scale=None, nullable=True):
        super(Decimal, self).__init__(nullable=nullable)
        if precision and precision > self._max_precision:
            raise ValueError("InvalidData: Precision(%d) is larger than %d." %
                             (precision, self._max_precision))
        if scale and scale > self._max_scale:
            raise ValueError("InvalidData: Scale(%d) is larger than %d." %
                             (scale, self._max_scale))
        if precision is None and scale is not None:
            raise ValueError('InvalidData: Scale should be provided along with precision.')
        self.precision = precision
        self.scale = scale

    @property
    def name(self):
        if self.precision is None:
            return type(self).__name__.lower()
        elif self.scale is None:
            return '{0}({1})'.format(type(self).__name__.lower(),
                                     self.precision)
        else:
            return '{0}({1},{2})'.format(type(self).__name__.lower(),
                                         self.precision, self.scale)

    def __hash__(self):
        if not hasattr(self, '_hash'):
            self._hash = hash((type(self), self.nullable, self.precision, self.scale))
        return self._hash

    def _equals(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return DataType._equals(self, other) and \
               self.precision == other.precision and \
               self.scale == other.scale

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        if isinstance(other, (BaseInteger, BaseFloat, String)):
            return True
        return super(Decimal, self).can_implicit_cast(other)

    def validate_value(self, val):
        if val is None and self.nullable:
            return True

        if not isinstance(val, _decimal.Decimal) and isinstance(val, DECIMAL_TYPES):
            val = _decimal.Decimal(str(val))
        to_scale = _decimal.Decimal(str(10 ** -self._max_scale))
        scaled_val = val.quantize(to_scale, _decimal.ROUND_HALF_UP, self._decimal_ctx)
        int_len = len(str(scaled_val)) - self._max_scale - 1
        if int_len > self._max_precision:
            raise ValueError(
                'decimal value %s overflow, max integer digit number is %s.' %
                (val, self._max_precision))
        return True

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        if six.PY3 and isinstance(value, six.binary_type):
            value = value.decode('utf-8')
        return _builtin_decimal.Decimal(value)

    @classmethod
    def parse_composite(cls, args):
        if len(args) > 2:
            raise ValueError('%s() accepts no more than two arguments.' % cls.__name__.upper())
        try:
            return cls(*[int(v) for v in args])
        except TypeError:
            raise ValueError('%s() only accept integers as arguments.' % cls.__name__.upper())

    def validate_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if not isinstance(value, _builtin_decimal.Decimal):
            value = self.cast_value(value, infer_primitive_data_type(value))
        self.validate_value(value)
        return value


class Array(CompositeMixin):
    __slots__ = 'nullable', 'value_type', '_hash'

    def __init__(self, value_type, nullable=True):
        super(Array, self).__init__(nullable=nullable)
        value_type = validate_data_type(value_type)
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
            self._hash = hash((type(self), self.nullable, hash(self.value_type)))
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

    @classmethod
    def parse_composite(cls, args):
        if len(args) != 1:
            raise ValueError('%s<> only accept one type.' % cls.__name__.upper())
        return cls(args[0])

    def validate_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if not isinstance(value, list):
            raise ValueError('Array data type requires `list`, instead of %s' % value)
        element_data_type = self.value_type
        return [validate_value(element, element_data_type) for element in value]


class Map(CompositeMixin):
    __slots__ = 'nullable', 'key_type', 'value_type', '_hash'

    def __init__(self, key_type, value_type, nullable=True):
        super(Map, self).__init__(nullable=nullable)
        key_type = validate_data_type(key_type)
        value_type = validate_data_type(value_type)
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
            self._hash = hash((type(self), self.nullable,
                               hash(self.key_type), hash(self.value_type)))
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

    @classmethod
    def parse_composite(cls, args):
        if len(args) != 2:
            raise ValueError('%s<> should be supplied with exactly two types.' % cls.__name__.upper())
        return cls(*args)

    def validate_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if not isinstance(value, dict):
            raise ValueError('Map data type requires `dict`, instead of %s' % value)
        key_data_type = self.key_type
        value_data_type = self.value_type

        convert = lambda k, v: (validate_value(k, key_data_type),
                                validate_value(v, value_data_type))
        return compat.OrderedDict(convert(k, v) for k, v in six.iteritems(value))


class Struct(CompositeMixin):
    __slots__ = 'nullable', 'field_types', '_hash'

    def __init__(self, field_types, nullable=True):
        super(Struct, self).__init__(nullable=nullable)
        self.field_types = OrderedDict()
        if isinstance(field_types, dict):
            field_types = six.iteritems(field_types)
        for k, v in field_types:
            self.field_types[k] = validate_data_type(v)

    @property
    def name(self):
        parts = ','.join('`%s`:%s' % (k, v.name) for k, v in six.iteritems(self.field_types))
        return '{0}<{1}>'.format(type(self).__name__.lower(), parts)

    def _equals(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return isinstance(other, Struct) and \
            len(self.field_types) == len(other.field_types) and \
            all(self.field_types[k] == other.field_types[k] for k in six.iterkeys(self.field_types))

    def __hash__(self):
        if not hasattr(self, '_hash'):
            fields_hash = hash(tuple((hash(k), hash(v)) for k, v in six.iteritems(self.field_types)))
            self._hash = hash((type(self), self.nullable, fields_hash))
        return self._hash

    def can_implicit_cast(self, other):
        if isinstance(other, six.string_types):
            other = validate_data_type(other)

        return isinstance(other, Struct) and self == other and \
               self.nullable == other.nullable

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)
        return value

    @classmethod
    def parse_composite(cls, args):
        if any(not isinstance(a, tuple) and ':' not in a
               for a in args):
            raise ValueError('Every field defined in STRUCT should be given a name.')

        def conv_type_tuple(type_tuple):
            if isinstance(type_tuple, tuple):
                return type_tuple
            else:
                return tuple(v.strip('`') for v in type_tuple.split(':', 1))

        return cls(conv_type_tuple(a) for a in args)

    def validate_composite_values(self, value):
        if value is None and self.nullable:
            return value
        if isinstance(value, tuple) and hasattr(value, '_fields'):
            value = OrderedDict(compat.izip(value._fields, value))
        if not isinstance(value, dict):
            raise ValueError('Struct data type requires `dict`, instead of %s' % value)

        convert = lambda k, v, tp: (validate_value(k, string), validate_value(v, tp))
        return compat.OrderedDict(convert(k, value[k], tp) for k, tp in six.iteritems(self.field_types))


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
interval_day_time = IntervalDayTime()
interval_year_month = IntervalYearMonth()
date = Date()

_odps_primitive_data_types = dict(
    [(t.name, t) for t in (
        tinyint, smallint, int_, bigint, float_, double, string, datetime, date,
        boolean, binary, timestamp, interval_day_time, interval_year_month,
    )]
)


_composite_handlers = dict(
    varchar=Varchar,
    char=Char,
    decimal=Decimal,
    array=Array,
    map=Map,
    struct=Struct,
)


def parse_composite_types(type_str, handlers=None):
    handlers = handlers or _composite_handlers

    def _create_composite_type(typ, *args):
        prefix = None
        if ':' in typ:
            prefix, typ = typ.split(':')
            prefix = prefix.strip().strip('`')
            typ = typ.strip()
        if typ not in handlers:
            raise ValueError('Composite type %s not supported.' % typ.upper())
        ctype = handlers[typ].parse_composite(args)

        if prefix is None:
            return ctype
        else:
            return prefix, ctype

    token_stack = []
    bracket_stack = []
    token_start = 0
    type_str = type_str.strip()
    quoted = False

    for idx, ch in enumerate(type_str):
        if ch == '`':
            quoted = not quoted
        elif not quoted:
            if ch == '<' or ch == '(':
                bracket_stack.append(len(token_stack))
                token = type_str[token_start:idx].strip()
                token_stack.append(token)
                token_start = idx + 1
            elif ch == '>' or ch == ')':
                token = type_str[token_start:idx].strip()
                if token:
                    token_stack.append(token)
                bracket_pos = bracket_stack.pop()
                ctype = _create_composite_type(*token_stack[bracket_pos:])
                token_stack = token_stack[:bracket_pos]
                token_stack.append(ctype)
                token_start = idx + 1
            elif ch == ',':
                token = type_str[token_start:idx].strip()
                if token:
                    token_stack.append(token)
                token_start = idx + 1
    if len(token_stack) != 1:
        try:
            return _create_composite_type(type_str)
        except ValueError:
            return None
    return token_stack[0]


def validate_data_type(data_type):
    if isinstance(data_type, DataType):
        return data_type

    if isinstance(data_type, six.string_types):
        data_type = data_type.lower()
        if data_type in _odps_primitive_data_types:
            return _odps_primitive_data_types[data_type]

        composite_type = parse_composite_types(data_type)
        if composite_type:
            return composite_type

    raise ValueError('Invalid data type: %s' % repr(data_type))


integer_builtins = six.integer_types
float_builtins = (float,)
try:
    import numpy as np
    integer_builtins += (np.integer,)
    float_builtins += (np.float_,)
except ImportError:
    pass

_odps_primitive_to_builtin_types = compat.OrderedDict((
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
))


integer_types = (tinyint, smallint, int_, bigint)


def infer_primitive_data_type(value):
    for data_type, builtin_types in six.iteritems(_odps_primitive_to_builtin_types):
        if isinstance(value, builtin_types):
            return data_type


def _patch_pd_types(data_type):
    if timestamp not in _odps_primitive_to_builtin_types and isinstance(data_type, Timestamp):
        try:
            import pandas as pd
            _odps_primitive_to_builtin_types[timestamp] = pd.Timestamp
        except ImportError:
            raise ImportError('To use TIMESTAMP in pyodps, you need to install pandas.')
    if interval_day_time not in _odps_primitive_to_builtin_types and isinstance(data_type, IntervalDayTime):
        try:
            import pandas as pd
            _odps_primitive_to_builtin_types[interval_day_time] = pd.Timedelta
        except ImportError:
            raise ImportError('To use TIMESTAMP in pyodps, you need to install pandas.')


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
    _patch_pd_types(data_type)

    if data_type in _odps_primitive_to_builtin_types:
        res = _validate_primitive_value(value, data_type)
    elif isinstance(data_type, CompositeMixin):
        res = data_type.validate_composite_values(value)
    else:
        raise ValueError('Unknown data type: %s' % data_type)

    data_type.validate_value(res)
    return res
