#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import absolute_import
import inspect
import operator
from collections import defaultdict, Iterable

from .core import Node, NodeMetaclass
from .errors import ExpressionError
from .utils import get_attrs, is_called_by_inspector
from .. import types
from ...compat import reduce, six
from ...config import options
from ...errors import DependencyNotInstalledError
from ...utils import TEMP_TABLE_PREFIX


def run_at_once(func):
    func.run_at_once = True
    return func


def repr_obj(obj):
    if hasattr(obj, '_repr'):
        try:
            return obj._repr()
        except:
            return object.__repr__(obj)
    elif isinstance(obj, (tuple, list)):
        return ','.join(repr_obj(it) for it in obj)

    return obj


class Expr(Node):
    __slots__ = '__execution', '__ban_optimize', '_engine', '_cache_data', '_need_cache'

    def _init(self, *args, **kwargs):
        self._init_attr('_Expr__ban_optimize', False)
        self._init_attr('_engine', None)
        self._init_attr('_Expr__execution', None)

        self._init_attr('_cache_data', None)
        self._init_attr('_need_cache', False)

        super(Expr, self)._init(*args, **kwargs)

    def __repr__(self):
        if not options.interactive or is_called_by_inspector():
            return self._repr()
        else:
            if self.__execution is None:
                try:
                    self.__execution = self.execute()
                except Exception as e:
                    self.__execution = e
            return self.__execution.__repr__()

    def _repr_html_(self):
        if not options.interactive:
            return '<code>' + repr(self) + '</code>'
        else:
            if self.__execution is None:
                self.__execution = self.execute()
            else:
                if isinstance(self.__execution, Exception):
                    try:
                        raise self.__execution
                    finally:
                        self.__execution = None
            if hasattr(self.__execution, '_repr_html_'):
                return self.__execution._repr_html_()
            return repr(self.__execution)

    def visualize(self):
        from ..engines import get_default_engine

        engine = get_default_engine(self)
        return engine.visualize(self)

    @run_at_once
    def execute(self, **kwargs):
        """
        :return: execution result
        :rtype: :class:`odps.df.backends.frame.ResultFrame`
        """

        from ..engines import get_default_engine

        engine = get_default_engine(self)
        self.__execution = engine.execute(self, **kwargs)
        return self.__execution

    def compile(self):
        """
        Compile this expression into an ODPS SQL

        :return: compiled DAG
        :rtype: str
        """

        from ..engines import get_default_engine

        engine = get_default_engine(self)
        return engine.compile(self)

    @run_at_once
    def persist(self, name, partitions=None, lifecycle=None, project=None, **kwargs):
        """
        Persist the execution into a new table. If `partitions` not specfied,
        will create a new table without partitions, and insert the SQL result into it.
        If `partitions` are specified, they will be the partition fields of the new table.

        :param name: table name
        :param partitions: list of string, the partition fields
        :type partitions: list
        :param int lifecycle: table lifecycle. If absent, `options.lifecycle` will be used.
        :return: :class:`odps.df.DataFrame`

        :Example:

        >>> df = df['name', 'id', 'ds']
        >>> df.persist('odps_new_table')
        >>> df.persist('odps_new_table', partitions=['ds'])
        """

        from ..engines import get_default_engine

        engine = get_default_engine(self)

        if lifecycle is None and options.lifecycle is not None:
            lifecycle = \
                options.lifecycle if not name.startswith(TEMP_TABLE_PREFIX) \
                    else options.temp_lifecycle
        return engine.persist(self, name, partitions=partitions, lifecycle=lifecycle,
                              project=project, **kwargs)

    def cache(self):
        self._need_cache = True
        self.__ban_optimize = True
        return self

    def verify(self):
        """
        Verify if this expression can be compiled into ODPS SQL.

        :return: True if compilation succeed else False
        :rtype: bool
        """

        try:
            self.compile()

            return True
        except:
            return False

    def _repr(self):
        from .formatter import ExprFormatter

        formatter = ExprFormatter(self)
        return formatter()

    def ast(self):
        """
        Return the AST string.

        :return: AST tree
        :rtype: str
        """

        return self._repr()

    def __getattribute__(self, attr):
        try:
            if attr != '_get_attr' and attr in self._get_attr('_args'):
                return self._get_arg(attr)
            return object.__getattribute__(self, attr)
        except AttributeError as e:
            if not attr.startswith('_'):
                new_attr = '_%s' % attr
                if new_attr in self._get_attr('_args'):
                    return self._get_arg(new_attr)
            raise e

    def __setattr__(self, key, value):
        if self._get_attr('_args_indexes', True) and key in self._args_indexes \
                and hasattr(self, '_cached_args') and self._cached_args:
            cached_args = list(self._cached_args)
            cached_args[self._args_indexes[key]] = value
            if isinstance(value, six.string_types):
                values = [value, ]
            elif not isinstance(value, Iterable):
                values = [value]
            else:
                values = value
            for v in values:
                if isinstance(v, Expr):
                    v._add_parent(self)

            self._cached_args = type(self._cached_args)(cached_args)

        super(Expr, self).__setattr__(key, value)

    def _defunc(self, field):
        return field(self) if inspect.isfunction(field) else field

    @property
    def optimize_banned(self):
        return self.__ban_optimize

    @optimize_banned.setter
    def optimize_banned(self, val):
        self.__ban_optimize = val

    def __hash__(self):
        # As we will not create a new Expr unless its args are brand-new,
        # so the expr is just different to any other exprs
        return id(self) * hash(Expr)

    def __eq__(self, other):
        try:
            return self._eq(other)
        except AttributeError:
            # Due to current complexity of parent's eq,
            # by now, every expression is unequal
            return self is other

    def __ne__(self, other):
        try:
            return self._ne(other)
        except AttributeError:
            return not super(Expr, self).__eq__(other)

    def __lt__(self, other):
        return self._lt(other)

    def __le__(self, other):
        return self._le(other)

    def __gt__(self, other):
        return self._gt(other)

    def __ge__(self, other):
        return self._ge(other)

    def __add__(self, other):
        return self._add(other)

    def __radd__(self, other):
        return self._radd(other)

    def __mul__(self, other):
        return self._mul(other)

    def __rmul__(self, other):
        return self._rmul(other)

    def __div__(self, other):
        return self._div(other)

    def __rdiv__(self, other):
        return self._rdiv(other)

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __floordiv__(self, other):
        return self._floordiv(other)

    def __rfloordiv__(self, other):
        return self._rfloordiv(other)

    def __sub__(self, other):
        return self._sub(other)

    def __rsub__(self, other):
        return self._rsub(other)

    def __pow__(self, power):
        return self._pow(power)

    def __rpow__(self, power):
        return self._rpow(power)

    def __or__(self, other):
        return self._or(other)

    def __ror__(self, other):
        return self._ror(other)

    def __and__(self, other):
        return self._and(other)

    def __rand__(self, other):
        return self._rand(other)

    def __neg__(self):
        return self._neg()

    def __invert__(self):
        return self._invert()

    def __abs__(self):
        return self._abs()


class CollectionExpr(Expr):
    """
    Collection represents for the two-dimensions data.

    :Example:

    >>> # projection
    >>> df = DataFrame(o.get_table('my_table')) # DataFrame is actually a CollectionExpr
    >>> df['name', 'id']  # projection some columns
    >>> df[[df.name, df.id]]  # projection
    >>> df[df]  # means nothing, but get all the columns
    >>> df[df, df.name.lower().rename('name2')]  # projection a new columns `name2` besides all the original columns
    >>> df.select(df, name2=df.name.lower())  # projection by `select`
    >>> df.exclude('name')  # projection all columns but `name`
    >>> df[df.exclude('name'), df.name.lower()]  # `name` will not conflict any more
    >>>
    >>> # filter
    >>> df[(df.id < 3) & (df.name != 'test')]
    >>> df.filter(df.id < 3, df.name != 'test')
    >>>
    >>> # slice
    >>> df[: 10]
    >>> df.limit(10)
    >>>
    >>> # Sequence
    >>> df.name # an instance of :class:`odps.df.expr.expressions.SequenceExpr`
    >>>
    >>> # schema or dtypes
    >>> df.dtypes
    odps.Schema {
      name    string
      id      int64
    }
    >>> df.schema
    odps.Schema {
      name    string
      id      int64
    }
    """

    __slots__ = '_schema', '_source_data'
    node_name = 'Collection'

    def _init(self, *args, **kwargs):
        super(CollectionExpr, self)._init(*args, **kwargs)
        if hasattr(self, '_schema') and any(it is None for it in self._schema.names):
            raise TypeError('Schema cannot has field which name is None')
        self._source_data = getattr(self, '_source_data', None)

    def __getitem__(self, item):
        item = self._defunc(item)
        if isinstance(item, tuple):
            item = list(item)
        if isinstance(item, CollectionExpr):
            item = [item, ]

        if isinstance(item, six.string_types):
            return self._get_field(item)
        elif isinstance(item, list) and all(isinstance(it, Scalar) for it in item):
            return self._summary(item)
        elif isinstance(item, slice):
            if item.start is None and item.stop is None and item.step is None:
                return self
            return self._slice(item)
        elif isinstance(item, list):
            return self._project(item)
        else:
            field = self._get_field(item)
            if isinstance(field, BooleanSequenceExpr):
                return self.filter(item)

        if isinstance(item, SequenceExpr):
            raise ExpressionError('No boolean sequence found for filtering, '
                                  'a tuple or list is required for projection')
        raise ExpressionError('Not supported projection: collection[%s]' % repr_obj(item))

    def filter(self, *predicates):
        """
        Filter the data by predicates

        :param predicates: the conditions to filter
        :return: new collection
        :rtype: :class:`odps.df.expr.expressions.CollectionExpr`
        """
        predicates = self._get_fields(predicates)

        predicate = reduce(operator.and_, predicates)
        return FilterCollectionExpr(self, predicate, _schema=self._schema)

    def _validate_field(self, field):
        if not isinstance(field, SequenceExpr):
            return True

        if not field.is_ancestor(self):
            return False

        for path in field.all_path(self):
            if any(isinstance(n, CollectionExpr) for n in path[1: -1]):
                return False

            from .reduction import GroupedSequenceReduction
            if any(isinstance(n, GroupedSequenceReduction) for n in path):
                return False
        return True

    def _get_field(self, field):
        field = self._defunc(field)

        if isinstance(field, six.string_types):
            if field not in self._schema:
                raise ValueError('Field(%s) does not exist, please check schema' % field)
            return Column(self, _name=field, _data_type=self._schema[field].type)

        if not self._validate_field(field):
            raise ExpressionError('Cannot support projection on %s' % repr_obj(field))

        return field

    def _get_fields(self, fields):
        selects = []

        for field in fields:
            field = self._defunc(field)
            if isinstance(field, CollectionExpr):
                if any(c is self for c in field.children()):
                    selects.extend(self._get_fields(field._project_fields))
                else:
                    selects.extend(self._get_fields(field._fetch_fields()))
            else:
                selects.append(self._get_field(field))

        return selects

    def _project(self, fields):
        selects = self._get_fields(fields)

        names = [select.name for select in selects]
        typos = [select.dtype for select in selects]

        if len(names) != len(set(names)):
            counts = defaultdict(lambda: 0)
            for n in names:
                counts[n] += 1
            raise ExpressionError('Duplicate column names: %s' %
                                  ', '.join(n for n in counts if counts[n] > 1))

        return ProjectCollectionExpr(self, _fields=selects,
                                     _schema=types.Schema.from_lists(names, typos))

    def select(self, *fields, **kw):
        """
        Projection columns. Remember to avoid column names' conflict.

        :param fields: columns to project
        :param kw: columns and their names to project
        :return: new collection
        :rtype: :class:`odps.df.expr.expression.CollectionExpr`
        """
        if len(fields) == 1 and isinstance(fields[0], list):
            fields = fields[0]
        else:
            fields = list(fields)
        if kw:
            def handle(it):
                it = self._defunc(it)
                if not isinstance(it, Expr):
                    it = Scalar(it)
                return it
            fields.extend([handle(f).rename(new_name)
                           for new_name, f in six.iteritems(kw)])

        return self._project(fields)

    def exclude(self, *fields):
        """
        Projection columns which not included in the fields

        :param fields: field names
        :return: new collection
        :rtype: :class:`odps.df.expr.expression.CollectionExpr`
        """

        if len(fields) == 1 and isinstance(fields[0], list):
            exclude_fields = fields[0]
        else:
            exclude_fields = list(fields)

        exclude_fields = [self._defunc(it) for it in exclude_fields]
        exclude_fields = [field.name if not isinstance(field, six.string_types) else field
                          for field in exclude_fields]

        fields = [name for name in self._schema.names
                  if name not in exclude_fields]

        return self._project(fields)

    def _summary(self, fields):
        names = [field if isinstance(field, six.string_types) else field.name
                 for field in fields]
        typos = [self._schema.get_type(field) if isinstance(field, six.string_types)
                 else field.dtype for field in fields]
        if None in names:
            raise ExpressionError('Column does not have a name, '
                                  'please specify one by `rename`')
        return Summary(_input=self, _fields=fields,
                       _schema=types.Schema.from_lists(names, typos))

    def _slice(self, slices):
        return SliceCollectionExpr(self, _indexes=slices, _schema=self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def columns(self):
        """
        :return: columns
        :rtype: list which each element is an instance of :class:`odps.models.Column`
        """

        return [self[n] for n in self._schema.names]

    def _fetch_fields(self):
        return [self._get_field(name) for name in self._schema.names]

    @property
    def _project_fields(self):
        return self._fetch_fields()

    def _data_source(self):
        if hasattr(self, '_source_data') and self._source_data is not None:
            yield self._source_data

    def __getattr__(self, attr):
        try:
            obj = object.__getattribute__(self, attr)

            return obj
        except AttributeError as e:
            if attr in object.__getattribute__(self, '_schema'):
                return self[attr]

            raise e

    def output_type(self):
        return 'collection'

    def limit(self, n):
        return self[:n]

    @run_at_once
    def head(self, n=None, **kwargs):
        """
        Return the first n rows. Execute at once.

        :param n:
        :return: result frame
        :rtype: :class:`odps.df.backends.frame.ResultFrame`
        """
        if n is None:
            n = options.display.max_rows

        from ..engines import get_default_engine

        engine = get_default_engine(self)
        return engine.execute(self, head=n, **kwargs)

    @run_at_once
    def tail(self, n=None, **kwargs):
        """
        Return the last n rows. Execute at once.

        :param n:
        :return: result frame
        :rtype: :class:`odps.df.backends.frame.ResultFrame`
        """
        if n is None:
            n = options.display.max_rows

        from ..engines import get_default_engine

        engine = get_default_engine(self)
        return engine.execute(self, tail=n, **kwargs)

    @run_at_once
    def to_pandas(self, wrap=False, **kwargs):
        """
        Convert to pandas DataFrame. Execute at once.

        :param wrap: if True, wrap the pandas DataFrame into a PyOdps DataFrame
        :return: pandas DataFrame
        """

        try:
            import pandas as pd
        except ImportError:
            raise DependencyNotInstalledError(
                    'to_pandas requires `pandas` library')

        res = self.execute(**kwargs).values
        if wrap:
            from .. import DataFrame
            return DataFrame(res)
        return res

    @property
    def dtypes(self):
        return self.schema

    def view(self):
        """
        Clone a same collection. useful for self-join.

        :return:
        """

        kv = dict((attr, getattr(self, attr)) for attr in get_attrs(self))
        return type(self)(**kv)

    def accept(self, visitor):
        if self._source_data is not None:
            visitor.visit_source_collection(self)
        else:
            raise NotImplementedError


_cached_typed_expr = dict()


class TypedExpr(Expr):
    __slots__ = '_name', '_source_name'

    @classmethod
    def _get_type(cls, *args, **kwargs):
        # return the data type which extracted from args and kwargs
        raise NotImplementedError

    @classmethod
    def _typed_classes(cls, *args, **kwargs):
        # return allowed data types
        raise NotImplementedError

    @classmethod
    def _base_class(cls, *args, **kwargs):
        # base class, SequenceExpr or Scalar
        raise NotImplementedError

    @classmethod
    def _new_cls(cls, *args, **kwargs):
        data_type = cls._get_type(*args, **kwargs)
        if data_type:
            base_class = cls._base_class(*args, **kwargs)
            typed_classes = cls._typed_classes(*args, **kwargs)

            data_type = types.validate_data_type(data_type)
            name = data_type.CLASS_NAME + base_class.__name__
            # get the typed class, e.g. Int64SequenceExpr, StringScalar
            typed_cls = globals().get(name)
            assert typed_cls is not None

            if issubclass(cls, typed_cls):
                return cls
            elif cls == base_class:
                return typed_cls
            elif cls in typed_classes:
                return typed_cls

            keys = (cls, typed_cls)
            if keys in _cached_typed_expr:
                return _cached_typed_expr[keys]

            mros = inspect.getmro(cls)
            has_data_type = len([sub for sub in mros if sub in typed_classes]) > 0
            if has_data_type:
                mros = mros[1:]
            subs = [sub for sub in mros if sub not in typed_classes]
            subs.insert(1, typed_cls)

            bases = list()
            for sub in subs[::-1]:
                for i in range(len(bases)):
                    if bases[i] is None:
                        continue
                    if issubclass(sub, bases[i]):
                        bases[i] = None
                bases.append(sub)
            bases = tuple(base for base in bases if base is not None)

            dic = dict()
            if hasattr(cls, '_args'):
                dic['_args'] = cls._args
            dic['__slots__'] = cls.__slots__ + getattr(cls, '_slots', ())
            dic['_add_args_slots'] = True
            try:
                accept_cls = next(c for c in bases if c.__name__ == cls.__name__)
                if hasattr(accept_cls, 'accept'):
                    dic['accept'] = accept_cls.accept
            except StopIteration:
                pass
            clz = type(cls.__name__, bases, dic)
            _cached_typed_expr[keys] = clz
            return clz
        else:
            return cls

    def __new__(cls, *args, **kwargs):
        clz = cls._new_cls(*args, **kwargs)
        return super(TypedExpr, clz).__new__(clz)

    @classmethod
    def _new(cls, *args, **kwargs):
        return cls._new_cls(*args, **kwargs)(*args, **kwargs)

    def is_renamed(self):
        return self._name is not None and self._source_name is not None and \
               self._name != self._source_name

    def rename(self, new_name):
        if new_name == self._name:
            return self

        attr_dict = dict((attr, getattr(self, attr, None)) for attr in get_attrs(self))
        attr_dict['_source_name'] = self._source_name
        attr_dict['_name'] = new_name

        return type(self)(**attr_dict)

    @property
    def name(self):
        return self._name

    @property
    def source_name(self):
        return self._source_name

    def astype(self, data_type):
        raise NotImplementedError

    def cast(self, t):
        return self.astype(t)


class SequenceExpr(TypedExpr):
    """
    Sequence represents for 1-dimension data.
    """

    __slots__ = '_data_type', '_source_data_type'

    @classmethod
    def _get_type(cls, *args, **kwargs):
        return types.validate_data_type(kwargs.get('_data_type'))

    @classmethod
    def _typed_classes(cls, *args, **kwargs):
        return _typed_sequence_exprs

    @classmethod
    def _base_class(cls, *args, **kwargs):
        return SequenceExpr

    def _init(self, *args, **kwargs):
        self._name = None
        super(SequenceExpr, self)._init(*args, **kwargs)

        if '_data_type' in kwargs:
            self._data_type = types.validate_data_type(kwargs.get('_data_type'))

        if '_source_name' not in kwargs:
            self._source_name = self._name

        if '_source_data_type' in kwargs:
            self._source_data_type = types.validate_data_type(kwargs.get('_source_data_type'))
        else:
            self._source_data_type = self._data_type

    def cache(self):
        raise ExpressionError('Cache operation does not support for sequence.')

    @run_at_once
    def head(self, n=None, **kwargs):
        """
        Return first n rows. Execute at once.

        :param n:
        :return: result frame
        :rtype: :class:`odps.df.expr.expressions.CollectionExpr`
        """

        if n is None:
            n = options.display.max_rows

        from ..engines import get_default_engine

        engine = get_default_engine(self)
        return engine.execute(self, head=n, **kwargs)

    @run_at_once
    def tail(self, n=None, **kwargs):
        """
        Return the last n rows. Execute at once.

        :param n:
        :return:
        """

        if n is None:
            n = options.display.max_rows

        from ..engines import get_default_engine

        engine = get_default_engine(self)
        return engine.execute(self, tail=n, **kwargs)

    @run_at_once
    def to_pandas(self, wrap=False, **kwargs):
        """
        Convert to pandas Series. Execute at once.

        :param wrap: if True, wrap the pandas DataFrame into a PyOdps DataFrame
        :return: pandas Series
        """

        try:
            import pandas as pd
        except ImportError:
            raise DependencyNotInstalledError(
                    'to_pandas requires for `pandas` library')

        df = self.execute(**kwargs).values
        if wrap:
            from .. import DataFrame
            df = DataFrame(df)
        return df[self.name]

    @property
    def data_type(self):
        return self._data_type

    @property
    def source_data_type(self):
        return self._source_data_type

    @property
    def dtype(self):
        """
        Return the data type. Available types:
        int8, int16, int32, int64, float32, float64, boolean, string, decimal, datetime

        :return: the data type
        """
        return self._data_type

    def astype(self, data_type):
        """
        Cast to a new data type.

        :param data_type: the new data type
        :return: casted sequence

        :Example:

        >>> df.id.astype('float')
        """

        data_type = types.validate_data_type(data_type)

        if data_type == self._data_type:
            return self

        attr_dict = dict()
        attr_dict['_data_type'] = data_type
        attr_dict['_source_data_type'] = self._source_data_type
        attr_dict['_input'] = self

        new_sequence = AsTypedSequenceExpr(**attr_dict)

        return new_sequence

    def output_type(self):
        return 'sequence(%s)' % repr(self._data_type)

    def accept(self, visitor):
        visitor.visit_sequence(self)


class BooleanSequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(BooleanSequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.boolean


class Int8SequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(Int8SequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.int8


class Int16SequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(Int16SequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.int16


class Int32SequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(Int32SequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.int32


class Int64SequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(Int64SequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.int64


class Float32SequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(Float32SequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.float32


class Float64SequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(Float64SequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.float64


class DecimalSequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(DecimalSequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.decimal


class StringSequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(StringSequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.string


class DatetimeSequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(DatetimeSequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.datetime


_typed_sequence_exprs = [globals()[t.__class__.__name__ + SequenceExpr.__name__]
                         for t in types._data_types.values()]


class AsTypedSequenceExpr(SequenceExpr):
    _args = '_input',
    node_name = "TypedSequence"

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_cast(self)

    @property
    def name(self):
        return self._name or self._input.name

    @property
    def source_name(self):
        return self._source_name or self._input.source_name

    @property
    def dtype(self):
        return self._data_type or self._input.data_type

    @property
    def source_type(self):
        return self._source_data_type or self._input._source_data_type


class Column(SequenceExpr):
    _args = '_input',

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_column(self)


class Scalar(TypedExpr):
    """
    Represent for the scalar type.
    """

    __slots__ = '_value', '_value_type', '_source_value_type'

    @classmethod
    def _get_type(cls, *args, **kwargs):
        value = args[0] if len(args) > 0 else None
        value_type = args[1] if len(args) > 1 else None

        val = kwargs.get('_value')
        if val is None:
            val = value

        value_type = kwargs.get('_value_type', None) or value_type

        if val is None and value_type is None:
            raise ValueError('Either value or value_type should be provided')

        if not isinstance(val, NodeMetaclass):
            return types.validate_value_type(val, value_type)
        else:
            return value_type

    @classmethod
    def _typed_classes(cls, *args, **kwargs):
        return _typed_scalar_exprs

    @classmethod
    def _base_class(cls, *args, **kwargs):
        return Scalar

    @classmethod
    def _transform(cls, *args, **kwargs):
        value = args[0] if len(args) > 0 else None
        value_type = args[1] if len(args) > 1 else None

        if ('_value' not in kwargs or kwargs['_value'] is None) and \
                value is not None:
            kwargs['_value'] = value
        if ('_value_type' not in kwargs or kwargs['_value_type'] is None) and \
                value_type is not None:
            kwargs['_value_type'] = types.validate_data_type(value_type)

        kwargs['_value_type'] = types.validate_value_type(kwargs.get('_value'),
                                                          kwargs.get('_value_type'))

        if '_source_name' not in kwargs:
            kwargs['_source_name'] = kwargs.get('_name')

        if '_source_value_type' in kwargs:
            kwargs['_source_value_type'] = types.validate_data_type(kwargs['_source_value_type'])
        else:
            kwargs['_source_value_type'] = kwargs['_value_type']
        return kwargs

    def __new__(cls, *args, **kwargs):
        kwargs = cls._transform(*args, **kwargs)
        return super(Scalar, cls).__new__(cls, **kwargs)

    def _init(self, *args, **kwargs):
        self._init_attr('_name', None)
        self._init_attr('_value', None)

        kwargs = self._transform(*args, **kwargs)
        super(Scalar, self)._init(**kwargs)

    def equals(self, other):
        return super(Scalar, self).equals(other)

    @property
    def value(self):
        return self._value

    @property
    def value_type(self):
        return self._value_type

    @property
    def dtype(self):
        return self._value_type

    def output_type(self):
        return 'Scalar[%s]' % repr(self._value_type)

    def astype(self, value_type):
        value_type = types.validate_data_type(value_type)

        if value_type == self._value_type:
            return self

        attr_dict = dict()

        attr_dict['_input'] = self
        attr_dict['_value_type'] = value_type
        attr_dict['_source_value_type'] = self._source_value_type

        new_scalar = AsTypedScalar(**attr_dict)

        return new_scalar

    def to_sequence(self):
        if self._value is None:
            attr_values = dict((attr, getattr(self, attr)) for attr in get_attrs(self))

            attr_values['_data_type'] = attr_values.pop('_value_type')
            if '_source_value_type' in attr_values:
                attr_values['_source_data_type'] = attr_values.pop('_source_value_type')
            del attr_values['_value']

            cls = next(c for c in inspect.getmro(type(self))[1:]
                       if c.__name__ == type(self).__name__ and not issubclass(c, Scalar))
            seq = cls._new(**attr_values)
            return seq

        raise ExpressionError('Cannot convert valued scalar to sequence')

    def accept(self, visitor):
        if self._value is not None:
            visitor.visit_scalar(self)
        else:
            raise NotImplementedError


class AsTypedScalar(Scalar):
    _args = '_input',
    node_name = "TypedScalar"

    def accept(self, visitor):
        return visitor.visit_cast(self)

    @property
    def name(self):
        return self._name or self._input.name

    @property
    def source_name(self):
        return self._source_name or self._input.source_name

    @property
    def value(self):
        return self._value

    @property
    def value_type(self):
        return self._value_type

    @property
    def dtype(self):
        return self._value_type

    @property
    def source_type(self):
        return self._source_value_type


class BooleanScalar(Scalar):
    def _init(self, *args, **kwargs):
        super(BooleanScalar, self)._init(*args, **kwargs)
        self._value_type = types.boolean


class Int8Scalar(Scalar):
    def _init(self, *args, **kwargs):
        super(Int8Scalar, self)._init(*args, **kwargs)
        self._value_type = types.int8


class Int16Scalar(Scalar):
    def _init(self, *args, **kwargs):
        super(Int16Scalar, self)._init(*args, **kwargs)
        self._value_type = types.int16


class Int32Scalar(Scalar):
    def _init(self, *args, **kwargs):
        super(Int32Scalar, self)._init(*args, **kwargs)
        self._value_type = types.int32


class Int64Scalar(Scalar):
    def _init(self, *args, **kwargs):
        super(Int64Scalar, self)._init(*args, **kwargs)
        self._value_type = types.int64


class Float32Scalar(Scalar):
    def _init(self, *args, **kwargs):
        super(Float32Scalar, self)._init(*args, **kwargs)
        self._value_type = types.float32


class Float64Scalar(Scalar):
    def __int__(self, *args, **kwargs):
        super(Float64Scalar, self)._init(*args, **kwargs)
        self._value_type = types.float64


class DecimalScalar(Scalar):
    def _init(self, *args, **kwargs):
        super(DecimalScalar, self)._init(*args, **kwargs)
        self._value_type = types.decimal


class StringScalar(Scalar):
    def _init(self, *args, **kwargs):
        super(StringScalar, self)._init(*args, **kwargs)
        self._value_type = types.string


class DatetimeScalar(Scalar):
    def _init(self, *args, **kwargs):
        super(DatetimeScalar, self)._init(*args, **kwargs)
        self._value_type = types.datetime


_typed_scalar_exprs = [globals()[t.__class__.__name__ + Scalar.__name__]
                       for t in types._data_types.values()]


class BuiltinFunction(Scalar):
    __slots__ = '_func_name', '_func_args', '_func_kwargs'

    def __init__(self, name=None, rtype=None, args=(), **kwargs):
        rtype = rtype or kwargs.pop('_value_type', types.string)
        rtype = types.validate_data_type(rtype)
        func_name = name or kwargs.pop('_func_name', None)
        func_args = args or kwargs.pop('_func_args', ())
        super(BuiltinFunction, self).__init__(_func_name=func_name,
                                              _func_args=func_args,
                                              _value_type=rtype,
                                              **kwargs)

    def accept(self, visitor):
        visitor.visit_builtin_function(self)


class FilterCollectionExpr(CollectionExpr):
    _args = '_input', '_predicate'
    node_name = 'Filter'

    def _init(self, *args, **kwargs):
        super(FilterCollectionExpr, self)._init(*args, **kwargs)

        if self._schema is None:
            self._schema = self._input.schema

    def iter_args(self):
        for it in zip(['collection', 'predicate'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        visitor.visit_filter_collection(self)


class ProjectCollectionExpr(CollectionExpr):
    _args = '_input', '_fields'
    node_name = 'Projection'

    def _init(self, *args, **kwargs):
        fields = kwargs.get('_fields')
        if fields is None and len(args) >= 2:
            fields = args[1]
        for field in fields:
            if field.name is None:
                raise ExpressionError('Column does not have a name, '
                                      'please specify one by `rename`: %s' % repr_obj(field._repr()))

        super(ProjectCollectionExpr, self)._init(*args, **kwargs)

    @property
    def _project_fields(self):
        return self._fields

    def iter_args(self):
        for it in zip(['collection', 'selections'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    @property
    def fields(self):
        return self._fields

    def accept(self, visitor):
        visitor.visit_project_collection(self)


class SliceCollectionExpr(CollectionExpr):

    _args = '_input', '_indexes'
    node_name = 'Slice'

    def _init(self, *args, **kwargs):
        super(SliceCollectionExpr, self)._init(*args, **kwargs)

        if isinstance(self._indexes, slice):
            scalar = lambda v: Scalar(_value=v) if v is not None else None
            self._indexes = scalar(self._indexes.start), \
                            scalar(self._indexes.stop), scalar(self._indexes.step)

    @property
    def start(self):
        res = self._indexes[0]
        return res.value if res is not None else None

    @property
    def stop(self):
        res = self._indexes[1]
        return res.value if res is not None else None

    @property
    def step(self):
        res = self._indexes[2]
        return res.value if res is not None else None

    @property
    def input(self):
        return self._input

    def iter_args(self):
        args = [self._input] + list(self._indexes)
        for it in zip(['collection', 'start', 'stop', 'step'], args):
            yield it

    def accept(self, visitor):
        visitor.visit_slice_collection(self)


class Summary(Expr):
    __slots__ = '_schema',
    _args = '_input', '_fields'

    def _init(self, *args, **kwargs):
        super(Summary, self)._init(*args, **kwargs)
        if hasattr(self, '_schema') and any(it is None for it in self._schema.names):
            raise TypeError('Schema cannot has field which name is None')

    @property
    def input(self):
        return self._input

    @property
    def fields(self):
        return self._fields

    @property
    def schema(self):
        return self._schema

    def iter_args(self):
        for it in zip(['collection', 'fields'], self.args):
            yield it

    def accept(self, visitor):
        visitor.visit_project_collection(self)


from . import element
from . import arithmetic
from . import reduction
from . import groupby
from . import collections
from . import window
from . import math
from . import strings
from . import datetimes
from . import merge
from ..tools import *


# hack for count
def _count(expr, *args, **kwargs):
    if len(args) + len(kwargs) > 0:
        from .strings import _count
        return _count(expr, *args, **kwargs)
    else:
        from .reduction import count
        return count(expr)


StringSequenceExpr.count = _count
