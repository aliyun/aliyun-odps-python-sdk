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

from __future__ import absolute_import
import inspect
import functools
import operator
import sys
from collections import defaultdict

from .core import Node, NodeMetaclass
from .errors import ExpressionError
from .utils import get_attrs, is_called_by_inspector, highest_precedence_data_type, new_id, \
    is_changed, get_proxied_expr
from .. import types
from ...compat import reduce, isvalidattr, dir2, OrderedDict, lkeys, six, futures, Iterable
from ...config import options
from ...errors import NoSuchObject, DependencyNotInstalledError
from ...utils import TEMP_TABLE_PREFIX, to_binary, deprecated, survey
from ...models import Schema


class ReprWrapper(object):
    def __init__(self, func, repr):
        self._func = func
        self._repr = repr
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __repr__(self):
        return self._repr(self._func)


def _wrap_method_repr(func):
    def inner(*args, **kwargs):
        obj = func(*args, **kwargs)
        if inspect.ismethod(obj):
            def _repr(x):
                instance = getattr(x, 'im_self', getattr(x, '__self__', None))
                method = 'bound method' if instance is not None else 'unbound method'
                if instance is not None:
                    return '<%(method)s %(instance)s.%(name)s>' % {
                        'method': method,
                        'instance': getattr(instance, 'node_name', instance.__class__.__name__),
                        'name': x.__name__
                    }
                else:
                    return '<function __main__.%s>' % x.__name__

            return ReprWrapper(obj, _repr)
        return obj

    return inner


def make_max_pt_function(expr=None):
    from ...models import Table
    from .. import func

    def max_pt(table_name=None):
        if isinstance(getattr(expr, 'data', None), Table):
            table_name = table_name or expr.data.name
        return func.max_pt(table_name)

    return max_pt


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
    __slots__ = '_deps', '_ban_optimize', '_engine', '_need_cache', '_mem_cache', '__execution', '_id'

    def _init(self, *args, **kwargs):
        """
        _deps is used for common dependencies.
        When a expr depend on other exprs, and the expr is not calculated from the others,
        the _deps are specified to identify the dependencies.
        """
        self._init_attr('_deps', None)

        self._init_attr('_ban_optimize', False)
        self._init_attr('_engine', None)
        self._init_attr('_Expr__execution', None)

        self._init_attr('_need_cache', False)
        self._init_attr('_mem_cache', False)

        if '_id' not in kwargs:
            kwargs['_id'] = new_id()

        super(Expr, self)._init(*args, **kwargs)

    def __repr__(self):
        if not options.interactive or is_called_by_inspector():
            return self._repr()
        else:
            if isinstance(self.__execution, Exception):
                self.__execution = None
            if self.__execution is None:
                try:
                    self.__execution = self.execute()
                except Exception as e:
                    self.__execution = e
                    raise
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
                        return
                    finally:
                        self.__execution = None
            if hasattr(self.__execution, '_repr_html_'):
                return self.__execution._repr_html_()
            return repr(self.__execution)

    def _handle_delay_call(self, method, *args, **kwargs):
        delay = kwargs.pop('delay', None)
        if delay is not None:
            future = delay.register_item(method, *args, **kwargs)
            return future
        else:
            from ..engines import get_default_engine
            engine = get_default_engine(self)

            wrapper = kwargs.pop('wrapper', None)
            result = getattr(engine, method)(*args, **kwargs)
            if wrapper is None:
                return result
            async_ = kwargs.get('async_', kwargs.get('async', False))
            if async_:
                user_future = futures.Future()

                def _relay(f):
                    try:
                        user_future.set_result(wrapper(f.result()))
                    except:
                        if hasattr(f, 'exception_info'):
                            user_future.set_exception_info(*f.exception_info())
                        else:
                            user_future.set_exception(f.exception())

                result.add_done_callback(_relay)
                return user_future
            else:
                return wrapper(result)

    def visualize(self):
        from ..engines import get_default_engine

        engine = get_default_engine(self)
        return engine.visualize(self)

    def execute(self, **kwargs):
        """
        :param hints: settings for SQL, e.g. `odps.sql.mapper.split.size`
        :type hints: dict
        :param priority: instance priority, 9 as default
        :type priority: int
        :param running_cluster: cluster to run this instance
        :return: execution result
        :rtype: :class:`odps.df.backends.frame.ResultFrame`
        """
        _wrapper = kwargs.pop('wrapper', None)

        def wrapper(result):
            self.__execution = result
            if _wrapper is not None:
                return _wrapper(result)
            else:
                return result

        return self._handle_delay_call('execute', self, wrapper=wrapper, **kwargs)

    def compile(self):
        """
        Compile this expression into an ODPS SQL

        :return: compiled DAG
        :rtype: str
        """

        from ..engines import get_default_engine

        engine = get_default_engine(self)
        return engine.compile(self)

    def persist(self, name, partitions=None, partition=None, lifecycle=None, project=None, **kwargs):
        """
        Persist the execution into a new table. If `partitions` not specified,
        will create a new table without partitions if the table does not exist,
        and insert the SQL result into it.
        If `partitions` are specified, they will be the partition fields of the new table.
        If `partition` is specified, the data will be inserted into the exact partition of the table.

        :param name: table name
        :param partitions: list of string, the partition fields
        :type partitions: list
        :param partition: persist to a specified partition
        :type partition: string or PartitionSpec
        :param lifecycle: table lifecycle. If absent, `options.lifecycle` will be used.
        :type lifecycle: int
        :param project: project name, if not provided, will be the default project
        :param hints: settings for SQL, e.g. `odps.sql.mapper.split.size`
        :type hints: dict
        :param priority: instance priority, 9 as default
        :type priority: int
        :param running_cluster: cluster to run this instance
        :param overwrite: overwrite the table, True as default
        :type overwrite: bool
        :param drop_table: drop table if exists, False as default
        :type drop_table: bool
        :param create_table: create table first if not exits, True as default
        :type create_table: bool
        :param drop_partition: drop partition if exists, False as default
        :type drop_partition: bool
        :param create_partition: create partition if not exists, None as default
        :type create_partition: bool
        :param cast: cast all columns' types as the existed table, False as default
        :type cast: bool
        :return: :class:`odps.df.DataFrame`

        :Example:

        >>> df = df['name', 'id', 'ds']
        >>> df.persist('odps_new_table')
        >>> df.persist('odps_new_table', partition='pt=test')
        >>> df.persist('odps_new_table', partitions=['ds'])
        """
        if lifecycle is None and options.lifecycle is not None:
            lifecycle = \
                options.lifecycle if not name.startswith(TEMP_TABLE_PREFIX) \
                    else options.temp_lifecycle

        return self._handle_delay_call('persist', self, name, partitions=partitions, partition=partition,
                                       lifecycle=lifecycle, project=project, **kwargs)

    def cache(self, mem=False):
        self._need_cache = True
        self._mem_cache = mem
        self._ban_optimize = True
        return self

    def uncache(self):
        self._need_cache = False
        self._mem_cache = False

        from ..backends.context import context
        context.uncache(self)

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

    @_wrap_method_repr
    def __getattribute__(self, attr):
        try:
            return super(Expr, self).__getattribute__(attr)
        except AttributeError as e:
            if not attr.startswith('_'):
                new_attr = '_%s' % attr
                if new_attr in object.__getattribute__(self, '_args_indexes'):
                    try:
                        return object.__getattribute__(self, new_attr)
                    except AttributeError:
                        return

            raise

    def _defunc(self, field):
        return field(self) if inspect.isfunction(field) else field

    @property
    def optimize_banned(self):
        return self._ban_optimize

    @optimize_banned.setter
    def optimize_banned(self, val):
        self._ban_optimize = val

    @property
    def args(self):
        if not self._deps:
            return super(Expr, self).args
        return super(Expr, self).args + self.deps

    @property
    def deps(self):
        if self._deps is None:
            return
        return tuple(dep if not isinstance(dep, tuple) else dep[0]
                     for dep in self._deps)

    def add_deps(self, *deps):
        dependencies = []
        if len(deps) == 1 and isinstance(deps[0], Iterable):
            dependencies.append([d for d in dependencies
                                 if isinstance(d, Expr)])
        else:
            dependencies.extend(deps)

        if getattr(self, '_deps', None) is None:
            self._deps = dependencies
        else:
            self._deps.extend(dependencies)

    def substitute(self, old_arg, new_arg, dag=None):
        super(Expr, self).substitute(old_arg, new_arg, dag=dag)

        new_deps = []
        if self._deps is not None:
            for dep in self._deps:
                if isinstance(dep, tuple):
                    node = dep[0]
                    if node is old_arg:
                        new_deps.append((new_arg, ) + dep[1:])
                    else:
                        new_deps.append(dep)
                else:
                    if dep is old_arg:
                        new_deps.append(new_arg)
                    else:
                        new_deps.append(dep)

        self._deps = new_deps

    def rebuild(self):
        # used in the dynamic setting, do nothing by default
        # `rebuild` will copy itself, and apply all changes to the new one
        return self.copy()

    def __hash__(self):
        return self._node_id * hash(Expr)

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

    def __mod__(self, other):
        return self._mod(other)

    def __rmod__(self, other):
        return self._rmod(other)

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

    __slots__ = (
        '_schema', '_source_data', '_proxy',
        '_ml_fields_cache', '_ml_uplink', '_ml_operations',
    )
    node_name = 'Collection'

    def _init(self, *args, **kwargs):
        self._init_attr('_source_data', None)
        self._init_attr('_proxy', None)

        self._init_attr('_ml_fields_cache', None)
        self._init_attr('_ml_uplink', [])
        self._init_attr('_ml_operations', [])

        super(CollectionExpr, self)._init(*args, **kwargs)

        if hasattr(self, '_schema') and any(it is None for it in self._schema.names):
            raise TypeError('Schema cannot has field whose name is None')

    def __dir__(self):
        dir_set = set(dir2(self)) | set([c.name for c in self.schema if isvalidattr(c.name)])
        return sorted(dir_set)

    def __getitem__(self, item):
        item = self._defunc(item)
        if isinstance(item, tuple):
            item = list(item)
        if isinstance(item, CollectionExpr):
            item = [item, ]

        if isinstance(item, six.string_types):
            return self._get_field(item)
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

    def _set_field(self, value, value_dag=None):
        expr = self.copy() if self._proxy is None else self._proxy._input

        if value_dag is None:
            value_dag = value.to_dag(copy=False, validate=False)

        if value_dag.contains_node(self):
            value_dag.substitute(self, expr)
            value = value_dag.root

        if self._proxy is not None:
            self._proxy._setitem(value, value_dag=value_dag)
            return

        fields = [f if f != value.name else value for f in self._schema.names]
        if value.name not in self._schema:
            fields.append(value)
        # make the isinstance to check the proxy type
        self.__class__ = type(self.__class__.__name__, (self.__class__,), {})
        self._proxy = expr.select(fields)

    def __setitem__(self, key, value):
        if not isinstance(value, Expr) and value is not None:
            value = Scalar(value)

        if not isinstance(key, tuple):
            if value is None:
                raise ValueError('Cannot determine type for column %s with None value.' % key)
            column_name = key
            value = value.rename(column_name)
        else:
            conds = key[:-1]
            conds = [self._defunc(c) for c in conds]
            if not all(isinstance(c.dtype, types.Boolean) for c in conds):
                raise ValueError('Conditions should be boolean expressions or boolean columns')
            if len(conds) == 1:
                cond = conds[0]
            else:
                cond = reduce(operator.and_, conds)
            column_name = key[-1]
            if column_name not in self._schema:
                if value is None:
                    raise ValueError('Cannot determine type for column %s with None value.' % key)
                default_col = Scalar(_value_type=value.dtype)
            else:
                default_col = self[column_name]
                if value is None:
                    value = Scalar(_value_type=default_col.dtype)
            value = cond.ifelse(value, default_col).rename(column_name)
        self._set_field(value)

    def __delitem__(self, key):
        if key not in self.schema:
            raise KeyError('Field({0}) does not exist'.format(key))

        if self._proxy is not None:
            return self._proxy._delitem(key)

        fields = [n for n in self._schema.names if n != key]
        # make the instance to check the proxy type
        self.__class__ = type(self.__class__.__name__, (self.__class__,), {})
        self._proxy = self.copy().select(fields)

    def __delattr__(self, item):
        if item in self._schema:
            return self.__delitem__(item)

        super(CollectionExpr, self).__delattr__(item)

    def __setattr__(self, key, value):
        try:
            if key != '_proxy' and object.__getattribute__(self, '_proxy'):
                return setattr(self._proxy, key, value)
        except AttributeError:
            pass

        Expr.__setattr__(self, key, value)

    def __getattribute__(self, attr):
        try:
            proxy = object.__getattribute__(self, '_proxy')
            if attr == '_proxy':
                return proxy

            if proxy:
                # delegate everything to proxy object
                return getattr(proxy, attr)
        except AttributeError:
            pass

        try:
            if attr in object.__getattribute__(self, '_schema')._name_indexes:
                cls_attr = getattr(type(self), attr, None)
                if cls_attr is None or inspect.ismethod(cls_attr) or inspect.isfunction(cls_attr):
                    return self[attr]
        except AttributeError:
            pass

        return super(CollectionExpr, self).__getattribute__(attr)

    def query(self, expr):
        """
        Query the data with a boolean expression.

        :param expr: the query string, you can use '@' character refer to environment variables.
        :return: new collection
        :rtype: :class:`odps.df.expr.expressions.CollectionExpr`

        """
        from .query import CollectionVisitor

        if not isinstance(expr, six.string_types):
            raise ValueError('expr must be a string')

        frame = sys._getframe(2).f_locals
        try:
            env = frame.copy()
        finally:
            del frame

        env['max_pt'] = ReprWrapper(make_max_pt_function(self), repr)

        visitor = CollectionVisitor(self, env)
        predicate = visitor.eval(expr)
        return self.filter(predicate)

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

    @deprecated('The function `filter_partition` is deprecated, please use `filter_parts` instead'
                "and change the predicate parameter into `pt1=1,pt2=2/pt1=2,pt2=1` form.")
    @survey
    def filter_partition(self, predicate='', exclude=True):
        if isinstance(predicate, six.string_types):
            part_reprs = '/'.join(','.join(p.split('/')) for p in predicate.split(','))
        else:
            part_reprs = predicate
        return self.filter_parts(part_reprs, exclude)

    def filter_parts(self, predicate='', exclude=True):
        """
        Filter the data by partition string. A partition string looks like `pt1=1,pt2=2/pt1=2,pt2=1`, where
        comma (,) denotes 'and', while (/) denotes 'or'.

        :param str|Partition predicate: predicate string of partition filter
        :param bool exclude: True if you want to exclude partition fields, otherwise False. True for default.
        :return: new collection
        :rtype: :class:`odps.df.expr.expressions.CollectionExpr`
        """
        source = self._source_data
        if source is None:
            raise ExpressionError('Can only filter on data sources.')

        def _parse_partition_predicate(p):
            if '=' not in p:
                raise ExpressionError('Illegal partition predicate.')
            field_name, field_value = [s.strip() for s in p.split('=', 1)]
            if not hasattr(source, 'schema'):
                raise ExpressionError('filter_partition can only be applied on ODPS DataFrames')
            if field_name not in source.schema:
                raise ExpressionError('Column `%s` not exists in input collection' % field_name)
            if field_name not in source.schema._partition_schema:
                raise ExpressionError('`%s` is not a partition column' % field_name)
            part_col = self[field_name]
            if field_value.startswith('\'') or field_value.startswith('\"'):
                encoding = 'string-escape' if six.PY2 else 'unicode-escape'
                field_value = to_binary(field_value.strip('"\'')).decode(encoding)

            if isinstance(part_col.data_type, types.Integer):
                field_value = int(field_value)
            elif isinstance(part_col.data_type, types.Float):
                field_value = float(field_value)
            return part_col == field_value

        from ...models.partition import Partition
        from ...types import PartitionSpec

        if isinstance(predicate, Partition):
            predicate = predicate.partition_spec
        if isinstance(predicate, PartitionSpec):
            predicate = ','.join("%s='%s'" % (k, v) for k, v in six.iteritems(predicate.kv))

        if isinstance(predicate, list):
            predicate = '/'.join(str(s) for s in predicate)
        elif not isinstance(predicate, six.string_types):
            raise ExpressionError('Only accept string predicates.')

        if not predicate:
            predicate_obj = None
        else:
            part_formatter = lambda p: reduce(operator.and_, map(_parse_partition_predicate, p.split(',')))
            predicate_obj = reduce(operator.or_, map(part_formatter, predicate.split('/')))

        if not source.schema.partitions:
            raise ExpressionError('No partition columns in the collection.')
        if exclude:
            columns = [c for c in self.schema if c.name not in source.schema._partition_schema]
            new_schema = types.Schema.from_lists([c.name for c in columns], [c.type for c in columns])
            return FilterPartitionCollectionExpr(self, predicate_obj, _schema=new_schema, _predicate_string=predicate)
        else:
            return self.filter(predicate_obj)

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

    @staticmethod
    def _backtrack_field(field, collection):
        from .window import RankOp

        for col in field.traverse(top_down=True, unique=True,
                                  stop_cond=lambda x: isinstance(x, (Column, RankOp))):
            if isinstance(col, Column):
                changed = is_changed(collection, col)
                if changed or changed is None:
                    return
                elif col.source_name not in collection.schema:
                    return
                else:
                    col.substitute(col._input, collection)
            elif isinstance(col, RankOp) and col._input is not collection:
                col.substitute(col._input, collection)

        return field

    def _get_field(self, field):
        from .reduction import GroupedSequenceReduction

        field = self._defunc(field)

        if isinstance(field, six.string_types):
            if field not in self._schema:
                raise ValueError('Field(%s) does not exist, please check schema' % field)
            cls = Column
            if callable(getattr(type(self), field, None)):
                cls = CallableColumn
            return cls(self, _name=field, _data_type=self._schema[field].type)

        if not self._validate_field(field):
            new_field = None
            if not isinstance(field, GroupedSequenceReduction):
                # the reduction is not allowed
                new_field = self._backtrack_field(field, self)
            if new_field is None:
                raise ExpressionError('Cannot support projection on %s' % repr_obj(field))
            field = new_field
        return field

    def _get_fields(self, fields, ret_raw_fields=False):
        selects = []
        raw_selects = []

        for field in fields:
            field = self._defunc(field)
            if isinstance(field, CollectionExpr):
                if any(c is self for c in field.children()):
                    selects.extend(self._get_fields(field._project_fields))
                else:
                    selects.extend(self._get_fields(field._fetch_fields()))
                raw_selects.append(field)
            else:
                select = self._get_field(field)
                selects.append(select)
                raw_selects.append(select)

        if ret_raw_fields:
            return selects, raw_selects
        return selects

    @classmethod
    def _backtrack_lateral_view(cls, lv, collection):
        if isinstance(lv.input, ProjectCollectionExpr):
            src_inputs = (lv.input, lv.input.input)
        else:
            src_inputs = (lv.input,)
        lv_copy = lv.copy(_id=None, _lateral_view=True)

        cur = collection
        while cur not in src_inputs and isinstance(cur, (ProjectCollectionExpr, FilterCollectionExpr)):
            cur = cur.input
        if cur not in src_inputs:
            raise ExpressionError("Input of 'apply' in lateral views can only be "
                                  "simple column selections")
        if cur is lv.input:
            apply_src = lv_copy
        else:
            apply_src = lv_copy.input

        if collection is apply_src.input:
            return lv_copy

        for f in apply_src._fields:
            cls._backtrack_field(f, collection)
        lv_copy.substitute(apply_src.input, collection)
        return lv_copy

    def _filter_lateral_views(self, fields):
        from .collections import RowAppliedCollectionExpr
        from .merge import JoinFieldMergedCollectionExpr, JoinCollectionExpr, UnionCollectionExpr

        walk_through_collections = (JoinCollectionExpr, UnionCollectionExpr,
                                    JoinFieldMergedCollectionExpr)

        lateral_views = []
        for field in fields:
            if not isinstance(field, RowAppliedCollectionExpr):
                continue
            else:
                is_lv = True
                stop_cond = lambda n: n is not self and isinstance(n, CollectionExpr) \
                                      and not isinstance(n, walk_through_collections)
                for coll in self.traverse(unique=True, stop_cond=stop_cond):
                    if coll is field:
                        is_lv = False
                        break
                if is_lv:
                    lateral_views.append(field)
        return lateral_views

    def _project(self, fields):
        field_lvs = self._filter_lateral_views(fields)
        lv_id_set = set(id(f) for f in field_lvs)

        selects = []
        lateral_views = []
        for idx, field in enumerate(fields):
            if id(field) not in lv_id_set:
                s = self._get_fields([field])
                selects.extend(s)
            else:
                lv = self._backtrack_lateral_view(field, self)
                lateral_views.append(lv)
                selects.extend(lv.columns)

        names = [f.name for f in selects]
        typos = [f.dtype for f in selects]

        if not lateral_views and all(isinstance(it, Scalar) for it in selects):
            return self._summary(selects)

        if len(names) != len(set(names)):
            counts = defaultdict(lambda: 0)
            for n in names:
                counts[n] += 1
            raise ExpressionError('Duplicate column names: %s' %
                                  ', '.join(n for n in counts if counts[n] > 1))

        if lateral_views:
            return LateralViewCollectionExpr(self, _fields=selects, _lateral_views=lateral_views,
                                             _schema=types.Schema.from_lists(names, typos))
        else:
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
    def odps_schema(self):
        from ..backends.odpssql.types import df_schema_to_odps_schema
        return df_schema_to_odps_schema(self.schema)

    @property
    def columns(self):
        """
        :return: columns
        :rtype: list which each element is a Column
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
            obj = super(CollectionExpr, self).__getattribute__(attr)

            return obj
        except AttributeError as e:
            if attr in object.__getattribute__(self, '_schema')._name_indexes:
                return self[attr]

            raise e

    def output_type(self):
        return 'collection'

    def limit(self, n):
        """
        limit n records

        :param n: n records
        :return:
        """

        return self[:n]

    def head(self, n=None, **kwargs):
        """
        Return the first n rows. Execute at once.

        :param n:
        :return: result frame
        :rtype: :class:`odps.df.backends.frame.ResultFrame`
        """
        if n is None:
            n = options.display.max_rows
        return self._handle_delay_call('execute', self, head=n, **kwargs)

    def tail(self, n=None, **kwargs):
        """
        Return the last n rows. Execute at once.

        :param n:
        :return: result frame
        :rtype: :class:`odps.df.backends.frame.ResultFrame`
        """
        if n is None:
            n = options.display.max_rows
        return self._handle_delay_call('execute', self, tail=n, **kwargs)

    def to_pandas(self, wrap=False, **kwargs):
        """
        Convert to pandas DataFrame. Execute at once.

        :param wrap: if True, wrap the pandas DataFrame into a PyODPS DataFrame
        :return: pandas DataFrame
        """

        try:
            import pandas as pd
        except ImportError:
            raise DependencyNotInstalledError(
                    'to_pandas requires `pandas` library')

        def wrapper(result):
            res = result.values
            if wrap:
                from .. import DataFrame
                return DataFrame(res, schema=self.schema)
            return res

        return self.execute(wrapper=wrapper, **kwargs)

    @property
    def dtypes(self):
        return self.schema

    def view(self):
        """
        Clone a same collection. useful for self-join.

        :return:
        """
        proxied = get_proxied_expr(self)
        kv = dict((attr, getattr(proxied, attr)) for attr in get_attrs(proxied))
        return type(proxied)(**kv)

    def describe(self):
        from .. import output

        numeric_methods = OrderedDict([
            ('count', lambda f: f.count().rename(col.name + '_count')),
            ('mean', lambda f: f.mean()),
            ('std', lambda f: f.std()),
            ('min', lambda f: f.min()),
            ('quantile_25', lambda f: f.quantile(0.25).rename(col.name + '_quantile_25')),
            ('quantile_50', lambda f: f.quantile(0.5).rename(col.name + '_quantile_50')),
            ('quantile_75', lambda f: f.quantile(0.75).rename(col.name + '_quantile_75')),
            ('max', lambda f: f.max()),
        ])
        string_methods = OrderedDict([
            ('count', lambda f: f.count().rename(col.name + '_count')),
            ('unique', lambda f: f.nunique().rename(col.name + '_unique')),
        ])

        aggs = []
        output_names = []
        output_types = []

        def produce_stat_fields(col, methods):
            output_names.append(col.name)
            fields = []
            tps = []
            for func in six.itervalues(methods):
                field = func(self[col.name])
                fields.append(field)
                tps.append(field.dtype)
            t = highest_precedence_data_type(*tps)
            output_types.append(t)
            aggs.extend([f.astype(t) if t == types.decimal else f for f in fields])

        if any(types.is_number(col.type) for col in self.schema.columns):
            methods = numeric_methods
            for col in self.schema.columns:
                if types.is_number(col.type):
                    produce_stat_fields(col, numeric_methods)
        else:
            methods = string_methods
            for col in self.schema.columns:
                produce_stat_fields(col, string_methods)

        summary = self[aggs]
        methods_names = lkeys(methods)

        @output(['type'] + output_names, ['string'] + output_types)
        def to_methods(row):
            for method in methods_names:
                values = [None] * len(output_names)
                for i, field_name in enumerate(output_names):
                    values[i] = getattr(row, '%s_%s' % (field_name, method.split('(', 1)[0]))
                yield [method, ] + values

        return summary.apply(to_methods, axis=1)

    def accept(self, visitor):
        if self._source_data is not None:
            visitor.visit_source_collection(self)
        else:
            raise NotImplementedError

    def get_cached(self, data):
        try:
            if hasattr(data, 'reload'):
                data.reload()
        except NoSuchObject:
            from ..backends.context import context
            context.uncache(self)
            return None

        coll = CollectionExpr(_source_data=data, _schema=self._schema)
        for attr in CollectionExpr.__slots__:
            if not attr.startswith('_ml_'):
                continue
            if hasattr(self, attr):
                setattr(coll, attr, getattr(self, attr))
        return coll


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
            if typed_cls is None:
                raise TypeError("Name {} doesn't exist".format(name))

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

            bases = []
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

    def eval(self, str_expr, rewrite=False):
        from .query import SequenceVisitor

        if not isinstance(str_expr, six.string_types):
            raise ValueError('expr must be a string')

        frame = sys._getframe(2).f_locals
        try:
            env = frame.copy()
        finally:
            del frame

        env['max_pt'] = ReprWrapper(make_max_pt_function(self), repr)

        visitor = SequenceVisitor(self, env)
        return visitor.eval(str_expr, rewrite=rewrite)


class SequenceExpr(TypedExpr):
    """
    Sequence represents for 1-dimension data.
    """

    __slots__ = (
        '_data_type', '_source_data_type',
        '_ml_fields_cache', '_ml_uplink', '_ml_operations',
    )

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
        self._init_attr('_name', None)

        self._init_attr('_ml_fields_cache', None)
        self._init_attr('_ml_uplink', [])
        self._init_attr('_ml_operations', [])
        super(SequenceExpr, self)._init(*args, **kwargs)

        if '_data_type' in kwargs:
            self._data_type = types.validate_data_type(kwargs.get('_data_type'))

        if '_source_name' not in kwargs:
            self._source_name = self._name

        if '_source_data_type' in kwargs:
            self._source_data_type = types.validate_data_type(kwargs.get('_source_data_type'))
        else:
            self._source_data_type = self._data_type

    def cache(self, mem=False):
        raise ExpressionError('Cache operation does not support for sequence.')

    def head(self, n=None, **kwargs):
        """
        Return first n rows. Execute at once.

        :param n:
        :return: result frame
        :rtype: :class:`odps.df.expr.expressions.CollectionExpr`
        """
        if n is None:
            n = options.display.max_rows
        return self._handle_delay_call('execute', self, head=n, **kwargs)

    def tail(self, n=None, **kwargs):
        """
        Return the last n rows. Execute at once.

        :param n:
        :return:
        """

        if n is None:
            n = options.display.max_rows
        return self._handle_delay_call('execute', self, tail=n, **kwargs)

    def to_pandas(self, wrap=False, **kwargs):
        """
        Convert to pandas Series. Execute at once.

        :param wrap: if True, wrap the pandas DataFrame into a PyODPS DataFrame
        :return: pandas Series
        """

        try:
            import pandas as pd
        except ImportError:
            raise DependencyNotInstalledError(
                    'to_pandas requires for `pandas` library')

        def wrapper(result):
            df = result.values
            if wrap:
                from .. import DataFrame
                df = DataFrame(df)
            return df[self.name]

        return self.execute(wrapper=wrapper, **kwargs)

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


class BinarySequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(BinarySequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.binary


class ListSequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(ListSequenceExpr, self)._init(*args, **kwargs)


class DictSequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(DictSequenceExpr, self)._init(*args, **kwargs)


class DateSequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(DateSequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.date


class TimestampSequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(TimestampSequenceExpr, self)._init(*args, **kwargs)
        self._data_type = types.timestamp


class UnknownSequenceExpr(SequenceExpr):
    def _init(self, *args, **kwargs):
        super(UnknownSequenceExpr, self)._init(*args, **kwargs)
        if not isinstance(self._data_type, types.Unknown):
            self._data_type = types.Unknown()


_typed_sequence_exprs = [globals()[t.__class__.__name__ + SequenceExpr.__name__]
                         for t in types._data_types.values()]

number_sequences = [globals().get(repr(t).capitalize() + SequenceExpr.__name__)
                    for t in types.number_types()]

int_number_sequences = [globals().get(repr(t).capitalize() + SequenceExpr.__name__)
                        for t in types.number_types() if repr(t).startswith('int')]


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

    def rebuild(self):
        attr_dict = self._attr_dict()
        tp = self._copy_type()
        if isinstance(attr_dict['_source_data_type'], types.Unknown):
            attr_dict['_source_data_type'] = self._input.dtype
        return tp(**attr_dict)


class Column(SequenceExpr):
    _args = '_input',

    @property
    def input(self):
        return self._input

    def rebuild(self):
        attr_dict = self._attr_dict()
        tp = self._copy_type()
        new_col = self.input[self.source_name]
        for attr in ('_source_data_type', '_data_type'):
            if isinstance(attr_dict[attr], types.Unknown):
                attr_dict[attr] = new_col.dtype
        return tp(**attr_dict)

    def accept(self, visitor):
        return visitor.visit_column(self)


class CallableColumn(Column):
    def __call__(self, *args, **kwargs):
        return getattr(type(self._input), self.source_name)(self._input, *args, **kwargs)


class Scalar(TypedExpr):
    """
    Represent for the scalar type.

    :param _value: value of the scalar
    :param _value_type: value type of the scalar

    :Example:
    >>> df[df, Scalar(4).rename('append_const')]
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

        if val is not None and not isinstance(val, NodeMetaclass):
            return types.validate_value_type(val, value_type)
        else:
            return types.validate_data_type(value_type)

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

        if kwargs.get('_value') is not None:
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
        visitor.visit_scalar(self)

    def get_cached(self, data):
        return Scalar(_value=data, _value_type=self.dtype)


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

    def rebuild(self):
        attr_dict = self._attr_dict()
        tp = self._copy_type()
        if isinstance(attr_dict['_source_value_type'], types.Unknown):
            attr_dict['_source_value_type'] = self._input.dtype
        return tp(**attr_dict)

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
    def _init(self, *args, **kwargs):
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


class DateScalar(Scalar):
    def _init(self, *args, **kwargs):
        super(DateScalar, self)._init(*args, **kwargs)
        self._value_type = types.date


class DatetimeScalar(Scalar):
    def _init(self, *args, **kwargs):
        super(DatetimeScalar, self)._init(*args, **kwargs)
        self._value_type = types.datetime


class BinaryScalar(Scalar):
    def _init(self, *args, **kwargs):
        super(BinaryScalar, self)._init(*args, **kwargs)
        self._value_type = types.binary


class ListScalar(Scalar):
    pass


class DictScalar(Scalar):
    pass


class UnknownScalar(Scalar):
    def _init(self, *args, **kwargs):
        super(UnknownScalar, self)._init(*args, **kwargs)
        if not isinstance(self._value_type, types.Unknown):
            self._value_type = types.Unknown()


_typed_scalar_exprs = [globals()[t.__class__.__name__ + Scalar.__name__]
                       for t in types._data_types.values()]

number_scalars = [globals().get(repr(t).capitalize() + Scalar.__name__)
                  for t in types.number_types()]

int_number_scalars = [globals().get(repr(t).capitalize() + Scalar.__name__)
                      for t in types.number_types() if repr(t).startswith('int')]


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


class RandomScalar(BuiltinFunction):
    """
    Represent for the random scalar type.

    :param seed: random seed, None by default

    :Example:
    >>> df[df, RandomScalar().rename('append_random')]
    """
    def __new__(cls, seed=None, **kw):
        args = (seed, ) if seed is not None else kw.get('_func_args', ())
        kw.update(dict(_func_name='rand', _value_type='float', _func_args=args))
        return BuiltinFunction.__new__(cls, **kw)

    def __init__(self, seed=None, **kw):
        args = (seed, ) if seed is not None else kw.get('_func_args', ())
        kw.update(dict(_func_name='rand', _value_type='float', _func_args=args))
        super(RandomScalar, self).__init__(**kw)


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

    def rebuild(self):
        rebuilt = super(FilterCollectionExpr, self).rebuild()
        rebuilt._schema = rebuilt.input.schema
        return rebuilt

    def accept(self, visitor):
        visitor.visit_filter_collection(self)


class ProjectCollectionExpr(CollectionExpr):
    __slots__ = '_raw_fields',
    _args = '_input', '_fields'
    _extra_args = '_raw_fields',
    node_name = 'Projection'

    def _init(self, *args, **kwargs):
        fields = kwargs.get('_fields')
        if fields is None and len(args) >= 2:
            fields = args[1]
        for field in fields:
            if field.name is None:
                raise ExpressionError('Column does not have a name, '
                                      'please specify one by `rename`: %s' % repr_obj(field._repr()))

        self._init_attr('_raw_fields', None)
        super(ProjectCollectionExpr, self)._init(*args, **kwargs)

    def _set_field(self, value, value_dag=None):
        from ..backends.context import context
        from .window import Window

        if context.is_cached(self):
            super(ProjectCollectionExpr, self)._set_field(value, value_dag=value_dag)
            return

        if value_dag is None:
            value_dag = value.to_dag(copy=False, validate=False)

        for n in value.traverse(top_down=True, unique=True,
                                stop_cond=lambda x: isinstance(x, Column)):
            if isinstance(n, Column) and (n.input is self or n.input._proxy is self):
                source_name = n.source_name
                idx = self._schema._name_indexes[source_name]
                field = self._fields[idx]
                if field.name != n.name:
                    field = field.rename(n.name)
                value_dag.substitute(n, field)
            elif isinstance(n, Window) and n.input is self:
                # Window object like rank will point to collection directly instead of through column
                n._input = self.input

        value = value_dag.root
        fields = [field if field.name != value.name else value for field in self._fields]
        if value.name not in self._schema:
            fields.append(value)

        self._schema = Schema.from_lists([f.name for f in fields],
                                         [f.dtype for f in fields])
        self._fields = fields

    def _delitem(self, key):
        from ..backends.context import context

        if context.is_cached(self):
            return super(ProjectCollectionExpr, self).__delitem__(key)

        fields = [f for f in self._fields if f.name != key]
        self._schema = Schema.from_lists([f.name for f in fields],
                                         [f.dtype for f in fields])
        self._fields = fields

    def __delitem__(self, key):
        if key not in self._schema:
            raise KeyError('Field({0}) does not exist'.format(key))

        self._delitem(key)

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

    def rebuild(self):
        if self._raw_fields:
            return self._input.select(*self._raw_fields)
        rebuilt = super(ProjectCollectionExpr, self).rebuild()
        rebuilt._schema = Schema.from_lists(
            [f.name for f in rebuilt._fields],
            [f.dtype for f in rebuilt._fields]
        )
        return rebuilt

    def accept(self, visitor):
        visitor.visit_project_collection(self)


class LateralViewCollectionExpr(ProjectCollectionExpr):
    _args = '_input', '_fields', '_lateral_views'
    node_name = 'LateralView'

    def _init(self, *args, **kwargs):
        self._init_attr('_lateral_views', None)
        super(LateralViewCollectionExpr, self)._init(*args, **kwargs)
        self._redirect_lateral_views()

    def _redirect_lateral_views(self):
        # dealing with cases like df[df['a', 'b'].apply(..., axis=1), df.c]
        for lv in self.lateral_views:
            if lv.input is self.input:
                continue
            for f, f_o in zip(lv._fields, lv.input._fields):
                lv.substitute(f, f_o)
            lv.substitute(lv.input, self.input)

    def iter_args(self):
        for it in zip(['collection', 'selections', 'lateral_views'], self.args):
            yield it

    @property
    def lateral_views(self):
        return self._lateral_views

    def accept(self, visitor):
        visitor.visit_lateral_view(self)


class FilterPartitionCollectionExpr(CollectionExpr):
    __slots__ = '_predicate_string',

    _args = '_input', '_predicate', '_fields'
    node_name = 'FilterPartition'

    def _init(self, *args, **kwargs):
        super(FilterPartitionCollectionExpr, self)._init(*args, **kwargs)
        self._fields = [self._input[col.name] for col in self._schema.columns]
        self._predicate_string = kwargs.get('_predicate_string')

    @property
    def _project_fields(self):
        return self._fields

    def iter_args(self):
        for it in zip(['collection', 'predicate', 'selections'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    @property
    def fields(self):
        return self._fields

    @property
    def predicate_string(self):
        return self._predicate_string

    def accept(self, visitor):
        visitor.visit_filter_partition_collection(self)

    @property
    def data_table(self):
        return self.input.data

    @property
    def data(self):
        from ...types import PartitionSpec
        from .arithmetic import Or

        if isinstance(self._predicate, Or):
            raise AttributeError('Cannot get data when predicate contains multiple partition specs.')
        spec = PartitionSpec(self.predicate_string)
        return self.data_table.partitions[spec]


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

    def rebuild(self):
        rebuilt = super(SliceCollectionExpr, self).rebuild()
        rebuilt._schema = self.input.schema
        return rebuilt

    def accept(self, visitor):
        visitor.visit_slice_collection(self)


class Summary(CollectionExpr):
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
from . import composites
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
