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

import operator
import inspect
from .expressions import CollectionExpr, ProjectCollectionExpr, \
    Column, BooleanSequenceExpr, SequenceExpr, repr_obj
from .arithmetic import Equal
from .errors import ExpressionError
from ...compat import reduce, six
from ...models import Schema


class JoinCollectionExpr(CollectionExpr):
    __slots__ = '_how', '_left_suffix', '_right_suffix', '_column_origins', \
                '_renamed_columns', '_column_conflict', '_mapjoin'
    _args = '_lhs', '_rhs', '_predicate'

    def _init(self, *args, **kwargs):
        self._left_suffix = None
        self._right_suffix = None
        self._column_origins = dict()
        self._renamed_columns = dict()
        self._column_conflict = False

        super(JoinCollectionExpr, self)._init(*args, **kwargs)
        if not isinstance(self._lhs, CollectionExpr):
            raise TypeError('Can only join collection expressions, got %s for left expr.' % type(self._lhs))
        if not isinstance(self._rhs, CollectionExpr):
            raise TypeError('Can only join collection expressions, got %s for right expr.' % type(self._rhs))

        if self._rhs is self._lhs:
            self._rhs = self._rhs.view()
        if isinstance(self._lhs, JoinCollectionExpr) and \
                (self._rhs is self._lhs._lhs or self._rhs is self._lhs._rhs):
            self._rhs = self._rhs.view()

        if self._left_suffix is None and self._right_suffix is None:
            overlaps = set(self._lhs.schema.names).intersection(self._rhs.schema.names)
            if len(overlaps) > 0:
                raise ValueError(
                    'Column conflict exists in join, overlap columns: %s' % ','.join(overlaps))

        self._set_schema()
        self._validate_predicates(self._predicate)
        self._how = self._how.upper()

    def _defunc(self, field):
        if inspect.isfunction(field):
            if six.get_function_code(field).co_argcount == 1:
                return field(self)
            else:
                return field(self._lhs, self._rhs)
        return field

    def _get_child(self, expr):
        while isinstance(expr, JoinProjectCollectionExpr):
            expr = expr.input

        return expr

    def _is_column(self, col, expr):
        return col.input is expr or col.input is self._get_child(expr)

    def _get_fields(self, fields):
        selects = []

        for field in fields:
            field = self._defunc(field)
            if isinstance(field, CollectionExpr):
                if any(c is self for c in field.children()) or \
                        any(c is self._get_child(self._lhs) for c in field.children()) or \
                        any(c is self._get_child(self._rhs) for c in field.children()):
                    selects.extend(self._get_fields(field._project_fields))
                else:
                    selects.extend(self._get_fields(field._fetch_fields()))
            else:
                selects.append(self._get_field(field))

        return selects

    def _get_field(self, field):
        field = self._defunc(field)

        if isinstance(field, six.string_types):
            if field not in self._schema:
                raise ValueError('Field(%s) does not exist' % field)
            return Column(self, _name=field, _data_type=self._schema[field].type)

        root = field
        subs = dict()

        collections = self, self._get_child(self._lhs), self._get_child(self._rhs)
        collections_ids = set(id(c) for c in collections)
        for collection in collections:
            for path in root.all_path(collection, strict=True):
                for i, node in enumerate(path):
                    if id(node) in subs:
                        path[i] = subs[id(node)]
                        continue
                    if id(node) in collections_ids:
                        break

                    if isinstance(node, Column):
                        if self._is_column(node, self):
                            subs[id(node)] = node
                            continue
                        idx = -1
                        if self._is_column(node, self._lhs):
                            idx = 0
                        elif self._is_column(node, self._rhs):
                            idx = 1

                        if idx == -1:
                            raise ExpressionError('field must come from Join collection '
                                                  'or its left and right child collection: %s'
                                                  % repr_obj(field))

                        name = node.source_name
                        if name in self._renamed_columns:
                            name = self._renamed_columns[name][idx]
                        to_sub = self._get_field(name)
                        if node.is_renamed():
                            to_sub = to_sub.rename(node.name)

                        path[i] = to_sub
                        subs[id(node)] = to_sub
                        if i == 0:
                            root = to_sub
                        else:
                            path[i - 1].substitute(node, to_sub)

        return root

    def origin_collection(self, column_name):
        idx, name = self._column_origins[column_name]
        return [self._lhs, self._rhs][idx], name

    @property
    def node_name(self):
        return self.__class__.__name__

    def iter_args(self):
        for it in zip(['collection(left)', 'collection(right)', 'on'], self.args):
            yield it

    @property
    def column_conflict(self):
        return self._column_conflict

    def accept(self, visitor):
        visitor.visit_join(self)

    def _get_non_suffixes_fields(self):
        return set()

    def _set_schema(self):
        names, typos = [], []

        non_suffixes_fields = self._get_non_suffixes_fields()

        for col in self._lhs.schema.columns:
            name = col.name
            if col.name in self._rhs.schema:
                self._column_conflict = True
            if col.name in self._rhs.schema and col.name not in non_suffixes_fields:
                name = '%s%s' % (col.name, self._left_suffix)
                self._renamed_columns[col.name] = (name,)
            names.append(name)
            typos.append(col.type)

            self._column_origins[name] = 0, col.name
        for col in self._rhs.schema.columns:
            name = col.name
            if col.name in self._lhs.schema:
                self._column_conflict = True
            if col.name in self._lhs.schema and col.name not in non_suffixes_fields:
                name = '%s%s' % (col.name, self._right_suffix)
                self._renamed_columns[col.name] = \
                    self._renamed_columns[col.name][0], name
            if name in non_suffixes_fields:
                continue
            names.append(name)
            typos.append(col.type)

            self._column_origins[name] = 1, col.name

        self._schema = Schema.from_lists(names, typos)

    def _validate_equal(self, equal_expr):
        # FIXME: sometimes may be wrong, e.g. t3 = t1.join(t2, 'name')); t4.join(t3, t4.id == t3.id)
        return (equal_expr.lhs.is_ancestor(self._get_child(self._lhs)) and
                equal_expr.rhs.is_ancestor(self._get_child(self._rhs))) or \
               (equal_expr.lhs.is_ancestor(self._get_child(self._rhs)) and
                equal_expr.rhs.is_ancestor(self._get_child(self._lhs)))

    def _validate_predicates(self, predicates):
        is_validate = False
        subs = []
        if self._mapjoin:
            is_validate = True
        for p in predicates:
            if (isinstance(p, tuple) and len(p) == 2) or isinstance(p, six.string_types):
                if isinstance(p, six.string_types):
                    left_name, right_name = p, p
                else:
                    left_name, right_name = p

                left_col = self._lhs._get_field(left_name)
                right_col = self._rhs._get_field(right_name)

                if not left_col.is_ancestor(self._lhs) or not right_col.is_ancestor(self._rhs):
                    raise ExpressionError('Invalid predicate: {0!s}'.format(repr_obj(p)))
                subs.append(left_col == right_col)

                is_validate = True
            elif isinstance(p, BooleanSequenceExpr):
                if not is_validate:
                    it = (expr for expr in p.traverse(top_down=True, unique=True)
                          if isinstance(expr, Equal))
                    while not is_validate:
                        try:
                            validate = self._validate_equal(next(it))
                            if validate:
                                is_validate = True
                                break
                        except StopIteration:
                            break
            else:
                if not is_validate:
                    raise ExpressionError('Invalid predicate: {0!s}'.format(repr_obj(p)))
        if not is_validate:
            raise ExpressionError('Invalid predicate: no validate predicate assigned')

        for p in predicates:
            if isinstance(p, BooleanSequenceExpr):
                subs.append(p)

        if len(subs) ==0:
            self._predicate = None
        else:
            self._predicate = reduce(operator.and_, subs)


class InnerJoin(JoinCollectionExpr):
    def _init(self, *args, **kwargs):
        self._how = 'INNER'
        super(InnerJoin, self)._init(*args, **kwargs)

    def _get_non_suffixes_fields(self):
        non_suffixes = set()

        for p in self._predicate:
            if isinstance(p, six.string_types):
                non_suffixes.add(p)
            elif isinstance(p, (tuple, Equal)):
                if isinstance(p, Equal):
                    p = p.lhs, p.rhs
                if isinstance(p, tuple) and len(p) != 2:
                    continue

                left_name = None
                if isinstance(p[0], six.string_types):
                    left_name = p[0]
                elif isinstance(p[0], Column) and not p[0].is_renamed():
                    left_name = p[0].name

                if left_name is None:
                    continue

                right_name = None
                if isinstance(p[1], six.string_types):
                    right_name = p[1]
                elif isinstance(p[1], Column) and not p[1].is_renamed():
                    right_name = p[1].name

                if left_name == right_name:
                    non_suffixes.add(left_name)

        return non_suffixes


class LeftJoin(JoinCollectionExpr):
    def _init(self, *args, **kwargs):
        self._how = 'LEFT OUTER'
        super(LeftJoin, self)._init(*args, **kwargs)


class RightJoin(JoinCollectionExpr):
    def _init(self, *args, **kwargs):
        self._how = 'RIGHT OUTER'
        super(RightJoin, self)._init(*args, **kwargs)


class OuterJoin(JoinCollectionExpr):
    def _init(self, *args, **kwargs):
        self._how = 'FULL OUTER'
        super(OuterJoin, self)._init(*args, **kwargs)


class JoinProjectCollectionExpr(ProjectCollectionExpr):
    """
    Only for analyzer, project join should generate normal `ProjectCollectionExpr`.
    """
    __slots__ = ()


_join_dict = {
    'INNER': InnerJoin,
    'LEFT': LeftJoin,
    'RIGHT': RightJoin,
    'OUTER': OuterJoin
}


def join(left, right, on=None, how='inner', suffixes=('_x', '_y'), mapjoin=False):
    """
    Join two collections.

    If `on` is not specified, we will find the common fields of the left and right collection.
    `suffixes` means that if column names conflict, the suffixes will be added automatically.
    For example, both left and right has a field named `col`,
    there will be col_x, and col_y in the joined collection.

    :param left: left collection
    :param right: right collection
    :param on: fields to join on
    :param how: 'inner', 'left', 'right', or 'outer'
    :param suffixes: when name conflict, the suffix will be added to both columns.
    :param mapjoin: set use mapjoin or not, default value False.
    :return: collection

    :Example:

    >>> df.dtypes.names
    ['name', 'id']
    >>> df2.dtypes.names
    ['name', 'id1']
    >>> df.join(df2)
    >>> df.join(df2, on='name')
    >>> df.join(df2, on=('id', 'id1'))
    >>> df.join(df2, on=['name', ('id', 'id1')])
    >>> df.join(df2, on=[df.name == df2.name, df.id == df2.id1])
    >>> df.join(df2, mapjoin=False)
    """
    if on is None and not mapjoin:
        on = [name for name in left.schema.names if name in right.schema]

    if isinstance(suffixes, (tuple, list)) and len(suffixes) == 2:
        left_suffix, right_suffix = suffixes
    else:
        raise ValueError('suffixes must be a tuple or list with two elements, got %s' % suffixes)
    if not isinstance(on, list):
        on = [on, ]
    for i in range(len(on)):
        it = on[i]
        if inspect.isfunction(it):
            on[i] = it(left, right)

    try:
        return _join_dict[how.upper()](_lhs=left, _rhs=right, _predicate=on, _left_suffix=left_suffix,
                                       _right_suffix=right_suffix, _mapjoin=mapjoin)
    except KeyError:
        return JoinCollectionExpr(_lhs=left, _rhs=right, _predicate=on, _how=how, _left_suffix=left_suffix,
                                  _right_suffix=right_suffix, _mapjoin=mapjoin)


def inner_join(left, right, on=None, suffixes=('_x', '_y'), mapjoin=False):
    """
    Inner join two collections.

    If `on` is not specified, we will find the common fields of the left and right collection.
    `suffixes` means that if column names conflict, the suffixes will be added automatically.
    For example, both left and right has a field named `col`,
    there will be col_x, and col_y in the joined collection.

    :param left: left collection
    :param right: right collection
    :param on: fields to join on
    :param suffixes: when name conflict, the suffixes will be added to both columns.
    :return: collection

    :Example:

    >>> df.dtypes.names
    ['name', 'id']
    >>> df2.dtypes.names
    ['name', 'id1']
    >>> df.inner_join(df2)
    >>> df.inner_join(df2, on='name')
    >>> df.inner_join(df2, on=('id', 'id1'))
    >>> df.inner_join(df2, on=['name', ('id', 'id1')])
    >>> df.inner_join(df2, on=[df.name == df2.name, df.id == df2.id1])
    """

    return join(left, right, on, suffixes=suffixes, mapjoin=mapjoin)


def left_join(left, right, on=None, suffixes=('_x', '_y'), mapjoin=False):
    """
    Left join two collections.

    If `on` is not specified, we will find the common fields of the left and right collection.
    `suffixes` means that if column names conflict, the suffixes will be added automatically.
    For example, both left and right has a field named `col`,
    there will be col_x, and col_y in the joined collection.

    :param left: left collection
    :param right: right collection
    :param on: fields to join on
    :param suffixes: when name conflict, the suffixes will be added to both columns.
    :param mapjoin: set use mapjoin or not, default value False.
    :return: collection

    :Example:

    >>> df.dtypes.names
    ['name', 'id']
    >>> df2.dtypes.names
    ['name', 'id1']
    >>> df.left_join(df2)
    >>> df.left_join(df2, on='name')
    >>> df.left_join(df2, on=('id', 'id1'))
    >>> df.left_join(df2, on=['name', ('id', 'id1')])
    >>> df.left_join(df2, on=[df.name == df2.name, df.id == df2.id1])
    """

    return join(left, right, on, how='left', suffixes=suffixes, mapjoin=mapjoin)


def right_join(left, right, on=None, suffixes=('_x', '_y'), mapjoin=False):
    """
    Right join two collections.

    If `on` is not specified, we will find the common fields of the left and right collection.
    `suffixes` means that if column names conflict, the suffixes will be added automatically.
    For example, both left and right has a field named `col`,
    there will be col_x, and col_y in the joined collection.

    :param left: left collection
    :param right: right collection
    :param on: fields to join on
    :param suffixes: when name conflict, the suffixes will be added to both columns.
    :param mapjoin: set use mapjoin or not, default value False.
    :return: collection

    :Example:

    >>> df.dtypes.names
    ['name', 'id']
    >>> df2.dtypes.names
    ['name', 'id1']
    >>> df.right_join(df2)
    >>> df.right_join(df2, on='name')
    >>> df.right_join(df2, on=('id', 'id1'))
    >>> df.right_join(df2, on=['name', ('id', 'id1')])
    >>> df.right_join(df2, on=[df.name == df2.name, df.id == df2.id1])
    """

    return join(left, right, on, how='right', suffixes=suffixes, mapjoin=mapjoin)


def outer_join(left, right, on=None, suffixes=('_x', '_y'), mapjoin=False):
    """
    Outer join two collections.

    If `on` is not specified, we will find the common fields of the left and right collection.
    `suffixes` means that if column names conflict, the suffixes will be added automatically.
    For example, both left and right has a field named `col`,
    there will be col_x, and col_y in the joined collection.

    :param left: left collection
    :param right: right collection
    :param on: fields to join on
    :param suffixes: when name conflict, the suffixes will be added to both columns.
    :param mapjoin: set use mapjoin or not, default value False.
    :return: collection

    :Example:

    >>> df.dtypes.names
    ['name', 'id']
    >>> df2.dtypes.names
    ['name', 'id1']
    >>> df.outer_join(df2)
    >>> df.outer_join(df2, on='name')
    >>> df.outer_join(df2, on=('id', 'id1'))
    >>> df.outer_join(df2, on=['name', ('id', 'id1')])
    >>> df.outer_join(df2, on=[df.name == df2.name, df.id == df2.id1])
    """

    return join(left, right, on, how='outer', suffixes=suffixes, mapjoin=mapjoin)


CollectionExpr.join = join
CollectionExpr.inner_join = inner_join
CollectionExpr.left_join = left_join
CollectionExpr.right_join = right_join
CollectionExpr.outer_join = outer_join


class UnionCollectionExpr(CollectionExpr):
    __slots__ = '_distinct',
    _args = '_lhs', '_rhs',
    node_name = 'Union'

    def _init(self, *args, **kwargs):
        super(UnionCollectionExpr, self)._init(*args, **kwargs)

        self._validate()
        self._schema = self._clean_schema()

    def _get_sequence_source_collection(self, expr):
        return next(it for it in expr.traverse(top_down=True, unique=True) if isinstance(it, CollectionExpr))

    def _validate_collection_child(self):
        if self._lhs.schema.names == self._rhs.schema.names and self._lhs.schema.types == self._rhs.schema.types:
            return True
        elif set(self._lhs.schema.names) == set(self._rhs.schema.names) and set(self._lhs.schema.types) == set(
            self._rhs.schema.types):
            self._rhs = self._rhs[self._lhs.schema.names]
            return True
        else:
            return False

    def _validate(self):
        if isinstance(self._lhs, SequenceExpr):
            source_collection = self._get_sequence_source_collection(self._lhs)
            self._lhs = source_collection[[self._lhs]]
        if isinstance(self._rhs, SequenceExpr):
            source_collection = self._get_sequence_source_collection(self._rhs)
            self._rhs = source_collection[[self._rhs]]

        if isinstance(self._lhs, CollectionExpr) and isinstance(self._rhs, CollectionExpr):
            if not self._validate_collection_child():
                raise ExpressionError('Table schemas must be equal to form union')
        else:
            raise ExpressionError('Column schemas must be equal to form union')

    def _clean_schema(self):

        return Schema.from_lists(self._lhs.schema.names, self._lhs.schema.types)

    def accept(self, visitor):
        return visitor.visit_union(self)

    def iter_args(self):
        for it in zip(['collection(left)', 'collection(right)'], self.args):
            yield it


def union(left, right, distinct=False):
    """
    Union two collections.

    :param left: left collection
    :param right: right collection
    :param distinct:
    :return: collection

    :Example:
    >>> df['name', 'id'].union(df2['id', 'name'])
    """

    return UnionCollectionExpr(_lhs=left, _rhs=right, _distinct=distinct)


CollectionExpr.union = union
SequenceExpr.union = union
CollectionExpr.concat = union
SequenceExpr.concat = union
