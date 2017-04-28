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

import inspect
from .expressions import CollectionExpr, ProjectCollectionExpr, \
    Column, BooleanSequenceExpr, SequenceExpr, Expr, Scalar, repr_obj
from .core import Node, ExprDictionary
from .arithmetic import Equal
from .errors import ExpressionError
from ...compat import six, reduce
from ...models import Schema


class JoinCollectionExpr(CollectionExpr):
    __slots__ = '_how', '_left_suffix', '_right_suffix', '_column_origins', \
                '_renamed_columns', '_column_conflict', '_mapjoin'
    _args = '_lhs', '_rhs', '_predicate'

    def _init(self, *args, **kwargs):
        self._init_attr('_left_suffix', None)
        self._init_attr('_right_suffix', None)
        self._init_attr('_column_origins', dict())
        self._init_attr('_renamed_columns', dict())
        self._init_attr('_column_conflict', False)

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

    def _get_fields(self, fields, ret_raw_fields=False):
        selects = []
        raw_selects = []

        for field in fields:
            field = self._defunc(field)
            if isinstance(field, CollectionExpr):
                if any(c is self for c in field.children()) or \
                        any(c is self._get_child(self._lhs) for c in field.children()) or \
                        any(c is self._get_child(self._rhs) for c in field.children()):
                    selects.extend(self._get_fields(field._project_fields))
                elif field is self:
                    selects.extend(self._get_fields(self._schema.names))
                elif field is self._get_child(self._lhs):
                    fields = [self._renamed_columns.get(n, [n])[0]
                              for n in field.schema.names]
                    selects.extend(self._get_fields(fields))
                elif field is self._get_child(self._rhs):
                    fields = [self._renamed_columns.get(n, [None, n])[1]
                              for n in field.schema.names]
                    selects.extend(self._get_fields(fields))
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

    def _get_field(self, field):
        field = self._defunc(field)

        if isinstance(field, six.string_types):
            if field not in self._schema:
                raise ValueError('Field(%s) does not exist' % field)
            return Column(self, _name=field, _data_type=self._schema[field].type)

        root = field
        has_path = False

        for expr in root.traverse(top_down=True, unique=True,
                                  stop_cond=lambda x: isinstance(x, Column) or x is self):
            if isinstance(expr, Column):
                if self._is_column(expr, self):
                    has_path = True
                    continue
                if self._is_column(expr, self._lhs):
                    has_path = True
                    idx = 0
                elif self._is_column(expr, self._rhs):
                    has_path = True
                    idx = 1
                elif isinstance(self._lhs, JoinCollectionExpr):
                    try:
                        expr = self._lhs._get_field(expr)
                    except ExpressionError:
                        continue
                    has_path = True
                    idx = 0
                elif isinstance(self._rhs, JoinCollectionExpr):
                    try:
                        expr = self._rhs._get_field(expr)
                    except ExpressionError:
                        continue
                    has_path = True
                    idx = 1
                else:
                    continue

                name = expr.source_name
                if name in self._renamed_columns:
                    name = self._renamed_columns[name][idx]
                to_sub = self._get_field(name)
                if expr.is_renamed():
                    to_sub = to_sub.rename(expr.name)

                to_sub.copy_to(expr)

        if isinstance(field, SequenceExpr) and not has_path:
            raise ExpressionError('field must come from Join collection '
                                  'or its left and right child collection: %s'
                                  % repr_obj(field))

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

    def _get_predicate_fields(self):
        predicate_fields = set()

        if not self._predicate:
            return predicate_fields

        for p in self._predicate:
            if isinstance(p, six.string_types):
                predicate_fields.add(p)
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
                    predicate_fields.add(left_name)

        return predicate_fields

    def _set_schema(self):
        names, typos = [], []

        non_suffixes_fields = self._get_non_suffixes_fields()

        for col in self._lhs.schema.columns:
            name = col.name
            if col.name in self._rhs.schema._name_indexes:
                self._column_conflict = True
            if col.name in self._rhs.schema._name_indexes and \
                    col.name not in non_suffixes_fields:
                name = '%s%s' % (col.name, self._left_suffix)
                self._renamed_columns[col.name] = (name,)
            names.append(name)
            typos.append(col.type)

            self._column_origins[name] = 0, col.name
        for col in self._rhs.schema.columns:
            name = col.name
            if col.name in self._lhs.schema._name_indexes:
                self._column_conflict = True
            if col.name in self._lhs.schema._name_indexes and \
                    col.name not in non_suffixes_fields:
                name = '%s%s' % (col.name, self._right_suffix)
                self._renamed_columns[col.name] = \
                    self._renamed_columns[col.name][0], name
            if name in non_suffixes_fields:
                continue
            names.append(name)
            typos.append(col.type)

            self._column_origins[name] = 1, col.name

        schema_type = type(self._lhs.schema) if issubclass(type(self._lhs.schema), Schema) \
            else type(self._rhs.schema)
        self._schema = schema_type.from_lists(names, typos)

    def _validate_equal(self, equal_expr):
        # FIXME: sometimes may be wrong, e.g. t3 = t1.join(t2, 'name')); t4.join(t3, t4.id == t3.id)
        return (equal_expr.lhs.is_ancestor(self._get_child(self._lhs)) and
                equal_expr.rhs.is_ancestor(self._get_child(self._rhs))) or \
               (equal_expr.lhs.is_ancestor(self._get_child(self._rhs)) and
                equal_expr.rhs.is_ancestor(self._get_child(self._lhs)))

    def _validate_predicates(self, predicates):
        if predicates is None:
            return

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

        if len(subs) == 0:
            self._predicate = None
        else:
            self._predicate = subs

    def _merge_joined_fields(self, merge_columns):
        if not merge_columns:
            return self

        predicate_fields = self._get_predicate_fields()
        if not predicate_fields:
            raise ValueError('No fields in predicate. Cannot merge columns.')

        src_map = self._column_origins
        rename_map = dict()
        for name, src in six.iteritems(src_map):
            src_id, src_name = src
            if src_name not in predicate_fields:
                continue
            if src_name not in rename_map:
                rename_map[src_name] = [None, None]
            rename_map[src_name][src_id] = name

        if merge_columns in ('auto', 'left', 'right') or (isinstance(merge_columns, bool) and merge_columns):
            merge_columns = dict((k, merge_columns) for k in six.iterkeys(rename_map))
        if isinstance(merge_columns, six.string_types):
            merge_columns = {merge_columns: 'auto'}
        if isinstance(merge_columns, list):
            merge_columns = dict((k, 'auto') for k in merge_columns)

        excludes = set()
        for col, action in six.iteritems(merge_columns):
            if col not in rename_map:
                raise ValueError('Column {0} not exists in join predicate.'.format(col))
            if isinstance(action, bool) and action:
                merge_columns[col] = 'auto'
            else:
                merge_columns[col] = action.lower()
            excludes.update(rename_map[col])

        selects = []
        merged = set()
        for col in self.schema.names:
            if col not in excludes:
                selects.append(self[col])
            else:
                src_name = src_map[col][1]
                if src_name in merged:
                    continue

                merged.add(src_name)

                left_name, right_name = rename_map[src_name]
                left_col = self[left_name]
                right_col = self[right_name]

                if merge_columns[src_name] == 'auto':
                    selects.append(left_col.isnull().ifelse(right_col, left_col).rename(src_name))
                elif merge_columns[src_name] == 'left':
                    selects.append(left_col.rename(src_name))
                elif merge_columns[src_name] == 'right':
                    selects.append(right_col.rename(src_name))
        selected = self.select(*selects)
        return JoinFieldMergedCollectionExpr(_input=self, _fields=selected._fields,
                                             _schema=selected._schema, _rename_map=rename_map)


class InnerJoin(JoinCollectionExpr):
    def _init(self, *args, **kwargs):
        self._how = 'INNER'
        super(InnerJoin, self)._init(*args, **kwargs)

    def _get_non_suffixes_fields(self):
        return self._get_predicate_fields()


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


class JoinFieldMergedCollectionExpr(ProjectCollectionExpr):
    __slots__ = '_rename_map',

    def _get_fields(self, fields, ret_raw_fields=False):
        selects = []
        raw_selects = []
        joined_expr = self._input

        for field in fields:
            field = self._defunc(field)
            if isinstance(field, CollectionExpr):
                if any(c is self for c in field.children()) or \
                        any(c is joined_expr._lhs for c in field.children()) or \
                        any(c is joined_expr._rhs for c in field.children()):
                    selects.extend(self._get_fields(field._project_fields))
                elif field is self:
                    selects.extend(self._get_fields(self._schema.names))
                elif field is joined_expr._lhs:
                    fields = [joined_expr._renamed_columns.get(n, [n])[0]
                              if n not in self._rename_map else n
                              for n in field.schema.names]
                    selects.extend(self._get_fields(fields))
                elif field is joined_expr._rhs:
                    fields = [joined_expr._renamed_columns.get(n, [None, n])[1]
                              if n not in self._rename_map else n
                              for n in field.schema.names]
                    selects.extend(self._get_fields(fields))
                else:
                    selects.extend(super(JoinFieldMergedCollectionExpr, self)._get_fields(field))
                raw_selects.append(field)
            else:
                select = self._get_field(field)
                selects.append(select)
                raw_selects.append(select)

        if ret_raw_fields:
            return selects, raw_selects
        return selects

    def _get_field(self, field):
        field = self._defunc(field)

        if isinstance(field, six.string_types):
            if field not in self._schema:
                raise ValueError('Field(%s) does not exist' % field)
            return Column(self, _name=field, _data_type=self._schema[field].type)

        joined_expr = self._input
        root = field
        has_path = False

        for expr in root.traverse(top_down=True, unique=True,
                                  stop_cond=lambda x: isinstance(x, Column) or x is self):
            if isinstance(expr, Column):
                if expr.input is self:
                    has_path = True
                    continue
                if expr.input is joined_expr._lhs:
                    has_path = True
                    idx = 0
                elif expr.input is joined_expr._rhs:
                    has_path = True
                    idx = 1
                elif isinstance(joined_expr._lhs, JoinCollectionExpr):
                    try:
                        expr = joined_expr._lhs._get_field(expr)
                    except ExpressionError:
                        continue
                    has_path = True
                    idx = 0
                elif isinstance(joined_expr._rhs, JoinCollectionExpr):
                    try:
                        expr = joined_expr._rhs._get_field(expr)
                    except ExpressionError:
                        continue
                    has_path = True
                    idx = 1
                else:
                    continue

                name = expr.source_name
                if name not in self._rename_map and name in joined_expr._renamed_columns:
                    name = joined_expr._renamed_columns[name][idx]
                to_sub = self._get_field(name)
                if expr.is_renamed():
                    to_sub = to_sub.rename(expr.name)

                to_sub.copy_to(expr)

        if isinstance(field, SequenceExpr) and not has_path:
            raise ExpressionError('field must come from Join collection '
                                  'or its left and right child collection: %s'
                                  % repr_obj(field))
        return root


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


def _make_different_sources(left, right, predicate=None):
    # TODO: move to analyzer, do it before analyze and optimize
    exprs = ExprDictionary()

    for n in left.traverse(unique=True):
        exprs[n] = True

    subs = ExprDictionary()

    dag = right.to_dag(copy=False, validate=False)
    for n in dag.traverse():
        if n in exprs:
            copied = subs.get(n, n.copy())
            for p in dag.successors(n):
                if p in exprs and p not in subs:
                    subs[p] = p.copy()
                subs.get(p, p).substitute(n, copied)
            subs[n] = copied

            if predicate and n is right:
                for p in predicate:
                    if not isinstance(p, Expr):
                        continue
                    p_dag = p.to_dag(copy=False, validate=False)
                    for p_n in p_dag.traverse(top_down=True):
                        if p_n is right:
                            p_dag.substitute(right, copied)

    return left, subs.get(right, right)


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
        on = [name for name in left.schema.names if name in right.schema._name_indexes]

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

    left, right = _make_different_sources(left, right, on)

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


def left_join(left, right, on=None, suffixes=('_x', '_y'), mapjoin=False, merge_columns=None):
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
    :param merge_columns: whether to merge columns with the same name into one column without suffix.
                          If the value is True, columns in the predicate with same names will be merged,
                          with non-null value. If the value is 'left' or 'right', the values of predicates
                          on the left / right collection will be taken. You can also pass a dictionary to
                          describe the behavior of each column, such as { 'a': 'auto', 'b': 'left' }.
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
    joined = join(left, right, on, how='left', suffixes=suffixes, mapjoin=mapjoin)
    return joined._merge_joined_fields(merge_columns)


def right_join(left, right, on=None, suffixes=('_x', '_y'), mapjoin=False, merge_columns=None):
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
    :param merge_columns: whether to merge columns with the same name into one column without suffix.
                          If the value is True, columns in the predicate with same names will be merged,
                          with non-null value. If the value is 'left' or 'right', the values of predicates
                          on the left / right collection will be taken. You can also pass a dictionary to
                          describe the behavior of each column, such as { 'a': 'auto', 'b': 'left' }.
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
    joined = join(left, right, on, how='right', suffixes=suffixes, mapjoin=mapjoin)
    return joined._merge_joined_fields(merge_columns)


def outer_join(left, right, on=None, suffixes=('_x', '_y'), mapjoin=False, merge_columns=None):
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
    :param merge_columns: whether to merge columns with the same name into one column without suffix.
                          If the value is True, columns in the predicate with same names will be merged,
                          with non-null value. If the value is 'left' or 'right', the values of predicates
                          on the left / right collection will be taken. You can also pass a dictionary to
                          describe the behavior of each column, such as { 'a': 'auto', 'b': 'left' }.
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
    joined = join(left, right, on, how='outer', suffixes=suffixes, mapjoin=mapjoin)
    return joined._merge_joined_fields(merge_columns)


CollectionExpr.join = join
CollectionExpr.inner_join = inner_join
CollectionExpr.left_join = left_join
CollectionExpr.right_join = right_join
CollectionExpr.outer_join = outer_join


def _get_sequence_source_collection(expr):
    return next(it for it in expr.traverse(top_down=True, unique=True) if isinstance(it, CollectionExpr))


class UnionCollectionExpr(CollectionExpr):
    __slots__ = '_distinct',
    _args = '_lhs', '_rhs',
    node_name = 'Union'

    def _init(self, *args, **kwargs):
        super(UnionCollectionExpr, self)._init(*args, **kwargs)

        self._validate()
        self._schema = self._clean_schema()

    def _validate_collection_child(self):
        if self._lhs.schema.names == self._rhs.schema.names and self._lhs.schema.types == self._rhs.schema.types:
            return True
        elif set(self._lhs.schema.names) == set(self._rhs.schema.names) and set(self._lhs.schema.types) == set(
            self._rhs.schema.types):
            self._rhs = self._rhs[self._lhs.schema.names]
            return self._lhs.schema.types == self._rhs.schema.types
        else:
            return False

    def _validate(self):
        if isinstance(self._lhs, SequenceExpr):
            source_collection = _get_sequence_source_collection(self._lhs)
            self._lhs = source_collection[[self._lhs]]
        if isinstance(self._rhs, SequenceExpr):
            source_collection = _get_sequence_source_collection(self._rhs)
            self._rhs = source_collection[[self._rhs]]

        if isinstance(self._lhs, CollectionExpr) and isinstance(self._rhs, CollectionExpr):
            if not self._validate_collection_child():
                raise ExpressionError('Table schemas must be equal to form union')
        else:
            raise ExpressionError('Both inputs should be collections or sequences.')

    def _clean_schema(self):
        return Schema.from_lists(self._lhs.schema.names, self._lhs.schema.types)

    def accept(self, visitor):
        return visitor.visit_union(self)

    def iter_args(self):
        for it in zip(['collection(left)', 'collection(right)'], self.args):
            yield it


class ConcatCollectionExpr(CollectionExpr):
    _args = '_lhs', '_rhs',
    node_name = 'Concat'

    def _init(self, *args, **kwargs):
        super(ConcatCollectionExpr, self)._init(*args, **kwargs)

        self._schema = self._clean_schema()

    def iter_args(self):
        for it in zip(['collection(left)', 'collection(right)'], self.args):
            yield it

    def _clean_schema(self):
        return Schema.from_lists(
            self._lhs.schema.names + self._rhs.schema.names,
            self._lhs.schema.types + self._rhs.schema.types,
        )

    @staticmethod
    def _get_sequence_source_collection(expr):
        return next(it for it in expr.traverse(top_down=True, unique=True) if isinstance(it, CollectionExpr))

    @classmethod
    def validate_input(cls, *inputs):
        new_inputs = []
        for i in inputs:
            if isinstance(i, SequenceExpr):
                source_collection = cls._get_sequence_source_collection(i)
                new_inputs.append(source_collection[[i]])
            else:
                new_inputs.append(i)

        if any(not isinstance(i, CollectionExpr) for i in new_inputs):
            raise ExpressionError('Inputs should be collections or sequences.')

        unioned = reduce(lambda a, b: a | b, (set(i.schema.names) for i in inputs))
        total_fields = sum((len(i.schema.names) for i in inputs))
        if total_fields != len(unioned):
            raise ExpressionError('Column names in inputs should not collides with each other.')

    def accept(self, visitor):
        return visitor.visit_concat(self)


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
    left, right = _make_different_sources(left, right)
    return UnionCollectionExpr(_lhs=left, _rhs=right, _distinct=distinct)


def __horz_concat(left, rights):
    for right in rights:
        left, right = _make_different_sources(left, right)
        left = ConcatCollectionExpr(_lhs=left, _rhs=right)
    return left


def concat(left, rights, distinct=False, axis=0):
    """
    Concat collections.

    :param left: left collection
    :param rights: right collections, can be a DataFrame object or a list of DataFrames
    :param distinct: whether to remove duplicate entries. only available when axis == 0
    :param axis: when axis == 0, the DataFrames are merged vertically, otherwise horizontally.
    :return: collection

    Note that axis==1 can only be used under Pandas DataFrames or XFlow.

    :Example:
    >>> df['name', 'id'].concat(df2['score'], axis=1)
    """
    if isinstance(rights, Node):
        rights = [rights, ]
    if not rights:
        raise ValueError('At least one DataFrame should be provided.')

    if axis == 0:
        for right in rights:
            left = union(left, right, distinct=distinct)
        return left

    ConcatCollectionExpr.validate_input(left, *rights)

    if hasattr(left, '_xflow_concat'):
        return left._xflow_concat(rights)
    else:
        return __horz_concat(left, rights)


def _drop(expr, data, axis=0, columns=None):
    """
    Drop data from a DataFrame.
    
    :param expr: collection to drop data from
    :param data: data to be removed
    :param axis: 0 for deleting rows, 1 for columns.
    :param columns: columns of data to select, only useful when axis == 0
    :return: collection
    
    Example:
    >>> import pandas as pd
    >>> df1 = DataFrame(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}))
    >>> df2 = DataFrame(pd.DataFrame({'a': [2, 3], 'b': [5, 7]}))
    >>> df1.drop(df2)
       a  b  c
    0  1  4  7
    1  3  6  9
    >>> df1.drop(df2, columns='a')
       a  b  c
    0  1  4  7
    >>> df1.drop(['a'], axis=1)
       b  c
    0  4  7
    1  5  8
    2  6  9
    >>> df1.drop(df2, axis=1)
       c
    0  7
    1  8
    2  9
    """
    from ..utils import to_collection
    expr = to_collection(expr)

    if axis == 0:
        if not isinstance(data, (CollectionExpr, SequenceExpr)):
            raise ExpressionError('data should be a collection or sequence when axis == 1.')

        data = to_collection(data)
        if columns is None:
            columns = [n for n in data.schema.names]
        if isinstance(columns, six.string_types):
            columns = [columns, ]

        data = data.select(*columns).distinct()

        drop_predicates = [data[n].isnull() for n in data.schema.names]
        return expr.left_join(data, on=columns, suffixes=('', '_dp')).filter(*drop_predicates) \
            .select(*expr.schema.names)
    else:
        if isinstance(data, (CollectionExpr, SequenceExpr)):
            data = to_collection(data).schema.names
        return expr.exclude(data)


def setdiff(left, *rights, **kwargs):
    """
    Exclude data from a collection, like `except` clause in SQL. All collections involved should
    have same schema.
    
    :param left: collection to drop data from
    :param rights: collection or list of collections
    :param distinct: whether to preserve duplicate entries
    :return: collection
    
    Examples:
    >>> import pandas as pd
    >>> df1 = DataFrame(pd.DataFrame({'a': [1, 2, 3, 3, 3], 'b': [1, 2, 3, 3, 3]}))
    >>> df2 = DataFrame(pd.DataFrame({'a': [1, 3], 'b': [1, 3]}))
    >>> df1.setdiff(df2)
       a  b
    0  2  2
    1  3  3
    2  3  3
    >>> df1.setdiff(df2, distinct=True)
       a  b
    0  2  2
    """
    import time
    from ..utils import output

    distinct = kwargs.get('distinct', False)

    if isinstance(rights[0], list):
        rights = rights[0]

    cols = [n for n in left.schema.names]
    types = [n for n in left.schema.types]

    counter_col_name = 'exc_counter_%d' % int(time.time())
    left = left[left, Scalar(1).rename(counter_col_name)]
    rights = [r[r, Scalar(-1).rename(counter_col_name)] for r in rights]

    unioned = left
    for r in rights:
        unioned = unioned.union(r)

    if distinct:
        aggregated = unioned.groupby(*cols).agg(**{counter_col_name: unioned[counter_col_name].min()})
        return aggregated.filter(aggregated[counter_col_name] == 1).select(*cols)
    else:
        aggregated = unioned.groupby(*cols).agg(**{counter_col_name: unioned[counter_col_name].sum()})

        @output(cols, types)
        def exploder(row):
            import sys
            irange = xrange if sys.version_info[0] < 3 else range
            for _ in irange(getattr(row, counter_col_name)):
                yield row[:-1]

        return aggregated.map_reduce(mapper=exploder).select(*cols)


def intersect(left, *rights, **kwargs):
    """
    Calc intersection among datasets,
    
    :param left: collection
    :param rights: collection or list of collections
    :param distinct: whether to preserve duolicate entries
    :return: collection
    
    Examples:
    >>> import pandas as pd
    >>> df1 = DataFrame(pd.DataFrame({'a': [1, 2, 3, 3, 3], 'b': [1, 2, 3, 3, 3]}))
    >>> df2 = DataFrame(pd.DataFrame({'a': [1, 3, 3], 'b': [1, 3, 3]}))
    >>> df1.intersect(df2)
       a  b
    0  1  1 
    1  3  3
    2  3  3
    >>> df1.intersect(df2, distinct=True)
       a  b
    0  1  1 
    1  3  3
    """
    import time
    from ..utils import output

    distinct = kwargs.get('distinct', False)

    if isinstance(rights[0], list):
        rights = rights[0]

    cols = [n for n in left.schema.names]
    types = [n for n in left.schema.types]

    collections = (left, ) + rights

    idx_col_name = 'idx_%d' % int(time.time())
    counter_col_name = 'exc_counter_%d' % int(time.time())

    collections = [c[c, Scalar(idx).rename(idx_col_name)] for idx, c in enumerate(collections)]

    unioned = reduce(lambda a, b: a.union(b), collections)
    src_agg = unioned.groupby(*(cols + [idx_col_name])) \
        .agg(**{counter_col_name: unioned.count()})

    aggregators = {
        idx_col_name: src_agg[idx_col_name].nunique(),
        counter_col_name: src_agg[counter_col_name].min(),
    }
    final_agg = src_agg.groupby(*cols).agg(**aggregators)
    final_agg = final_agg.filter(final_agg[idx_col_name] == len(collections))

    if distinct:
        return final_agg.filter(final_agg[counter_col_name] > 0).select(*cols)
    else:
        @output(cols, types)
        def exploder(row):
            import sys
            irange = xrange if sys.version_info[0] < 3 else range
            for _ in irange(getattr(row, counter_col_name)):
                yield row[:-2]

        return final_agg.map_reduce(mapper=exploder).select(*cols)


CollectionExpr.union = union
SequenceExpr.union = union
CollectionExpr.__horz_concat = __horz_concat
SequenceExpr.__horz_concat = __horz_concat
CollectionExpr.concat = concat
SequenceExpr.concat = concat
CollectionExpr.drop = _drop
SequenceExpr.drop = _drop
CollectionExpr.setdiff = setdiff
SequenceExpr.setdiff = setdiff
CollectionExpr.except_ = setdiff
SequenceExpr.except_ = setdiff
CollectionExpr.intersect = intersect
SequenceExpr.intersect = intersect
