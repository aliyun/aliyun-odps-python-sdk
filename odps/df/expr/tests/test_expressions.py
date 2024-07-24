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

from collections import namedtuple

import pytest

from ....config import option_context
from ....models import TableSchema
from .. import errors
from ..arithmetic import Add
from ..core import ExprDictionary
from ..expressions import *
from ..tests.core import MockTable


@pytest.fixture
def exprs(config):
    schema = TableSchema.from_lists(
        ['name', 'id', 'fid'], [types.string, types.int64, types.float64]
    )
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema, client=config.odps.rest)
    expr = CollectionExpr(_source_data=table, _schema=schema)

    schema2 = TableSchema.from_lists(
        ['name', 'id', 'fid'], [types.string, types.int64, types.float64],
        ['part1', 'part2'], [types.string, types.int64],
    )
    table2 = MockTable(name='pyodps_test_expr_table2', table_schema=schema2, client=config.odps.rest)
    expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

    schema3 = TableSchema.from_lists(
        ['id', 'name', 'relatives', 'hobbies'],
        [types.int64, types.string, types.Dict(types.string, types.string),
         types.List(types.string)],
    )
    table3 = MockTable(name='pyodps_test_expr_table3', table_schema=schema3, client=config.odps.rest)
    expr3 = CollectionExpr(_source_data=table3, _schema=schema3)

    nt = namedtuple("NT", "expr, expr2, expr3")
    return nt(expr, expr2, expr3)


def test_dir(exprs):
    expr_dir = dir(exprs.expr)
    assert 'id' in expr_dir
    assert 'fid' in expr_dir

    new_df = exprs.expr[exprs.expr.id, exprs.expr.fid, exprs.expr.name.rename('if')]
    assert 'if' not in dir(new_df)

    assert exprs.expr._id == exprs.expr.copy()._id


def test_typed_expr_missing_attr(exprs):
    with pytest.raises(AttributeError) as ex_pack:
        getattr(exprs.expr.name, "mean")
    # need to show type of expression when certain method
    assert str(exprs.expr.name.dtype) in str(ex_pack.value)


def test_projection(exprs):
    projected = exprs.expr['name', exprs.expr.id.rename('new_id')]

    assert isinstance(projected, CollectionExpr)
    assert projected._schema == TableSchema.from_lists(['name', 'new_id'], [types.string, types.int64])

    projected = exprs.expr[[exprs.expr.name, exprs.expr.id.astype('string')]]

    assert isinstance(projected, ProjectCollectionExpr)
    assert projected._schema == TableSchema.from_lists(['name', 'id'], [types.string, types.string])

    projected = exprs.expr.select(exprs.expr.name, Scalar('abc').rename('word'), size=5)

    assert isinstance(projected, ProjectCollectionExpr)
    assert projected._schema == TableSchema.from_lists(
            ['name', 'word', 'size'], [types.string, types.string, types.int8]
        )
    assert isinstance(projected._fields[1], StringScalar)
    assert projected._fields[1].value == 'abc'
    assert isinstance(projected._fields[2], Int8Scalar)
    assert projected._fields[2].value == 5

    expr = exprs.expr[lambda x: x.exclude('id')]
    assert expr.schema.names == [n for n in expr.schema.names if n != 'id']

    pytest.raises(ExpressionError, lambda: exprs.expr[exprs.expr.distinct('id', 'fid'), 'name'])
    pytest.raises(ExpressionError, lambda: exprs.expr[[exprs.expr.id + exprs.expr.fid]])

    with option_context() as options:
        options.interactive = True

        exprs.expr['name', 'id'][[exprs.expr.name, ]]

    pytest.raises(ExpressionError, lambda: exprs.expr[exprs.expr.name])
    pytest.raises(ExpressionError, lambda: exprs.expr['name', exprs.expr.groupby('name').id.sum()])

    expr = exprs.expr.filter(exprs.expr.id < 0)
    expr[exprs.expr.name, exprs.expr.id]


def test_filter(exprs):
    filtered = exprs.expr[(exprs.expr.id < 10) & (exprs.expr.name == 'test')]

    assert isinstance(filtered, FilterCollectionExpr)

    filtered = exprs.expr.filter(exprs.expr.id < 10, exprs.expr.name == 'test')

    assert isinstance(filtered, FilterCollectionExpr)


def test_slice(exprs):
    sliced = exprs.expr[:100]

    assert isinstance(sliced, SliceCollectionExpr)
    assert sliced._schema == exprs.expr._schema
    assert isinstance(sliced._indexes, tuple)

    not_sliced = exprs.expr[:]

    assert not isinstance(not_sliced, SliceCollectionExpr)
    assert isinstance(not_sliced, CollectionExpr)


def test_as_type(exprs):
    fid = exprs.expr.id.astype('float')

    assert isinstance(fid._source_data_type, types.Int64)
    assert isinstance(fid._data_type, types.Float64)
    assert isinstance(fid, Float64SequenceExpr)
    assert not isinstance(fid, Int64SequenceExpr)

    int_fid = fid.astype('int')

    assert isinstance(int_fid._source_data_type, types.Int64)
    assert isinstance(int_fid._data_type, types.Int64)
    assert isinstance(int_fid, Int64SequenceExpr)
    assert not isinstance(int_fid, Float64SequenceExpr)

    float_fid = (fid + 1).astype('float32')

    assert isinstance(float_fid, Float32SequenceExpr)
    assert not isinstance(float_fid, Int32SequenceExpr)
    assert isinstance(float_fid, AsTypedSequenceExpr)


def test_rename(exprs):
    new_id = exprs.expr.id.rename('new_id')

    assert isinstance(new_id, SequenceExpr)
    assert new_id._source_name == 'id'
    assert new_id._name == 'new_id'

    double_new_id = new_id.rename('2new_id')

    assert isinstance(double_new_id, SequenceExpr)
    assert double_new_id._source_name == 'id'
    assert double_new_id._name == '2new_id'

    assert double_new_id is not new_id

    add_id = (exprs.expr.id + exprs.expr.fid).rename('add_id')
    assert isinstance(add_id, Float64SequenceExpr)
    assert not isinstance(add_id, Int64SequenceExpr)
    assert add_id._source_name is None
    assert isinstance(add_id, Add)
    assert add_id.name == 'add_id'
    assert isinstance(add_id._lhs, Int64SequenceExpr)
    assert isinstance(add_id._rhs, Float64SequenceExpr)
    assert add_id._lhs._source_name == 'id'
    assert add_id._rhs._source_name == 'fid'

    add_scalar_id = (exprs.expr.id + 5).rename('add_s_id')
    assert not isinstance(add_scalar_id, Float64SequenceExpr)
    assert isinstance(add_scalar_id, Int64SequenceExpr)
    assert isinstance(add_scalar_id, Add)
    assert add_scalar_id.name == 'add_s_id'
    assert add_scalar_id._lhs._source_name == 'id'


def test_new_sequence(exprs):
    column = Column(_data_type='int32')

    assert Int32SequenceExpr in type(column).mro()
    assert isinstance(column, Int32SequenceExpr)

    column = type(column)._new(_data_type='string')
    assert Int32SequenceExpr not in type(column).mro()
    assert StringSequenceExpr in type(column).mro()
    assert isinstance(column, StringSequenceExpr)
    assert not isinstance(column, Int32SequenceExpr)
    assert isinstance(column, Column)

    seq = SequenceExpr(_data_type='int64')
    assert isinstance(seq, Int64SequenceExpr)

    seq = BooleanSequenceExpr(_data_type='boolean')
    assert isinstance(seq, BooleanSequenceExpr)

    seq = DatetimeSequenceExpr(_data_type='float32')
    assert isinstance(seq, Float32SequenceExpr)

    class Int64Column(Column):
        __slots__ = 'test',

    column = Int64Column(_data_type='float64', test='value')

    assert isinstance(column, Float64SequenceExpr)
    assert not isinstance(column, Int64SequenceExpr)

    column = type(column)._new(_data_type='int8', test=column.test)
    assert column.test == 'value'
    assert isinstance(column, Int8SequenceExpr)
    assert not isinstance(column, Float64SequenceExpr)
    assert not isinstance(column, Int64SequenceExpr)
    assert isinstance(column, Int64Column)

    class Int64Column(Int64SequenceExpr):
        pass

    column = Int64Column(_data_type='float64')

    assert isinstance(column, Float64SequenceExpr)
    assert not isinstance(column, Int64SequenceExpr)

    column = type(column)._new(_data_type='int8')
    assert isinstance(column, Int8SequenceExpr)
    assert not isinstance(column, Float64SequenceExpr)
    assert not isinstance(column, Int64SequenceExpr)
    assert not isinstance(column, Int64Column)


def test_sequence_cache(exprs):
    df = exprs.expr.name
    pytest.raises(ExpressionError, lambda: df.cache())


def test_expr_field_validation(exprs):
    df = exprs.expr
    pytest.raises(errors.ExpressionError, lambda: df[df[:10].id])

    df2 = exprs.expr[['id']]
    pytest.raises(errors.ExpressionError, lambda: df[df2.id])


def test_filter_parts(exprs):
    pytest.raises(ExpressionError, lambda: exprs.expr.filter_parts(None))
    pytest.raises(ExpressionError, lambda: exprs.expr.filter_parts('part3=a'))
    pytest.raises(ExpressionError, lambda: exprs.expr.filter_parts('part1=a,part2=1/part1=b,part2=2'))
    pytest.raises(ExpressionError, lambda: exprs.expr2.filter_parts('part1,part2=1/part1=b,part2=2'))

    filtered1 = exprs.expr2.filter_parts('part1=a,part2=1/part1=b,part2=2')
    assert isinstance(filtered1, FilterPartitionCollectionExpr)
    assert filtered1.schema == exprs.expr.schema
    assert filtered1.predicate_string == 'part1=a,part2=1/part1=b,part2=2'

    filtered2 = exprs.expr2.filter_parts('part1=a,part2=1/part1=b,part2=2', exclude=False)
    assert isinstance(filtered2, FilterCollectionExpr)

    try:
        import pandas as pd
        from ... import DataFrame
        pd_df = pd.DataFrame([['Col1', 1], ['Col2', 2]], columns=['Field1', 'Field2'])
        df = DataFrame(pd_df)
        pytest.raises(ExpressionError, lambda: df.filter_parts('Fieldd2=2'))
    except ImportError:
        pass


def test_dep_expr(exprs):
    expr1 = Scalar('1')
    expr2 = exprs.expr['id']
    expr2.add_deps(expr1)

    assert expr1 in expr2.deps


def test_backtrack_field(exprs):
    expr = exprs.expr.filter(exprs.expr.id < 3)[exprs.expr.id + 1, exprs.expr.name.rename('name2')]

    assert expr._fields[0].lhs.input is expr.input
    assert expr._fields[1].input is expr.input
    assert expr._fields[1].name == 'name2'

    with pytest.raises(ExpressionError):
        exprs.expr[exprs.expr.id + 1, exprs.expr.name][exprs.expr.name, exprs.expr.id]

    expr = exprs.expr[exprs.expr.id + 1, 'name'].filter(exprs.expr.name == 'a')

    assert expr._predicate.lhs.input is expr.input

    with pytest.raises(ExpressionError):
        exprs.expr[exprs.expr.id + 1, exprs.expr.name][exprs.expr2.name,]

    expr1 = exprs.expr['name', (exprs.expr.id + 1).rename('id2')]
    expr = expr1[expr1.name.notnull()][
        expr1.name.rename('name2'),
        expr1.id2.rename('id3'),
        expr1.groupby('id2').sort('id2').rank()
    ]
    assert expr._fields[1].input is expr.input
    assert expr._fields[2].input is expr.input


def test_setitem_field(config, exprs):
    from ..groupby import GroupByCollectionExpr
    from ..merge import JoinFieldMergedCollectionExpr

    expr = exprs.expr.copy()

    expr['new_id'] = expr.id + 1

    assert 'new_id' in expr.schema.names
    assert expr._fields[-1].lhs.input is expr.input

    assert expr.schema.names == ['name', 'id', 'fid', 'new_id']

    expr['new_id2'] = expr.id + 2

    assert 'new_id2' in expr.schema.names
    assert expr._fields[-1].lhs.input is expr.input

    assert expr.schema.names == ['name', 'id', 'fid', 'new_id', 'new_id2']
    assert expr._input._proxy is None

    expr['new_id2'] = expr.new_id

    expr['new_id3'] = expr.id + expr.new_id2
    assert expr._fields[-1].lhs.input is expr.input
    assert expr._fields[-1].rhs.lhs.input is expr.input

    assert isinstance(expr, ProjectCollectionExpr)
    assert isinstance(expr, ProjectCollectionExpr) is True

    expr2 = expr.groupby('name').agg(expr.id.sum())
    expr2['new_id2'] = expr2.id_sum + 1
    assert isinstance(expr2, ProjectCollectionExpr)
    assert not isinstance(expr2, GroupByCollectionExpr)
    assert not isinstance(expr2, FilterCollectionExpr)

    schema = TableSchema.from_lists(
        ['name', 'id', 'fid2', 'fid3'],
        [types.string, types.int64, types.float64, types.float64],
    )
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)
    table._client = config.odps.rest
    expr3 = CollectionExpr(_source_data=table, _schema=schema)

    expr4 = expr.left_join(expr3, on=[expr.name == expr3.name, expr.id == expr3.id],
                           merge_columns=True)
    expr4['fid_1'] = expr4.groupby('id').sort('fid2').row_number()
    assert isinstance(expr4, JoinFieldMergedCollectionExpr)
    assert expr4._proxy is None

    expr5 = expr[expr]
    expr5['name_2'] = expr5.apply(lambda row: row.name, axis=1, reduce=True)
    assert isinstance(expr5, ProjectCollectionExpr)
    assert expr5._proxy is None


def test_setitem_condition_field(exprs):
    from ..arithmetic import And
    from ..element import IfElse

    expr = exprs.expr.copy()

    pytest.raises(ValueError, expr.__setitem__, (expr.id, 'new_id'), 0)
    pytest.raises(ValueError, expr.__setitem__, (expr.id, expr.name, 'new_id'), 0)

    expr[expr.id < 10, 'new_id'] = expr.id + 1
    assert 'new_id' in expr.schema.names
    assert isinstance(expr._fields[-1], IfElse)

    expr[expr.id > 5, 'new_id'] = None
    assert 'new_id' in expr.schema.names
    assert isinstance(expr._fields[-1], IfElse)

    expr[expr.id < 5, expr.name == 'test', 'new_id2'] = expr.id + 2
    assert 'new_id2' in expr.schema.names
    assert isinstance(expr._fields[-1], IfElse)
    assert isinstance(expr._fields[-1].input, And)

    expr[expr.id >= 5, expr.name == 'test', 'new_id2'] = expr.id + 2
    assert 'new_id2' in expr.schema.names
    assert isinstance(expr._fields[-1], IfElse)
    assert isinstance(expr._fields[-1].input, And)
    assert isinstance(expr._fields[-1]._else, IfElse)

    expr2 = expr['id', 'name']
    expr2[expr2.id >= 5, expr2.name == 'test', 'new_id3'] = expr.id + 2
    assert 'new_id3' in expr2.schema.names
    assert isinstance(expr._fields[-1], IfElse)
    assert isinstance(expr._fields[-1].input, And)
    assert isinstance(expr._fields[-1]._else, IfElse)


def test_delitem_field(exprs):
    from ..collections import DistinctCollectionExpr
    from ..groupby import GroupByCollectionExpr

    expr = exprs.expr.copy()

    del expr['fid']

    assert 'fid' not in expr.schema
    assert expr.schema.names == ['name', 'id']
    assert isinstance(expr, ProjectCollectionExpr)

    expr['id2'] = exprs.expr.id + 1
    del expr['id2']

    assert 'id2' not in expr.schema
    assert expr.schema.names == ['name', 'id']

    expr['id3'] = expr.id
    del expr['id']

    assert 'id' not in expr.schema
    assert 'id3' in expr.schema
    assert expr.schema.names == ['name', 'id3']

    expr2 = expr.groupby('name').agg(expr.id3.sum().rename('id'))
    del expr2.name

    assert 'name' not in expr2.schema
    assert 'id' in expr2.schema
    assert expr2.schema.names == ['id']
    assert isinstance(expr2, ProjectCollectionExpr)
    assert not isinstance(expr2, GroupByCollectionExpr)

    expr3 = expr2.distinct()
    expr3['new_id'] = expr3.id + 1
    expr3['new_id2'] = expr3.new_id * 2
    del expr3['new_id']

    assert 'new_id' not in expr3.schema
    assert 'new_id2' in expr3.schema
    assert expr3.schema.names == ['id', 'new_id2']
    assert isinstance(expr3, ProjectCollectionExpr)
    assert not isinstance(expr3, DistinctCollectionExpr)


def test_lateral_view(exprs):
    from ..collections import RowAppliedCollectionExpr

    expr = exprs.expr3.copy()

    expr1 = expr[expr.id, expr.relatives.explode(), expr.hobbies.explode()]
    assert isinstance(expr1, LateralViewCollectionExpr)
    assert len(expr1.lateral_views) == 2
    assert isinstance(expr1.lateral_views[0], RowAppliedCollectionExpr)
    assert expr1.lateral_views[0]._lateral_view is True
    assert isinstance(expr1.lateral_views[1], RowAppliedCollectionExpr)
    assert expr1.lateral_views[1]._lateral_view is True

    expr2 = expr.relatives.explode(['r_key', 'r_value'])
    expr2 = expr2[expr2.r_key.rename('rk'), expr2]
    assert isinstance(expr2, ProjectCollectionExpr)
    assert not isinstance(expr2, LateralViewCollectionExpr)

    left = expr.relatives.explode(['r_key', 'r_value'])
    joined = left.join(expr, on=(left.r_key == expr.name))
    expr3 = joined['name', left]
    assert isinstance(expr3, ProjectCollectionExpr)
    assert not isinstance(expr3, LateralViewCollectionExpr)

    left = expr.relatives.explode(['name', 'r_value'])
    joined = left.left_join(expr, on='name', merge_columns=True)
    expr4 = joined['id', left]
    assert isinstance(expr4, ProjectCollectionExpr)
    assert not isinstance(expr4, LateralViewCollectionExpr)

    u1 = expr.relatives.explode(['name', 'r_value'])
    u2 = expr.relatives.explode(['name', 'r_value'])
    u3 = expr[expr.name, expr.hobbies.explode('r_value')]
    unioned = u1.union(u2).union(u3)
    expr5 = unioned[Scalar('unioned').rename('scalar'), u1]
    assert isinstance(expr5, ProjectCollectionExpr)
    assert not isinstance(expr5, LateralViewCollectionExpr)
