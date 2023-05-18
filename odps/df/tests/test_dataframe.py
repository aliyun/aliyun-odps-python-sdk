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

from datetime import datetime
from decimal import Decimal

import pytest

from ...tests.core import get_result, tn, pandas_case, global_locked
from ...models import TableSchema
from ...utils import to_text
from ...errors import ODPSError, DependencyNotInstalledError
from .. import DataFrame, Delay
from ..backends.context import context


@pytest.fixture
def table(odps):
    test_table_name = tn('pyodps_test_dataframe')
    schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'])

    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema)

    with table.open_writer() as w:
        w.write([[1, 'name1'], [2, 'name2'], [3, 'name3']])

    try:
        yield table
    finally:
        table.drop()


def test_dataframe(table):
    df = DataFrame(table)

    assert 3 == df.count().execute()
    assert 1 == df[df.name == 'name1'].count().execute()

    res = df[df.name.contains('中文')].execute()
    assert len(res) >= 0


@pandas_case
def test_dataframe_from_pandas():
    import pandas as pd

    pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [None, None, None]})

    pytest.raises(TypeError, lambda: DataFrame(pd_df))

    df = DataFrame(pd_df, unknown_as_string=True)
    assert df.schema.get_type('b').name == 'string'

    df = DataFrame(pd_df[['a']], as_type={'a': 'string'})
    assert df.schema.get_type('a').name == 'string'

    df = DataFrame(pd_df, as_type={'b': 'int'})
    assert df.schema.get_type('b').name == 'int64'

    pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [[1, 2], [3, 4, 5], [6]]})

    pytest.raises(TypeError, DataFrame, pd_df)

    df = DataFrame(pd_df, as_type={'b': 'list<int64>'})
    assert df.schema.get_type('b').name == 'list<int64>'

    df = DataFrame(pd_df, as_type={'b': 'list<string>'})
    assert df.schema.get_type('b').name == 'list<string>'

    pd_df = pd.DataFrame({'a': [1, 2, 3],
                          'b': [{1: 'a', 2: 'b'}, {3: 'c', 4: 'd', 5: None}, {6: 'f'}]})

    pytest.raises(TypeError, DataFrame, pd_df)

    df = DataFrame(pd_df, as_type={'b': 'dict<int64, string>'})
    assert df.schema.get_type('b').name == 'dict<int64,string>'

    df = DataFrame(pd_df, as_type={'b': 'dict<string, string>'})
    assert df.schema.get_type('b').name == 'dict<string,string>'


def test_head_and_tail(table):
    df = DataFrame(table)

    assert 1 == len(df.head(1))
    assert 2 == len(df.head(2))
    assert [3, 'name3'] == list(df.tail(1)[0])

    r = df[df.name == 'name2'].head(1)
    assert 1 == len(r)
    assert [2, 'name2'] == list(r[0])


@global_locked('odps_instance_method_repr')
def test_instance_method_repr(table):
    from odps import options

    class CannotExecuteDataFrame(DataFrame):
        def execute(self, **kwargs):
            raise RuntimeError('DataFrame cannot be executed')

    options.interactive = True

    try:
        df = CannotExecuteDataFrame(table)

        assert repr(df.count) == '<bound method Collection.count>'
        assert repr(df.name.count) == '<bound method Column._count>'
    finally:
        options.interactive = False


@pandas_case
def test_to_pandas(odps):
    table_name = tn('pyodps_test_mixed_engine_to_pandas')
    odps.delete_table(table_name, if_exists=True)
    table2 = odps.create_table(
        name=table_name,
        table_schema=TableSchema.from_lists(
            ['col%s' % i for i in range(7)],
            ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal', 'datetime']
        )
    )
    expr2 = DataFrame(table2)

    data2 = [
        [1234567, 3.14, 'test', datetime(2016, 6, 1), True, Decimal('3.14'), None]
    ]
    odps.write_table(table2, 0, data2)

    pd_df = expr2.to_pandas()
    assert list(data2[0]) == list(pd_df.iloc[0].tolist())

    wrapped_pd_df = expr2.to_pandas(wrap=True)
    assert list(data2[0]) == list(next(wrapped_pd_df.execute()))

    pd_df_col = expr2.col0.to_pandas()
    assert list([data2[0][0]]) == list(pd_df_col.tolist())

    wrapped_pd_df_col = expr2.col0.to_pandas(wrap=True)
    assert list([data2[0][0]]) == list(next(wrapped_pd_df_col.execute()))

    pd_df_future = expr2.to_pandas(async_=True)
    assert list(data2[0]) == list(pd_df_future.result().iloc[0].tolist())

    wrapped_pd_df_future = expr2.to_pandas(async_=True, wrap=True)
    assert list(data2[0]) == list(next(wrapped_pd_df_future.result().execute()))

    delay = Delay()
    pd_df_future = expr2.to_pandas(delay=delay)
    delay.execute()
    assert list(data2[0]) == list(pd_df_future.result().iloc[0].tolist())

    def raiser(_x):
        raise ValueError

    exc_future = expr2.col0.map(raiser).to_pandas(async_=True)
    pytest.raises(ODPSError, exc_future.result)


@pandas_case
def test_unicode_pd_dataframe():
    import pandas as pd

    pd_df = pd.DataFrame([['中文'], [to_text('中文2')]], columns=[to_text('字段')])
    df = DataFrame(pd_df)

    r = df['字段'].execute()
    assert to_text('中文') == to_text(r[0][0])
    assert to_text('中文2') == to_text(r[1][0])


@pandas_case
def test_pandas_groupby_filter():
    import pandas as pd

    data = [
        [2001, 1],
        [2002, 2],
        [2003, 3]
    ]
    df = DataFrame(pd.DataFrame(data, columns=['id', 'fid']))

    df2 = df.groupby('id').agg(df.fid.sum())
    df3 = df2[df2.id == 2003]

    expected = [
        [2003, 3]
    ]

    assert df3.execute().values.values.tolist() == expected

    df2 = df.groupby('id').agg(df.fid.sum())
    df2.execute()
    assert context.is_cached(df2) is True
    df3 = df2[df2.id == 2003]

    assert df3.execute().values.values.tolist() == expected
    assert df3.execute().values.values.tolist() == expected

    df4 = df.fid.sum()
    assert df4.execute() == 6
    assert df4.execute() == 6


def test_create_dataframe_from_partition(odps):
    from ...types import PartitionSpec

    test_table_name = tn('pyodps_test_dataframe_partition')
    schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds'], ['string'])

    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema)

    with table.open_writer('ds=today', create_partition=True) as w:
        w.write([[1, 'name1'], [2, 'name2'], [3, 'name3']])

    try:
        df = DataFrame(table.get_partition('ds=today'))
        assert df.count().execute() == 3

        df = table.get_partition('ds=today').to_df()
        partition = df.data
        assert partition.table is table
        assert partition.partition_spec == PartitionSpec('ds=today')
        assert df.count().execute() == 3
    finally:
        table.drop()


def test_set_item(table):
    df = DataFrame(table)
    df['id2'] = df.id + 1
    assert len(df.execute()) == 3

    df['id3'] = df['id2'] * 2
    assert len(df.execute()[0]) == 4

    del df['id2']
    res = df.execute()
    result = get_result(res)

    expected = [
        [1, 'name1', 4],
        [2, 'name2', 6],
        [3, 'name3', 8]
    ]
    assert expected == result

    df = DataFrame(table)
    df['id2'] = df.id

    try:
        res = df.to_pandas()
        result = get_result(res)

        expected = [
            [1, 'name1', 1],
            [2, 'name2', 2],
            [3, 'name3', 3]
        ]
        assert expected == result
    except (DependencyNotInstalledError, ImportError):
        pass

    df = DataFrame(table)
    df[df.id <= 2, 'id2'] = df.id

    try:
        res = df.to_pandas()
        result = get_result(res)

        expected = [
            [1, 'name1', 1],
            [2, 'name2', 2],
            [3, 'name3', None]
        ]
        assert expected == result
    except (DependencyNotInstalledError, ImportError):
        pass

    df = DataFrame(table)
    df[df.id <= 2, 'id2'] = df.id
    df[df.id > 1, 'id2'] = None

    try:
        res = df.to_pandas()
        result = get_result(res)

        expected = [
            [1, 'name1', 1],
            [2, 'name2', None],
            [3, 'name3', None]
        ]
        assert expected == result
    except (DependencyNotInstalledError, ImportError):
        pass

    df = DataFrame(table)
    df[df.id < 2, 'id2'] = df.id
    df[df.id > 2, df.name == 'name3', 'id2'] = df.id + 1

    try:
        res = df.to_pandas()
        result = get_result(res)

        expected = [
            [1, 'name1', 1],
            [2, 'name2', None],
            [3, 'name3', 4]
        ]
        assert expected == result
    except (DependencyNotInstalledError, ImportError):
        pass


def test_repeat_set_item(table):
    df = DataFrame(table)

    df['rank'] = df.groupby('name').sort('id').id.rank()
    df['rank'] = df.groupby('name').sort('id').id.rank()

    assert len(df.execute()) == 3


def test_dataframe_with_col_head(odps, table):
    test_table_name2 = tn('pyodps_test_dataframe_with_head')
    schema = TableSchema.from_lists(['id', 'head'], ['bigint', 'string'])

    odps.delete_table(test_table_name2, if_exists=True)
    loc_table = odps.create_table(test_table_name2, schema)

    with loc_table.open_writer() as w:
        w.write([[1, 'name1'], [2, 'name2'], [3, 'name3']])

    df = DataFrame(loc_table)
    df2 = DataFrame(table)
    df3 = df.join(df2, on=('head', 'name'))
    df3.head(10)


def test_fillna(odps):
    test_table_name = tn('pyodps_test_dataframe_fillna')
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(
        test_table_name,
        TableSchema.from_lists(
            ['val1', 'val2', 'val3', 'val4'], ['bigint'] * 4, ['name'], ['string']
        ),
    )
    table.create_partition('name=a')

    df = DataFrame(table.get_partition('name=a'))

    columns = df.columns[:3]
    df2 = df[columns].fillna(0, subset=columns[:2])
    df2.head()

    def sum_val(row):
        return sum(row)

    df2['new_field'] = df2.apply(sum_val, axis=1, reduce=True, rtype='int')
    df2.head()


def test_join_partition_dataframe(odps, table):
    test_table_name = tn('pyodps_test_join_partition_dataframe')
    schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds'], ['string'])
    odps.delete_table(test_table_name, if_exists=True)
    loc_table = odps.create_table(test_table_name, schema)
    loc_table.create_partition('ds=today')

    test_table_name2 = tn('pyodps_test_join_partition_dataframe2')
    odps.delete_table(test_table_name2, if_exists=True)
    table2 = odps.create_table(test_table_name2, schema)
    table2.create_partition('ds=today')

    df = DataFrame(loc_table.get_partition('ds=today'))
    df2 = DataFrame(table2.get_partition('ds=today'))
    df3 = DataFrame(table)

    df4 = df2.join(df, on=[df2.id.astype('string') == df.id.astype('string')])
    df5 = df3.join(df, on=[df3.id.astype('string') == df.id.astype('string')])
    df4.left_join(df5, on=[df4.id_y.astype('string') == df5.id_y.astype('string')]).head()
