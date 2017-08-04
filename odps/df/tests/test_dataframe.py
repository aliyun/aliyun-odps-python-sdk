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

from datetime import datetime
from decimal import Decimal

from odps.tests.core import TestBase, tn, pandas_case, global_locked
from odps.df.backends.context import context
from odps.compat import unittest
from odps.models import Schema
from odps.df import DataFrame
from odps.utils import to_text
from odps.errors import DependencyNotInstalledError


class Test(TestBase):

    def setup(self):
        test_table_name = tn('pyodps_test_dataframe')
        schema = Schema.from_lists(['id', 'name'], ['bigint', 'string'])

        self.odps.delete_table(test_table_name, if_exists=True)
        self.table = self.odps.create_table(test_table_name, schema)

        with self.table.open_writer() as w:
            w.write([[1, 'name1'], [2, 'name2'], [3, 'name3']])

    def teardown(self):
        self.table.drop()

    def testDataFrame(self):
        df = DataFrame(self.table)

        self.assertEqual(3, df.count().execute())
        self.assertEqual(1, df[df.name == 'name1'].count().execute())

        res = df[df.name.contains('中文')].execute()
        self.assertGreaterEqual(len(res), 0)

    @pandas_case
    def testDataFrameFromPandas(self):
        import pandas as pd

        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [None, None, None]})

        self.assertRaises(TypeError, lambda: DataFrame(pd_df))

        df = DataFrame(pd_df, unknown_as_string=True)
        self.assertEqual(df.schema.get_type('b').name, 'string')

        df = DataFrame(pd_df[['a']], as_type={'a': 'string'})
        self.assertEqual(df.schema.get_type('a').name, 'string')

        df = DataFrame(pd_df, as_type={'b': 'int'})
        self.assertEqual(df.schema.get_type('b').name, 'int64')

    def testHeadAndTail(self):
        df = DataFrame(self.table)

        self.assertEqual(1, len(df.head(1)))
        self.assertEqual(2, len(df.head(2)))
        self.assertEqual([3, 'name3'], list(df.tail(1)[0]))

        r = df[df.name == 'name2'].head(1)
        self.assertEqual(1, len(r))
        self.assertEqual([2, 'name2'], list(r[0]))

    @global_locked('odps_instance_method_repr')
    def testInstanceMethodRepr(self):
        from odps import options

        class CannotExecuteDataFrame(DataFrame):
            def execute(self, **kwargs):
                raise RuntimeError('DataFrame cannot be executed')

        options.interactive = True

        try:
            df = CannotExecuteDataFrame(self.table)

            self.assertEqual(repr(df.count), '<bound method Collection.count>')
            self.assertEqual(repr(df.name.count), '<bound method Column._count>')
        finally:
            options.interactive = False

    @pandas_case
    def testToPandas(self):
        table_name = tn('pyodps_test_mixed_engine_to_pandas')
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name,
                                        schema=Schema.from_lists(['col%s' % i for i in range(7)],
                                                                 ['bigint', 'double', 'string', 'datetime',
                                                                  'boolean', 'decimal', 'datetime']))
        expr2 = DataFrame(table2)

        data2 = [
            [1234567, 3.14, 'test', datetime(2016, 6, 1), True, Decimal('3.14'), None]
        ]
        self.odps.write_table(table2, 0, data2)

        pd_df = expr2.to_pandas()
        self.assertSequenceEqual(data2[0], pd_df.ix[0].tolist())

        wrapeed_pd_df = expr2.to_pandas(wrap=True)
        self.assertSequenceEqual(data2[0], list(next(wrapeed_pd_df.execute())))

    @pandas_case
    def testUnicodePdDataFrame(self):
        import pandas as pd

        pd_df = pd.DataFrame([['中文'], [to_text('中文2')]], columns=[to_text('字段')])
        df = DataFrame(pd_df)

        r = df['字段'].execute()
        self.assertEqual(to_text('中文'), to_text(r[0][0]))
        self.assertEqual(to_text('中文2'), to_text(r[1][0]))

    @pandas_case
    def testPandasGroupbyFilter(self):
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

        self.assertEqual(df3.execute().values.values.tolist(), expected)

        df2 = df.groupby('id').agg(df.fid.sum())
        df2.execute()
        self.assertTrue(context.is_cached(df2))
        df3 = df2[df2.id == 2003]

        self.assertEqual(df3.execute().values.values.tolist(), expected)
        self.assertEqual(df3.execute().values.values.tolist(), expected)

        df4 = df.fid.sum()
        self.assertEqual(df4.execute(), 6)
        self.assertEqual(df4.execute(), 6)

    def testCreateDataFrameFromPartition(self):
        from odps.types import PartitionSpec
        test_table_name = tn('pyodps_test_dataframe_partition')
        schema = Schema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds'], ['string'])

        self.odps.delete_table(test_table_name, if_exists=True)
        table = self.odps.create_table(test_table_name, schema)

        with table.open_writer('ds=today', create_partition=True) as w:
            w.write([[1, 'name1'], [2, 'name2'], [3, 'name3']])

        try:
            df = DataFrame(table.get_partition('ds=today'))
            self.assertEqual(df.count().execute(), 3)

            df = table.get_partition('ds=today').to_df()
            partition = df.data
            self.assertIs(partition.table, table)
            self.assertEqual(partition.partition_spec, PartitionSpec('ds=today'))
            self.assertEqual(df.count().execute(), 3)
        finally:
            table.drop()

    def testSetItem(self):
        df = DataFrame(self.table)
        df['id2'] = df.id + 1
        self.assertEqual(len(df.execute()), 3)

        df['id3'] = df['id2'] * 2
        self.assertEqual(len(next(df.execute())), 4)

        del df['id2']
        res = df.execute()
        result = self._get_result(res)

        expected = [
            [1, 'name1', 4],
            [2, 'name2', 6],
            [3, 'name3', 8]
        ]
        self.assertEqual(expected, result)

        df = DataFrame(self.table)
        df['id2'] = df.id

        try:
            res = df.to_pandas()
            result = self._get_result(res)

            expected = [
                [1, 'name1', 1],
                [2, 'name2', 2],
                [3, 'name3', 3]
            ]
            self.assertEqual(expected, result)
        except (DependencyNotInstalledError, ImportError):
            pass

        df = DataFrame(self.table)
        df[df.id <= 2, 'id2'] = df.id

        try:
            res = df.to_pandas()
            result = self._get_result(res)

            expected = [
                [1, 'name1', 1],
                [2, 'name2', 2],
                [3, 'name3', None]
            ]
            self.assertEqual(expected, result)
        except (DependencyNotInstalledError, ImportError):
            pass

        df = DataFrame(self.table)
        df[df.id < 2, 'id2'] = df.id
        df[df.id > 2, df.name == 'name3', 'id2'] = df.id + 1

        try:
            res = df.to_pandas()
            result = self._get_result(res)

            expected = [
                [1, 'name1', 1],
                [2, 'name2', None],
                [3, 'name3', 4]
            ]
            self.assertEqual(expected, result)
        except (DependencyNotInstalledError, ImportError):
            pass

    def testRepeatSetItem(self):
        df = DataFrame(self.table)

        df['rank'] = df.groupby('name').sort('id').id.rank()
        df['rank'] = df.groupby('name').sort('id').id.rank()

        self.assertEqual(len(df.execute()), 3)

    def testDataFrameWithColHead(self):
        test_table_name2 = tn('pyodps_test_dataframe_with_head')
        schema = Schema.from_lists(['id', 'head'], ['bigint', 'string'])

        self.odps.delete_table(test_table_name2, if_exists=True)
        table = self.odps.create_table(test_table_name2, schema)

        with table.open_writer() as w:
            w.write([[1, 'name1'], [2, 'name2'], [3, 'name3']])

        df = DataFrame(table)
        df2 = DataFrame(self.table)
        df3 = df.join(df2, on=('head', 'name'))
        df3.head(10)

if __name__ == '__main__':
    unittest.main()
