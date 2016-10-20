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

from datetime import datetime
from decimal import Decimal

from odps.tests.core import TestBase, tn, pandas_case, global_locked
from odps.compat import unittest
from odps.models import Schema
from odps.df import DataFrame
from odps.utils import to_text


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
        self.assertEqual(1, df[df.name == 'name1'].count())

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
        self.assertIsNotNone(df2._cache_data)
        df3 = df2[df2.id == 2003]

        self.assertEqual(df3.execute().values.values.tolist(), expected)
        self.assertEqual(df3.execute().values.values.tolist(), expected)

        df4 = df.fid.sum()
        self.assertEqual(df4.execute(), 6)
        self.assertEqual(df4.execute(), 6)

if __name__ == '__main__':
    unittest.main()
