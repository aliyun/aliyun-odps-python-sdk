# encoding: utf-8
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

import textwrap

from odps import options
from odps.df import DataFrame
from odps.runner import adapter_from_df
from odps.ml.adapter import merge_data
from odps.ml.tests.base import MLTestBase, tn

IRIS_TABLE = tn('pyodps_test_ml_iris')
TEMP_TABLE_1_NAME = tn('pyodps_test_mixin_test_table1')
TEMP_TABLE_2_NAME = tn('pyodps_test_mixin_test_table2')


def _df_roles(df):
    return dict((f.name, ','.join(r.name for r in f.role)) for f in adapter_from_df(df).fields)


def _df_continuity(df):
    return dict((f.name, f.continuity.name) for f in adapter_from_df(df).fields)


def _df_key_value(df):
    return dict((f.name, repr(f.kv_config) if f.kv_config else '') for f in adapter_from_df(df).fields)


class Test(MLTestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.create_iris(IRIS_TABLE)
        self.df = DataFrame(self.odps.get_table(IRIS_TABLE))

    def test_coll_field_operations(self):
        # select_features
        self.assertRaises(ValueError, lambda: self.df.select_features())
        df2 = self.df.select_features('sepal_length sepal_width petal_length')
        self.assertEqual(_df_roles(df2), dict(category='', sepal_width='FEATURE', sepal_length='FEATURE',
                                              petal_length='FEATURE', petal_width=''))
        df3 = df2.select_features('petal_width', add=True)
        self.assertEqual(_df_roles(df3), dict(category='', sepal_width='FEATURE', sepal_length='FEATURE',
                                              petal_length='FEATURE', petal_width='FEATURE'))
        # exclude_fields
        self.assertRaises(ValueError, lambda: self.df.exclude_fields())
        df4 = df3.exclude_fields('sepal_length sepal_width')
        self.assertEqual(_df_roles(df4), dict(category='', sepal_width='', sepal_length='',
                                              petal_length='FEATURE', petal_width='FEATURE'))
        # weight_field
        self.assertRaises(ValueError, lambda: self.df.weight_field(None))
        df5 = df3.weight_field('sepal_width')
        self.assertEqual(_df_roles(df5), dict(category='', sepal_width='WEIGHT', sepal_length='FEATURE',
                                              petal_length='FEATURE', petal_width='FEATURE'))
        # label_field
        self.assertRaises(ValueError, lambda: self.df.label_field(None))
        df6 = self.df.label_field('category')
        self.assertEqual(_df_roles(df6), dict(category='LABEL', sepal_width='FEATURE', sepal_length='FEATURE',
                                              petal_length='FEATURE', petal_width='FEATURE'))
        # roles
        self.assertIs(self.df, self.df.roles())
        df7 = self.df.roles(label='category', weight='sepal_width')
        self.assertEqual(_df_roles(df7), dict(category='LABEL', petal_length='FEATURE', petal_width='FEATURE',
                                              sepal_width='WEIGHT', sepal_length='FEATURE'))
        # discrete
        df8 = self.df.discrete('sepal_width, sepal_length')
        self.assertEqual(_df_continuity(df8), dict(category='DISCRETE', sepal_width='DISCRETE', sepal_length='DISCRETE',
                                                   petal_length='CONTINUOUS', petal_width='CONTINUOUS'))
        # continuous
        df9 = df8.continuous('sepal_width')
        self.assertEqual(_df_continuity(df9), dict(category='DISCRETE', sepal_width='CONTINUOUS', sepal_length='DISCRETE',
                                                   petal_length='CONTINUOUS', petal_width='CONTINUOUS'))
        # key_value
        df10 = self.df.key_value('sepal_length sepal_width')
        self.assertEqual(_df_key_value(df10), dict(category='', petal_length='', petal_width='',
                                                   sepal_width='KVConfig(kv=:, item=,)', sepal_length='KVConfig(kv=:, item=,)'))
        df11 = df10.key_value('sepal_length', kv='-', item=';')
        self.assertEqual(_df_key_value(df11), dict(category='', petal_length='', petal_width='',
                                                   sepal_width='KVConfig(kv=:, item=,)', sepal_length='KVConfig(kv=-, item=;)'))
        # erase_key_value
        df12 = df10.erase_key_value('sepal_width')
        self.assertEqual(_df_key_value(df12), dict(category='', petal_length='', petal_width='',
                                                   sepal_width='', sepal_length='KVConfig(kv=:, item=,)'))

    def test_seq_field_operations(self):
        seq = self.df.sepal_length
        # roles
        seq1 = seq.role('weight')
        self.assertEqual(_df_roles(seq1), dict(sepal_length='WEIGHT'))
        # discrete
        seq2 = seq.discrete()
        self.assertEqual(_df_continuity(seq2), dict(sepal_length='DISCRETE'))
        # continuous
        seq3 = seq.continuous()
        self.assertEqual(_df_continuity(seq3), dict(sepal_length='CONTINUOUS'))
        # key_value
        seq4 = seq.key_value()
        self.assertEqual(_df_key_value(seq4), dict(sepal_length='KVConfig(kv=:, item=,)'))
        seq5 = seq4.key_value(kv='-', item=';')
        self.assertEqual(_df_key_value(seq5), dict(sepal_length='KVConfig(kv=-, item=;)'))
        # erase_key_value
        seq6 = seq5.erase_key_value()
        self.assertEqual(_df_key_value(seq6), dict(sepal_length=''))

    def test_coll_df_operations(self):
        from odps.ml.nodes import transform_nodes as tnodes

        splited = self.df.split(0.75)
        self.assertEqual(len(splited), 2)
        self.assertEqual(_df_roles(splited[0]), _df_roles(splited[1]))
        split_node = adapter_from_df(splited[0])._bind_node
        self.assertEqual(split_node.code_name, 'Split')
        self.assertEqual(split_node.parameters['fraction'], 0.75)

        id_appended = self.df.append_id()
        self.assertEqual(_df_roles(id_appended), dict(category='FEATURE', petal_length='FEATURE', petal_width='FEATURE',
                                                      sepal_width='FEATURE', sepal_length='FEATURE', append_id=''))
        append_id_node = adapter_from_df(id_appended)._bind_node
        self.assertEqual(append_id_node.code_name, 'AppendID')
        self.assertEqual(append_id_node.parameters['IDColName'], 'append_id')

        summary_ep = self.df._create_summary_adapter()
        summary_node = summary_ep._bind_node
        self.assertIsInstance(summary_node, tnodes.SummaryNode)

    def test_dtypes(self):
        rstrip_lines = lambda s: '\n'.join(l.rstrip() for l in s.splitlines())
        old_dtypes_repr = rstrip_lines(textwrap.dedent("""
        odps.Schema {
          sepal_length            float64
          sepal_width             float64
          petal_length            float64
          petal_width             float64
          category                string
        }
        """)).strip()
        self.assertEqual(rstrip_lines(repr(self.df.dtypes)).strip(), old_dtypes_repr)
        new_df = self.df.roles(label='category').key_value('sepal_length')
        new_dtypes_repr = rstrip_lines(textwrap.dedent("""
        odps.Schema {
          sepal_length            KV(':', ',')   FEATURE
          sepal_width             float64        FEATURE
          petal_length            float64        FEATURE
          petal_width             float64        FEATURE
          category                string         LABEL
        }
        """)).strip()
        self.assertEqual(rstrip_lines(repr(new_df.dtypes)).strip(), new_dtypes_repr)

    def test_merge(self):
        self.odps.delete_table(TEMP_TABLE_1_NAME, if_exists=True)
        self.odps.execute_sql('create table {0} (col11 string, col12 string) lifecycle 1'.format(TEMP_TABLE_1_NAME))
        self.odps.delete_table(TEMP_TABLE_2_NAME, if_exists=True)
        self.odps.execute_sql('create table {0} (col21 string, col22 string) lifecycle 1'.format(TEMP_TABLE_2_NAME))

        df1 = DataFrame(self.odps.get_table(TEMP_TABLE_1_NAME))
        df2 = DataFrame(self.odps.get_table(TEMP_TABLE_2_NAME))

        self.assertRaises(ValueError, lambda: merge_data(df1))

        merged1 = merge_data(df1, df2)
        self.assertEqual(_df_roles(merged1), dict(col21='FEATURE', col11='FEATURE', col12='FEATURE', col22='FEATURE'))

        merged2 = merge_data((df1, 'col11'), (df2, 'col21', True))
        self.assertEqual(_df_roles(merged2), dict(col11='FEATURE', col22='FEATURE'))

        merged3 = merge_data((df1, 'col11'), (df2, 'col21', True), auto_rename=True)
        self.assertEqual(_df_roles(merged3), dict(t0_col11='FEATURE', t1_col22='FEATURE'))

        merged4 = df1.merge_with(df2)
        self.assertEqual(_df_roles(merged4), dict(col21='FEATURE', col11='FEATURE', col12='FEATURE', col22='FEATURE'))

    def test_sample(self):
        num_sampled = self.df.sample(n=20)
        adapter = adapter_from_df(num_sampled)
        self.assertIsInstance(num_sampled, DataFrame)
        self.assertEqual(adapter._bind_node.code_name, 'RandomSample')

        frac_sampled = self.df.sample(frac=0.5)
        adapter = adapter_from_df(frac_sampled)
        self.assertIsInstance(frac_sampled, DataFrame)
        self.assertEqual(adapter._bind_node.code_name, 'RandomSample')

        weighted_sampled = self.df.sample(frac=0.5, weights=self.df.sepal_length)
        adapter = adapter_from_df(weighted_sampled)
        self.assertIsInstance(weighted_sampled, DataFrame)
        self.assertEqual(adapter._bind_node.code_name, 'WeightedSample')
        self.assertEqual(adapter._bind_node.parameters['probCol'], 'sepal_length')

        stratified_sampled = self.df.sample(frac={'Iris-setosa': 0.5}, strata='category')
        adapter = adapter_from_df(stratified_sampled)
        self.assertIsInstance(stratified_sampled, DataFrame)
        self.assertEqual(adapter._bind_node.code_name, 'StratifiedSample')

    def test_batch_persist(self):
        options.runner.dry_run = False
        call_seq = []

        dfs = []
        tables = []
        for idx in range(3):
            write_str = 'F%d' % idx
            def gen_fun(wobj):
                return lambda _: call_seq.append(wobj)

            f = gen_fun((write_str, 'U'))
            df_upper = self.mock_action(self.df, action=f)
            f = gen_fun((write_str, 'D'))
            df_lower = self.mock_action(df_upper, action=f)

            dfs.append(df_lower)
            tables.append('TN' + str(idx))

        DataFrame.batch_persist(dfs, tables)

        for idx in range(3):
            write_str = 'F%d' % idx
            self.assertListEqual([p[1] for p in call_seq if p[0] == write_str], list('UD'))
        for dir in 'UD':
            self.assertListEqual(sorted(p[0] for p in call_seq if p[1] == dir), ['F0', 'F1', 'F2'])

