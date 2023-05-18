# encoding: utf-8
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

import textwrap

import pytest

from .... import options
from ....df import DataFrame
from ...tests.base import MLTestUtil, tn

IRIS_TABLE = tn('pyodps_test_ml_iris')
TEMP_TABLE_1_NAME = tn('pyodps_test_mixin_test_table1')
TEMP_TABLE_2_NAME = tn('pyodps_test_mixin_test_table2')


def _df_roles(df):
    return dict((f.name, ','.join(r.name for r in f.role)) for f in df._ml_fields)


def _df_continuity(df):
    return dict((f.name, f.continuity.name) for f in df._ml_fields)


def _df_key_value(df):
    return dict((f.name, repr(f.kv_config) if f.kv_config else '') for f in df._ml_fields)


@pytest.fixture
def utils(odps, tunnel):
    util = MLTestUtil(odps, tunnel)
    util.create_iris(IRIS_TABLE)
    util.df = DataFrame(odps.get_table(IRIS_TABLE))
    return util


def test_collection_labelling(utils):
    # select_features
    pytest.raises(ValueError, lambda: utils.df.select_features())
    df2 = utils.df.select_features('sepal_length sepal_width petal_length')
    assert _df_roles(df2) == dict(category='', sepal_width='FEATURE', sepal_length='FEATURE',
                                          petal_length='FEATURE', petal_width='')
    df3 = df2.select_features('petal_width', add=True)
    assert _df_roles(df3) == dict(category='', sepal_width='FEATURE', sepal_length='FEATURE',
                                          petal_length='FEATURE', petal_width='FEATURE')
    # exclude_fields
    pytest.raises(ValueError, lambda: utils.df.exclude_fields())
    df4 = df3.exclude_fields('sepal_length sepal_width')
    assert _df_roles(df4) == dict(category='', sepal_width='', sepal_length='',
                                          petal_length='FEATURE', petal_width='FEATURE')
    # weight_field
    pytest.raises(ValueError, lambda: utils.df.weight_field(None))
    df5 = df3.weight_field('sepal_width')
    assert _df_roles(df5) == dict(category='', sepal_width='WEIGHT', sepal_length='FEATURE',
                                          petal_length='FEATURE', petal_width='FEATURE')
    # label_field
    pytest.raises(ValueError, lambda: utils.df.label_field(None))
    df6 = utils.df.label_field('category')
    assert _df_roles(df6) == dict(category='LABEL', sepal_width='FEATURE', sepal_length='FEATURE',
                                          petal_length='FEATURE', petal_width='FEATURE')
    # roles
    assert utils.df is utils.df.roles()
    df7 = utils.df.roles(label='category', weight='sepal_width')
    assert _df_roles(df7) == dict(category='LABEL', petal_length='FEATURE', petal_width='FEATURE',
                                          sepal_width='WEIGHT', sepal_length='FEATURE')
    # discrete
    df8 = utils.df.discrete('sepal_width, sepal_length')
    assert _df_continuity(df8) == dict(category='DISCRETE', sepal_width='DISCRETE', sepal_length='DISCRETE',
                                               petal_length='CONTINUOUS', petal_width='CONTINUOUS')
    # continuous
    df9 = df8.continuous('sepal_width')
    assert _df_continuity(df9) == dict(category='DISCRETE', sepal_width='CONTINUOUS', sepal_length='DISCRETE',
                          petal_length='CONTINUOUS', petal_width='CONTINUOUS')
    # key_value
    df10 = utils.df.key_value('sepal_length sepal_width')
    assert _df_key_value(df10) == dict(category='', petal_length='', petal_width='',
                                               sepal_width='KVConfig(kv=:, item=,)',
                                               sepal_length='KVConfig(kv=:, item=,)')
    df11 = df10.key_value('sepal_length', kv='-', item=';')
    assert _df_key_value(df11) == dict(category='', petal_length='', petal_width='',
                                               sepal_width='KVConfig(kv=:, item=,)',
                                               sepal_length='KVConfig(kv=-, item=;)')
    # erase_key_value
    df12 = df10.erase_key_value('sepal_width')
    assert _df_key_value(df12) == dict(category='', petal_length='', petal_width='',
                                               sepal_width='', sepal_length='KVConfig(kv=:, item=,)')


def test_seq_field_operations(utils):
    seq = utils.df.sepal_length
    # roles
    seq1 = seq.role('weight')
    assert _df_roles(seq1) == dict(sepal_length='WEIGHT')
    # discrete
    seq2 = seq.discrete()
    assert _df_continuity(seq2) == dict(sepal_length='DISCRETE')
    # continuous
    seq3 = seq.continuous()
    assert _df_continuity(seq3) == dict(sepal_length='CONTINUOUS')
    # key_value
    seq4 = seq.key_value()
    assert _df_key_value(seq4) == dict(sepal_length='KVConfig(kv=:, item=,)')
    seq5 = seq4.key_value(kv='-', item=';')
    assert _df_key_value(seq5) == dict(sepal_length='KVConfig(kv=-, item=;)')
    # erase_key_value
    seq6 = seq5.erase_key_value()
    assert _df_key_value(seq6) == dict(sepal_length='')


def test_collection_operations(utils):
    splited = utils.df.split(0.75)
    assert len(splited) == 2
    assert _df_roles(splited[0]) == _df_roles(splited[1])
    assert splited[0]._algo == 'Split'
    assert splited[0]._params['fraction'] == 0.75

    id_appended = utils.df.append_id()
    assert _df_roles(id_appended) == dict(category='FEATURE', petal_length='FEATURE', petal_width='FEATURE',
                                                  sepal_width='FEATURE', sepal_length='FEATURE', append_id='')
    assert id_appended._algo == 'AppendID'
    assert id_appended._params['IDColName'] == 'append_id'


def test_d_types(utils):
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
    assert rstrip_lines(repr(utils.df.dtypes)).strip() == old_dtypes_repr
    new_df = utils.df.roles(label='category').key_value('sepal_length')
    new_dtypes_repr = rstrip_lines(textwrap.dedent("""
    odps.Schema {
      sepal_length            KV(':', ',')   FEATURE
      sepal_width             float64        FEATURE
      petal_length            float64        FEATURE
      petal_width             float64        FEATURE
      category                string         LABEL
    }
    """)).strip()
    assert rstrip_lines(repr(new_df.dtypes)).strip() == new_dtypes_repr


def test_merge(odps, utils):
    from ..mixin import merge_data

    odps.delete_table(TEMP_TABLE_1_NAME, if_exists=True)
    odps.execute_sql('create table {0} (col11 string, col12 string) lifecycle 1'.format(TEMP_TABLE_1_NAME))
    odps.delete_table(TEMP_TABLE_2_NAME, if_exists=True)
    odps.execute_sql('create table {0} (col21 string, col22 string) lifecycle 1'.format(TEMP_TABLE_2_NAME))

    df1 = DataFrame(odps.get_table(TEMP_TABLE_1_NAME))
    df2 = DataFrame(odps.get_table(TEMP_TABLE_2_NAME))

    pytest.raises(ValueError, lambda: merge_data(df1))

    merged1 = merge_data(df1, df2)
    assert _df_roles(merged1) == dict(col21='FEATURE', col11='FEATURE', col12='FEATURE', col22='FEATURE')

    merged2 = merge_data((df1, 'col11'), (df2, 'col21', True))
    assert _df_roles(merged2) == dict(col11='FEATURE', col22='FEATURE')

    merged3 = merge_data((df1, 'col11'), (df2, 'col21', True), auto_rename=True)
    assert _df_roles(merged3) == dict(t0_col11='FEATURE', t1_col22='FEATURE')

    merged4 = df1.merge_with(df2)
    assert _df_roles(merged4) == dict(col21='FEATURE', col11='FEATURE', col12='FEATURE', col22='FEATURE')

    options.ml.dry_run = True
    merged4._add_case(utils.gen_check_params_case({
        'autoRenameCol': 'False',
        'outputTableName': 'merged_table',
        'inputTableNames': odps.project + '.' + TEMP_TABLE_1_NAME + ',' + odps.project + '.' + TEMP_TABLE_2_NAME,
        'selectedColNamesList': 'col11,col12;col21,col22'}
    ))
    merged4.persist('merged_table')


def test_sample_class(odps, utils):
    from ..core import AlgoExprMixin
    num_sampled = utils.df.sample(n=20)
    assert isinstance(num_sampled, AlgoExprMixin)
    assert num_sampled._algo == 'RandomSample'

    frac_sampled = utils.df.sample(frac=0.5)
    assert isinstance(frac_sampled, AlgoExprMixin)
    assert frac_sampled._algo == 'RandomSample'

    weighted_sampled = utils.df.sample(frac=0.5, weights=utils.df.sepal_length)
    assert isinstance(weighted_sampled, AlgoExprMixin)
    assert weighted_sampled._algo == 'WeightedSample'
    assert weighted_sampled._params['probCol'] == 'sepal_length'

    stratified_sampled = utils.df.sample(frac={'Iris-setosa': 0.5}, strata='category')
    assert isinstance(stratified_sampled, AlgoExprMixin)
    assert stratified_sampled._algo == 'StratifiedSample'
