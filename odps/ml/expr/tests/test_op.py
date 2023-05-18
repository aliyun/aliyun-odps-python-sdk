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

import functools

import pytest

from ....df.expr.expressions import CollectionExpr
from ....df.expr.tests.core import MockTable
from ....df.types import validate_data_type
from ....models.table import TableSchema
from ..op import *
from ...tests.base import MLTestUtil, tn
from ...utils import KVConfig

TEMP_TABLE_1_NAME = tn('pyodps_test_ops_test_table1')
TEMP_TABLE_2_NAME = tn('pyodps_test_ops_test_table2')

datatypes = lambda *types: [validate_data_type(t) for t in types]


def assertFieldsEqual(df1, df2, func=repr):
    def repr_fields(fields):
        if isinstance(fields, CollectionExpr):
            fields = fields._ml_fields
        if len(fields) == 0:
            return []
        if isinstance(fields[0], MLField):
            return [func(f) for f in fields]
        else:
            return fields

    assert repr_fields(df1) == repr_fields(df2)


def get_table1_df():
    schema = TableSchema.from_lists(['col11', 'col12'], datatypes('string', 'string'))
    table = MockTable(name=TEMP_TABLE_1_NAME, table_schema=schema)
    return CollectionExpr(_source_data=table, _schema=schema)


def get_table2_df():
    schema = TableSchema.from_lists(['col21', 'col22'], datatypes('string', 'string'))
    table = MockTable(name=TEMP_TABLE_2_NAME, table_schema=schema)
    return CollectionExpr(_source_data=table, _schema=schema)


def exec_op(op, dfs, target):
    op.execute(dfs, target)


@pytest.fixture
def utils(odps, tunnel):
    util = MLTestUtil(odps, tunnel)
    return util


def test_base_methods(utils):
    fields = [MLField('f%02d' % fid, 'string', FieldRole.FEATURE) for fid in range(5)]
    fields_set_singleton = list(DFOperation._set_singleton_role(fields, {'f00': FieldRole.WEIGHT}))
    assert fields_set_singleton[0].role == set([FieldRole.FEATURE, FieldRole.WEIGHT])

    fields_set_singleton2 = list(DFOperation._set_singleton_role(fields_set_singleton, {'f01': FieldRole.WEIGHT}))
    assert fields_set_singleton2[0].role == set([FieldRole.FEATURE, ])
    assert fields_set_singleton2[1].role == set([FieldRole.FEATURE, FieldRole.WEIGHT])

    fields_set_singleton_expect = list(DFOperation._set_singleton_role(fields_set_singleton2,
                                                                       {'category': FieldRole.LABEL}))
    assert fields_set_singleton_expect[-1].role == set([FieldRole.LABEL, ])
    assert fields_set_singleton_expect[-1].name == 'category'
    assert fields_set_singleton_expect[-1].type == 'expected'

    fields_remove_role = list(DFOperation._remove_field_roles(fields_set_singleton2, ['f01', ], FieldRole.WEIGHT))
    assert fields_remove_role[1].role == set([FieldRole.FEATURE, ])
    fields_remove_role = list(DFOperation._remove_field_roles(fields_set_singleton2, 'f01', FieldRole.WEIGHT))
    assert fields_remove_role[1].role == set([FieldRole.FEATURE, ])

    fields_clear_role = list(DFOperation._clear_field_roles(fields_set_singleton, 'f00'))
    assert fields_clear_role[0].role == set()
    fields_clear_role = list(DFOperation._clear_field_roles(fields_set_singleton, 'category'))
    assert fields_clear_role[-1].role == set()
    assert fields_set_singleton_expect[-1].name == 'category'
    assert fields_set_singleton_expect[-1].type == 'expected'

    fields_add_role = list(DFOperation._add_field_roles(fields_set_singleton, 'f01', FieldRole.WEIGHT, True))
    assert fields_add_role[0].role == set([FieldRole.WEIGHT, FieldRole.FEATURE])
    assert fields_add_role[1].role == set([FieldRole.WEIGHT, FieldRole.FEATURE])
    fields_add_role2 = list(DFOperation._add_field_roles(fields_add_role, 'f01,f02', FieldRole.WEIGHT, False))
    assert fields_add_role2[0].role == set([FieldRole.FEATURE, ])
    assert fields_add_role2[1].role == set([FieldRole.WEIGHT, FieldRole.FEATURE])
    assert fields_add_role2[2].role == set([FieldRole.WEIGHT, FieldRole.FEATURE])
    fields_add_role3 = list(DFOperation._add_field_roles(fields_add_role, 'category', FieldRole.LABEL, True))
    assert fields_add_role3[-1].role == set([FieldRole.LABEL, ])
    assert fields_add_role3[-1].name == 'category'
    assert fields_add_role3[-1].type == 'expected'


def test_base_copy_operation(utils):
    ds1 = get_table1_df()
    target = utils.mock_action(ds1)
    exec_op(DFOperation(), [ds1, ], target)
    assertFieldsEqual(ds1, target)


def test_operations(utils):
    df1 = get_table1_df()
    src_fields1 = copy.deepcopy([f for f in df1._ml_fields])
    df2 = get_table2_df()
    src_fields2 = copy.deepcopy([f for f in df2._ml_fields])

    target = utils.mock_action(df1)
    exec_op(BatchRoleOperation(['col11', 'col12'], FieldRole.WEIGHT, True), [df1, ], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(target, [set([FieldRole.FEATURE, FieldRole.WEIGHT]), ] * 2, lambda f: f.role)

    target = utils.mock_action(df1)
    exec_op(ExcludeFieldsOperation(['col12', ]), [df1, ], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(target, [set([FieldRole.FEATURE, ]), set()], lambda f: f.role)

    target = utils.mock_action(df1)
    exec_op(SingletonRoleOperation({'col11': FieldRole.WEIGHT, 'col12': FieldRole.LABEL}), [df1, ], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(target,
                           [set([FieldRole.FEATURE, FieldRole.WEIGHT]), set([FieldRole.FEATURE, FieldRole.LABEL])],
                           lambda f: f.role)

    target = utils.mock_action(df1)
    exec_op(FieldContinuityOperation(dict(col11=True, col12=False, col13=True)), [df1, ], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(target, [FieldContinuity.CONTINUOUS, FieldContinuity.DISCRETE, FieldContinuity.CONTINUOUS],
                           lambda f: f.continuity)
    assert target._ml_fields[-1].name == 'col13'
    assert target._ml_fields[-1].type == 'expected'

    target = utils.mock_action(df1)
    kv_config_vals = [KVConfig(':', ','), KVConfig('_', '+'), KVConfig('*', '%')]
    kv_config = dict(zip(['col11', 'col12', 'col13'], kv_config_vals))
    exec_op(FieldKVConfigOperation(kv_config), [df1, ], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(target, kv_config_vals, lambda f: f.kv_config)
    assert target._ml_fields[-1].name == 'col13'
    assert target._ml_fields[-1].type == 'expected'

    target = utils.mock_action(df1)
    exec_op(StaticFieldChangeOperation([MLField('col13', 'bigint', FieldRole.FEATURE),
                                             MLField('col14', 'bigint', FieldRole.FEATURE)]), [df1, ], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(target, ['col13', 'col14'], lambda f: f.name)

    target = utils.mock_action(df1)
    exec_op(StaticFieldChangeOperation([MLField('col13', 'bigint', FieldRole.FEATURE),
                                             MLField('col14', 'bigint', FieldRole.FEATURE)],
                                            is_append=True), [df1, ], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(target, ['col11', 'col12', 'col13', 'col14'], lambda f: f.name)

    def test_generator(params, fields):
        assert params == dict(message='TestMsg')
        assertFieldsEqual(df1, fields[0])
        return 'field1:string:label,field2:bigint'

    target = utils.mock_action(df1, msg='TestMsg')
    exec_op(ProgrammaticFieldChangeOperation(
        functools.partial(test_generator, target._params, {0: df1._ml_fields}),
        is_append=False), [df1, ], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(target, [
        MLField('field1', 'string', FieldRole.LABEL, FieldContinuity.DISCRETE),
        MLField('field2', 'bigint', FieldRole.FEATURE, FieldContinuity.CONTINUOUS),
    ])

    target = utils.mock_action(df1, msg='TestMsg')
    exec_op(ProgrammaticFieldChangeOperation(
        functools.partial(test_generator, target._params, {0: df1._ml_fields}),
        is_append=True), [df1, ], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(target, df1._ml_fields + [
        MLField('field1', 'string', FieldRole.LABEL, FieldContinuity.DISCRETE),
        MLField('field2', 'bigint', FieldRole.FEATURE, FieldContinuity.CONTINUOUS),
    ])

    sel_cols = {0: [f.name for f in df1._ml_fields], 1: [df2._ml_fields[0].name, ]}
    exc_cols = {0: [], 1: [df2._ml_fields[1].name, ]}
    target = utils.mock_action([df1, df2])
    exec_op(MergeFieldsOperation(False, sel_cols, exc_cols), [df1, df2], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(df2, src_fields2)
    assertFieldsEqual(target, df1._ml_fields + [df2._ml_fields[0], ])

    target = utils.mock_action([df1, df2])
    new_table_names = ['t0_%s' % f.name for f in df1._ml_fields] + ['t1_%s' % df2._ml_fields[0].name, ]
    exec_op(MergeFieldsOperation(True, sel_cols, exc_cols), [df1, df2], target)
    assertFieldsEqual(df1, src_fields1)
    assertFieldsEqual(df2, src_fields2)
    assertFieldsEqual(target, new_table_names, lambda f: f.name)


def test_rename_operation(utils):
    df = get_table1_df()
    labeled_df = df.roles(label='col12')

    df2 = labeled_df[labeled_df.col11, labeled_df.col12.rename('col12_renamed')]
    df2_sel = df2[df2.col12_renamed, ]
    assertFieldsEqual(df2_sel, [
        MLField('col12_renamed', 'string', FieldRole.LABEL, FieldContinuity.DISCRETE),
    ])
