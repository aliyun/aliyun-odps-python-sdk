# encoding: utf-8
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

from __future__ import print_function

from odps.df import DataFrame
from odps.df.backends.odpssql.types import df_schema_to_odps_schema
from odps.tests.core import tn
from odps.ml.tests.base import MLTestBase
from odps.models import Partition, Schema
from odps import types as odps_types

IRIS_TABLE = tn('pyodps_test_ml_iris')
SIMPLE_PERSIST_TABLE = tn('pyodps_test_ml_simple_persist_table')
STATIC_PART_TABLE = tn('pyodps_test_ml_static_part_table')
DYNAMIC_PART_TABLE = tn('pyodps_test_ml_dynamic_part_table')


class Test(MLTestBase):
    def testSimplePersist(self):
        self.create_iris(IRIS_TABLE)
        df = DataFrame(self.odps.get_table(IRIS_TABLE))
        df.append_id().persist(SIMPLE_PERSIST_TABLE, lifecycle=1, drop_table=True)
        self.assertTrue(self.odps.exist_table(SIMPLE_PERSIST_TABLE))

    def testStaticPartition(self):
        self.create_iris(IRIS_TABLE)
        df = DataFrame(self.odps.get_table(IRIS_TABLE))
        id_df = df.append_id()

        src_schema = df_schema_to_odps_schema(id_df.schema)
        schema = Schema(columns=src_schema.simple_columns,
                        partitions=[Partition(name='ds', type=odps_types.string)])
        self.odps.delete_table(STATIC_PART_TABLE, if_exists=True)
        dest_table = self.odps.create_table(STATIC_PART_TABLE, schema, lifecycle=1)

        id_df.persist(STATIC_PART_TABLE, partition='ds=20170314', lifecycle=1)
        self.assertTrue(dest_table.exist_partition('ds=20170314'))

    def testDynamicPartition(self):
        self.create_iris(IRIS_TABLE)
        df = DataFrame(self.odps.get_table(IRIS_TABLE))
        id_df = df.append_id()
        self.odps.delete_table(DYNAMIC_PART_TABLE, if_exists=True)
        id_df.persist(DYNAMIC_PART_TABLE, partitions='category', lifecycle=1, drop_table=True)

        self.assertTrue(self.odps.exist_table(DYNAMIC_PART_TABLE))
        t = self.odps.get_table(DYNAMIC_PART_TABLE)
        self.assertIn('category', [pt.name for pt in t.schema.partitions])
