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

from datetime import datetime

from odps.tests.core import TestBase
from odps.compat import unittest
from odps.models import Schema
from odps import types


class Test(TestBase):
    def testPartitions(self):
        test_table_name = 'pyodps_t_tmp_partitions_table'
        partitions = ['s=%s' % i for i in range(3)]
        schema = Schema.from_lists(['id', ], ['string', ], ['s', ], ['string', ])

        self.odps.delete_table(test_table_name, if_exists=True)
        table = self.odps.create_table(test_table_name, schema)
        for partition in partitions:
            table.create_partition(partition)

        self.assertEqual(sorted([str(types.PartitionSpec(p)) for p in partitions]),
                         sorted([str(p.partition_spec) for p in table.partitions]))

        table.get_partition(partitions[0]).drop()
        self.assertEqual(sorted([str(types.PartitionSpec(p)) for p in partitions[1:]]),
                         sorted([str(p.partition_spec) for p in table.partitions]))

        self.odps.delete_table(test_table_name)

    def testSubPartitions(self):
        test_table_name = 'pyodps_t_tmp_sub_partitions_table'
        root_partition = 'type=test'
        sub_partitions = ['s=%s' % i for i in range(3)]
        schema = Schema.from_lists(['id', ], ['string', ], ['type', 's'], ['string', 'string'])

        self.odps.delete_table(test_table_name, if_exists=True)
        table = self.odps.create_table(test_table_name, schema)
        partitions = [root_partition+','+p for p in sub_partitions]
        partitions.append('type=test2,s=0')
        for partition in partitions:
            table.create_partition(partition)

        self.assertEqual(sorted([str(types.PartitionSpec(p)) for p in partitions]),
                         sorted([str(p.partition_spec) for p in table.partitions]))

        self.assertEqual(len(list(table.iterate_partitions(root_partition))), 3)

        table.delete_partition(partitions[0])
        self.assertEqual(sorted([str(types.PartitionSpec(p)) for p in partitions[1:]]),
                         sorted([str(p.partition_spec) for p in table.partitions]))

        self.odps.delete_table(test_table_name)

    def testPartition(self):
        test_table_name = 'pyodps_t_tmp_partition_table'
        partition = 's=1'
        schema = Schema.from_lists(['id', ], ['string', ], ['s', ], ['string', ])

        self.odps.delete_table(test_table_name, if_exists=True)
        table = self.odps.create_table(test_table_name, schema)
        partition = table.create_partition(partition)

        self.assertFalse(partition._getattr('_is_extend_info_loaded'))
        self.assertFalse(partition._getattr('_loaded'))

        self.assertIsNone(partition._getattr('creation_time'))
        self.assertIsNone(partition._getattr('last_meta_modified_time'))
        self.assertIsNone(partition._getattr('last_modified_time'))
        self.assertIsNone(partition._getattr('size'))
        self.assertIsNone(partition._getattr('is_archived'))
        self.assertIsNone(partition._getattr('is_exstore'))
        self.assertIsNone(partition._getattr('lifecycle'))
        self.assertIsNone(partition._getattr('physical_size'))
        self.assertIsNone(partition._getattr('file_num'))

        self.assertIsInstance(partition.is_archived, bool)
        self.assertIsInstance(partition.is_exstore, bool)
        self.assertIsInstance(partition.lifecycle, int)
        self.assertIsInstance(partition.physical_size, int)
        self.assertIsInstance(partition.file_num, int)
        self.assertIsInstance(partition.creation_time, datetime)
        self.assertIsInstance(partition.last_meta_modified_time, datetime)
        self.assertIsInstance(partition.last_modified_time, datetime)
        self.assertIsInstance(partition.size, int)

        self.assertTrue(partition._is_extend_info_loaded)
        self.assertTrue(partition.is_loaded)

        self.assertTrue(table.exist_partition(partition))
        self.assertFalse(table.exist_partition('s=a_non_exist_partition'))

        self.odps.delete_table(test_table_name)
        self.assertFalse(table.exist_partition(partition))

if __name__ == '__main__':
    unittest.main()