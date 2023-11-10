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

from datetime import datetime

import mock
import pytest

from ...tests.core import tn
from ... import types
from .. import TableSchema
from ..storage_tier import StorageTier


def test_partitions(odps):
    test_table_name = tn('pyodps_t_tmp_partitions_table')
    partitions = ['s=%s' % i for i in range(3)]
    schema = TableSchema.from_lists(['id', ], ['string', ], ['s', ], ['string', ])

    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema)
    for partition in partitions:
        table.create_partition(partition)

    assert (
        sorted([str(types.PartitionSpec(p)) for p in partitions])
        == sorted([str(p.partition_spec) for p in table.partitions])
    )

    table.get_partition(partitions[0]).drop()
    assert (
        sorted([str(types.PartitionSpec(p)) for p in partitions[1:]])
        == sorted([str(p.partition_spec) for p in table.partitions])
    )

    p = next(table.partitions)
    assert len(p.columns) > 0
    p.reload()
    assert len(p.columns) > 0

    assert len(list(p.head(5))) == 0

    p.truncate()

    odps.delete_table(test_table_name)


def test_sub_partitions(odps):
    test_table_name = tn('pyodps_t_tmp_sub_partitions_table')
    root_partition = 'type=test'
    sub_partitions = ['s=%s' % i for i in range(3)]
    schema = TableSchema.from_lists(['id', ], ['string', ], ['type', 's'], ['string', 'string'])

    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema)
    partitions = [root_partition+','+p for p in sub_partitions]
    partitions.append('type=test2,s=0')
    for partition in partitions:
        table.create_partition(partition)

    assert sorted([str(types.PartitionSpec(p)) for p in partitions]) == sorted([str(p.partition_spec) for p in table.partitions])

    assert len(list(table.iterate_partitions(root_partition))) == 3
    assert table.exist_partitions('type=test2') is True
    assert table.exist_partitions('type=test3') is False

    table.delete_partition(partitions[0])
    assert sorted([str(types.PartitionSpec(p)) for p in partitions[1:]]) == sorted([str(p.partition_spec) for p in table.partitions])

    odps.delete_table(test_table_name)


def test_partition(odps):
    test_table_name = tn('pyodps_t_tmp_partition_table')
    partition = 's=1'
    schema = TableSchema.from_lists(['id', ], ['string', ], ['s', ], ['string', ])

    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema)
    partition = table.create_partition(partition)

    assert partition._getattr('_is_extend_info_loaded') is False
    assert partition._getattr('_loaded') is False

    assert partition._getattr('creation_time') is None
    assert partition._getattr('last_meta_modified_time') is None
    assert partition._getattr('last_data_modified_time') is None
    assert partition._getattr('size') is None
    assert partition._getattr('is_archived') is None
    assert partition._getattr('is_exstore') is None
    assert partition._getattr('lifecycle') is None
    assert partition._getattr('physical_size') is None
    assert partition._getattr('file_num') is None

    assert isinstance(partition.is_archived, bool)
    assert isinstance(partition.is_exstore, bool)
    assert isinstance(partition.lifecycle, int)
    assert isinstance(partition.physical_size, int)
    assert isinstance(partition.file_num, int)
    assert isinstance(partition.creation_time, datetime)
    assert isinstance(partition.last_meta_modified_time, datetime)
    assert isinstance(partition.last_data_modified_time, datetime)
    with pytest.deprecated_call():
        assert isinstance(partition.last_modified_time, datetime)
    assert isinstance(partition.size, int)

    assert partition._is_extend_info_loaded is True
    assert partition.is_loaded is True

    assert table.exist_partition(partition) is True
    assert table.exist_partition('s=a_non_exist_partition') is False

    row_contents = ['index', '1']
    with partition.open_writer() as writer:
        writer.write([row_contents])
    with partition.open_reader() as reader:
        for rec in reader:
            assert row_contents == list(rec.values)

    odps.delete_table(test_table_name)
    assert table.exist_partition(partition) is False


def test_iter_partition_condition(odps):
    from ...types import PartitionSpec
    from ..partitions import PartitionSpecCondition

    test_table_name = tn('pyodps_t_tmp_cond_partition_table')
    odps.delete_table(test_table_name, if_exists=True)
    tb = odps.create_table(test_table_name, ("col string", "pt1 string, pt2 string"))

    tb.create_partition("pt1=1,pt2=1")
    tb.create_partition("pt1=1,pt2=2")
    tb.create_partition("pt1=2,pt2=1")
    tb.create_partition("pt1=2,pt2=2")

    with pytest.raises(ValueError):
        list(tb.iterate_partitions("pt3=1"))
    with pytest.raises(ValueError):
        list(tb.iterate_partitions("pt1"))
    with pytest.raises(ValueError):
        list(tb.iterate_partitions("pt1~1"))

    orig_init = PartitionSpecCondition.__init__
    part_prefix = [None]

    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        part_prefix[0] = self.partition_spec

    with mock.patch("odps.models.partitions.PartitionSpecCondition.__init__", new=new_init):
        # filter with predicates
        parts = list(tb.iterate_partitions("pt1=1"))
        assert part_prefix[0] == PartitionSpec("pt1=1")
        assert len(parts) == 2
        assert [str(pt) for pt in parts] == ["pt1='1',pt2='1'", "pt1='1',pt2='2'"]

        # filter with sub partitions
        parts = list(tb.iterate_partitions("pt2=1"))
        assert part_prefix[0] is None
        assert len(parts) == 2
        assert [str(pt) for pt in parts] == ["pt1='1',pt2='1'", "pt1='2',pt2='1'"]

        # filter with inequalities
        parts = list(tb.iterate_partitions("pt2!=1"))
        assert part_prefix[0] is None
        assert len(parts) == 2
        assert [str(pt) for pt in parts] == ["pt1='1',pt2='2'", "pt1='2',pt2='2'"]

    tb.drop()


def test_tiered_partition(odps_with_storage_tier):
    odps = odps_with_storage_tier

    test_table_name = tn('pyodps_t_tmp_parted_tiered')
    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(
        test_table_name, ("col string", "pt string"), lifecycle=1
    )
    part = table.create_partition("pt=20230711")
    part.set_storage_tier("standard")
    assert part.storage_tier_info.storage_tier == StorageTier.STANDARD
    part.set_storage_tier("LowFrequency")
    assert part.storage_tier_info.storage_tier == StorageTier.LOWFREQENCY
    table.drop()
