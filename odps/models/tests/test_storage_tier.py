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

import datetime
import time

from ..storage_tier import StorageTier, StorageTierInfo


def test_storage_tier_parse():
    modified_time = int(time.time())

    assert StorageTierInfo.deserial({}) is None

    table_tier_info = {
        "StorageTier": "Standard",
        "StorageLastModifiedTime": modified_time,
        "StandardSize": 1024,
        "LowFrequencySize": 2048,
        "LongTermSize": 1024,
    }
    info = StorageTierInfo.deserial(table_tier_info)
    assert info.storage_tier == StorageTier.STANDARD
    assert info.last_modified_time == datetime.datetime.fromtimestamp(modified_time)
    assert info.storage_size == {
        StorageTier.STANDARD: 1024,
        StorageTier.LOWFREQENCY: 2048,
        StorageTier.LONGTERM: 1024,
    }

    project_tier_info = {
        "chargableAliveDataSize": 1024,
        "chargableLowFreqStorageSize": 2048,
        "chargableLongTermStorageSize": 1024,
        "chargableRecycleBinSize": 1024,
    }
    info = StorageTierInfo.deserial(project_tier_info)
    assert info.storage_size == {
        StorageTier.STANDARD: 1024,
        StorageTier.LOWFREQENCY: 2048,
        StorageTier.LONGTERM: 1024,
    }
