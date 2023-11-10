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

from .. import serializers
from ..compat import Enum


class StorageTier(Enum):
    STANDARD = "standard"
    LOWFREQENCY = "lowfrequency"
    LONGTERM = "longterm"


class StorageTierInfo(serializers.JSONSerializableModel):
    __slots__ = "storage_size",

    _storage_tier_size_names = {
        StorageTier.STANDARD: "StandardSize",
        StorageTier.LOWFREQENCY: "LowFrequencySize",
        StorageTier.LONGTERM: "LongTermSize",
    }

    _storage_tier_charge_size_names = {
        StorageTier.STANDARD: "chargableAliveDataSize",
        StorageTier.LOWFREQENCY: "chargableLowFreqStorageSize",
        StorageTier.LONGTERM: "chargableLongTermStorageSize",
    }

    storage_tier = serializers.JSONNodeField(
        'StorageTier', parse_callback=lambda x: StorageTier(x.lower()) if x else None
    )
    last_modified_time = serializers.JSONNodeField(
        'StorageLastModifiedTime',
        parse_callback=lambda x: datetime.datetime.fromtimestamp(int(x)),
    )

    def __init__(self, **kwargs):
        super(StorageTierInfo, self).__init__(**kwargs)
        self.storage_size = dict()

    @classmethod
    def deserial(cls, content, obj=None, **kw):
        res = super(StorageTierInfo, cls).deserial(content, obj=obj, **kw)
        for key_defs in (cls._storage_tier_size_names, cls._storage_tier_charge_size_names):
            for tier, key in key_defs.items():
                if key not in content:
                    continue
                res.storage_size[tier] = int(content[key])
        return res if res.storage_tier is not None or res.storage_size else None
