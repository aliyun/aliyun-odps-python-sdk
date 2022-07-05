# !/usr/bin/env python
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

import json
import time

from ....utils import to_str


def wait_all_schedulers_ready(kv_store, scheduler_keys):
    schedulers = scheduler_keys.split(",")
    while True:
        scheduler_endpoints = []
        for scheduler_key in schedulers:
            json_val = to_str(kv_store.get(scheduler_key))
            if json_val:
                config = json.loads(to_str(json_val))
                scheduler_endpoints.append(to_str(config["endpoint"]))
        if len(scheduler_endpoints) == len(schedulers):
            break
        time.sleep(1)
    return scheduler_endpoints
