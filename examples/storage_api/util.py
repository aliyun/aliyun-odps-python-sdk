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

import os

from odps import ODPS
from odps.apis.storage_api import *

o = ODPS(
    os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"),
    os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
    project="your-default-project",
    endpoint="your-end-point",
)

table = "<table to access>"
quota_name = "<quota name>"


def get_arrow_client():
    odps_table = o.get_table(table)
    client = StorageApiArrowClient(odps=o, table=odps_table, quota_name=quota_name)

    return client
