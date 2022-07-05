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

import os

from ...utils import to_str


def _load_bearer_token_environ():
    from .cupid_service import CupidServiceClient

    if "CUPID_SERVICE_SOCKET" not in os.environ:
        return
    bearer_token = CupidServiceClient().get_bearer_token()
    os.environ["ODPS_BEARER_TOKEN"] = to_str(bearer_token)
    os.environ["ODPS_ENDPOINT"] = os.environ["ODPS_RUNTIME_ENDPOINT"]


_load_bearer_token_environ()
