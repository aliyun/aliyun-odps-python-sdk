# -*- coding: utf-8 -*-
# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from .utils import suspend_option_errors


def patch_compat():
    from odps import accounts
    from odps import core as odps_core_mod
    from odps import options

    # make legacy AliyunAccount class compatible
    if hasattr(accounts, "CloudAccount") and not hasattr(accounts, "AliyunAccount"):
        accounts.AliyunAccount = accounts.CloudAccount

    _odps_core_defaults = {
        "DEFAULT_ENDPOINT": "http://service.odps.aliyun.com/api",
        "DEFAULT_REGION_NAME": "cn",
        "LOGVIEW_HOST_DEFAULT": "http://logview.aliyun.com",
    }

    for _attr_item in _odps_core_defaults.items():
        if not getattr(odps_core_mod, _attr_item[0], None):
            setattr(odps_core_mod, _attr_item[0], _attr_item[1])
    if odps_core_mod.DEFAULT_ENDPOINT is None:
        odps_core_mod.DEFAULT_ENDPOINT = _odps_core_defaults["DEFAULT_ENDPOINT"]

    with suspend_option_errors():
        options.signature_prefix = "aliyun_v4"
    with suspend_option_errors():
        options.enable_v4_sign = False
