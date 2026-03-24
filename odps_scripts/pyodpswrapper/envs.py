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

import os

from .utils import suspend_option_errors

CONFIG_FILE = "PYODPS_CONFIG_FILE"
PICKLE_ACCOUNT = "PYODPS_PICKLE_ACCOUNT"
IS_EXT = "IS_EXT"
ACCESS_ID_ENV = "SKYNET_ACCESSID"
ACCESS_KEY_ENV = "SKYNET_ACCESSKEY"
PROJECT_ENV = "SKYNET_PACKAGEID"
ENDPOINT_ENV = "SKYNET_ENDPOINT"
SYSTEM_ENV = "SKYNET_SYSTEM_ENV"
RUNNING_QUOTA_ENV = "SKYNET_ODPS_RUNNING_QUOTA"
CONNECTION = "SKYNET_CONNECTION"
REGION_ENV = "SKYNET_REGION"
PRIORITY = "SKYNET_PRIORITY"
ACCESS_KEY_ENCRYPT = "PYODPS_ACESS_KEY_ENCRYPT"
ACCESS_KEY_ENCRYPT_KEY = "PYODPS_ACCESS_KEY_ENCRYPT_KEY"
BIZ_ID_KEYS = ["SKYNET_ID", "SKYNET_BIZDATE", "SKYNET_TASKID", "SKYNET_JOBID"]
SKYNET_SETTING_EXCLUDE = {ACCESS_ID_ENV, ACCESS_KEY_ENV, ACCESS_KEY_ENCRYPT}
SKYNET_SETTING_EXCLUDE.update(
    {
        "SKYNET_ACCESSKEY",
        "SKYNET_ALISA_TIME_KEY",
        "SKYNET_ALISA_GW_KEY",
        "SKYNET_ENGINE_AK",
        "SKYNET_CONNECTION",
    }
)


def is_internal():
    return os.environ.get(IS_EXT, "false") == "false"


def is_production():
    system_env_value = (os.environ.get(SYSTEM_ENV) or "dev").lower()
    assert system_env_value in ("dev", "prod")
    return system_env_value == "prod"


def get_biz_id(project_name):
    sql_seq = 1
    on_duty = os.getenv("SKYNET_ONDUTY", "-")
    on_duty = (
        "_" + on_duty.replace("\\", "#", 1) + "_" + os.getenv("SKYNET_TENANT_ID", "-")
    )
    app_name = "_" + project_name
    return (
        "_".join(os.getenv(key, "-") for key in BIZ_ID_KEYS)
        + "_%d" % sql_seq
        + app_name
        + on_duty
    )


def get_priority():
    priority = os.getenv(PRIORITY, 0)
    return min(9, max(0, 9 - int(priority)))


def set_skynet_to_odps_options():
    from odps import options

    skynet_settings = {
        key: val
        for key, val in os.environ.items()
        if key.startswith("SKYNET_") and key not in SKYNET_SETTING_EXCLUDE
    }
    if is_internal() and RUNNING_QUOTA_ENV in os.environ:
        skynet_settings["odps.running.quotagroup"] = skynet_settings.pop(
            RUNNING_QUOTA_ENV
        )
    with suspend_option_errors():
        if options.default_task_settings is None:
            options.default_task_settings = skynet_settings
        else:
            options.default_task_settings.update(skynet_settings)
