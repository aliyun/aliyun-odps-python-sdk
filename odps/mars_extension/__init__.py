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

from distutils.version import LooseVersion

import mars

try:
    mars_version = mars.__version__
except AttributeError:
    raise ImportError("Mars package broken")

if LooseVersion(mars_version) >= LooseVersion("0.7"):
    from .oscar import dataframe
    from .oscar.core import (
        create_mars_cluster,
        to_mars_dataframe,
        persist_mars_dataframe,
        run_script_in_mars,
        run_mars_job,
        list_mars_instances,
        sql_to_mars_dataframe,
    )
    from .oscar.deploy.client import MarsCupidClient, CUPID_APP_NAME, NOTEBOOK_NAME
else:
    from .legacy import (
        create_mars_cluster,
        dataframe,
        to_mars_dataframe,
        persist_mars_dataframe,
        run_script_in_mars,
        run_mars_job,
        list_mars_instances,
        sql_to_mars_dataframe,
    )
    from .legacy.deploy.client import MarsCupidClient, CUPID_APP_NAME, NOTEBOOK_NAME


INTERNAL_PATTERN = "\/[^\.]+\.[^\.-]+\.[^\.-]+\-[^\.-]+\."
