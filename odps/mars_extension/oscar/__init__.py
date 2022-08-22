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

from .run_script import execute_with_odps_context
from .filesystem import VolumeFileSystem

from mars import __version__ as mars_version
from mars.lib.filesystem.core import register_filesystem
from mars.remote.run_script import RunScript
from mars.remote.core import RemoteFunction


RemoteFunction.register_executor(execute_with_odps_context(RemoteFunction.execute))
RunScript.register_executor(execute_with_odps_context(RunScript.execute))
register_filesystem("odps", VolumeFileSystem)


# hotfix v0.8 bug
if mars_version.startswith("0.8"):
    import pandas as pd
    from mars import utils as mars_utils

    def on_serialize_nsplits(value):
        if value is None:
            return None
        new_nsplits = []
        for dim_splits in value:
            new_nsplits.append(tuple(None if pd.isna(v) else v for v in dim_splits))
        return tuple(new_nsplits)

    mars_utils.on_serialize_nsplits.__code__ = on_serialize_nsplits.__code__

try:
    from pandas._libs import lib as _pandas__libs_lib
    from mars import utils as mars_utils

    if not hasattr(_pandas__libs_lib, "NoDefault"):
        _pandas__libs_lib.NoDefault = mars_utils.NoDefault
        _pandas__libs_lib.no_default = mars_utils.no_default
except ImportError:
    pass
