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

MAX_SCRIPT_LOAD_SEC = 5
SCRIPT_LOAD_CHECK_INTERVAL = 0.01


def build_trait(trait_cls, default_value=None, **metadata):
    """Build a trait with metadata.

    `trait_cls` comes from the caller, so this helper works whenever the caller
    managed to import traitlets - it does not depend on the widget state below.
    Keeping it unconditional matters because this module is imported before an
    IPython front end exists, while callers may be (re)imported afterwards.
    """
    if default_value:
        return trait_cls(default_value).tag(**metadata)
    else:
        return trait_cls().tag(**metadata)


try:
    from ..console import (  # noqa: F401  (in_ipython_frontend re-exported)
        get_ipython_major_version,
        get_widgets,
        in_ipython_frontend,
    )

    widgets = get_widgets()
    ipython_major_version = get_ipython_major_version()
    if widgets is None or ipython_major_version is None:
        raise ImportError

    from traitlets import List, Unicode  # noqa: F401
except Exception:
    InstancesProgress = None
else:
    last_ipython_msg_id = None
