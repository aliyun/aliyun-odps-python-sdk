# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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


MAX_SCRIPT_LOAD_SEC = 5
SCRIPT_LOAD_CHECK_INTERVAL = 0.01

COMMON_JS = 'common'
CSS_REGISTER_JS = """
require(['pyodps'], function(p) { p.register_css('##CSS_STR##'); });
""".strip()


try:
    from ..console import widgets, ipython_major_version, in_ipython_frontend, is_widgets_available
    if any(v is None for v in (widgets, ipython_major_version, in_ipython_frontend)):
        raise ImportError

    if ipython_major_version < 4:
        from IPython.utils.traitlets import Unicode, List
        traitlets_version = (3, 0)
    else:
        from traitlets import Unicode, List
        from traitlets import version_info as traitlets_version
    from IPython.display import display
except Exception:
    InstancesProgress = None
    build_trait = None
    in_ipython_frontend = None
else:
    _script_content = None

    last_ipython_msg_id = None

    def build_trait(trait_cls, default_value=None, **metadata):
        if tuple(traitlets_version[:2]) < (4, 1):  # old-fashioned call
            if default_value:
                return trait_cls(default_value, **metadata)
            else:
                return trait_cls(**metadata)
        else:
            if default_value:
                return trait_cls(default_value).tag(**metadata)
            else:
                return trait_cls().tag(**metadata)

    class HTMLNotifier(widgets.Widget):
        _view_name = build_trait(Unicode, 'HTMLNotifier', sync=True)
        _view_module = build_trait(Unicode, 'pyodps/html-notify', sync=True)
        msg = build_trait(Unicode, 'msg', sync=True)

        def notify(self, msg):
            self.msg = json.dumps(dict(body=msg))


def html_notify(msg):
    if in_ipython_frontend and in_ipython_frontend():
        notifier = HTMLNotifier()
        if is_widgets_available():
            display(notifier)
        notifier.notify(msg)
