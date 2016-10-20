# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

try:
    from ..console import widgets, ipython_major_version, in_ipython_frontend, is_widgets_available
    if ipython_major_version < 4:
        from IPython.utils.traitlets import Unicode
    else:
        from traitlets import Unicode
    from IPython import get_ipython
    from IPython.display import display

    from ..ui.common import build_trait
except Exception:
    display_retry_button = None
    show_ml_retry_button = None
    register_retry_magic = None
else:
    if in_ipython_frontend():
        class MLRetryButton(widgets.DOMWidget):
            _view_name = build_trait(Unicode, 'MLRetryButton', sync=True)
            _view_module = build_trait(Unicode, 'pyodps/ml-retry', sync=True)
            msg = build_trait(Unicode, 'msg', sync=True)

            def update(self):
                self.msg = 'update'

    def show_ml_retry_button():
        if in_ipython_frontend():
            btn = MLRetryButton()
            if is_widgets_available():
                display(btn)
            btn.update()


    retry_via_magic = False

    def register_retry_magic():
        from .runner import get_retry_mode, set_retry_mode
        try:
            if in_ipython_frontend():
                from IPython import get_ipython
                from IPython.core.magic import register_line_magic

                @register_line_magic
                def retry(_):
                    global retry_via_magic

                    if not get_retry_mode():
                        retry_via_magic = True
                        set_retry_mode(True)
                    return ''

                del retry

                def auto_cancel_retry():
                    global retry_via_magic
                    if retry_via_magic:
                        retry_via_magic = False
                        set_retry_mode(False)

                get_ipython().events.register('post_execute', auto_cancel_retry)
        except ImportError:
            pass
