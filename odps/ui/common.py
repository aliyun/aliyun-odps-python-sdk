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

import json
import os
import time
import codecs

from odps.utils import load_static_file_paths, load_static_text_files
from odps.compat import six, quote


MAX_SCRIPT_LOAD_SEC = 5
SCRIPT_LOAD_CHECK_INTERVAL = 0.01

COMMON_JS = 'common'
CSS_REGISTER_JS = """
require(['pyodps'], function(p) { p.register_css('##CSS_STR##'); });
""".strip()


try:
    from ..console import widgets, ipython_major_version, in_ipython_frontend
    if ipython_major_version < 4:
        from IPython.utils.traitlets import Unicode, List
    else:
        from traitlets import Unicode, List
    from IPython.display import display
except Exception:
    InstancesProgress = None
    init_frontend_scripts = None
    build_unicode_control = None
    in_ipython_frontend = None
else:
    def minify_scripts():
        try:
            import slimit
            for fn in load_static_file_paths('ui/*.js'):
                if 'packages' in fn or not os.access(os.path.dirname(fn), os.W_OK):
                    return
                if fn.endswith('.min.js'):
                    continue
                min_fn = fn.replace('.js', '.min.js')
                if os.path.exists(min_fn) and os.path.getmtime(fn) <= os.path.getmtime(min_fn):
                    continue
                with codecs.open(fn, encoding='utf-8') as inf:
                    content = inf.read()
                    minified = slimit.minify(content, mangle=True, mangle_toplevel=True)
                    with codecs.open(min_fn, 'w', encoding='utf-8') as outf:
                        outf.write(minified)

            import csscompressor
            for fn in load_static_file_paths('ui/*.css'):
                if 'packages' in fn or not os.access(os.path.dirname(fn), os.W_OK):
                    return
                if fn.endswith('.min.css'):
                    continue
                min_fn = fn.replace('.css', '.min.css')
                if os.path.exists(min_fn) and os.path.getmtime(fn) <= os.path.getmtime(min_fn):
                    continue
                with codecs.open(fn, encoding='utf-8') as inf:
                    content = inf.read()
                    minified = csscompressor.compress(content)
                    with codecs.open(min_fn, 'w', encoding='utf-8') as outf:
                        outf.write(minified)
        except ImportError:
            pass

    _script_content = None

    def load_script():
        global _script_content
        if _script_content is None:
            # load static resources, .min.* come first.
            def load_static_resources(ext):
                file_dict = load_static_text_files('ui/*.min.' + ext)
                file_names = set([fn.replace('.min.' + ext, '') for fn in six.iterkeys(file_dict)])
                file_dict.update(load_static_text_files('ui/*.' + ext, lambda fn: '.min.' not in fn and
                                                                                  not fn.replace('.' + ext, '') in file_names))
                return file_dict

            minify_scripts()

            js_dict = load_static_resources('js') if in_ipython_frontend() else None

            if js_dict:
                css_dict = load_static_resources('css') if in_ipython_frontend() else None

                css_contents = ''
                if css_dict:
                    css_contents = '\n'.join(six.itervalues(css_dict))

                common_js_file = None
                if COMMON_JS + '.min.js' in js_dict:
                    common_js_file = COMMON_JS + '.min.js'
                elif COMMON_JS + '.js' in js_dict:
                    common_js_file = COMMON_JS + '.js'

                js_contents = ''

                if common_js_file:
                    js_common = js_dict.pop(common_js_file)
                    js_contents += '\n' + js_common.replace('##WIDGET_NUM##', str(len(js_dict))) \
                        .replace('##MAX_SCRIPT_LOAD_SEC##', str(MAX_SCRIPT_LOAD_SEC)) + '\n'

                if css_contents:
                    js_contents += '\n' + CSS_REGISTER_JS.replace('##CSS_STR##', quote(css_contents)) + '\n'
                js_contents += '\n'.join(six.itervalues(js_dict))
                _script_content = js_contents
            else:
                _script_content = None
        return _script_content

    last_ipython_msg_id = None

    def init_frontend_scripts():
        global last_ipython_msg_id
        if in_ipython_frontend():
            from IPython import get_ipython
            pheader = get_ipython().parent_header
            ipython_msg_id = pheader['msg_id'] if 'msg_id' in pheader else None
            if ipython_msg_id is not None and ipython_msg_id == last_ipython_msg_id:
                return
            last_ipython_msg_id = ipython_msg_id

            js = widgets.HTML()
            js.value = '<script type="type/javascript">\n%s\n</script>' % load_script()
            display(js)
            # Wait for interrupt signal from the front end
            try:
                for _ in range(int(round(MAX_SCRIPT_LOAD_SEC / SCRIPT_LOAD_CHECK_INTERVAL))):
                    time.sleep(SCRIPT_LOAD_CHECK_INTERVAL)
            except KeyboardInterrupt:
                pass
            js.close()

    def build_unicode_control(default_value=None, **metadata):
        from traitlets import version_info as traitlets_version, Unicode
        if traitlets_version[0] <= 4 and traitlets_version[1] < 1:  # old-fashioned call
            if default_value:
                return Unicode(default_value, **metadata)
            else:
                return Unicode(**metadata)
        else:
            if default_value:
                return Unicode(default_value).tag(**metadata)
            else:
                return Unicode().tag(**metadata)

    class HTMLNotifier(widgets.Widget):
        _view_name = build_unicode_control('HTMLNotifier', sync=True)
        _view_module = build_unicode_control('pyodps/html-notify', sync=True)
        msg = build_unicode_control('msg', sync=True)

        def __init__(self, **kwargs):
            init_frontend_scripts()

            widgets.Widget.__init__(self, **kwargs)  # Call the base.

        def notify(self, msg):
            self.msg = json.dumps(dict(body=msg))


def html_notify(msg):
    if in_ipython_frontend and in_ipython_frontend():
        notifier = HTMLNotifier()
        display(notifier)
        notifier.notify(msg)
