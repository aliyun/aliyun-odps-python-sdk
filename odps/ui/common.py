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
from six import iterkeys, itervalues
from six.moves.urllib.parse import quote

from odps.utils import load_static_file_paths, load_static_text_files


MAX_SCRIPT_LOAD_SEC = 5
SCRIPT_LOAD_CHECK_INTERVAL = 0.01

INTERRUPTER_JS = """
if ('undefined' === typeof pyodps) pyodps = {};
pyodps.load_counter = 0; pyodps.last_load_time = new Date();
pyodps.loaded = function() {
    pyodps.load_counter++;
    if (pyodps.load_counter == ##SCRIPT_NUM## && (new Date() - pyodps.last_load_time) / 1000.0 <= ##MAX_SCRIPT_LOAD_SEC## - 0.5)
        IPython.notebook.kernel.interrupt();
}
"""


CSS_REGISTER_JS = """
require(["jquery"], function($) {
    if ($('style[data-pyodps-styles="_"]').length > 0) return;
    css_str = decodeURIComponent('##CSS_STR##');
    $('head').append('<style type="text/css" data-pyodps-styles="_">' + css_str + '</style>');
});
"""


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

    # load static resources, .min.* come first.
    def load_static_resources(ext):
        file_dict = load_static_text_files('ui/*.min.' + ext)
        file_names = set([fn.replace('.min.' + ext, '') for fn in iterkeys(file_dict)])
        file_dict.update(load_static_text_files('ui/*.' + ext, lambda fn: '.min.' not in fn and
                                                                          not fn.replace('.' + ext, '') in file_names))
        return file_dict

    minify_scripts()

    js_dict = load_static_resources('js') if in_ipython_frontend() else None

    if js_dict:
        css_dict = load_static_resources('css') if in_ipython_frontend() else None

        css_contents = ''
        if css_dict:
            css_contents = '\n'.join(itervalues(css_dict))

        js_contents = INTERRUPTER_JS.replace('##SCRIPT_NUM##', str(len(js_dict)))\
            .replace('##MAX_SCRIPT_LOAD_SEC##', str(MAX_SCRIPT_LOAD_SEC))
        if css_contents:
            js_contents += '\n' + CSS_REGISTER_JS.replace('##CSS_STR##', quote(css_contents))
        js_contents += '\n'.join(itervalues(js_dict))

    _script_loaded = False

    class init_frontend_scripts(object):
        def __init__(self):
            self.run()

        def __enter__(self):
            global _script_loaded
            if _script_loaded:
                return self
            return self.run()

        def __exit__(self, *_):
            global _script_loaded
            _script_loaded = False

        def run(self):
            if in_ipython_frontend():
                js = widgets.HTML()
                js.value = '<script type="type/javascript">\n%s\n</script>' % js_contents
                display(js)
                # Wait for interrupt signal from the front end
                try:
                    for _ in range(int(round(MAX_SCRIPT_LOAD_SEC / SCRIPT_LOAD_CHECK_INTERVAL))):
                        time.sleep(SCRIPT_LOAD_CHECK_INTERVAL)
                except KeyboardInterrupt:
                    pass
                js.close()
                return self
            else:
                return self

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

    class HTMLNotifier(widgets.DOMWidget):
        _view_name = build_unicode_control('HTMLNotifier', sync=True)

        def __init__(self, **kwargs):
            init_frontend_scripts()

            widgets.DOMWidget.__init__(self, **kwargs)  # Call the base.
            self.errors = widgets.CallbackDispatcher(accepted_nargs=[0, 1])

        def notify(self, msg):
            self.send(json.dumps(dict(body=msg)))


def html_notify(msg):
    if in_ipython_frontend and in_ipython_frontend():
        notifier = HTMLNotifier()
        display(notifier)
        notifier.notify(msg)
        notifier.close()
