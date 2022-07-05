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

from __future__ import print_function

import re
from collections import namedtuple

from ..compat import six, getargspec
from .. import ODPS, options, utils
from ..inter import list_rooms

PROJECT_REGEX = re.compile(r'.*project *= *(?P<project>[^\(\),]+)')
NAME_REGEX = re.compile(r'.*name *= *(?P<name>[^\(\),]+)')
TEMP_TABLE_PREFIXES = [
    utils.TEMP_TABLE_PREFIX,
    'jdbc_temp_tbl_',
    'pai_temp_',
    'temp_xlib_table_'
]


class RoomCompleter(object):
    def __init__(self, ipython=None):
        self._ipython = ipython
        self._regex_str = r'^%(enter|setup|teardown|stores) +'

    def register(self):
        self._ipython.set_hook('complete_command', self, re_key=self._regex_str)

    def __call__(self, completer, event):
        cursor_text = event.text_until_cursor
        _, prefix = cursor_text.split(' ', 1)
        prefix = prefix.strip()
        rooms = [n for n in list_rooms() if n.startswith(prefix)]
        return rooms[:options.completion_size]


class BaseCompleter(object):
    def __init__(self, ipython=None):
        self._ipython = ipython
        self._regex_str = self.build_regex()

    def build_regex(self):
        pass

    def get_list_call(self, cursor_str, full_line=None):
        pass

    def register(self):
        self._ipython.set_hook('complete_command', self, re_key=self._regex_str)

    def __call__(self, completer, event):
        cursor_text = event.text_until_cursor
        full_line = event.line
        call_tuple = self.get_list_call(cursor_text, full_line)
        if call_tuple is None:
            return None
        code, quote = call_tuple
        if quote is None:
            quote = '\''
        else:
            quote = ''

        try:
            list_gen = self._ipython.ev(code)
        except:
            return None

        def _is_temp_table(tn):
            return any(tn.startswith(p) for p in TEMP_TABLE_PREFIXES)

        def render_object(o):
            if hasattr(o, 'name'):
                if _is_temp_table(o.name):
                    return None
                name = o.name
            else:
                name = str(o)
            return quote + utils.str_to_printable(name, auto_quote=False) + quote

        names = [render_object(o) for idx, o in enumerate(list_gen)
                 if idx < options.completion_size]
        return [n for n in names if n]


class ObjectCompleter(BaseCompleter):
    @staticmethod
    def iter_methods():
        odps_methods = [m_name for m_name in dir(ODPS) if callable(getattr(ODPS, m_name))
                        and not m_name.startswith('_')]
        odps_listers = [m_name for m_name in odps_methods if m_name.startswith('list_')]

        for m_name in odps_methods:
            for prefix in ('get_', 'delete_', 'write_', 'read_'):
                if m_name.startswith(prefix):
                    lister = None
                    lister_prefix = m_name.replace(prefix, 'list_')
                    for l in odps_listers:
                        if l.startswith(lister_prefix):
                            lister = l
                            break
                    if lister:
                        yield m_name, lister
                    break

    def build_regex(self):
        self._methods = {}
        method_type = namedtuple('MethodType', 'use_prefix list_method')

        for m_name, lister in self.iter_methods():
            arg_tuple = getargspec(getattr(ODPS, lister))
            use_prefix = 'prefix' in arg_tuple[0]
            self._methods[m_name] = method_type(use_prefix=use_prefix, list_method=lister)

        _regex_str = '(^|.*[\(\)\s,=]+)(?P<odps>[^\(\)\s,]+)\.(?P<getfn>' + '|'.join(six.iterkeys(self._methods)) + ')\('
        self._regex = re.compile(_regex_str + r'(?P<args>[^\(\)]*)$')
        return _regex_str

    def get_list_call(self, cursor_str, full_line=None):
        full_line = full_line or cursor_str

        cmatch = self._regex.match(cursor_str)
        if cmatch is None:
            return None
        odps_obj = cmatch.group('odps')
        get_cmd = cmatch.group('getfn')
        arg_str = cmatch.group('args').strip()

        project = 'None'
        arg_start, arg_cursor = cmatch.span('args')
        arg_body = full_line[arg_start:]
        pmatch = PROJECT_REGEX.match(arg_body)
        if pmatch:
            project = pmatch.group('project')

        nmatch = NAME_REGEX.match(arg_str)
        name_str = nmatch.group('name') if nmatch else arg_str

        quote = None
        if name_str != '' and not (name_str.startswith('\'') or name_str.startswith('\"')):
            return None
        if name_str.endswith('\"') or name_str.endswith('\''):
            return None
        if name_str:
            quote = name_str[0]
            name_str = name_str[1:]
        if '"' in name_str or "'" in name_str:
            return None

        if name_str and self._methods[get_cmd].use_prefix:
            formatter = '{odps}.{func}(prefix="{prefix}", project={project})'
        else:
            formatter = '{odps}.{func}(project={project})'
        return formatter.format(odps=odps_obj, func=self._methods[get_cmd].list_method,
                                prefix=name_str, project=project), quote


def load_ipython_extension(ipython):
    ObjectCompleter(ipython).register()
    RoomCompleter(ipython).register()
