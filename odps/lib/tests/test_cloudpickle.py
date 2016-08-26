#!/usr/bin/env python
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

from __future__ import print_function

import base64
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import uuid

from odps.compat import six, unittest, PY27
from odps.lib.cloudpickle import loads, dumps
from odps.utils import to_binary
from odps.tests.core import TestBase

CROSS_VAR_PICKLE_CODE = """
import base64
import json
import sys
import os

try:
    os.unlink(os.path.realpath(__file__))
except Exception:
    pass

import_paths = json.loads(r\"\"\"
{import_paths}
\"\"\".strip())
sys.path.extend(import_paths)

from odps.lib.cloudpickle import dumps
from odps.utils import to_str
from {module_name} import {method_ref}
print(to_str(base64.b64encode(dumps({method_ref}()))))
""".replace('{module_name}', __name__)


def pickled_runner(q, pickled, args, kwargs, **kw):
    wrapper = kw.pop('wrapper', None)
    impl = kwargs.pop('impl', 'CP3')
    if wrapper:
        wrapper = loads(wrapper)
    else:
        wrapper = lambda v, a, kw: v(*a, **kw)
    deserial = loads(base64.b64decode(pickled), impl=impl)
    q.put(wrapper(deserial, args, kwargs))


def run_pickled(pickled, *args, **kwargs):
    wrapper_kw = {}
    if 'wrapper' in kwargs:
        wrapper_kw['wrapper'] = dumps(kwargs.pop('wrapper'))
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=pickled_runner, args=(queue, pickled, args, kwargs), kwargs=wrapper_kw)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise SystemError('Pickle process exited abnormally.')
    try:
        return queue.get()
    except:
        return None


def _gen_nested_yield_obj():
    out_closure = 10

    class _NestClass(object):
        def __init__(self):
            self._o_closure = out_closure

        def nested_method(self, add_val):
            return self._o_closure + add_val

    class _FuncClass(object):
        def __init__(self):
            self.nest = _NestClass()

        def __call__(self, add_val):
            yield self.nest.nested_method(add_val)

    return _FuncClass


def _gen_nested_fun():
    out_closure = 10

    def _gen_nested_obj():
        # class NestedClass(object):
        def nested_method(add_val):
            return out_closure + add_val

        return nested_method

    return lambda v: _gen_nested_obj()(v)


class Test(TestBase):
    @staticmethod
    def _invoke_other_python_pickle(executable, method_ref):
        paths = [path for path in sys.path if 'odps' in path.lower()]
        if callable(method_ref):
            method_ref = method_ref.__name__
        script_text = CROSS_VAR_PICKLE_CODE.format(import_paths=json.dumps(paths), method_ref=method_ref)
        tf_name = os.path.join(tempfile.gettempdir(), 'pyodps_pk_cross_test_{0}.py'.format(str(uuid.uuid4())))
        with open(tf_name, 'w') as out_file:
            out_file.write(script_text)
            out_file.close()
        proc = subprocess.Popen([executable, tf_name], stdout=subprocess.PIPE)
        sio = six.StringIO()
        while True:
            out = proc.stdout.read(1)
            if out == '' and proc.poll() != None:
                if proc.poll() != 0:
                    raise SystemError('Pickle error occured!')
                break
            sio.write(out)
        return sio.getvalue().strip()

    def testNestedFunc(self):
        func = _gen_nested_fun()
        obj_serial = base64.b64encode(dumps(func))
        deserial = loads(base64.b64decode(obj_serial))
        self.assertEqual(deserial(20), func(20))

    @unittest.skipIf(not PY27, 'Ignored under Python 3')
    def test3to2NestedFunc(self):
        executable = self.config.get('test', 'py3_executable')
        if not executable:
            return
        func = _gen_nested_fun()
        py3_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_nested_fun))
        self.assertEqual(run_pickled(py3_serial, 20), func(20))


    def testNestedClassObj(self):
        func = _gen_nested_yield_obj()
        obj_serial = base64.b64encode(dumps(func))
        deserial = loads(base64.b64decode(obj_serial))
        self.assertEqual(sum(deserial()(20)), sum(func()(20)))

    @unittest.skipIf(not PY27, 'Only runnable under Python 2.7')
    def test3to27NestedClassObj(self):
        try:
            executable = self.config.get('test', 'py3_executable')
            if not executable:
                return
        except:
            return
        func = _gen_nested_yield_obj()
        py3_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_nested_yield_obj))
        self.assertEqual(run_pickled(py3_serial, 20, wrapper=lambda fun, a, kw: sum(fun()(*a, **kw)), impl='CP3'),
                         sum(func()(20)))

    @unittest.skipIf(not PY27, 'Only runnable under Python 2.7')
    def test26to27NestedClassObj(self):
        try:
            executable = self.config.get('test', 'py26_executable')
            if not executable:
                return
        except:
            return
        func = _gen_nested_yield_obj()
        py26_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_nested_yield_obj))
        self.assertEqual(run_pickled(py26_serial, 20, wrapper=lambda fun, a, kw: sum(fun()(*a, **kw)), impl='CP26'),
                         sum(func()(20)))
