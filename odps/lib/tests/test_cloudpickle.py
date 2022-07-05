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

import base64
import json
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
import uuid

from odps.compat import six, unittest, PY27
from odps.lib.cloudpickle import loads, dumps
from odps.utils import to_binary
from odps.tests.core import TestBase, numpy_case

# if bytecode needed in debug, switch it on
DUMP_CODE = False

CROSS_VAR_PICKLE_CODE = """
import base64
import json
import sys
import platform
import os
import pickle

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

client_impl = (sys.version_info[0],
               sys.version_info[1],
               platform.python_implementation().lower())
result_obj = {method_ref}()
result_tuple = (
    base64.b64encode(dumps(result_obj, dump_code={dump_code})),
    client_impl,
)
with open(r'{pickled_file}', 'w') as f:
    f.write(to_str(base64.b64encode(pickle.dumps(result_tuple, protocol=0))))
    f.close()
""".replace('{module_name}', __name__).replace('{dump_code}', repr(DUMP_CODE))


def pickled_runner(out_file, pickled, args, kwargs, **kw):
    try:
        wrapper = kw.pop('wrapper', None)
        impl = kwargs.pop('impl', (3, 5, 'cpython'))
        if wrapper:
            wrapper = loads(wrapper)
        else:
            wrapper = lambda v, a, kw: v(*a, **kw)
        deserial = loads(base64.b64decode(pickled), impl=impl, dump_code=DUMP_CODE)
        with open(out_file, "wb") as out_file:
            out_file.write(pickle.dumps(wrapper(deserial, args, kwargs)))
    except:
        traceback.print_exc()
        raise


def run_pickled(pickled, *args, **kwargs):
    pickled, kwargs['impl'] = pickle.loads(base64.b64decode(pickled))
    wrapper_kw = {}
    if 'wrapper' in kwargs:
        wrapper_kw['wrapper'] = dumps(kwargs.pop('wrapper'))

    tmp_dir = tempfile.mkdtemp(prefix="test_pyodps_")
    try:
        fn = os.path.join(tmp_dir, "pickled_result.bin")
        proc = multiprocessing.Process(
            target=pickled_runner, args=(fn, pickled, args, kwargs), kwargs=wrapper_kw
        )
        proc.start()
        proc.join()
        with open(fn, "rb") as in_file:
            return pickle.loads(in_file.read())
    finally:
        shutil.rmtree(tmp_dir)


def _gen_nested_yield_obj():
    out_closure = 10

    class _NestClass(object):
        inner_gain = 5

        def __init__(self):
            self._o_closure = out_closure

        def nested_method(self, add_val):
            if add_val < 5:
                return self._o_closure + add_val * 2 + self.inner_gain
            else:
                return self._o_closure + add_val + self.inner_gain

    class _FuncClass(object):
        def __init__(self):
            self.nest = _NestClass()

        def __call__(self, add_val):
            yield self.nest.nested_method(add_val)

    return _FuncClass


def _gen_from_import_func():
    def fun(val):
        from numpy import sinh
        return float(sinh(val))

    return fun


class BuildMeta(type):
    pass


class BuildBase(object):
    pass


if six.PY2:
    def _gen_class_builder_func():
        out_closure = 10

        def _gen_nested_class_obj():
            class BuildCls(BuildBase):
                __metaclass__ = BuildMeta
                a = out_closure

                def b(self, add_val):
                    print(self.a)
                    return self.a + add_val + out_closure

            return BuildCls
        return _gen_nested_class_obj
else:
    py3_code = textwrap.dedent("""
    def _gen_class_builder_func():
        out_closure = 10

        def _gen_nested_class_obj():
            class BuildCls(BuildBase, metaclass=BuildMeta):
                a = out_closure

                def b(self, add_val):
                    print(self.a)
                    return self.a + add_val + out_closure

            return BuildCls
        return _gen_nested_class_obj
    """)
    my_locs = locals().copy()
    six.exec_(py3_code, globals(), my_locs)
    _gen_class_builder_func = my_locs.get('_gen_class_builder_func')


if sys.version_info[:2] < (3, 6):
    def _gen_format_string_func():
        out_closure = 4.0

        def _format_fun(arg):
            return 'Formatted stuff {0}: {1:>5}'.format(arg, out_closure)

        return _format_fun
else:
    py36_code = textwrap.dedent("""
    def _gen_format_string_func():
        out_closure = 4.0

        def _format_fun(arg):
            return f'Formatted stuff {arg}: {out_closure:>5}'

        return _format_fun
    """)
    my_locs = locals().copy()
    six.exec_(py36_code, globals(), my_locs)
    _gen_format_string_func = my_locs.get('_gen_format_string_func')


if sys.version_info[:2] < (3, 6):
    def _gen_build_unpack_func():
        out_closure = (1, 2, 3)

        def merge_kws(a, b, *args, **kwargs):
            kwargs.update(dict(a=a, b=b))
            kwargs.update((str(idx), v) for idx, v in enumerate(args))
            return kwargs

        def _gen_fun(arg):
            t = out_closure + (4, ) + (5, 6, 7) + (arg, )
            l = list(out_closure) + [4, ] + [5, 6, 7]
            s = set(out_closure) | set([4]) | set([5, 6, 7])
            m = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
            wk = merge_kws(3, 4, 5, *(out_closure + (1, 2, 3)), **dict(m=1, n=2, p=3, q=4, r=5))
            return t, l, s, m, wk

        return _gen_fun
else:
    py36_code = textwrap.dedent("""
    def _gen_build_unpack_func():
        out_closure = (1, 2, 3)

        def merge_kws(a, b, *args, **kwargs):
            kwargs.update(dict(a=a, b=b))
            kwargs.update((str(idx), v) for idx, v in enumerate(args))
            return kwargs

        def _gen_fun(arg):
            t = (*out_closure, *(4, ), *(5, 6, 7), *(arg, ))
            l = [*out_closure, *(4, ), *[5, 6, 7]]
            s = {*out_closure, *[4], *[5, 6, 7]}
            m = {**dict(a=1, b=2), **dict(c=3), **dict(d=4, e=5)}
            wk = merge_kws(3, 4, 5, *out_closure, *[1, 2, 3], **dict(m=1, n=2), **dict(p=3, q=4, r=5))
            return t, l, s, m, wk

        return _gen_fun
    """)
    my_locs = locals().copy()
    six.exec_(py36_code, globals(), my_locs)
    _gen_build_unpack_func = my_locs.get('_gen_build_unpack_func')


if sys.version_info[:2] < (3, 6):
    def _gen_matmul_func():
        out_closure = [[4, 9, 2], [3, 5, 7], [8, 1, 6]]

        def _gen_fun(arg):
            import numpy as np
            a = np.array(out_closure)
            b = np.array([9, 5, arg])
            c = np.dot(a, b)
            return repr(c)

        return _gen_fun
else:
    py36_code = textwrap.dedent("""
    def _gen_matmul_func():
        out_closure = [[4, 9, 2], [3, 5, 7], [8, 1, 6]]

        def _gen_fun(arg):
            import numpy as np
            a = np.array(out_closure)
            b = np.array([9, 5, arg])
            c = a @ b
            return repr(c)

        return _gen_fun
    """)
    my_locs = locals().copy()
    six.exec_(py36_code, globals(), my_locs)
    _gen_matmul_func = my_locs.get('_gen_matmul_func')


def _gen_try_except_func():
    out_closure = dict(k=12.0)

    def _gen_fun(arg):
        ex = None
        agg = arg

        def _cl():
            print(ex)

        try:
            agg *= out_closure['not_exist']
        except KeyError as ex:
            agg += 1

        try:
            agg -= out_closure['k']
        except KeyError as ex:
            _cl()
            agg /= 10
        return agg

    return _gen_fun


def _gen_nested_fun():
    out_closure = 10

    def _gen_nested_obj():
        # class NestedClass(object):
        def nested_method(add_val):
            return out_closure + add_val

        return nested_method

    return lambda v: _gen_nested_obj()(*(v, ))


class Test(TestBase):
    @staticmethod
    def _invoke_other_python_pickle(executable, method_ref):
        paths = [path for path in sys.path if 'odps' in path.lower()]
        if callable(method_ref):
            method_ref = method_ref.__name__
        ts_name = os.path.join(tempfile.gettempdir(), 'pyodps_pk_cross_test_{0}.py'.format(str(uuid.uuid4())))
        tp_name = os.path.join(tempfile.gettempdir(), 'pyodps_pk_cross_pickled_{0}'.format(str(uuid.uuid4())))
        script_text = CROSS_VAR_PICKLE_CODE.format(import_paths=json.dumps(paths), method_ref=method_ref,
                                                   pickled_file=tp_name)
        with open(ts_name, 'w') as out_file:
            out_file.write(script_text)
            out_file.close()
        proc = subprocess.Popen([executable, ts_name])
        proc.wait()
        if not os.path.exists(tp_name):
            raise SystemError('Pickle error occurred!')
        else:
            with open(tp_name, 'r') as f:
                pickled = f.read().strip()
                f.close()
            os.unlink(tp_name)

            if not pickled:
                raise SystemError('Pickle error occurred!')
        return pickled

    def testRangeObject(self):
        obj_serial = dumps(range(10))
        deserial = loads(obj_serial)
        self.assertListEqual(list(range(10)), list(deserial))

    def testNestedFunc(self):
        func = _gen_nested_fun()
        obj_serial = base64.b64encode(dumps(func))
        deserial = loads(base64.b64decode(obj_serial))
        self.assertEqual(deserial(20), func(20))

    @unittest.skipIf(not PY27, 'Ignored under Python 3')
    @numpy_case
    def testFromImport(self):
        executable = self.config.get('test', 'py3_executable')
        if not executable:
            return
        func = _gen_from_import_func()
        py3_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_from_import_func))
        self.assertEqual(run_pickled(py3_serial, 20), func(20))

    @unittest.skipIf(not PY27, 'Ignored under Python 3')
    def test3to2FormatString(self):
        executable = self.config.get('test', 'py3_executable')
        if not executable:
            return
        func = _gen_format_string_func()
        py3_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_format_string_func))
        self.assertEqual(run_pickled(py3_serial, 20), func(20))

    @unittest.skipIf(not PY27, 'Ignored under Python 3')
    def test3to2BuildUnpack(self):
        executable = self.config.get('test', 'py3_executable')
        if not executable:
            return
        func = _gen_build_unpack_func()
        py3_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_build_unpack_func))
        self.assertEqual(run_pickled(py3_serial, 20), func(20))

    @unittest.skipIf(not PY27, 'Ignored under Python 3')
    @numpy_case
    def test3to2MatMul(self):
        executable = self.config.get('test', 'py3_executable')
        if not executable:
            return
        func = _gen_matmul_func()
        py3_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_matmul_func))
        self.assertEqual(run_pickled(py3_serial, 20), func(20))

    @unittest.skipIf(not PY27, 'Ignored under Python 3')
    def test3to2TryExcept(self):
        executable = self.config.get('test', 'py3_executable')
        if not executable:
            return
        func = _gen_try_except_func()
        py3_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_try_except_func))
        self.assertEqual(run_pickled(py3_serial, 20), func(20))

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
    def test3to27NestedYieldObj(self):
        try:
            executable = self.config.get('test', 'py3_executable')
            if not executable:
                return
        except:
            return
        func = _gen_nested_yield_obj()
        py3_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_nested_yield_obj))
        self.assertEqual(run_pickled(py3_serial, 20, wrapper=lambda fun, a, kw: sum(fun()(*a, **kw))),
                         sum(func()(20)))

    @unittest.skipIf(not PY27, 'Only runnable under Python 2.7')
    def test26to27NestedYieldObj(self):
        try:
            executable = self.config.get('test', 'py26_executable')
            if not executable:
                return
        except:
            return
        func = _gen_nested_yield_obj()
        py26_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_nested_yield_obj))
        self.assertEqual(run_pickled(py26_serial, 20, wrapper=lambda fun, a, kw: sum(fun()(*a, **kw))),
                         sum(func()(20)))

    @unittest.skipIf(not PY27, 'Only runnable under Python 2.7')
    def test3to27NestedClassObj(self):
        try:
            executable = self.config.get('test', 'py3_executable')
            if not executable:
                return
        except:
            return
        cls = _gen_class_builder_func()()
        py3_serial = to_binary(self._invoke_other_python_pickle(executable, _gen_class_builder_func))
        self.assertEqual(run_pickled(py3_serial, 5, wrapper=lambda cls, a, kw: cls()().b(*a, **kw)),
                         cls().b(5))
