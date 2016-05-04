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

import os
import base64

from .types import df_type_to_odps_type
from .cloudpickle import dumps
from ...expr.element import MappedExpr
from ...expr.collections import RowAppliedCollectionExpr
from ...expr.groupby import GroupbyAppliedCollectionExpr
from ....compat import OrderedDict
from ....utils import to_str

dirname = os.path.dirname(os.path.abspath(__file__))
CLOUD_PICKLE_FILE = os.path.join(dirname, 'cloudpickle.py')
with open(CLOUD_PICKLE_FILE) as f:
    CLOUD_PICKLE = f.read()

UDF_TMPL = '''\
%(cloudpickle)s

import base64
from collections import namedtuple
import time

try:
    import numpy
except ImportError:
    pass

from odps.udf import annotate

@annotate('%(from_type)s->%(to_type)s')
class %(func_cls_name)s(object):

    def __init__(self):
        encoded = '%(func_str)s'
        f_str = base64.b64decode(encoded)
        self.f = loads(f_str)

        encoded_func_args = '%(func_args_str)s'
        func_args_str = base64.b64decode(encoded_func_args)
        self.args = loads(func_args_str) or tuple()

        encoded_func_kwargs = '%(func_kwargs_str)s'
        func_kwargs_str = base64.b64decode(encoded_func_kwargs)
        self.kwargs = loads(func_kwargs_str) or dict()

        self.names = tuple(it for it in '%(names_str)s'.split(',') if it)

        self.from_types = '%(from_type)s'.split(',')
        self.to_type = '%(to_type)s'

    def _handle_input(self, args):
        from datetime import datetime

        res = []
        for t, arg in zip(self.from_types, args):
            if t == 'datetime' and not isinstance(arg, datetime):
                res.append(datetime.fromtimestamp(arg / 1000.0))
            else:
                res.append(arg)
        return res

    def _to_milliseconds(self, dt):
        return int((time.mktime(dt.timetuple()) + dt.microsecond/1000000.0) * 1000)

    def _handle_output(self, arg):
        from datetime import datetime

        t = self.to_type

        if t == 'datetime' and isinstance(arg, datetime):
            return self._to_milliseconds(arg)
        else:
            return arg

    def evaluate(self, *args):
        args = self._handle_input(args)
        if not self.names:
            args = tuple(args) + tuple(self.args)
            res = self.f(*args, **self.kwargs)
            return self._handle_output(res)
        else:
            named_args = namedtuple('NamedArgs', self.names)
            res = self.f(named_args(*args), *self.args, **self.kwargs)
            return self._handle_output(res)
'''


UDTF_TMPL = '''\
%(cloudpickle)s

import base64
import inspect
from collections import namedtuple

try:
    import numpy
except ImportError:
    pass

from odps.udf import annotate
from odps.udf import BaseUDTF

@annotate('%(from_type)s->%(to_type)s')
class %(func_cls_name)s(BaseUDTF):
    def __init__(self):
        encoded = '%(func_str)s'
        f_str = base64.b64decode(encoded)
        self.f = loads(f_str)
        if inspect.isfunction(self.f):
            self.is_f_generator = inspect.isgeneratorfunction(self.f)
            self.close_f = None
            self.is_close_f_generator = False
        else:
            self.f = self.f()
            self.is_f_generator = inspect.isgeneratorfunction(self.f.__call__)
            self.close_f = getattr(self.f, 'close', None)
            self.is_close_f_generator = inspect.isgeneratorfunction(self.close_f)

        encoded_func_args = '%(func_args_str)s'
        func_args_str = base64.b64decode(encoded_func_args)
        self.args = loads(func_args_str) or tuple()

        encoded_func_kwargs = '%(func_kwargs_str)s'
        func_kwargs_str = base64.b64decode(encoded_func_kwargs)
        self.kwargs = loads(func_kwargs_str) or dict()

        self.names = tuple(it for it in '%(names_str)s'.split(',') if it)

        self.from_types = '%(from_type)s'.split(',')
        self.to_types = '%(to_type)s'.split(',')

    def _handle_input(self, args):
        from datetime import datetime

        res = []
        for t, arg in zip(self.from_types, args):
            if t == 'datetime' and not isinstance(arg, datetime):
                res.append(datetime.fromtimestamp(arg / 1000.0))
            else:
                res.append(arg)
        return res

    def _to_milliseconds(self, dt):
        return int((time.mktime(dt.timetuple()) + dt.microsecond/1000000.0) * 1000)

    def _handle_output(self, args):
        from datetime import datetime

        if len(self.to_types) != len(args):
            raise ValueError('Function ouput size should be' + str(len(self.to_types)))

        res = []
        for t, arg in zip(self.to_types, args):
            if t == 'datetime' and isinstance(arg, datetime):
                res.append(self._to_milliseconds(arg))
            else:
                res.append(arg)
        return type(args)(res)

    def process(self, *args):
        args = self._handle_input(args)
        if not self.names:
            args = tuple(args) + tuple(self.args)
        else:
            named_args = namedtuple('NamedArgs', self.names)
            args = (named_args(*args), ) + tuple(self.args)

        if self.is_f_generator:
            for r in self.f(*args, **self.kwargs):
                if not isinstance(r, (list, tuple)):
                    r = (r, )
                self.forward(*self._handle_output(r))
        else:
            res = self.f(*args, **self.kwargs)
            if res:
                if not isinstance(res, (list, tuple)):
                    res = (res, )
                self.forward(*self._handle_output(res))

    def close(self):
        if not self.close_f:
            return

        if self.is_close_f_generator:
            for r in self.close_f(*self.args, **self.kwargs):
                if not isinstance(r, (list, tuple)):
                    r = (r, )
                self.forward(*self._handle_output(r))
        else:
            res = self.close_f(*self.args, **self.kwargs)
            if res:
                if not isinstance(res, (list, tuple)):
                    res = (res, )
                self.forward(*self._handle_output(res))
'''


def gen_udf(expr, func_cls_name=None):
    func_to_udfs = OrderedDict()

    for node in expr.traverse(unique=True):
        func = getattr(node, '_func', None)
        if func is None:
            continue

        if isinstance(node, MappedExpr):
            names_str = ''
            if isinstance(node, MappedExpr) and node._multiple and \
                    all(f.name is not None for f in node.inputs):
                names_str = ','.join(f.name for f in node.inputs)
            from_type = ','.join(df_type_to_odps_type(t).name for t in node.input_types)
            func_to_udfs[func] = UDF_TMPL % {
                'cloudpickle': CLOUD_PICKLE,
                'from_type':  from_type,
                'to_type': df_type_to_odps_type(node.data_type).name,
                'func_cls_name': func_cls_name,
                'func_str': to_str(base64.b64encode(dumps(func))),
                'func_args_str': to_str(base64.b64encode(dumps(node._func_args))),
                'func_kwargs_str': to_str(base64.b64encode(dumps(node._func_kwargs))),
                'names_str': names_str
            }
        elif isinstance(node, (RowAppliedCollectionExpr, GroupbyAppliedCollectionExpr)):
            names_str = ','.join(f.name for f in node.fields)
            from_type = ','.join(df_type_to_odps_type(t).name for t in node.input_types)
            to_type = ','.join(df_type_to_odps_type(t).name for t in node.schema.types)
            func_to_udfs[func] = UDTF_TMPL % {
                'cloudpickle': CLOUD_PICKLE,
                'from_type': from_type,
                'to_type': to_type,
                'func_cls_name': func_cls_name,
                'func_str': to_str(base64.b64encode(dumps(func))),
                'func_args_str': to_str(base64.b64encode(dumps(node._func_args))),
                'func_kwargs_str': to_str(base64.b64encode(dumps(node._func_kwargs))),
                'close_func_str': to_str(base64.b64encode(dumps(getattr(node, '_close_func', None)))),
                'names_str': names_str
            }

    return func_to_udfs
