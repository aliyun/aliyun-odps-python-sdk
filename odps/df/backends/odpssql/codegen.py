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
import uuid

from .types import df_type_to_odps_type
from .cloudpickle import dumps
from ...expr.element import MappedExpr
from ...expr.collections import RowAppliedCollectionExpr
from ...expr.groupby import GroupbyAppliedCollectionExpr
from ...expr.utils import get_executed_collection_table_name
from ...expr.reduction import Aggregation, GroupedAggregation
from ...utils import make_copy
from ....compat import OrderedDict, six
from ....utils import to_str
from ....models import FileResource, TableResource

dirname = os.path.dirname(os.path.abspath(__file__))
CLOUD_PICKLE_FILE = os.path.join(dirname, 'cloudpickle.py')
with open(CLOUD_PICKLE_FILE) as f:
    CLOUD_PICKLE = f.read()

UDF_TMPL = '''\
%(cloudpickle)s

import base64
from collections import namedtuple
import time
import inspect

try:
    import numpy
except ImportError:
    pass

try:
    import numpy
except ImportError:
    pass

from odps.udf import annotate
from odps.distcache import get_cache_file, get_cache_table

@annotate('%(from_type)s->%(to_type)s')
class %(func_cls_name)s(object):

    def __init__(self):
        rs = loads(base64.b64decode('%(resources)s'))
        resources = []
        for t, n, fields in rs:
            if t == 'file':
                resources.append(get_cache_file(n))
            else:
                tb = get_cache_table(n)
                if fields:
                    named_args = namedtuple('NamedArgs', fields)
                    tb = (named_args(*args) for args in tb)
                resources.append(tb)

        encoded = '%(func_str)s'
        f_str = base64.b64decode(encoded)
        self.f = loads(f_str)

        if inspect.isfunction(self.f):
            if resources:
                self.f = self.f(resources)
        else:
            if resources:
                self.f = self.f(resources)
            else:
                self.f = self.f()

        self.names = tuple(it for it in '%(names_str)s'.split(',') if it)

        encoded_func_args = '%(func_args_str)s'
        func_args_str = base64.b64decode(encoded_func_args)
        self.args = loads(func_args_str) or tuple()

        encoded_func_kwargs = '%(func_kwargs_str)s'
        func_kwargs_str = base64.b64decode(encoded_func_kwargs)
        self.kwargs = loads(func_kwargs_str) or dict()

        self.from_types = '%(raw_from_type)s'.split(',')
        self.to_type = '%(to_type)s'

    def _handle_input(self, args):
        from datetime import datetime
        from decimal import Decimal

        res = []
        for t, arg in zip(self.from_types, args):
            if t == 'datetime' and arg is not None and not isinstance(arg, datetime):
                res.append(datetime.fromtimestamp(arg / 1000.0))
            elif t == 'decimal' and arg is not None and isinstance(arg, str):
                res.append(Decimal(arg))
            else:
                res.append(arg)
        return res

    def _to_milliseconds(self, dt):
        return int((time.mktime(dt.timetuple()) + dt.microsecond/1000000.0) * 1000)

    def _handle_output(self, arg):
        from datetime import datetime
        from decimal import Decimal

        t = self.to_type

        if t == 'datetime' and isinstance(arg, datetime):
            return self._to_milliseconds(arg)
        elif t == 'string' and isinstance(arg, Decimal):
            return str(arg)
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
from odps.distcache import get_cache_file, get_cache_table

@annotate('%(from_type)s->%(to_type)s')
class %(func_cls_name)s(BaseUDTF):
    def __init__(self):
        rs = loads(base64.b64decode('%(resources)s'))
        resources = []
        for t, n, fields in rs:
            if t == 'file':
                resources.append(get_cache_file(n))
            else:
                tb = get_cache_table(n)
                if fields:
                    named_args = namedtuple('NamedArgs', fields)
                    tb = (named_args(*args) for args in tb)
                resources.append(tb)

        encoded = '%(func_str)s'
        f_str = base64.b64decode(encoded)
        self.f = loads(f_str)
        if inspect.isfunction(self.f):
            if resources:
                self.f = self.f(resources)

            self.is_f_generator = inspect.isgeneratorfunction(self.f)
            self.close_f = None
            self.is_close_f_generator = False
        else:
            if not resources:
                self.f = self.f()
            else:
                self.f = self.f(resources)
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

        self.from_types = '%(raw_from_type)s'.split(',')
        self.to_types = '%(to_type)s'.split(',')

    def _handle_input(self, args):
        from datetime import datetime
        from decimal import Decimal

        res = []
        for t, arg in zip(self.from_types, args):
            if t == 'datetime' and arg is not None and not isinstance(arg, datetime):
                res.append(datetime.fromtimestamp(arg / 1000.0))
            elif t == 'decimal' and arg is not None and not isinstance(arg, Decimal):
                res.append(Decimal(arg))
            else:
                res.append(arg)
        return res

    def _to_milliseconds(self, dt):
        return int((time.mktime(dt.timetuple()) + dt.microsecond/1000000.0) * 1000)

    def _handle_output(self, args):
        from datetime import datetime
        from decimal import Decimal

        if len(self.to_types) != len(args):
            raise ValueError('Function ouput size should be' + str(len(self.to_types)))

        res = []
        for t, arg in zip(self.to_types, args):
            if t == 'datetime' and isinstance(arg, datetime):
                res.append(self._to_milliseconds(arg))
            elif t == 'string' and isinstance(arg, Decimal):
                res.append(str(arg))
            else:
                res.append(arg)
        return res

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

UDAF_TMPL = '''\
%(cloudpickle)s

import base64
import inspect
from collections import namedtuple

try:
    import numpy
except ImportError:
    pass

from odps.udf import annotate
from odps.udf import BaseUDAF
from odps.distcache import get_cache_file, get_cache_table

@annotate('%(from_type)s->%(to_type)s')
class %(func_cls_name)s(BaseUDAF):
    def __init__(self):
        rs = loads(base64.b64decode('%(resources)s'))
        resources = []
        for t, n, fields in rs:
            if t == 'file':
                resources.append(get_cache_file(n))
            else:
                tb = get_cache_table(n)
                if fields:
                    named_args = namedtuple('NamedArgs', fields)
                    tb = (named_args(*args) for args in tb)
                resources.append(tb)

        encoded_func_args = '%(func_args_str)s'
        func_args_str = base64.b64decode(encoded_func_args)
        args = loads(func_args_str) or tuple()

        encoded_func_kwargs = '%(func_kwargs_str)s'
        func_kwargs_str = base64.b64decode(encoded_func_kwargs)
        kwargs = loads(func_kwargs_str) or dict()

        encoded = '%(func_str)s'
        f_str = base64.b64decode(encoded)
        agg = loads(f_str)
        if resources:
            if not args and not kwargs:
                self.f = agg(resources)
            else:
                kwargs['resources'] = resources
                self.f = agg(*args, **kwargs)
        else:
            self.f = agg(*args, **kwargs)

        self.from_types = '%(raw_from_type)s'.split(',')
        self.to_type = '%(to_type)s'

    def _handle_input(self, args):
        from datetime import datetime
        from decimal import Decimal

        res = []
        for t, arg in zip(self.from_types, args):
            if t == 'datetime' and arg is not None and not isinstance(arg, datetime):
                res.append(datetime.fromtimestamp(arg / 1000.0))
            elif t == 'decimal' and arg is not None and not isinstance(arg, Decimal):
                res.append(Decimal(arg))
            else:
                res.append(arg)
        return res

    def _to_milliseconds(self, dt):
        return int((time.mktime(dt.timetuple()) + dt.microsecond/1000000.0) * 1000)

    def _handle_output(self, arg):
        from datetime import datetime
        from decimal import Decimal

        t = self.to_type

        if t == 'datetime' and isinstance(arg, datetime):
            return self._to_milliseconds(arg)
        elif t == 'string' and isinstance(arg, Decimal):
            return str(arg)
        else:
            return arg

    def new_buffer(self):
        return self.f.buffer()

    def iterate(self, buffer, *args):
        args = self._handle_input(args)
        self.f(buffer, *args)

    def merge(self, buffer, pbuffer):
        self.f.merge(buffer, pbuffer)

    def terminate(self, buffer):
        res = self.f.getvalue(buffer)
        return self._handle_output(res)
'''


def gen_udf(expr, func_cls_name=None):
    func_to_udfs = OrderedDict()
    func_to_resources = OrderedDict()

    for node in expr.traverse(unique=True):
        func = getattr(node, 'func', None)
        if func is None:
            continue
        if isinstance(func, six.string_types):
            continue
        if func in func_to_udfs:
            func = make_copy(func)
            node.func = func

        resources = []
        collection_idx = 0
        if hasattr(node, '_resources') and node._resources:
            for res in node._resources :
                if isinstance(res, FileResource):
                    tp = 'file'
                    name = res.name
                    fields = None
                    create = False
                    table_name = None
                elif isinstance(res, TableResource):
                    tp = 'table'
                    name = res.name
                    fields = [col.name for col in res.get_source_table().schema.columns]
                    create = False
                    table_name = None
                else:
                    res = node._collection_resources[collection_idx]
                    collection_idx += 1

                    tp = 'table'
                    name = 'tmp_pyodps_resource_%s' % (uuid.uuid4())
                    fields = res.schema.names
                    create = True
                    table_name = get_executed_collection_table_name(res)
                resources.append((tp, name, fields, create, table_name))

        if isinstance(node, MappedExpr):
            names_str = ''
            if isinstance(node, MappedExpr) and node._multiple and \
                    all(f.name is not None for f in node.inputs):
                names_str = ','.join(f.name for f in node.inputs)
            from_type = ','.join(df_type_to_odps_type(t).name for t in node.input_types)
            raw_from_type = ','.join(df_type_to_odps_type(t).name for t in node.raw_input_types)
            func_to_udfs[func] = UDF_TMPL % {
                'cloudpickle': CLOUD_PICKLE,
                'raw_from_type': raw_from_type,
                'from_type':  from_type,
                'to_type': df_type_to_odps_type(node.data_type).name,
                'func_cls_name': func_cls_name,
                'func_str': to_str(base64.b64encode(dumps(func))),
                'func_args_str': to_str(base64.b64encode(dumps(node._func_args))),
                'func_kwargs_str': to_str(base64.b64encode(dumps(node._func_kwargs))),
                'names_str': names_str,
                'resources': to_str(base64.b64encode(dumps([r[:3] for r in resources])))
            }
            if resources:
                func_to_resources[func] = resources
        elif isinstance(node, (RowAppliedCollectionExpr, GroupbyAppliedCollectionExpr)):
            names_str = ','.join(f.name for f in node.fields)
            from_type = ','.join(df_type_to_odps_type(t).name for t in node.input_types)
            raw_from_type = ','.join(df_type_to_odps_type(t).name for t in node.raw_input_types)
            to_type = ','.join(df_type_to_odps_type(t).name for t in node.schema.types)
            func_to_udfs[func] = UDTF_TMPL % {
                'cloudpickle': CLOUD_PICKLE,
                'raw_from_type': raw_from_type,
                'from_type': from_type,
                'to_type': to_type,
                'func_cls_name': func_cls_name,
                'func_str': to_str(base64.b64encode(dumps(func))),
                'func_args_str': to_str(base64.b64encode(dumps(node._func_args))),
                'func_kwargs_str': to_str(base64.b64encode(dumps(node._func_kwargs))),
                'close_func_str': to_str(base64.b64encode(dumps(getattr(node, '_close_func', None)))),
                'names_str': names_str,
                'resources': to_str(base64.b64encode(dumps([r[:3] for r in resources])))
            }
            if resources:
                func_to_resources[func] = resources
        elif isinstance(node, (Aggregation, GroupedAggregation)):
            func_to_udfs[func] = UDAF_TMPL % {
                'cloudpickle': CLOUD_PICKLE,
                'raw_from_type': df_type_to_odps_type(node.raw_input.dtype).name,
                'from_type': df_type_to_odps_type(node.input.dtype).name,
                'to_type': df_type_to_odps_type(node.dtype).name,
                'func_cls_name': func_cls_name,
                'func_str': to_str(base64.b64encode(dumps(func))),
                'func_args_str': to_str(base64.b64encode(dumps(node._func_args))),
                'func_kwargs_str': to_str(base64.b64encode(dumps(node._func_kwargs))),
                'resources': to_str(base64.b64encode(dumps([r[:3] for r in resources])))
            }
            if resources:
                func_to_resources[func] = resources

    return func_to_udfs, func_to_resources
