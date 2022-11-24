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

import base64
import os
import sys
import platform
import uuid

from .types import df_type_to_odps_type
from ...expr.collections import RowAppliedCollectionExpr
from ...expr.element import MappedExpr
from ...expr.reduction import Aggregation, GroupedAggregation
from ...expr.utils import get_executed_collection_project_table_name
from ...utils import make_copy
from ....config import options
from ....lib import cloudpickle
from ....compat import OrderedDict, six
from ....models import FileResource, TableResource, ArchiveResource
from ....utils import to_str

dirname = os.path.dirname(os.path.abspath(cloudpickle.__file__))
CLOUD_PICKLE_FILE = os.path.join(dirname, 'cloudpickle.py')
with open(CLOUD_PICKLE_FILE) as f:
    CLOUD_PICKLE = f.read()

IMPORT_FILE = os.path.join(dirname, 'importer.py')
with open(IMPORT_FILE) as f:
    MEM_IMPORT = f.read()

CLIENT_IMPL = '(%d, %d, "%s")' % (sys.version_info[0],
                                  sys.version_info[1],
                                  platform.python_implementation().lower())

X_NAMED_TUPLE_FILE = os.path.join(dirname, 'xnamedtuple.py')
with open(X_NAMED_TUPLE_FILE) as f:
    X_NAMED_TUPLE = f.read()


UDF_TMPL_HEADER = '''\
%(cloudpickle)s
%(memimport)s

%(xnamedtuple)s

try:
    # workaround for character not in range error
    import _strptime
except:
    pass

import base64
import inspect
import time
import os
import sys

from odps.udf import annotate
from odps.distcache import get_cache_file, get_cache_table, get_cache_archive

PY2 = sys.version_info[0] == 2

try:
    from odps.distcache import get_cache_archive_filenames
except ImportError:
    def get_cache_archive_filenames(name, relative_path='.'):
        from odps.distcache import WORK_DIR, DistributedCacheError

        def _is_parent(parent, child):
            return parent == child or child.startswith(parent + '/')

        if os.path.split(name)[0] != '':
            raise DistributedCacheError("Invalid resource name: " + name)
        ret_files = []

        # find the real resource path to avoid the symbol link in inner system
        resourcepath = os.path.realpath(os.path.join(WORK_DIR, name))

        # use realpath == abspath to check the symbol link
        dirpath = os.path.join(resourcepath, relative_path)
        if not os.path.exists(dirpath):
            raise DistributedCacheError("Invalid relative path, file not exists: " + relative_path)

        if os.path.realpath(dirpath) != os.path.abspath(dirpath):
            raise DistributedCacheError("Invalid relative path, relative path contains symlink: " + relative_path)

        if not _is_parent(resourcepath, dirpath):
            raise DistributedCacheError("Invalid relative path, path not correct in archive: " + relative_path)

        if not os.path.isdir(dirpath):
            return [dirpath]

        for root, dirs, files in os.walk(dirpath):
            for f in dirs:
                filename = os.path.join(root, f)
                if os.path.islink(filename):
                    relativename = os.path.relpath(filename, resourcepath)
                    raise DistributedCacheError("Invalid relative path, relative path contains symlink: " + relativename)

            for f in files:
                filename = os.path.join(root, f)
                if os.path.islink(filename):
                    relativename = os.path.relpath(filename, resourcepath)
                    raise DistributedCacheError("Invalid relative path, relative path contains symlink: " + relativename)
                ret_files.append(filename)
        return ret_files
        
    
def get_cache_archive_data(name, relative_path='.'):
    try:
        return [os.path.normpath(f) for f in get_cache_archive_filenames(name, relative_path)]
    except RuntimeError:
        return dict((os.path.normpath(fo.name), fo) for fo in get_cache_archive(name, relative_path))


class UnbufferedStream(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        if attr != 'stream':
            return getattr(self.stream, attr)
        else:
            return object.__getattribute__(self, 'stream')

    def __setattr__(self, attr, value):
        if attr != 'stream':
            return setattr(self.stream, attr, value)
        else:
            return object.__setattr__(self, 'stream', value)

sys.stdout = UnbufferedStream(sys.stdout)


try:
    import socket
except ImportError:
    class MockSocketModule(object):
        _GLOBAL_DEFAULT_TIMEOUT = object()
        def __getattr__(self, item):
            raise AttributeError('Accessing attribute `{0}` of module `socket` is prohibited by sandbox.'.format(item))
    sys.modules['socket'] = MockSocketModule()


def gen_resource_data(fields, tb):
    named_args = xnamedtuple('NamedArgs', fields)
    for args in tb:
        yield named_args(*args)


def read_lib(lib, f):
    if isinstance(f, (list, dict)):
        return f
    if lib.endswith('.zip') or lib.endswith('.egg') or lib.endswith('.whl'):
        if PY2 and hasattr(f, "read"):
            return zipfile.ZipFile(f)
        else:
            return zipfile.ZipFile(f.name)
    if lib.endswith('.tar') or lib.endswith('.tar.gz') or lib.endswith('.tar.bz2'):
        from io import BytesIO
        if lib.endswith('.tar'):
            mode = 'r'
        else:
            mode = 'r:gz' if lib.endswith('.tar.gz') else 'r:bz2'
        if PY2 and hasattr(f, "read"):
            return tarfile.open(name='', fileobj=f, mode=mode)
        else:
            return tarfile.open(name=f.name, mode=mode)
    if lib.endswith('.py'):
        if PY2 and hasattr(f, "read"):
            return {f.name: f}
        else:
            return {f.name: open(f.name, 'rb')}

    raise ValueError(
        'Unknown library type which should be one of zip(egg, wheel), tar, or tar.gz')


# Use this method to make testThirdPartyLibraries happy
np_generic = None
def load_np_generic():
    global np_generic

    try:
        from numpy import generic
        np_generic = generic
    except ImportError:
        class PseudoNpGeneric(object):
            pass
        np_generic = PseudoNpGeneric
''' % {
    'cloudpickle': CLOUD_PICKLE,
    'memimport': MEM_IMPORT,
    'xnamedtuple': X_NAMED_TUPLE,
}


UDF_TMPL = '''

@annotate('%(from_type)s->%(to_type)s')
class %(func_cls_name)s(object):
    def __init__(self):
        unpickler_kw = dict(impl=%(implementation)s, dump_code=%(dump_code)s)
        rs = loads(base64.b64decode('%(resources)s'), **unpickler_kw)
        resources = []
        for t, n, fields in rs:
            if t == 'file':
                resources.append(get_cache_file(str(n)))
            elif t == 'archive':
                resources.append(get_cache_archive(str(n)))
            else:
                tb = get_cache_table(str(n))
                if fields:
                    tb = gen_resource_data(fields, tb)
                resources.append(tb)
                
        libraries = (l for l in '%(libraries)s'.split(',') if len(l) > 0)
        files = []
        for lib in libraries:
            if lib.startswith('a:'):
                lib = lib[2:]
                f = get_cache_archive_data(lib)
            else:
                f = get_cache_file(lib)
            files.append(read_lib(lib, f))
        sys.meta_path.append(CompressImporter(*files, supersede=%(supersede_libraries)r))

        load_np_generic()

        encoded = '%(func_str)s'
        f_str = base64.b64decode(encoded)
        self.f = loads(f_str, **unpickler_kw)

        if inspect.isclass(self.f):
            if resources:
                self.f = self.f(resources)
            else:
                self.f = self.f()
        else:
            if resources:
                self.f = self.f(resources)

        self.names = tuple(it for it in '%(names_str)s'.split(',') if it)
        if self.names:
            self.named_args = xnamedtuple('NamedArgs', self.names)

        encoded_func_args = '%(func_args_str)s'
        func_args_str = base64.b64decode(encoded_func_args)
        self.args = loads(func_args_str, **unpickler_kw) or tuple()

        encoded_func_kwargs = '%(func_kwargs_str)s'
        func_kwargs_str = base64.b64decode(encoded_func_kwargs)
        self.kwargs = loads(func_kwargs_str, **unpickler_kw) or dict()

        self.from_types = '%(raw_from_type)s'.split(',')
        self.to_type = '%(to_type)s'

    def _handle_input(self, args):
        from datetime import datetime
        from decimal import Decimal

        res = []
        for t, arg in zip(self.from_types, args):
            if PY2 and t == 'datetime' and arg is not None and not isinstance(arg, datetime):
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

        if PY2 and t == 'datetime' and isinstance(arg, datetime):
            if isinstance(arg, np_generic):
                arg = arg.item()
            return self._to_milliseconds(arg)
        elif t == 'string' and isinstance(arg, Decimal):
            return str(arg)
        else:
            if isinstance(arg, np_generic):
                arg = arg.item()
            return arg

    def evaluate(self, %(input_args)s):
        args = %(input_args)s,
        args = self._handle_input(args)
        if not self.names:
            args = tuple(args) + tuple(self.args)
            res = self.f(*args, **self.kwargs)
            return self._handle_output(res)
        else:
            res = self.f(self.named_args(*args), *self.args, **self.kwargs)
            return self._handle_output(res)
'''


UDTF_TMPL = '''
import functools
from odps.udf import BaseUDTF

if PY2:
    string_type = unicode
    byte_type = str
else:
    string_type = str
    byte_type = (bytes, bytearray)

@annotate('%(from_type)s->%(to_type)s')
class %(func_cls_name)s(BaseUDTF):
    def __init__(self):
        unpickler_kw = dict(impl=%(implementation)s, dump_code=%(dump_code)s)
        rs = loads(base64.b64decode('%(resources)s'), **unpickler_kw)
        resources = []
        for t, n, fields in rs:
            if t == 'file':
                resources.append(get_cache_file(str(n)))
            elif t == 'archive':
                resources.append(get_cache_archive(str(n)))
            else:
                tb = get_cache_table(str(n))
                if fields:
                    tb = gen_resource_data(fields, tb)
                resources.append(tb)

        libraries = (l for l in '%(libraries)s'.split(',') if len(l) > 0)
        files = []
        for lib in libraries:
            if lib.startswith('a:'):
                lib = lib[2:]
                f = get_cache_archive_data(lib)
            else:
                f = get_cache_file(lib)
            files.append(read_lib(lib, f))
        sys.meta_path.append(CompressImporter(*files, supersede=%(supersede_libraries)r))

        load_np_generic()

        encoded = '%(func_str)s'
        f_str = base64.b64decode(encoded)
        self.f = loads(f_str, **unpickler_kw)

        if inspect.isclass(self.f):
            if not resources:
                self.f = self.f()
            else:
                self.f = self.f(resources)
            self.is_f_generator = inspect.isgeneratorfunction(self.f.__call__)
            self.close_f = getattr(self.f, 'close', None)
            self.is_close_f_generator = inspect.isgeneratorfunction(self.close_f)
        else:
            if resources:
                self.f = self.f(resources)

            if isinstance(self.f, functools.partial):
                self.is_f_generator = inspect.isgeneratorfunction(self.f.func)
            else:
                self.is_f_generator = inspect.isgeneratorfunction(self.f)
            self.close_f = None
            self.is_close_f_generator = False

        encoded_func_args = '%(func_args_str)s'
        func_args_str = base64.b64decode(encoded_func_args)
        self.args = loads(func_args_str, **unpickler_kw) or tuple()

        encoded_func_kwargs = '%(func_kwargs_str)s'
        func_kwargs_str = base64.b64decode(encoded_func_kwargs)
        self.kwargs = loads(func_kwargs_str, **unpickler_kw) or dict()

        self.names = tuple(it for it in '%(names_str)s'.split(',') if it)
        if self.names:
            self.name_args = xnamedtuple('NamedArgs', self.names)

        self.from_types = '%(raw_from_type)s'.split(',')
        self.to_types = '%(to_type)s'.split(',')

    def _handle_input(self, args):
        from datetime import datetime
        from decimal import Decimal

        res = []
        for t, arg in zip(self.from_types, args):
            if PY2 and t == 'datetime' and arg is not None and not isinstance(arg, datetime):
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
            raise ValueError('Function output size should be ' + str(len(self.to_types))
                + ', got ' + str(args))

        res = []
        for t, arg in zip(self.to_types, args):
            if PY2 and t == 'datetime' and isinstance(arg, datetime):
                if isinstance(arg, np_generic):
                    arg = arg.item()
                res.append(self._to_milliseconds(arg))
            elif t == 'string' and isinstance(arg, Decimal):
                res.append(str(arg))
            elif PY2 and t == 'string' and isinstance(arg, string_type):
                res.append(arg.encode('utf-8'))
            elif PY3 and t == 'string' and isinstance(arg, byte_type):
                res.append(arg.decode('utf-8'))
            else:
                if isinstance(arg, np_generic):
                    arg = arg.item()
                res.append(arg)
        return res

    def process(self, %(input_args)s):
        args = %(input_args)s,
        args = self._handle_input(args)
        if not self.names:
            args = tuple(args) + tuple(self.args)
        else:
            args = (self.name_args(*args), ) + tuple(self.args)

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

UDAF_TMPL = '''
from odps.udf import BaseUDAF

@annotate('%(from_type)s->%(to_type)s')
class %(func_cls_name)s(BaseUDAF):
    def __init__(self):
        unpickler_kw = dict(impl=%(implementation)s, dump_code=%(dump_code)s)
        rs = loads(base64.b64decode('%(resources)s'), **unpickler_kw)
        resources = []
        for t, n, fields in rs:
            if t == 'file':
                resources.append(get_cache_file(str(n)))
            elif t == 'archive':
                resources.append(get_cache_archive(str(n)))
            else:
                tb = get_cache_table(str(n))
                if fields:
                    tb = gen_resource_data(fields, tb)
                resources.append(tb)
                
        libraries = (l for l in '%(libraries)s'.split(',') if len(l) > 0)
        files = []
        for lib in libraries:
            if lib.startswith('a:'):
                lib = lib[2:]
                f = get_cache_archive_data(lib)
            else:
                f = get_cache_file(lib)
            files.append(read_lib(lib, f))
        sys.meta_path.append(CompressImporter(*files, supersede=%(supersede_libraries)r))
        
        load_np_generic()

        encoded_func_args = '%(func_args_str)s'
        func_args_str = base64.b64decode(encoded_func_args)
        args = loads(func_args_str, **unpickler_kw) or tuple()

        encoded_func_kwargs = '%(func_kwargs_str)s'
        func_kwargs_str = base64.b64decode(encoded_func_kwargs)
        kwargs = loads(func_kwargs_str, **unpickler_kw) or dict()

        encoded = '%(func_str)s'
        f_str = base64.b64decode(encoded)
        agg = loads(f_str, **unpickler_kw)
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
            if PY2 and t == 'datetime' and arg is not None and not isinstance(arg, datetime):
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

        if PY2 and t == 'datetime' and isinstance(arg, datetime):
            if isinstance(arg, np_generic):
                arg = arg.item()
            return self._to_milliseconds(arg)
        elif t == 'string' and isinstance(arg, Decimal):
            return str(arg)
        else:
            if isinstance(arg, np_generic):
                arg = arg.item()
            return arg

    def new_buffer(self):
        return self.f.buffer()

    def iterate(self, buffer, %(input_args)s):
        args = %(input_args)s,
        args = self._handle_input(args)
        self.f(buffer, *args)

    def merge(self, buffer, pbuffer):
        self.f.merge(buffer, pbuffer)

    def terminate(self, buffer):
        res = self.f.getvalue(buffer)
        return self._handle_output(res)
'''


def _gen_map_udf(node, func_cls_name, libraries, func, resources,
                 func_to_udfs, func_to_resources, func_params):
    names_str = ''
    if isinstance(node, MappedExpr) and node._multiple and \
            all(f.name is not None for f in node.inputs):
        names_str = ','.join(f.name for f in node.inputs)
    from_type = ','.join(df_type_to_odps_type(t).name for t in node.input_types)
    to_type = df_type_to_odps_type(node.dtype).name
    raw_from_type = ','.join(df_type_to_odps_type(t).name for t in node.raw_input_types)
    func_args_str = to_str(
        base64.b64encode(cloudpickle.dumps(node._func_args, dump_code=options.df.dump_udf)))
    func_kwargs_str = to_str(
        base64.b64encode(cloudpickle.dumps(node._func_kwargs, dump_code=options.df.dump_udf)))

    key = (from_type, to_type, func, tuple(resources), names_str, func_args_str, func_kwargs_str)
    if key in func_params:
        return
    else:
        if func in func_to_udfs:
            func = make_copy(func)
            node.func = func

        func_params.add(key)

    func_to_udfs[func] = UDF_TMPL_HEADER + UDF_TMPL % {
        'raw_from_type': raw_from_type,
        'from_type': from_type,
        'to_type': to_type,
        'func_cls_name': func_cls_name,
        'func_str': to_str(base64.b64encode(cloudpickle.dumps(func, dump_code=options.df.dump_udf))),
        'func_args_str': func_args_str,
        'func_kwargs_str': func_kwargs_str,
        'names_str': names_str,
        'resources': to_str(
            base64.b64encode(cloudpickle.dumps([r[:3] for r in resources], dump_code=options.df.dump_udf))),
        'implementation': CLIENT_IMPL,
        'dump_code': options.df.dump_udf,
        'input_args': ', '.join('arg{0}'.format(i) for i in range(len(node.input_types))),
        'libraries': ','.join(libraries if libraries is not None else []),
        'supersede_libraries': options.df.supersede_libraries,
    }
    if resources:
        func_to_resources[func] = resources


def _gen_apply_udf(node, func_cls_name, libraries, func, resources,
                   func_to_udfs, func_to_resources, func_params):
    names_str = ','.join(f.name for f in node.fields)
    from_type = ','.join(df_type_to_odps_type(t).name for t in node.input_types)
    raw_from_type = ','.join(df_type_to_odps_type(t).name for t in node.raw_input_types)
    to_type = ','.join(df_type_to_odps_type(t).name for t in node.schema.types)
    func_args_str = to_str(
        base64.b64encode(cloudpickle.dumps(node._func_args, dump_code=options.df.dump_udf)))
    func_kwargs_str = to_str(
        base64.b64encode(cloudpickle.dumps(node._func_kwargs, dump_code=options.df.dump_udf)))

    key = (from_type, to_type, func, tuple(resources), names_str, func_args_str, func_kwargs_str)
    if key in func_params:
        return
    else:
        if func in func_to_udfs:
            func = make_copy(func)
            node.func = func

        func_params.add(key)

    func_to_udfs[func] = UDF_TMPL_HEADER + UDTF_TMPL % {
        'raw_from_type': raw_from_type,
        'from_type': from_type,
        'to_type': to_type,
        'func_cls_name': func_cls_name,
        'func_str': to_str(base64.b64encode(cloudpickle.dumps(func, dump_code=options.df.dump_udf))),
        'func_args_str': func_args_str,
        'func_kwargs_str': func_kwargs_str,
        'close_func_str': to_str(
            base64.b64encode(cloudpickle.dumps(getattr(node, '_close_func', None), dump_code=options.df.dump_udf))),
        'names_str': names_str,
        'resources': to_str(base64.b64encode(cloudpickle.dumps([r[:3] for r in resources]))),
        'implementation': CLIENT_IMPL,
        'dump_code': options.df.dump_udf,
        'input_args': ', '.join('arg{0}'.format(i) for i in range(len(node.input_types))),
        'libraries': ','.join(libraries if libraries is not None else []),
        'supersede_libraries': options.df.supersede_libraries,
    }
    if resources:
        func_to_resources[func] = resources


def _gen_agg_udf(node, func_cls_name, libraries, func, resources,
                 func_to_udfs, func_to_resources, func_params):
    from_type = ','.join(df_type_to_odps_type(t).name for t in node.input_types)
    raw_from_type = ','.join(df_type_to_odps_type(t).name for t in node.raw_input_types)
    to_type = df_type_to_odps_type(node.dtype).name
    func_args_str = to_str(
        base64.b64encode(cloudpickle.dumps(node._func_args, dump_code=options.df.dump_udf)))
    func_kwargs_str = to_str(
        base64.b64encode(cloudpickle.dumps(node._func_kwargs, dump_code=options.df.dump_udf)))

    key = (from_type, to_type, func, tuple(resources), func_args_str, func_kwargs_str)
    if key in func_params:
        return
    else:
        if func in func_to_udfs:
            func = make_copy(func)
            node.func = func

        func_params.add(key)

    func_to_udfs[func] = UDF_TMPL_HEADER + UDAF_TMPL % {
        'raw_from_type': raw_from_type,
        'from_type': from_type,
        'to_type': to_type,
        'func_cls_name': func_cls_name,
        'func_str': to_str(base64.b64encode(cloudpickle.dumps(func, dump_code=options.df.dump_udf))),
        'func_args_str': func_args_str,
        'func_kwargs_str': func_kwargs_str,
        'resources': to_str(
            base64.b64encode(cloudpickle.dumps([r[:3] for r in resources], dump_code=options.df.dump_udf))),
        'implementation': CLIENT_IMPL,
        'dump_code': options.df.dump_udf,
        'input_args': ', '.join('arg{0}'.format(i) for i in range(len(node.input_types))),
        'libraries': ','.join(libraries if libraries is not None else []),
        'supersede_libraries': options.df.supersede_libraries,
    }
    if resources:
        func_to_resources[func] = resources


def gen_udf(expr, func_cls_name=None, libraries=None):
    func_to_udfs = OrderedDict()
    func_to_resources = OrderedDict()
    func_params = set()
    if libraries is not None:
        def _get_library_name(res):
            if isinstance(res, six.string_types):
                return res
            elif isinstance(res, ArchiveResource):
                return 'a:' + res.name
            else:
                return res.name

        libraries = [_get_library_name(lib) for lib in libraries]

    for node in expr.traverse(unique=True):
        func = getattr(node, 'func', None)
        if func is None:
            continue
        if isinstance(func, six.string_types):
            continue

        resources = []
        collection_idx = 0
        if hasattr(node, '_resources') and node._resources:
            for res in node._resources:
                if isinstance(res, ArchiveResource):
                    tp = 'archive'
                    name = res.name
                    fields = None
                    create = False
                    table_name = None
                elif isinstance(res, FileResource):
                    tp = 'file'
                    name = res.name
                    fields = None
                    create = False
                    table_name = None
                elif isinstance(res, TableResource):
                    tp = 'table'
                    name = res.name
                    fields = tuple(col.name for col in res.get_source_table().schema.simple_columns)
                    create = False
                    table_name = None
                else:
                    res = node._collection_resources[collection_idx]
                    collection_idx += 1

                    tp = 'table'
                    name = 'tmp_pyodps_resource_%s' % (uuid.uuid4())
                    fields = tuple(res.schema.names)
                    create = True
                    table_name = get_executed_collection_project_table_name(res)
                resources.append((tp, name, fields, create, table_name))

        if isinstance(node, MappedExpr):
            _gen_map_udf(node, func_cls_name, libraries, func, resources,
                         func_to_udfs, func_to_resources, func_params)
        elif isinstance(node, RowAppliedCollectionExpr):
            _gen_apply_udf(node, func_cls_name, libraries, func, resources,
                           func_to_udfs, func_to_resources, func_params)
        elif isinstance(node, (Aggregation, GroupedAggregation)):
            _gen_agg_udf(node, func_cls_name, libraries, func, resources,
                         func_to_udfs, func_to_resources, func_params)

    return func_to_udfs, func_to_resources
