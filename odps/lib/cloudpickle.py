#!/usr/bin/env python
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

"""
This class is defined to override standard pickle functionality

The goals of it follow:
-Serialize lambdas and nested functions to compiled byte code
-Deal with main module correctly
-Deal with other non-serializable objects

It does not include an unpickler, as standard python unpickling suffices.

This module was extracted from the `cloud` package, developed by `PiCloud, Inc.
<http://www.picloud.com>`_.

Copyright (c) 2012, Regents of the University of California.
Copyright (c) 2009 `PiCloud, Inc. <http://www.picloud.com>`_.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of California, Berkeley nor the
      names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import print_function
from __future__ import absolute_import

import operator
import os
import io
import pickle
import struct
import sys
import types
from functools import partial
import itertools
import dis
import traceback
import platform
import opcode

if sys.version < '3':
    from pickle import Pickler, Unpickler
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
    PY3 = False
else:
    types.ClassType = type
    from pickle import _Pickler as Pickler
    from io import BytesIO as StringIO
    PY3 = True

PY27 = not PY3 and sys.version_info[1] == 7

PYPY = platform.python_implementation().lower() == 'pypy'
JYTHON = platform.python_implementation().lower() == 'jython'


#relevant opcodes
BUILD_CLASS = opcode.opmap.get('BUILD_CLASS')
BUILD_LIST_FROM_ARG_PYPY = 203
BUILD_LIST = opcode.opmap['BUILD_LIST']
BUILD_MAP = opcode.opmap.get('BUILD_MAP')
CALL_FUNCTION = opcode.opmap['CALL_FUNCTION']
CALL_METHOD_PYPY = 202
COMPARE_OP = opcode.opmap.get('COMPARE_OP')
DUP_TOP = dis.opmap.get('DUP_TOP')
DUP_TOPX = dis.opmap.get('DUP_TOPX')
EXTENDED_ARG = dis.EXTENDED_ARG
EXTENDED_ARG_PY3 = 144
FOR_ITER = opcode.opmap['FOR_ITER']
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
IMPORT_FROM = opcode.opmap.get('IMPORT_FROM')
IMPORT_NAME = opcode.opmap.get('IMPORT_NAME')
JUMP_ABSOLUTE = opcode.opmap['JUMP_ABSOLUTE']
JUMP_FORWARD = opcode.opmap['JUMP_FORWARD']
JUMP_IF_FALSE_OR_POP = opcode.opmap.get('JUMP_IF_FALSE_OR_POP')
JUMP_IF_NOT_DEBUG_PYPY = 204
JUMP_IF_TRUE_OR_POP = opcode.opmap.get('JUMP_IF_TRUE_OR_POP')
LIST_APPEND = opcode.opmap['LIST_APPEND']
LIST_APPEND_PY26 = 18
LOAD_ATTR = opcode.opmap['LOAD_ATTR']
LOAD_BUILD_CLASS_PY3 = 71
LOAD_CONST = opcode.opmap['LOAD_CONST']
LOAD_FAST = opcode.opmap['LOAD_FAST']
LOAD_LOCALS = opcode.opmap.get('LOAD_LOCALS')
LOOKUP_METHOD_PYPY = 201
MAKE_CLOSURE = opcode.opmap.get('MAKE_CLOSURE')
MAKE_FUNCTION = opcode.opmap['MAKE_FUNCTION']
NOP = opcode.opmap['NOP']
POP_JUMP_IF_TRUE = opcode.opmap.get('POP_JUMP_IF_TRUE')
POP_JUMP_IF_FALSE = opcode.opmap.get('POP_JUMP_IF_FALSE')
POP_TOP = opcode.opmap.get('POP_TOP')
RETURN_VALUE = opcode.opmap['RETURN_VALUE']
ROT_TWO = opcode.opmap['ROT_TWO']
ROT_THREE = opcode.opmap.get('ROT_THREE')
ROT_FOUR = opcode.opmap.get('ROT_FOUR')
STORE_NAME = opcode.opmap.get('STORE_NAME')

DELETE_GLOBAL = dis.opname.index('DELETE_GLOBAL')
LOAD_GLOBAL = dis.opname.index('LOAD_GLOBAL')
STORE_GLOBAL = dis.opname.index('STORE_GLOBAL')
GLOBAL_OPS = [STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL]


def islambda(func):
    return getattr(func, '__name__') == '<lambda>'


_BUILTIN_TYPE_NAMES = {}
for k, v in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k


def _builtin_type(name):
    return getattr(types, name)


def _extract_code_args(obj):
    if PY3:
        return (
            obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
            obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames,
            obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars,
            obj.co_cellvars
        )
    else:
        return (
            obj.co_argcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code,
            obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name,
            obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars
        )


class CloudPickler(Pickler):

    dispatch = Pickler.dispatch.copy()

    def __init__(self, file, protocol=None, dump_code=False):
        Pickler.__init__(self, file, protocol)
        # set of modules to unpickle
        self.modules = set()
        # map ids to dictionary. used to ensure that functions can share global env
        self.globals_ref = {}
        self.dump_code = dump_code

    def dump(self, obj):
        self.inject_addons()
        try:
            return Pickler.dump(self, obj)
        except RuntimeError as e:
            if 'recursion' in e.args[0]:
                msg = """Could not pickle object as excessively deep recursion required."""
                raise pickle.PicklingError(msg)

    def save_memoryview(self, obj):
        """Fallback to save_string"""
        Pickler.save_string(self, str(obj))

    def save_buffer(self, obj):
        """Fallback to save_string"""
        Pickler.save_string(self,str(obj))
    if PY3:
        dispatch[memoryview] = save_memoryview
    else:
        dispatch[buffer] = save_buffer

    def save_unsupported(self, obj):
        raise pickle.PicklingError("Cannot pickle objects of type %s" % type(obj))
    dispatch[types.GeneratorType] = save_unsupported

    # itertools objects do not pickle!
    for v in itertools.__dict__.values():
        if type(v) is type:
            dispatch[v] = save_unsupported

    def save_module(self, obj):
        """
        Save a module as an import
        """
        self.modules.add(obj)
        self.save_reduce(subimport, (obj.__name__,), obj=obj)
    dispatch[types.ModuleType] = save_module

    def save_codeobject(self, obj):
        """
        Save a code object
        """
        args = _extract_code_args(obj)
        if self.dump_code:
            print(obj.co_name)
            dis.dis(obj.co_code)
        self.save_reduce(types.CodeType, args, obj=obj)
    dispatch[types.CodeType] = save_codeobject

    def save_function(self, obj, name=None):
        """ Registered with the dispatch to handle all function types.

        Determines what kind of function obj is (e.g. lambda, defined at
        interactive prompt, etc) and handles the pickling appropriately.
        """
        write = self.write

        if name is None:
            name = obj.__name__
        modname = pickle.whichmodule(obj, name)
        # print('which gives %s %s %s' % (modname, obj, name))
        try:
            themodule = sys.modules[modname]
        except KeyError:
            # eval'd items such as namedtuple give invalid items for their function __module__
            modname = '__main__'

        if modname == '__main__':
            themodule = None

        if themodule:
            self.modules.add(themodule)
            if getattr(themodule, name, None) is obj:
                return self.save_global(obj, name)

        # if func is lambda, def'ed at prompt, is in main, or is nested, then
        # we'll pickle the actual function object rather than simply saving a
        # reference (as is done in default pickler), via save_function_tuple.
        if islambda(obj) or obj.__code__.co_filename == '<stdin>' or themodule is None:
            #print("save global", islambda(obj), obj.__code__.co_filename, modname, themodule)
            self.save_function_tuple(obj)
            return
        else:
            # func is nested
            klass = getattr(themodule, name, None)
            if klass is None or klass is not obj:
                self.save_function_tuple(obj)
                return

        if obj.__dict__:
            # essentially save_reduce, but workaround needed to avoid recursion
            self.save(_restore_attr)
            write(pickle.MARK + pickle.GLOBAL + modname + '\n' + name + '\n')
            self.memoize(obj)
            self.save(obj.__dict__)
            write(pickle.TUPLE + pickle.REDUCE)
        else:
            write(pickle.GLOBAL + modname + '\n' + name + '\n')
            self.memoize(obj)
    dispatch[types.FunctionType] = save_function

    def save_function_tuple(self, func):
        """  Pickles an actual func object.

        A func comprises: code, globals, defaults, closure, and dict.  We
        extract and save these, injecting reducing functions at certain points
        to recreate the func object.  Keep in mind that some of these pieces
        can contain a ref to the func itself.  Thus, a naive save on these
        pieces could trigger an infinite loop of save's.  To get around that,
        we first create a skeleton func object using just the code (this is
        safe, since this won't contain a ref to the func), and memoize it as
        soon as it's created.  The other stuff can then be filled in later.
        """
        save = self.save
        write = self.write

        code, f_globals, defaults, closure, dct, base_globals = self.extract_func_data(func)

        save(_fill_function)  # skeleton function updater
        write(pickle.MARK)    # beginning of tuple that _fill_function expects

        # create a skeleton function object and memoize it
        save(_make_skel_func)
        save((code, closure, base_globals))
        write(pickle.REDUCE)
        self.memoize(func)

        # save the rest of the func data needed by _fill_function
        save(f_globals)
        save(defaults)
        save(dct)
        write(pickle.TUPLE)
        write(pickle.REDUCE)  # applies _fill_function on the tuple

    @staticmethod
    def extract_code_globals(co):
        """
        Find all globals names read or written to by codeblock co
        """
        code = co.co_code
        if not PY3:
            code = [ord(c) for c in code]
        names = co.co_names
        out_names = set()

        n = len(code)
        i = 0
        extended_arg = 0
        while i < n:
            op = code[i]

            i += 1
            if op >= HAVE_ARGUMENT:
                oparg = code[i] + code[i+1] * 256 + extended_arg
                extended_arg = 0
                i += 2
                if op == EXTENDED_ARG:
                    extended_arg = oparg*65536
                if op in GLOBAL_OPS:
                    out_names.add(names[oparg])

        # see if nested function have any global refs
        if co.co_consts:
            for const in co.co_consts:
                if type(const) is types.CodeType:
                    out_names |= CloudPickler.extract_code_globals(const)

        return out_names

    def extract_func_data(self, func):
        """
        Turn the function into a tuple of data necessary to recreate it:
            code, globals, defaults, closure, dict
        """
        code = func.__code__

        # extract all global ref's
        func_global_refs = self.extract_code_globals(code)

        # process all variables referenced by global environment
        f_globals = {}
        for var in func_global_refs:
            if var in func.__globals__:
                f_globals[var] = func.__globals__[var]

        # defaults requires no processing
        defaults = func.__defaults__

        # process closure
        closure = [c.cell_contents for c in func.__closure__] if func.__closure__ else []

        # save the dict
        dct = func.__dict__

        base_globals = self.globals_ref.get(id(func.__globals__), {})
        self.globals_ref[id(func.__globals__)] = base_globals

        return (code, f_globals, defaults, closure, dct, base_globals)

    def save_builtin_function(self, obj):
        if obj.__module__ is "__builtin__":
            return self.save_global(obj)
        return self.save_function(obj)
    dispatch[types.BuiltinFunctionType] = save_builtin_function

    def save_global(self, obj, name=None, pack=struct.pack):
        if obj.__module__ == "__builtin__" or obj.__module__ == "builtins":
            if obj in _BUILTIN_TYPE_NAMES:
                return self.save_reduce(_builtin_type, (_BUILTIN_TYPE_NAMES[obj],), obj=obj)

        if name is None:
            name = obj.__name__

        modname = getattr(obj, "__module__", None)
        if modname is None:
            modname = pickle.whichmodule(obj, name)

        if modname == '__main__':
            themodule = None
        else:
            __import__(modname)
            themodule = sys.modules[modname]
            self.modules.add(themodule)

        if hasattr(themodule, name) and getattr(themodule, name) is obj:
            return Pickler.save_global(self, obj, name)

        typ = type(obj)
        if typ is not obj and isinstance(obj, (type, types.ClassType)):
            d = dict(obj.__dict__)  # copy dict proxy to a dict
            if not isinstance(d.get('__dict__', None), property):
                # don't extract dict that are properties
                d.pop('__dict__', None)
            d.pop('__weakref__', None)

            # hack as __new__ is stored differently in the __dict__
            new_override = d.get('__new__', None)
            if new_override:
                d['__new__'] = obj.__new__

            self.save_reduce(typ, (obj.__name__, obj.__bases__, d), obj=obj)
        else:
            raise pickle.PicklingError("Can't pickle %r" % obj)

    dispatch[type] = save_global
    dispatch[types.ClassType] = save_global

    def save_instancemethod(self, obj):
        # Memoization rarely is ever useful due to python bounding
        if PY3:
            self.save_reduce(types.MethodType, (obj.__func__, obj.__self__), obj=obj)
        else:
            self.save_reduce(types.MethodType, (obj.__func__, obj.__self__, obj.__self__.__class__),
                         obj=obj)
    dispatch[types.MethodType] = save_instancemethod

    def save_inst(self, obj):
        """Inner logic to save instance. Based off pickle.save_inst
        Supports __transient__"""
        cls = obj.__class__

        memo = self.memo
        write = self.write
        save = self.save

        if hasattr(obj, '__getinitargs__'):
            args = obj.__getinitargs__()
            len(args)  # XXX Assert it's a sequence
            pickle._keep_alive(args, memo)
        else:
            args = ()

        write(pickle.MARK)

        if self.bin:
            save(cls)
            for arg in args:
                save(arg)
            write(pickle.OBJ)
        else:
            for arg in args:
                save(arg)
            write(pickle.INST + cls.__module__ + '\n' + cls.__name__ + '\n')

        self.memoize(obj)

        try:
            getstate = obj.__getstate__
        except AttributeError:
            stuff = obj.__dict__
            #remove items if transient
            if hasattr(obj, '__transient__'):
                transient = obj.__transient__
                stuff = stuff.copy()
                for k in list(stuff.keys()):
                    if k in transient:
                        del stuff[k]
        else:
            stuff = getstate()
            pickle._keep_alive(stuff, memo)
        save(stuff)
        write(pickle.BUILD)

    if not PY3:
        dispatch[types.InstanceType] = save_inst

    def save_property(self, obj):
        # properties not correctly saved in python
        self.save_reduce(property, (obj.fget, obj.fset, obj.fdel, obj.__doc__), obj=obj)
    dispatch[property] = save_property

    def save_classmethod(self, obj):
        try:
            orig_func = obj.__func__
        except AttributeError:  # Python 2.6
            orig_func = obj.__get__(None, object)
            if isinstance(obj, classmethod):
                orig_func = orig_func.__func__  # Unbind
        self.save_reduce(type(obj), (orig_func,), obj=obj)
    dispatch[classmethod] = save_classmethod
    dispatch[staticmethod] = save_classmethod

    def save_itemgetter(self, obj):
        """itemgetter serializer (needed for namedtuple support)"""
        class Dummy:
            def __getitem__(self, item):
                return item
        items = obj(Dummy())
        if not isinstance(items, tuple):
            items = (items, )
        return self.save_reduce(operator.itemgetter, items)

    if type(operator.itemgetter) is type:
        dispatch[operator.itemgetter] = save_itemgetter

    def save_attrgetter(self, obj):
        """attrgetter serializer"""
        class Dummy(object):
            def __init__(self, attrs, index=None):
                self.attrs = attrs
                self.index = index
            def __getattribute__(self, item):
                attrs = object.__getattribute__(self, "attrs")
                index = object.__getattribute__(self, "index")
                if index is None:
                    index = len(attrs)
                    attrs.append(item)
                else:
                    attrs[index] = ".".join([attrs[index], item])
                return type(self)(attrs, index)
        attrs = []
        obj(Dummy(attrs))
        return self.save_reduce(operator.attrgetter, tuple(attrs))

    if type(operator.attrgetter) is type:
        dispatch[operator.attrgetter] = save_attrgetter

    def save_reduce(self, func, args, state=None,
                    listitems=None, dictitems=None, obj=None):
        """Modified to support __transient__ on new objects
        Change only affects protocol level 2 (which is always used by PiCloud"""
        # Assert that args is a tuple or None
        if not isinstance(args, tuple):
            raise pickle.PicklingError("args from reduce() should be a tuple")

        # Assert that func is callable
        if not hasattr(func, '__call__'):
            raise pickle.PicklingError("func from reduce should be callable")

        save = self.save
        write = self.write

        # Protocol 2 special case: if func's name is __newobj__, use NEWOBJ
        if self.proto >= 2 and getattr(func, "__name__", "") == "__newobj__":
            #Added fix to allow transient
            cls = args[0]
            if not hasattr(cls, "__new__"):
                raise pickle.PicklingError(
                    "args[0] from __newobj__ args has no __new__")
            if obj is not None and cls is not obj.__class__:
                raise pickle.PicklingError(
                    "args[0] from __newobj__ args has the wrong class")
            args = args[1:]
            save(cls)

            #Don't pickle transient entries
            if hasattr(obj, '__transient__'):
                transient = obj.__transient__
                state = state.copy()

                for k in list(state.keys()):
                    if k in transient:
                        del state[k]

            save(args)
            write(pickle.NEWOBJ)
        else:
            save(func)
            save(args)
            write(pickle.REDUCE)

        # modify here to avoid assert error
        if obj is not None and id(obj) not in self.memo:
            self.memoize(obj)

        # More new special cases (that work with older protocols as
        # well): when __reduce__ returns a tuple with 4 or 5 items,
        # the 4th and 5th item should be iterators that provide list
        # items and dict items (as (key, value) tuples), or None.

        if listitems is not None:
            self._batch_appends(listitems)

        if dictitems is not None:
            self._batch_setitems(dictitems)

        if state is not None:
            save(state)
            write(pickle.BUILD)

    def save_partial(self, obj):
        """Partial objects do not serialize correctly in python2.x -- this fixes the bugs"""
        self.save_reduce(_genpartial, (obj.func, obj.args, obj.keywords))

    if sys.version_info < (2,7):  # 2.7 supports partial pickling
        dispatch[partial] = save_partial


    def save_file(self, obj):
        """Save a file"""
        try:
            import StringIO as pystringIO #we can't use cStringIO as it lacks the name attribute
        except ImportError:
            import io as pystringIO

        if not hasattr(obj, 'name') or  not hasattr(obj, 'mode'):
            raise pickle.PicklingError("Cannot pickle files that do not map to an actual file")
        if obj is sys.stdout:
            return self.save_reduce(getattr, (sys,'stdout'), obj=obj)
        if obj is sys.stderr:
            return self.save_reduce(getattr, (sys,'stderr'), obj=obj)
        if obj is sys.stdin:
            raise pickle.PicklingError("Cannot pickle standard input")
        if obj.closed:
            raise pickle.PicklingError("Cannot pickle closed files")
        if hasattr(obj, 'isatty') and obj.isatty():
            raise pickle.PicklingError("Cannot pickle files that map to tty objects")
        if 'r' not in obj.mode and '+' not in obj.mode:
            raise pickle.PicklingError("Cannot pickle files that are not opened for reading: %s" % obj.mode)

        name = obj.name

        retval = pystringIO.StringIO()

        try:
            # Read the whole file
            curloc = obj.tell()
            obj.seek(0)
            contents = obj.read()
            obj.seek(curloc)
        except IOError:
            raise pickle.PicklingError("Cannot pickle file %s as it cannot be read" % name)
        retval.write(contents)
        retval.seek(curloc)

        retval.name = name
        self.save(retval)
        self.memoize(obj)

    if PY3:
        dispatch[io.TextIOWrapper] = save_file
    else:
        dispatch[file] = save_file

    """Special functions for Add-on libraries"""
    def inject_addons(self):
        """Plug in system. Register additional pickling functions if modules already loaded"""
        pass


# Shorthands for legacy support

def dump(obj, file, protocol=2, dump_code=False):
    CloudPickler(file, protocol, dump_code).dump(obj)


def dumps(obj, protocol=2, dump_code=False):
    file = StringIO()

    cp = CloudPickler(file, protocol, dump_code)
    cp.dump(obj)

    return file.getvalue()

# including pickles unloading functions in this namespace
if PY3:
    from pickle import _Unpickler as Unpickler
else:
    from pickle import Unpickler

class CloudUnpickler(Unpickler):
    def __init__(self, *args, **kwargs):
        self._src_major, self._src_minor, self._src_impl = kwargs.pop('impl', None) or (2, 7, 'cpython')
        self._src_version = (self._src_major, self._src_minor)

        self._dump_code = kwargs.pop('dump_code', False)
        Unpickler.__init__(self, *args, **kwargs)

        self._override_dispatch(pickle.BININT, self.load_binint)
        self._override_dispatch(pickle.BININT2, self.load_binint2)
        self._override_dispatch(pickle.LONG4, self.load_long4)
        self._override_dispatch(pickle.BINSTRING, self.load_binstring)
        self._override_dispatch(pickle.BINUNICODE, self.load_binunicode)
        self._override_dispatch(pickle.EXT2, self.load_ext2)
        self._override_dispatch(pickle.EXT4, self.load_ext4)
        self._override_dispatch(pickle.LONG_BINGET, self.load_long_binget)
        self._override_dispatch(pickle.LONG_BINPUT, self.load_long_binput)
        self._override_dispatch(pickle.REDUCE, self.load_reduce)

    def _override_dispatch(self, code, func_name):
        if callable(func_name):
            func_name = func_name.__name__
        lm = lambda x: getattr(x, func_name)()
        lm.__name__ = func_name
        self.dispatch[code] = lm

    def find_class(self, module, name):
        # Subclasses may override this
        try:
            if PY3 and module == '__builtin__':
                module = 'builtins'
            __import__(module)

            mod = sys.modules[module]
            klass = getattr(mod, name)
            return klass
        except ImportError as e:
            try:
                return globals()[name]
            except KeyError:
                raise ImportError(str(e) + ', name: ' + name)

    def load_binint(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by the ODPS python sandbox.
        self.append(struct.unpack('<i', self.read(4))[0])

    def load_binint2(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by the ODPS python sandbox.
        self.append(struct.unpack('<i', self.read(2) + '\000\000')[0])

    def load_long4(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by the ODPS python sandbox.
        n = struct.unpack('<i', self.read(4))[0]
        bytes = self.read(n)
        self.append(pickle.decode_long(bytes))

    def load_binstring(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by the ODPS python sandbox.
        len = struct.unpack('<i', self.read(4))[0]
        self.append(self.read(len))

    def load_binunicode(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by the ODPS python sandbox.
        len = struct.unpack('<i', self.read(4))[0]
        self.append(unicode(self.read(len), 'utf-8'))

    def load_ext2(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by the ODPS python sandbox.
        code = struct.unpack('<i', self.read(2) + '\000\000')[0]
        self.get_extension(code)

    def load_ext4(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by the ODPS python sandbox.
        code = struct.unpack('<i', self.read(4))[0]
        self.get_extension(code)

    def load_long_binget(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by the ODPS python sandbox.
        i = struct.unpack('<i', self.read(4))[0]
        self.append(self.memo[repr(i)])

    def load_long_binput(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by the ODPS python sandbox.
        i = struct.unpack('<i', self.read(4))[0]
        self.memo[repr(i)] = self.stack[-1]

    @staticmethod
    def _conv_string_tuples(tp):
        if not PY3:
            return tuple(s.encode('utf-8') if isinstance(s, unicode) else s for s in tp)
        else:
            return tuple(str(s) if isinstance(s, bytes) else s for s in tp)

    @staticmethod
    def _realign_opcode(code, new_to_old, old_to_new):
        # reassign labels
        i = 0
        sio = StringIO()
        while i < len(code):
            op = ord(code[i])
            if op >= HAVE_ARGUMENT:
                if op in opcode.hasjrel:
                    # relocate to new relative address
                    old_abs = new_to_old[i + 3] + ord(code[i + 1]) + (ord(code[i + 2]) << 8)
                    new_rel = old_to_new[old_abs] - i - 3
                    sio.write(code[i])
                    sio.write(chr(new_rel & 0xff))
                    sio.write(chr(new_rel >> 8))
                elif op in opcode.hasjabs:
                    # relocate to new absolute address
                    new_abs = old_to_new[ord(code[i + 1]) + (ord(code[i + 2]) << 8)]
                    sio.write(code[i])
                    sio.write(chr(new_abs & 0xff))
                    sio.write(chr(new_abs >> 8))
                else:
                    sio.write(code[i:i + 3])
                i += 3
            else:
                sio.write(code[i])
                i += 1
        return sio.getvalue()

    @classmethod
    def _code_compat_pypy2_to_cp27(cls, code_args):
        # only works under Python 2.7
        code_args = list(code_args)

        # translate byte codes
        byte_code = code_args[4]
        # build line mappings, extra space for new_to_old mapping, as code could be longer
        new_to_old = [0, ] * (2 + 2 * len(byte_code))
        old_to_new = [0, ] * (1 + len(byte_code))
        remapped = False
        i, ni = 0, 0
        sio = StringIO()
        while i < len(byte_code):
            opcode_ord = ord(byte_code[i])
            if opcode_ord == LOOKUP_METHOD_PYPY:
                sio.write(chr(LOAD_ATTR))
                sio.write(byte_code[i + 1:i + 3])
                i += 3
                ni += 3
            elif opcode_ord == CALL_METHOD_PYPY:
                sio.write(chr(CALL_FUNCTION))
                sio.write(byte_code[i + 1:i + 3])
                i += 3
                ni += 3
            elif opcode_ord == BUILD_LIST_FROM_ARG_PYPY:
                sio.write(chr(BUILD_LIST))
                sio.write(chr(0))
                sio.write(chr(0))
                sio.write(chr(ROT_TWO))
                i += 3
                ni += 4
            elif opcode_ord == JUMP_IF_NOT_DEBUG_PYPY:
                if not __debug__:
                    sio.write(chr(JUMP_FORWARD))
                    sio.write(byte_code[i + 1:i + 3])
                    ni += 3
                i += 3
            elif ord(byte_code[i]) >= HAVE_ARGUMENT:
                sio.write(byte_code[i:i + 3])
                i += 3
                ni += 3
            else:
                sio.write(byte_code[i])
                i += 1
                ni += 1

            if len(new_to_old) <= ni:
                new_to_old.extend([0, ] * len(byte_code))
            if i != ni:
                remapped = True
            new_to_old[ni] = i
            old_to_new[i] = ni
        byte_code = sio.getvalue()

        if not remapped:
            code_args[4] = byte_code
        else:
            code_args[4] = cls._realign_opcode(byte_code, new_to_old, old_to_new)
        return code_args

    @classmethod
    def _code_compat_3_to_27(cls, code_args):
        # only works under Python 2.7
        code_args = list(code_args)

        # translate byte codes
        byte_code = code_args[5]
        # build line mappings, extra space for new_to_old mapping, as code could be longer
        new_to_old = [0, ] * (2 * len(byte_code))
        old_to_new = [0, ] * len(byte_code)
        remapped = False
        # replace LOAD_FAST / LOAD_CONST / ROT_FOUR group
        i, ni = 0, 0
        sio = StringIO()
        while i < len(byte_code):
            if len(new_to_old) <= ni:
                new_to_old.extend([0, ] * len(byte_code))
            new_to_old[ni] = i
            old_to_new[i] = ni
            opcode_ord = ord(byte_code[i])
            if opcode_ord == LOAD_BUILD_CLASS_PY3:
                func_cid = len(code_args[6])
                code_args[6] = code_args[6] + (_py27_build_py3_class, )
                [sio.write(chr(c)) for c in (LOAD_CONST, func_cid & 0xff, func_cid >> 8)]
                remapped = True
                i += 1
                ni += 3
            elif opcode_ord == ROT_FOUR:
                # ROT_FOUR -> DUP_TOPX 2
                [sio.write(chr(c)) for c in (DUP_TOPX, 2, 0)]
                remapped = True
                i += 1
                ni += 3
            elif len(byte_code) - i >= 3 and opcode_ord in (MAKE_CLOSURE, MAKE_FUNCTION):
                # The TOSes in PY3 is changed for MAKE_CLOSURE
                sio.write(chr(POP_TOP))
                sio.write(byte_code[i:i + 3])
                remapped = True
                i += 3
                ni += 4
            elif opcode_ord == EXTENDED_ARG_PY3:
                sio.write(chr(EXTENDED_ARG))
                sio.write(byte_code[i + 1:i + 3])
                i += 3
                ni += 3
            elif opcode_ord == EXTENDED_ARG:
                sio.write(chr(LIST_APPEND))
                sio.write(byte_code[i + 1:i + 3])
                i += 3
                ni += 3
            elif opcode_ord >= HAVE_ARGUMENT:
                sio.write(byte_code[i:i + 3])
                i += 3
                ni += 3
            else:
                sio.write(byte_code[i])
                i += 1
                ni += 1
        byte_code = sio.getvalue()

        if not remapped:
            code_args[5] = byte_code
        else:
            code_args[5] = cls._realign_opcode(byte_code, new_to_old, old_to_new)

        # 7: co_names, 8: co_varnames, 13: co_freevars
        for col in (6, 7, 8, 13, 14):
            code_args[col] = cls._conv_string_tuples(code_args[col])
        # 9: co_filename, 10: co_name
        for col in (9, 10):
            code_args[col] = code_args[col].encode('utf-8') if isinstance(code_args[col], unicode) else code_args[col]

        return [code_args[0], ] + code_args[2:]

    @classmethod
    def _code_compat_26_to_27(cls, code_args):
        # only works under Python 2.7
        code_args = list(code_args)

        # translate byte codes
        byte_code = code_args[4]
        # build line mappings, extra space for new_to_old mapping, as code could be longer
        new_to_old = [0, ] * (2 * len(byte_code))
        old_to_new = [0, ] * len(byte_code)
        remapped = False
        i, ni = 0, 0
        sio = StringIO()
        while i < len(byte_code):
            if len(new_to_old) <= ni:
                new_to_old.extend([0, ] * len(byte_code))
            new_to_old[ni] = i
            old_to_new[i] = ni
            opcode_ord = ord(byte_code[i])
            if opcode_ord == JUMP_IF_TRUE_OR_POP:
                # JUMP_IF_TRUE -> DUP_TOP, POP_JUMP_IF_TRUE
                sio.write(chr(DUP_TOP))
                sio.write(chr(POP_JUMP_IF_TRUE))

                abs_pos = i + 3 + ord(byte_code[i + 1]) + (ord(byte_code[i + 2]) << 8)
                sio.write(chr(abs_pos & 0xff))
                sio.write(chr(abs_pos >> 8))

                remapped = True
                i += 3
                ni += 4
            elif opcode_ord == JUMP_IF_FALSE_OR_POP:
                # JUMP_IF_FALSE -> DUP_TOP, POP_JUMP_IF_FALSE
                sio.write(chr(DUP_TOP))
                sio.write(chr(POP_JUMP_IF_FALSE))

                abs_pos = i + 3 + ord(byte_code[i + 1]) + (ord(byte_code[i + 2]) << 8)
                sio.write(chr(abs_pos & 0xff))
                sio.write(chr(abs_pos >> 8))

                remapped = True
                i += 3
                ni += 4
            elif (opcode_ord + 1) in (COMPARE_OP, IMPORT_NAME, BUILD_MAP, LOAD_ATTR, IMPORT_FROM):
                sio.write(chr(opcode_ord + 1))
                sio.write(byte_code[i + 1:i + 3])
                i += 3
                ni += 3
            elif opcode_ord + 2 == EXTENDED_ARG:
                sio.write(chr(opcode_ord + 2))
                sio.write(byte_code[i + 1:i + 3])
                i += 3
                ni += 3
            elif opcode_ord == LIST_APPEND_PY26:
                [sio.write(chr(c)) for c in (LIST_APPEND, 1, 0, POP_TOP)]
                remapped = True
                i += 1
                ni += 4
            elif ord(byte_code[i]) >= HAVE_ARGUMENT:
                sio.write(byte_code[i:i + 3])
                i += 3
                ni += 3
            else:
                sio.write(byte_code[i])
                i += 1
                ni += 1
        byte_code = sio.getvalue()

        if not remapped:
            code_args[4] = byte_code
        else:
            code_args[4] = cls._realign_opcode(byte_code, new_to_old, old_to_new)
        return code_args

    def load_reduce(self):
        # Replace the internal implementation of pickle
        # cause code representation in Python 3 differs from that in Python 2
        stack = self.stack
        args = stack.pop()
        func = stack[-1]
        if func.__name__ == 'code':
            if PY27 and self._src_major == 3:  # src PY3, dest PY27
                args = self._code_compat_3_to_27(args)
            elif PY27 and self._src_version == (2, 6):  # src PY26, dest PY27
                args = self._code_compat_26_to_27(args)
            elif PY27 and not PYPY and self._src_impl == 'pypy':
                args = self._code_compat_pypy2_to_cp27(args)

            if self._dump_code:
                print(args[9 if not PY3 else 10])
                dis.dis(args[4 if not PY3 else 5])
                sys.stdout.flush()
        elif func.__name__ == 'tablecode':
            if JYTHON:
                from org.python.core import PyBytecode
                func = PyBytecode
        elif func.__name__ == 'type' or func.__name__ == 'classobj' or (isinstance(func, type) and issubclass(func, type)):
            if not PY3:
                args = list(args)
                args[0] = args[0].encode('utf-8') if isinstance(args[0], unicode) else args[0]
        try:
            value = func(*args)
        except Exception as exc:
            raise Exception('Failed to unpickle reduce. func=%s mod=%s args=%s msg="%s"' % (func.__name__, func.__module__, repr(args), str(exc)))
        stack[-1] = value


def load(file, impl=None, dump_code=False):
    return CloudUnpickler(file, impl=impl, dump_code=dump_code).load()


def loads(str, impl=None, dump_code=False):
    file = StringIO(str)
    return CloudUnpickler(file, impl=impl, dump_code=dump_code).load()


#hack for __import__ not working as desired
def subimport(name):
    __import__(name)
    return sys.modules[name]


# restores function attributes
def _restore_attr(obj, attr):
    for key, val in attr.items():
        setattr(obj, key, val)
    return obj


def _get_module_builtins():
    return pickle.__builtins__


def print_exec(stream):
    ei = sys.exc_info()
    traceback.print_exception(ei[0], ei[1], ei[2], None, stream)


def _modules_to_main(modList):
    """Force every module in modList to be placed into main"""
    if not modList:
        return

    main = sys.modules['__main__']
    for modname in modList:
        if type(modname) is str:
            try:
                mod = __import__(modname)
            except Exception as e:
                sys.stderr.write('warning: could not import %s\n.  '
                                 'Your function may unexpectedly error due to this import failing;'
                                 'A version mismatch is likely.  Specific error was:\n' % modname)
                print_exec(sys.stderr)
            else:
                setattr(main, mod.__name__, mod)


#object generators:
def _genpartial(func, args, kwds):
    if not args:
        args = ()
    if not kwds:
        kwds = {}
    return partial(func, *args, **kwds)


def _fill_function(func, globals, defaults, dict):
    """ Fills in the rest of function data into the skeleton function object
        that were created via _make_skel_func().
         """
    func.__globals__.update(globals)
    func.__defaults__ = defaults
    func.__dict__ = dict

    return func


def _make_cell(value):
    return (lambda: value).__closure__[0]


def _reconstruct_closure(values):
    return tuple([_make_cell(v) for v in values])


def _make_skel_func(code, closures, base_globals = None):
    """ Creates a skeleton function object that contains just the provided
        code and the correct number of cells in func_closure.  All other
        func attributes (e.g. func_globals) are empty.
    """
    closure = _reconstruct_closure(closures) if closures else None

    if base_globals is None:
        base_globals = {}
    base_globals['__builtins__'] = __builtins__

    return types.FunctionType(code, base_globals,
                              None, None, closure)


"""Constructors for 3rd party libraries
Note: These can never be renamed due to client compatibility issues"""

def _getobject(modname, attribute):
    mod = __import__(modname, fromlist=[attribute])
    return mod.__dict__[attribute]


def _py27_build_py3_class(func, name, *bases, **kwds):
    # Stack structure: (class_name, base_classes_tuple, method dictionary)
    metacls = kwds.pop('metaclass', None)
    bases = tuple(bases)
    code_args = list(_extract_code_args(func.__code__))

    # start translating code
    consts = code_args[5]
    n_consts = len(consts)
    name_cid, base_cid, metaclass_cid = range(n_consts, n_consts + 3)
    code_args[5] = consts + (name, bases, metacls)

    names = code_args[6]
    metaclass_nid = len(names)
    code_args[6] = names + ('__metaclass__', )

    aug_code = [POP_TOP, ]

    if metacls is not None:
        aug_code += [
            LOAD_CONST, metaclass_cid & 0xff, metaclass_cid >> 8,
            STORE_NAME, metaclass_nid & 0xff, metaclass_nid >> 8,
        ]

    aug_code += [
        LOAD_CONST, name_cid & 0xff, name_cid >> 8,
        LOAD_CONST, base_cid & 0xff, base_cid >> 8,
        LOAD_LOCALS,
        BUILD_CLASS,
        RETURN_VALUE
    ]
    if not hasattr(func, '_cp_original_len'):
        func._cp_original_len = len(code_args[4]) - 1

    sio = StringIO()
    sio.write(code_args[4][:func._cp_original_len])
    [sio.write(chr(o)) for o in aug_code]
    code_args[4] = sio.getvalue()

    code_obj = types.CodeType(*code_args)
    func.__code__ = code_obj
    return func()
