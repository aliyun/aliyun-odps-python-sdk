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
from __future__ import absolute_import
from __future__ import print_function

import dis
import inspect
import io
import itertools
import opcode
import operator
import pickle
import platform
import struct
import sys
import traceback
import types
import weakref
from functools import partial

if sys.version < '3':
    from pickle import Pickler, Unpickler
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
    BytesIO = StringIO
    irange = xrange
    lrange = range
    to_unicode = unicode
    to_ascii = str
    iteritems = lambda d: d.iteritems()
    PY3 = False
else:
    types.ClassType = type
    from pickle import _Pickler as Pickler
    from io import BytesIO as StringIO
    irange = range
    lrange = lambda x: list(range(x))
    to_unicode = str
    to_ascii = ascii
    iteritems = lambda d: d.items()
    PY3 = True

PY27 = not PY3 and sys.version_info[1] == 7
PY35LE = sys.version_info[:2] <= (3, 5)

PYPY = platform.python_implementation().lower() == 'pypy'
JYTHON = platform.python_implementation().lower() == 'jython'

def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})


# relevant opcodes
BINARY_MATRIX_MULTIPLY_PY3 = 16
BUILD_CLASS = opcode.opmap.get('BUILD_CLASS')
BUILD_LIST_FROM_ARG_PYPY = 203
BUILD_LIST = opcode.opmap['BUILD_LIST']
BUILD_TUPLE = opcode.opmap.get('BUILD_TUPLE')
BUILD_MAP = opcode.opmap.get('BUILD_MAP')
BUILD_CONST_KEY_MAP_PY36 = 156
BUILD_LIST_UNPACK_PY3 = 149
BUILD_MAP_UNPACK_PY3 = 150
BUILD_MAP_UNPACK_WITH_CALL_PY3 = 151
BUILD_SET_UNPACK_PY3 = 153
BUILD_STRING_PY36 = 157
BUILD_TUPLE_UNPACK_PY3 = 152
BUILD_TUPLE_UNPACK_WITH_CALL_PY36 = 158
CALL_FUNCTION = opcode.opmap['CALL_FUNCTION']
CALL_FUNCTION_EX_PY36 = 142
CALL_FUNCTION_KW = opcode.opmap.get('CALL_FUNCTION_KW')
CALL_METHOD_PYPY = 202
COMPARE_OP = opcode.opmap.get('COMPARE_OP')
DELETE_DEREF_PY3 = 138
DUP_TOP = dis.opmap.get('DUP_TOP')
DUP_TOP_TWO_PY3 = 5
DUP_TOPX = dis.opmap.get('DUP_TOPX')
EXTENDED_ARG = dis.EXTENDED_ARG
EXTENDED_ARG_PY26 = 143
EXTENDED_ARG_PY3 = 144
FORMAT_VALUE_PY36 = 155
FOR_ITER = opcode.opmap['FOR_ITER']
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
IMPORT_FROM = opcode.opmap.get('IMPORT_FROM')
IMPORT_NAME = opcode.opmap.get('IMPORT_NAME')
INPLACE_MATRIX_MULTIPLY_PY3 = 17
JUMP_ABSOLUTE = opcode.opmap['JUMP_ABSOLUTE']
JUMP_FORWARD = opcode.opmap['JUMP_FORWARD']
JUMP_IF_FALSE_OR_POP = opcode.opmap.get('JUMP_IF_FALSE_OR_POP')
JUMP_IF_NOT_DEBUG_PYPY = 204
JUMP_IF_TRUE_OR_POP = opcode.opmap.get('JUMP_IF_TRUE_OR_POP')
LIST_APPEND = opcode.opmap['LIST_APPEND']
LIST_APPEND_PY26 = 18
LIST_APPEND_PY3 = 145
LOAD_ATTR = opcode.opmap['LOAD_ATTR']
LOAD_BUILD_CLASS_PY3 = 71
LOAD_CLASSDEREF_PY3 = 148
LOAD_CONST = opcode.opmap['LOAD_CONST']
LOAD_DEREF = opcode.opmap['LOAD_DEREF']
LOAD_FAST = opcode.opmap['LOAD_FAST']
LOAD_LOCALS = opcode.opmap.get('LOAD_LOCALS')
LOOKUP_METHOD_PYPY = 201
MAKE_CLOSURE = opcode.opmap.get('MAKE_CLOSURE')
MAKE_FUNCTION = opcode.opmap['MAKE_FUNCTION']
NOP = opcode.opmap['NOP']
POP_EXCEPT_PY3 = 89
POP_JUMP_IF_TRUE = opcode.opmap.get('POP_JUMP_IF_TRUE')
POP_JUMP_IF_FALSE = opcode.opmap.get('POP_JUMP_IF_FALSE')
POP_TOP = opcode.opmap.get('POP_TOP')
RETURN_VALUE = opcode.opmap['RETURN_VALUE']
ROT_TWO = opcode.opmap['ROT_TWO']
ROT_THREE = opcode.opmap.get('ROT_THREE')
ROT_FOUR = opcode.opmap.get('ROT_FOUR')
STORE_ATTR = opcode.opmap.get('STORE_ATTR')
STORE_DEREF = opcode.opmap.get('STORE_DEREF')
STORE_NAME = opcode.opmap.get('STORE_NAME')
STORE_FAST = opcode.opmap.get('STORE_FAST')
UNPACK_SEQUENCE = opcode.opmap.get('UNPACK_SEQUENCE')

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


if sys.version_info < (3, 4):
    def _walk_global_ops(code):
        """
        Yield (opcode, argument number) tuples for all
        global-referencing instructions in *code*.
        """
        code = getattr(code, 'co_code', b'')
        if not PY3:
            code = map(ord, code)

        n = len(code)
        i = 0
        extended_arg = 0
        while i < n:
            op = code[i]
            i += 1
            if op >= HAVE_ARGUMENT:
                oparg = code[i] + code[i + 1] * 256 + extended_arg
                extended_arg = 0
                i += 2
                if op == EXTENDED_ARG:
                    extended_arg = oparg * 65536
                if op in GLOBAL_OPS:
                    yield op, oparg

else:
    def _walk_global_ops(code):
        """
        Yield (opcode, argument number) tuples for all
        global-referencing instructions in *code*.
        """
        for instr in dis.get_instructions(code):
            op = instr.opcode
            if op in GLOBAL_OPS:
                yield op, instr.arg

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
        except pickle.PickleError:
            raise
        except Exception as e:
            if "'i' format requires" in e.message:
                msg = "Object too large to serialize: " + e.message
            else:
                msg = "Could not serialize object: " + e.__class__.__name__ + ": " + e.message
            print_exec(sys.stderr)
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
        try:
            modname = pickle.whichmodule(obj, name)
        except Exception:
            modname = None
        # print('which gives %s %s %s' % (modname, obj, name))
        try:
            # whichmodule() could fail, see
            # https://bitbucket.org/gutworth/six/issues/63/importing-six-breaks-pickling
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
        save(func.__module__)
        write(pickle.TUPLE)
        write(pickle.REDUCE)  # applies _fill_function on the tuple

    _extract_code_globals_cache = (
        weakref.WeakKeyDictionary()
        if sys.version_info >= (2, 7) and not hasattr(sys, "pypy_version_info")
        else {})

    @classmethod
    def extract_code_globals(cls, co):
        """
        Find all globals names read or written to by codeblock co
        """
        out_names = cls._extract_code_globals_cache.get(co)
        if out_names is None:
            try:
                names = co.co_names
            except AttributeError:
                # PyPy "builtin-code" object
                out_names = set()
            else:
                out_names = set(names[oparg]
                                for op, oparg in _walk_global_ops(co))

                # see if nested function have any global refs
                if co.co_consts:
                    for const in co.co_consts:
                        if type(const) is types.CodeType:
                            out_names |= cls.extract_code_globals(const)

            cls._extract_code_globals_cache[co] = out_names

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
            try:
                # whichmodule() could fail, see
                # https://bitbucket.org/gutworth/six/issues/63/importing-six-breaks-pickling
                modname = pickle.whichmodule(obj, name)
            except Exception:
                modname = '__main__'

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
                raise ImportError(str(e) + ', name: ' + name +
                                  '\nYou need to use third-party library support '
                                  'to run this module in MaxCompute clusters.')

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

    @classmethod
    def _code_compat_36_to_27(cls, code_args):
        # only works under Python 2.7
        code_args = Cp36_Cp35(code_args).translate_code()
        return Cp35_Cp27(code_args).translate_code()

    def load_reduce(self):
        # Replace the internal implementation of pickle
        # cause code representation in Python 3 differs from that in Python 2
        stack = self.stack
        args = stack.pop()
        func = stack[-1]
        if func.__name__ == 'code':
            if PY27 and self._src_version >= (3, 6):  # src >= PY36, dest PY27
                args = self._code_compat_36_to_27(args)
            elif PY27 and self._src_major == 3 and self._src_version <= (3, 5):  # src PY3 && src <= PY35, dest PY27
                args = Cp35_Cp27(args).translate_code()
            elif PY27 and self._src_version == (2, 6):  # src PY26, dest PY27
                args = Cp26_Cp27(args).translate_code()
            elif PY27 and not PYPY and self._src_impl == 'pypy':
                args = Pypy2_Cp27(args).translate_code()

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
                sys.stderr.write('Warning: could not import %s\n.  '
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


def _fill_function(func, globals, defaults, dict, module):
    """ Fills in the rest of function data into the skeleton function object
        that were created via _make_skel_func().
         """
    func.__globals__.update(globals)
    func.__defaults__ = defaults
    func.__dict__ = dict
    func.__module__ = module

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


def op_translator(op):
    if not isinstance(op, (list, set)):
        ops = [op]
    else:
        ops = list(op)

    def _decorator(fun):
        func_args = inspect.getargs(fun.__code__).args

        def _wrapper(self, **kwargs):
            args = [kwargs[v] for v in func_args[1:]]
            return fun(self, *args)

        _wrapper._bind_ops = ops
        _wrapper.__name__ = fun.__name__
        _wrapper.__doc__ = fun.__doc__
        return _wrapper

    return _decorator


class CodeRewriterMeta(type):
    def __init__(cls, what, bases=None, d=None):
        type.__init__(cls, what, bases, d)

        translator_dict = dict()
        d = d or dict()
        for k, v in iteritems(d):
            if hasattr(v, '_bind_ops'):
                for op in v._bind_ops:
                    translator_dict[op] = v
        cls._translator = translator_dict


class CodeRewriter(with_metaclass(CodeRewriterMeta)):
    CO_NLOCALS_POS = None
    CO_CODE_POS = None
    CO_CONSTS_POS = None
    CO_NAMES_POS = None
    CO_VARNAMES_POS = None
    CO_FREEVARS_POS = None
    CO_CELLVARS_POS = None
    CO_LNOTAB_POS = None
    OP_EXTENDED_ARG = EXTENDED_ARG

    def __init__(self, code_args):
        self.code_args = list(code_args)
        self._const_poses = dict()
        self._name_poses = dict()
        self._varname_poses = dict()
        self.code_writer = BytesIO()

    def _patch_code_tuple(self, offset, reg, key, *args):
        poses = []
        patches = []
        patch_id = len(self.code_args[offset])
        for a in args:
            if key(a) not in reg:
                patches.append(a)
                reg[key(a)] = patch_id
                patch_id += 1
            poses.append(reg[key(a)])
        if patches:
            self.code_args[offset] += tuple(patches)
        return tuple(poses) if len(poses) != 1 else poses[0]

    @staticmethod
    def _reassign_targets(code, new_to_old, old_to_new):
        code_len = len(code)

        def get_group_end(start):
            end = start + 1
            while end < code_len and new_to_old[end] == 0:
                end += 1
            return end

        i = 0
        gstart = 0
        gend = get_group_end(0)
        sio = StringIO()
        while i < code_len:
            op = ord(code[i])
            if new_to_old[i]:
                gstart = i
                gend = get_group_end(i)
            if op >= HAVE_ARGUMENT:
                op_data = ord(code[i + 1]) + (ord(code[i + 2]) << 8)
                if op in opcode.hasjrel:
                    # relocate to new relative address
                    if gstart <= i + 3 + op_data < gend:
                        new_rel = op_data
                    else:
                        old_abs = new_to_old[gend] + op_data - (gend - i - 3)
                        new_rel = old_to_new[old_abs] - i - 3
                    sio.write(code[i])
                    sio.write(chr(new_rel & 0xff))
                    sio.write(chr(new_rel >> 8))
                elif op in opcode.hasjabs:
                    # relocate to new absolute address
                    old_rel = op_data - new_to_old[gstart]
                    if gstart <= old_rel + gstart < gend:
                        new_abs = old_rel + gstart
                    else:
                        new_abs = old_to_new[op_data - (i - gstart)]
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

    @staticmethod
    def _reassign_lnotab(lnotab, old_to_new):
        sio = StringIO()
        cur_old_pc = 0
        cur_new_pc, last_new_pc = 0, 0
        for pdelta, ldelta in zip(lnotab[::2], lnotab[1::2]):
            cur_old_pc += ord(pdelta)
            last_new_pc = cur_new_pc
            cur_new_pc = old_to_new[cur_old_pc]
            sio.write(bytes(bytearray([cur_new_pc - last_new_pc, ord(ldelta)])))
        return sio.getvalue()

    def patch_consts(self, *args):
        return self._patch_code_tuple(self.CO_CONSTS_POS, self._const_poses, id, *args)

    def patch_names(self, *args):
        return self._patch_code_tuple(self.CO_NAMES_POS, self._name_poses, lambda x: x, *args)

    def patch_varnames(self, *args):
        self.code_args[self.CO_NLOCALS_POS] += sum(1 for a in args if a not in self._varname_poses)
        return self._patch_code_tuple(self.CO_VARNAMES_POS, self._varname_poses, lambda x: x, *args)

    def write_replacement_call(self, func, stack_len=None, with_arg=None):
        func_cid = self.patch_consts(func)
        stack_len = stack_len if stack_len is not None else with_arg
        instructions = []

        if with_arg is not None:
            flag_cid = self.patch_consts(with_arg)
            instructions.extend([LOAD_CONST, flag_cid & 0xff, flag_cid >> 8])

        instructions.extend([
            BUILD_TUPLE, stack_len & 0xff, stack_len >> 8,
            LOAD_CONST, func_cid & 0xff, func_cid >> 8,
            ROT_TWO,
            CALL_FUNCTION, 1, 0
        ])

        self.code_writer.write(bytes(bytearray(instructions)))
        return len(instructions)

    def write_instruction(self, opcode, arg=None):
        inst_size = 0
        if arg is not None:
            arg_list = []
            if arg == 0:
                arg_list.append(arg)
            else:
                while arg > 0:
                    arg_list.append(arg & 0xffff)
                    arg >>= 16
            arg_list = list(reversed(arg_list))
            for ap in arg_list[:-1]:
                inst_size += self.translate_instruction(self.OP_EXTENDED_ARG, ap, None)
            arg = arg_list[-1]
            self.code_writer.write(bytes(bytearray([opcode, arg & 0xff, arg >> 8])))
            return inst_size + 3
        else:
            self.code_writer.write(chr(opcode))
            return inst_size + 1

    def translate_instruction(self, opcode, arg, pc):
        cls = type(self)
        if not hasattr(cls, '_translator') or opcode not in cls._translator:
            return self.write_instruction(opcode, arg)
        else:
            return cls._translator[opcode](self, op=opcode, arg=arg, pc=pc)

    def iter_code(self):
        idx = 0
        extended_arg = 0
        bytecode = self.code_args[self.CO_CODE_POS]
        while idx < len(bytecode):
            opcode = ord(bytecode[idx])
            if opcode < HAVE_ARGUMENT:
                idx += 1
                extended_arg = 0
                yield idx, opcode, None
            else:
                arg = (extended_arg << 16) + ord(bytecode[idx + 1]) + (ord(bytecode[idx + 2]) << 8)
                idx += 3
                if opcode == self.OP_EXTENDED_ARG:
                    extended_arg = arg
                else:
                    extended_arg = 0
                    yield idx, opcode, arg

    def translate_code(self):
        # translate byte codes
        byte_code = self.code_args[self.CO_CODE_POS]
        # build line mappings, extra space for new_to_old mapping, as code could be longer
        new_to_old = [0, ] * (2 + 2 * len(byte_code))
        old_to_new = [0, ] * (1 + len(byte_code))
        remapped = False
        ni = 0

        for pc, op, arg in self.iter_code():
            inst_size = self.translate_instruction(op, arg, pc)
            ni += inst_size

            if len(new_to_old) <= ni:
                new_to_old.extend([0, ] * (1 + len(byte_code)))
            new_to_old[ni] = pc
            old_to_new[pc] = ni

            if ni != pc:
                remapped = True

        byte_code = self.code_writer.getvalue()
        if not remapped:
            self.code_args[self.CO_CODE_POS] = self.code_writer.getvalue()
        else:
            self.code_args[self.CO_CODE_POS] = self._reassign_targets(byte_code, new_to_old, old_to_new)
            lnotab = self.code_args[self.CO_LNOTAB_POS]
            self.code_args[self.CO_LNOTAB_POS] = self._reassign_lnotab(lnotab, old_to_new)

        return self.code_args


class Py2CodeRewriter(CodeRewriter):
    CO_NLOCALS_POS = 1
    CO_CODE_POS = 4
    CO_CONSTS_POS = 5
    CO_NAMES_POS = 6
    CO_VARNAMES_POS = 7
    CO_LNOTAB_POS = 11
    CO_FREEVARS_POS = 12
    CO_CELLVARS_POS = 13
    OP_EXTENDED_ARG = EXTENDED_ARG


class Py3CodeRewriter(CodeRewriter):
    CO_NLOCALS_POS = 2
    CO_CODE_POS = 5
    CO_CONSTS_POS = 6
    CO_NAMES_POS = 7
    CO_VARNAMES_POS = 8
    CO_LNOTAB_POS = 12
    CO_FREEVARS_POS = 13
    CO_CELLVARS_POS = 14
    OP_EXTENDED_ARG = EXTENDED_ARG_PY3


class Py36CodeRewriter(Py3CodeRewriter):
    def iter_code(self):
        extended_arg = 0
        bytecode = self.code_args[self.CO_CODE_POS]
        for idx in irange(0, len(bytecode), 2):
            opcode = ord(bytecode[idx])
            if opcode < HAVE_ARGUMENT:
                extended_arg = 0
                yield idx + 2, opcode, None
            else:
                arg = (extended_arg << 8) + ord(bytecode[idx + 1])
                if opcode == self.OP_EXTENDED_ARG:
                    extended_arg = arg
                else:
                    extended_arg = 0
                    yield idx + 2, opcode, arg


class Cp26_Cp27(Py2CodeRewriter):
    @op_translator(JUMP_IF_TRUE_OR_POP)
    def handle_jump_if_true_or_pop(self, arg, pc):
        return sum([
            self.write_instruction(DUP_TOP),
            self.write_instruction(POP_JUMP_IF_TRUE, pc + arg + 1)
        ])

    @op_translator(JUMP_IF_FALSE_OR_POP)
    def handle_jump_if_false_or_pop(self, arg, pc):
        return sum([
            self.write_instruction(DUP_TOP),
            self.write_instruction(POP_JUMP_IF_FALSE, pc + arg + 1)
        ])

    @op_translator([v - 1 for v in (COMPARE_OP, IMPORT_NAME, BUILD_MAP, LOAD_ATTR, IMPORT_FROM)])
    def handle_op_increment(self, op, arg):
        return self.write_instruction(op + 1, arg)

    @op_translator(EXTENDED_ARG_PY26)
    def handle_extended_arg(self, arg):
        return self.write_instruction(EXTENDED_ARG, arg)

    @op_translator(LIST_APPEND_PY26)
    def handle_list_append(self):
        return sum([
            self.write_instruction(LIST_APPEND, 1),
            self.write_instruction(POP_TOP),
        ])


class Pypy2_Cp27(Py2CodeRewriter):
    @op_translator(LOOKUP_METHOD_PYPY)
    def handle_lookup_method(self, arg):
        return self.write_instruction(LOAD_ATTR, arg)

    @op_translator(CALL_METHOD_PYPY)
    def handle_call_method(self, arg):
        return self.write_instruction(CALL_FUNCTION, arg)

    @op_translator(BUILD_LIST_FROM_ARG_PYPY)
    def handle_build_list_from_arg(self):
        return sum([
            self.write_instruction(BUILD_LIST, 0),
            self.write_instruction(ROT_TWO),
        ])

    @op_translator(JUMP_IF_NOT_DEBUG_PYPY)
    def handle_jump_if_not_debug(self, arg):
        if not __debug__:
            return self.write_instruction(JUMP_FORWARD, arg)
        else:
            return 0


class Cp35_Cp27(Py3CodeRewriter):
    @staticmethod
    def _build_class(func, name, *bases, **kwds):
        # Stack structure: (class_name, base_classes_tuple, method dictionary)
        metacls = kwds.pop('metaclass', None)
        bases = tuple(bases)
        code_args = list(_extract_code_args(func.__code__))

        # start translating code
        consts = code_args[5]
        n_consts = len(consts)
        name_cid, base_cid, metaclass_cid = lrange(n_consts, n_consts + 3)
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

    @staticmethod
    def _build_tuple_unpack(args):
        tp = ()
        for a in args:
            tp += tuple(a)
        return tp

    @staticmethod
    def _build_list_unpack(args):
        tp = []
        for a in args:
            tp += list(a)
        return tp

    @staticmethod
    def _build_set_unpack(args):
        s = set()
        for a in args:
            s.update(a)
        return s

    @staticmethod
    def _build_map_unpack(args):
        d = dict()
        for a in args:
            d.update(a)
        return d

    @staticmethod
    def _matmul(args):
        import numpy as np
        return np.dot(args[0], args[1])

    @staticmethod
    def _imatmul(args):
        import numpy as np
        return np.dot(args[0], args[1], out=args[0])

    @op_translator(DELETE_DEREF_PY3)
    def handle_delete_deref(self, arg):
        none_cid = self.patch_consts(None)
        return sum([
            self.write_instruction(LOAD_CONST, none_cid),
            self.write_instruction(STORE_DEREF, arg),
        ])

    @op_translator(LOAD_CLASSDEREF_PY3)
    def handle_load_classderef(self, arg, pc):
        cellvars = self.code_args[self.CO_CELLVARS_POS]
        n_cellvars = len(cellvars) if cellvars else 0
        assert arg >= n_cellvars

        idx = arg - n_cellvars
        var_name = self.code_args[self.CO_FREEVARS_POS][idx]

        locals_cid = self.patch_consts(locals)
        var_name_cid = self.patch_consts(var_name)
        contains_nid = self.patch_names('__contains__')
        getitem_nid = self.patch_names('__getitem__')

        # pseudo-code of following byte-codes:
        # if var_name in locals():
        #     return LOAD_DEREF(arg)
        # else:
        #     return locals()[var_name]
        return sum([
            self.write_instruction(LOAD_CONST, locals_cid),     # + 0
            self.write_instruction(CALL_FUNCTION, 0),           # + 3
            self.write_instruction(DUP_TOP),                    # + 6
            self.write_instruction(LOAD_ATTR, contains_nid),    # + 7
            self.write_instruction(LOAD_CONST, var_name_cid),   # + 10
            self.write_instruction(CALL_FUNCTION, 1),           # + 13
            self.write_instruction(POP_JUMP_IF_TRUE, pc + 23),  # + 16
            self.write_instruction(POP_TOP),                    # + 19
            self.write_instruction(LOAD_DEREF, arg),            # + 20
            self.write_instruction(JUMP_FORWARD, 9),            # + 23
            self.write_instruction(LOAD_ATTR, getitem_nid),     # + 26
            self.write_instruction(LOAD_CONST, var_name_cid),   # + 29
            self.write_instruction(CALL_FUNCTION, 1),           # + 32
        ])

    @op_translator(LOAD_BUILD_CLASS_PY3)
    def handle_load_build_class(self):
        func_cid = self.patch_consts(self._build_class)
        return self.write_instruction(LOAD_CONST, func_cid)

    @op_translator(DUP_TOP_TWO_PY3)
    def handle_dup_top_two(self):
        return self.write_instruction(DUP_TOPX, 2)

    @op_translator([MAKE_CLOSURE, MAKE_FUNCTION])
    def handle_make_function(self, op, arg):
        return sum([
            self.write_instruction(POP_TOP),
            self.write_instruction(op, arg),
        ])

    @op_translator(EXTENDED_ARG_PY3)
    def handle_extended_arg(self, arg):
        return self.write_instruction(EXTENDED_ARG, arg)

    @op_translator(LIST_APPEND_PY3)
    def handle_list_append(self, arg):
        return self.write_instruction(LIST_APPEND, arg)

    @op_translator(BUILD_MAP_UNPACK_PY3)
    def handle_build_map_unpack(self, arg):
        return self.write_replacement_call(self._build_map_unpack, arg)

    @op_translator(BUILD_MAP_UNPACK_WITH_CALL_PY3)
    def handle_build_map_unpack_with_call(self, arg):
        return self.write_replacement_call(self._build_map_unpack, arg & 0xff)

    @op_translator(BUILD_TUPLE_UNPACK_PY3)
    def handle_build_tuple_unpack(self, arg):
        return self.write_replacement_call(self._build_tuple_unpack, arg)

    @op_translator(BUILD_LIST_UNPACK_PY3)
    def handle_build_list_unpack(self, arg):
        return self.write_replacement_call(self._build_list_unpack, arg)

    @op_translator(BUILD_SET_UNPACK_PY3)
    def handle_build_set_unpack(self, arg):
        return self.write_replacement_call(self._build_set_unpack, arg)

    @op_translator(BINARY_MATRIX_MULTIPLY_PY3)
    def handle_binary_matmul(self):
        return self.write_replacement_call(self._matmul, 2)

    @op_translator(INPLACE_MATRIX_MULTIPLY_PY3)
    def handle_inplace_matmul(self):
        return self.write_replacement_call(self._imatmul, 2)

    @op_translator(POP_EXCEPT_PY3)
    def handle_pop_except(self):
        # we can safely skip POP_EXCEPT in python 2
        return 0

    @staticmethod
    def _conv_string_tuples(tp):
        if not PY3:
            return tuple(s.encode('utf-8') if isinstance(s, unicode) else s for s in tp)
        else:
            return tuple(str(s) if isinstance(s, bytes) else s for s in tp)

    def translate_code(self):
        code_args = super(Cp35_Cp27, self).translate_code()

        # 7: co_names, 8: co_varnames, 13: co_freevars
        for col in (6, 7, 8, 13, 14):
            code_args[col] = self._conv_string_tuples(code_args[col])
        # 9: co_filename, 10: co_name
        for col in (9, 10):
            code_args[col] = code_args[col].encode('utf-8') if isinstance(code_args[col], unicode) else code_args[col]

        return [code_args[0], ] + code_args[2:]


class Cp36_Cp35(Py36CodeRewriter):
    @staticmethod
    def _call_function_kw(args):
        names = args[-1]
        func = args[0]
        f_args = args[1:-1 - len(names)]
        f_kw = dict(zip(names, args[-1 - len(names):-1]))
        return func(*f_args, **f_kw)

    @staticmethod
    def _build_string(args):
        return ''.join(args)

    @staticmethod
    def _format_value(args):
        flags = args[-1]
        s = args[0]
        if flags & 0x03 == 1:
            s = to_unicode(s)
        elif flags & 0x03 == 2:
            s = repr(s)
        elif flags & 0x03 == 3:
            s = to_ascii(s)
        if flags & 0x04:
            formatter = '{0:%s}' % args[1]
        else:
            formatter = '{0}'
        return formatter.format(s)

    @staticmethod
    def _build_const_key_map(args):
        return dict(zip(args[-1], args[:-1]))

    @staticmethod
    def _call_function_ex(args):
        flags = args[-1]
        func = args[0]
        f_args = args[1]
        if flags & 0x01:
            f_kw = args[2]
            return func(*f_args, **f_kw)
        else:
            return func(*f_args)

    @staticmethod
    def _build_tuple_unpack_with_call(args):
        ret_list = []
        for a in args:
            ret_list.extend(a)
        return ret_list

    @op_translator(MAKE_FUNCTION)
    def handle_make_function(self, op, arg):
        func_var_id = self.patch_varnames('_reg_func_name_%d' % id(op))
        instructions = []

        if arg & 0x08:
            instructions.extend([MAKE_CLOSURE, 0, 0])
        else:
            instructions.extend([MAKE_FUNCTION, 0, 0])
        instructions.extend([STORE_FAST, func_var_id & 0xff, func_var_id >> 8])

        if arg & 0x04:
            if PY3:
                annotation_id = self.patch_names('__annotations__')
                instructions.extend([
                    LOAD_FAST, func_var_id & 0xff, func_var_id >> 8,
                    STORE_ATTR, annotation_id & 0xff, annotation_id >> 8,
                ])
            else:
                instructions.append(POP_TOP)

        if arg & 0x02:
            if PY3:
                kwdefaults_id = self.patch_names('__kwdefaults__')
                instructions.extend([
                    LOAD_FAST, func_var_id & 0xff, func_var_id >> 8,
                    STORE_ATTR, kwdefaults_id & 0xff, kwdefaults_id >> 8,
                ])
            else:
                instructions.append(POP_TOP)

        if arg & 0x01:
            defaults_id = self.patch_names('__defaults__' if PY3 else 'func_defaults')
            instructions.extend([
                LOAD_FAST, func_var_id & 0xff, func_var_id >> 8,
                STORE_ATTR, defaults_id & 0xff, defaults_id >> 8,
            ])

        instructions.extend([LOAD_FAST, func_var_id & 0xff, func_var_id >> 8])

        self.code_writer.write(bytes(bytearray(instructions)))
        return len(instructions)

    @op_translator(CALL_FUNCTION_KW)
    def handle_call_function_kw(self, arg):
        return self.write_replacement_call(self._call_function_kw, 2 + arg)

    @op_translator(CALL_FUNCTION_EX_PY36)
    def handle_call_function_ex(self, arg):
        return self.write_replacement_call(self._call_function_ex, 3 + (arg & 0x01), arg)

    @op_translator(BUILD_CONST_KEY_MAP_PY36)
    def handle_build_const_key_map(self, arg):
        return self.write_replacement_call(self._build_const_key_map, 1 + arg)

    @op_translator(BUILD_TUPLE_UNPACK_WITH_CALL_PY36)
    def handle_build_tuple_unpack_with_call(self, arg):
        return self.write_replacement_call(self._build_tuple_unpack_with_call, arg)

    @op_translator(BUILD_STRING_PY36)
    def handle_build_string(self, arg):
        return self.write_replacement_call(self._build_string, arg)

    @op_translator(FORMAT_VALUE_PY36)
    def handle_format_value(self, arg):
        return self.write_replacement_call(self._format_value, 2 + (1 if arg & 0x04 else 0), arg)
