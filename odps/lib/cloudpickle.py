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

"""
This class is defined to override standard pickle functionality

The goals of it follow:
-Serialize lambdas and nested functions to compiled byte code
-Deal with main module correctly
-Deal with other non-serializable objects

It does not include an unpickler, as standard python unpickling suffices.

This module was extracted from the `cloud` package, developed by `PiCloud, Inc.
<https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.

Copyright (c) 2012, Regents of the University of California.
Copyright (c) 2009 `PiCloud, Inc. <https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.
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

import dis
import io
import inspect
import sys
import types
import opcode
import pickle
import struct
import logging
import weakref
import operator
import itertools
import traceback
from functools import partial


# we replace default config in MaxCompute to handle compatibility
# between different python versions
DEFAULT_PROTOCOL = 2  # pickle.HIGHEST_PROTOCOL

try:
    import _compat_pickle
except ImportError:
    _compat_pickle = None

try:
    import importlib
    imp = None
except ImportError:
    import imp
    importlib = None

if sys.version < '3':
    import __builtin__
    from pickle import Pickler, Unpickler

    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO

    string_types = basestring  # noqa: F821 basestring is removed in Python 3
    iteritems = lambda d: d.iteritems()
    irange = __builtin__.xrange
    to_ascii = __builtin__.str
    to_unicode = __builtin__.unicode
    PY3 = False
    PY38 = False
else:
    types.ClassType = type
    import builtins as __builtin__
    from pickle import _Pickler as Pickler, _Unpickler as Unpickler
    from io import BytesIO as StringIO

    string_types = (str, bytes)
    iteritems = lambda d: d.items()
    irange = __builtin__.range
    to_ascii = __builtin__.ascii
    to_unicode = __builtin__.str
    unicode = str
    PY3 = True
    PY38 = sys.version_info[:2] >= (3, 8)


def _with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})


# Container for the global namespace to ensure consistent unpickling of
# functions defined in dynamic modules (modules not registed in sys.modules).
_dynamic_modules_globals = weakref.WeakValueDictionary()


class _DynamicModuleFuncGlobals(dict):
    """Global variables referenced by a function defined in a dynamic module

    To avoid leaking references we store such context in a WeakValueDictionary
    instance.  However instances of python builtin types such as dict cannot
    be used directly as values in such a construct, hence the need for a
    derived class.
    """
    pass


def _make_cell_set_template_code():
    """Get the Python compiler to emit LOAD_FAST(arg); STORE_DEREF

    Notes
    -----
    In Python 3, we could use an easier function:

    .. code-block:: python

       def f():
           cell = None

           def _stub(value):
               nonlocal cell
               cell = value

           return _stub

        _cell_set_template_code = f()

    This function is _only_ a LOAD_FAST(arg); STORE_DEREF, but that is
    invalid syntax on Python 2. If we use this function we also don't need
    to do the weird freevars/cellvars swap below
    """
    def inner(value):
        lambda: cell  # make ``cell`` a closure so that we get a STORE_DEREF
        cell = value

    co = inner.__code__

    # NOTE: we are marking the cell variable as a free variable intentionally
    # so that we simulate an inner function instead of the outer function. This
    # is what gives us the ``nonlocal`` behavior in a Python 2 compatible way.
    if not PY3:
        return types.CodeType(
            co.co_argcount,
            co.co_nlocals,
            co.co_stacksize,
            co.co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_cellvars,  # this is the trickery
            (),
        )
    elif not PY38:
        return types.CodeType(
            co.co_argcount,
            co.co_kwonlyargcount,
            co.co_nlocals,
            co.co_stacksize,
            co.co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_cellvars,  # this is the trickery
            (),
        )
    else:
        return types.CodeType(
            co.co_argcount,
            co.co_posonlyargcount,
            co.co_kwonlyargcount,
            co.co_nlocals,
            co.co_stacksize,
            co.co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_cellvars,  # this is the trickery
            (),
        )


_cell_set_template_code = _make_cell_set_template_code()


def cell_set(cell, value):
    """Set the value of a closure cell.
    """
    return types.FunctionType(
        _cell_set_template_code,
        {},
        '_cell_set_inner',
        (),
        (cell,),
    )(value)


# relevant opcodes
STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
EXTENDED_ARG = dis.EXTENDED_ARG


def islambda(func):
    return getattr(func, '__name__') == '<lambda>'


_BUILTIN_TYPE_NAMES = {}
for k, v in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k


def _builtin_type(name):
    return getattr(types, name)


def _make__new__factory(type_):
    def _factory():
        return type_.__new__
    return _factory


# NOTE: These need to be module globals so that they're pickleable as globals.
_get_dict_new = _make__new__factory(dict)
_get_frozenset_new = _make__new__factory(frozenset)
_get_list_new = _make__new__factory(list)
_get_set_new = _make__new__factory(set)
_get_tuple_new = _make__new__factory(tuple)
_get_object_new = _make__new__factory(object)

# Pre-defined set of builtin_function_or_method instances that can be
# serialized.
_BUILTIN_TYPE_CONSTRUCTORS = {
    dict.__new__: _get_dict_new,
    frozenset.__new__: _get_frozenset_new,
    set.__new__: _get_set_new,
    list.__new__: _get_list_new,
    tuple.__new__: _get_tuple_new,
    object.__new__: _get_object_new,
}


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


class CloudPickler(Pickler):

    dispatch = Pickler.dispatch.copy()

    def __init__(self, file, protocol=None, dump_code=False):
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        Pickler.__init__(self, file, protocol=protocol)
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
            else:
                raise

    if sys.version_info < (2, 7):
        memoryview = bytearray  # to make flake8 happy
    else:
        def save_memoryview(self, obj):
            self.save(obj.tobytes())

        dispatch[memoryview] = save_memoryview

    if not PY3:
        def save_buffer(self, obj):
            self.save(str(obj))

        dispatch[buffer] = save_buffer  # noqa: F821 'buffer' was removed in Python 3

    def save_module(self, obj):
        """
        Save a module as an import
        """
        self.modules.add(obj)
        if _is_dynamic(obj):
            self.save_reduce(dynamic_subimport, (obj.__name__, vars(obj)),
                             obj=obj)
        else:
            self.save_reduce(subimport, (obj.__name__,), obj=obj)

    dispatch[types.ModuleType] = save_module

    @staticmethod
    def _extract_code_args(obj):
        if PY38:
            args = (
                obj.co_argcount, obj.co_posonlyargcount, obj.co_kwonlyargcount, obj.co_nlocals,
                obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts, obj.co_names,
                obj.co_varnames, obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab,
                obj.co_freevars, obj.co_cellvars
            )
        elif PY3:
            args = (
                obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
                obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames,
                obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars,
                obj.co_cellvars
            )
        else:
            args = (
                obj.co_argcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code,
                obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name,
                obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars
            )
        return args

    def save_codeobject(self, obj):
        """
        Save a code object
        """
        if self.dump_code:
            print(obj.co_name)
            dis.dis(obj.co_code)
        self.save_reduce(types.CodeType, self._extract_code_args(obj), obj=obj)

    dispatch[types.CodeType] = save_codeobject

    def save_function(self, obj, name=None):
        """ Registered with the dispatch to handle all function types.

        Determines what kind of function obj is (e.g. lambda, defined at
        interactive prompt, etc) and handles the pickling appropriately.
        """
        try:
            should_special_case = obj in _BUILTIN_TYPE_CONSTRUCTORS
        except TypeError:
            # Methods of builtin types aren't hashable in python 2.
            should_special_case = False

        if should_special_case:
            # We keep a special-cased cache of built-in type constructors at
            # global scope, because these functions are structured very
            # differently in different python versions and implementations (for
            # example, they're instances of types.BuiltinFunctionType in
            # CPython, but they're ordinary types.FunctionType instances in
            # PyPy).
            #
            # If the function we've received is in that cache, we just
            # serialize it as a lookup into the cache.
            return self.save_reduce(_BUILTIN_TYPE_CONSTRUCTORS[obj], (), obj=obj)

        write = self.write

        if name is None:
            name = obj.__name__
        try:
            # whichmodule() could fail, see
            # https://bitbucket.org/gutworth/six/issues/63/importing-six-breaks-pickling
            modname = pickle.whichmodule(obj, name)
        except Exception:
            modname = None
        # print('which gives %s %s %s' % (modname, obj, name))
        try:
            themodule = sys.modules[modname]
        except KeyError:
            # eval'd items such as namedtuple give invalid items for their function __module__
            modname = '__main__'

        if modname == '__main__':
            themodule = None

        try:
            lookedup_by_name = getattr(themodule, name, None)
        except Exception:
            lookedup_by_name = None

        if themodule:
            self.modules.add(themodule)
            if lookedup_by_name is obj:
                return self.save_global(obj, name)

        # a builtin_function_or_method which comes in as an attribute of some
        # object (e.g., itertools.chain.from_iterable) will end
        # up with modname "__main__" and so end up here. But these functions
        # have no __code__ attribute in CPython, so the handling for
        # user-defined functions below will fail.
        # So we pickle them here using save_reduce; have to do it differently
        # for different python versions.
        if not hasattr(obj, '__code__'):
            if PY3:
                rv = obj.__reduce_ex__(self.proto)
            else:
                if hasattr(obj, '__self__'):
                    rv = (getattr, (obj.__self__, name))
                else:
                    raise pickle.PicklingError("Can't pickle %r" % obj)
            return self.save_reduce(obj=obj, *rv)

        # if func is lambda, def'ed at prompt, is in main, or is nested, then
        # we'll pickle the actual function object rather than simply saving a
        # reference (as is done in default pickler), via save_function_tuple.
        if (islambda(obj)
                or getattr(obj.__code__, 'co_filename', None) == '<stdin>'
                or themodule is None):
            self.save_function_tuple(obj)
            return
        else:
            # func is nested
            if lookedup_by_name is None or lookedup_by_name is not obj:
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

    def _save_subimports(self, code, top_level_dependencies):
        """
        Ensure de-pickler imports any package child-modules that
        are needed by the function
        """

        # check if any known dependency is an imported package
        for x in top_level_dependencies:
            if isinstance(x, types.ModuleType) and hasattr(x, '__package__') and x.__package__:
                # check if the package has any currently loaded sub-imports
                prefix = x.__name__ + '.'
                # A concurrent thread could mutate sys.modules,
                # make sure we iterate over a copy to avoid exceptions
                for name in list(sys.modules):
                    # Older versions of pytest will add a "None" module to sys.modules.
                    if name is not None and name.startswith(prefix):
                        # check whether the function can address the sub-module
                        tokens = set(name[len(prefix):].split('.'))
                        if not tokens - set(code.co_names):
                            # ensure unpickler executes this import
                            self.save(sys.modules[name])
                            # then discards the reference to it
                            self.write(pickle.POP)

    def save_dynamic_class(self, obj):
        """
        Save a class that can't be stored as module global.

        This method is used to serialize classes that are defined inside
        functions, or that otherwise can't be serialized as attribute lookups
        from global modules.
        """
        clsdict = dict(obj.__dict__)  # copy dict proxy to a dict
        clsdict.pop('__weakref__', None)

        # For ABCMeta in python3.7+, remove _abc_impl as it is not picklable.
        # This is a fix which breaks the cache but this only makes the first
        # calls to issubclass slower.
        if "_abc_impl" in clsdict:
            import abc
            (registry, _, _, _) = abc._get_dump(obj)
            clsdict["_abc_impl"] = [subclass_weakref()
                                    for subclass_weakref in registry]

        # On PyPy, __doc__ is a readonly attribute, so we need to include it in
        # the initial skeleton class.  This is safe because we know that the
        # doc can't participate in a cycle with the original class.
        type_kwargs = {'__doc__': clsdict.pop('__doc__', None)}

        # If type overrides __dict__ as a property, include it in the type kwargs.
        # In Python 2, we can't set this attribute after construction.
        __dict__ = clsdict.pop('__dict__', None)
        if isinstance(__dict__, property):
            type_kwargs['__dict__'] = __dict__

        save = self.save
        write = self.write

        # We write pickle instructions explicitly here to handle the
        # possibility that the type object participates in a cycle with its own
        # __dict__. We first write an empty "skeleton" version of the class and
        # memoize it before writing the class' __dict__ itself. We then write
        # instructions to "rehydrate" the skeleton class by restoring the
        # attributes from the __dict__.
        #
        # A type can appear in a cycle with its __dict__ if an instance of the
        # type appears in the type's __dict__ (which happens for the stdlib
        # Enum class), or if the type defines methods that close over the name
        # of the type, (which is common for Python 2-style super() calls).

        # Push the rehydration function.
        save(_rehydrate_skeleton_class)

        # Mark the start of the args tuple for the rehydration function.
        write(pickle.MARK)

        # Create and memoize an skeleton class with obj's name and bases.
        tp = type(obj)
        self.save_reduce(tp, (obj.__name__, obj.__bases__, type_kwargs), obj=obj)

        # Now save the rest of obj's __dict__. Any references to obj
        # encountered while saving will point to the skeleton class.
        save(clsdict)

        # Write a tuple of (skeleton_class, clsdict).
        write(pickle.TUPLE)

        # Call _rehydrate_skeleton_class(skeleton_class, clsdict)
        write(pickle.REDUCE)

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
        if is_tornado_coroutine(func):
            self.save_reduce(_rebuild_tornado_coroutine, (func.__wrapped__,),
                             obj=func)
            return

        save = self.save
        write = self.write

        code, f_globals, defaults, closure_values, dct, base_globals = self.extract_func_data(func)

        save(_fill_function)  # skeleton function updater
        write(pickle.MARK)    # beginning of tuple that _fill_function expects

        self._save_subimports(
            code,
            itertools.chain(f_globals.values(), closure_values or ()),
        )

        # create a skeleton function object and memoize it
        save(_make_skel_func)
        save((
            code,
            len(closure_values) if closure_values is not None else -1,
            base_globals,
        ))
        write(pickle.REDUCE)
        self.memoize(func)

        # save the rest of the func data needed by _fill_function
        state = {
            'globals': f_globals,
            'defaults': defaults,
            'dict': dct,
            'closure_values': closure_values,
            'module': func.__module__,
            'name': func.__name__,
            'doc': func.__doc__,
        }
        if hasattr(func, '__annotations__') and sys.version_info >= (3, 7):
            state['annotations'] = func.__annotations__
        if hasattr(func, '__qualname__'):
            state['qualname'] = func.__qualname__
        save(state)
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
                out_names = set(names[oparg] for _, oparg in _walk_global_ops(co))

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
            code, globals, defaults, closure_values, dict
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
        closure = (
            list(map(_get_cell_contents, func.__closure__))
            if func.__closure__ is not None
            else None
        )

        # save the dict
        dct = func.__dict__

        base_globals = self.globals_ref.get(id(func.__globals__), None)
        if base_globals is None:
            # For functions defined in a well behaved module use
            # vars(func.__module__) for base_globals. This is necessary to
            # share the global variables across multiple pickled functions from
            # this module.
            if hasattr(func, '__module__') and func.__module__ is not None:
                base_globals = func.__module__
            else:
                base_globals = {}
        self.globals_ref[id(func.__globals__)] = base_globals

        return (code, f_globals, defaults, closure, dct, base_globals)

    def save_builtin_function(self, obj):
        if obj.__module__ == "__builtin__":
            return self.save_global(obj)
        return self.save_function(obj)

    dispatch[types.BuiltinFunctionType] = save_builtin_function

    def save_global(self, obj, name=None, pack=struct.pack):
        """
        Save a "global".

        The name of this method is somewhat misleading: all types get
        dispatched here.
        """
        if obj is type(None):
            return self.save_reduce(type, (None,), obj=obj)
        elif obj is type(Ellipsis):
            return self.save_reduce(type, (Ellipsis,), obj=obj)
        elif obj is type(NotImplemented):
            return self.save_reduce(type, (NotImplemented,), obj=obj)

        if obj.__module__ == "__main__":
            return self.save_dynamic_class(obj)

        try:
            return Pickler.save_global(self, obj, name=name)
        except Exception:
            if obj.__module__ == "__builtin__" or obj.__module__ == "builtins":
                if obj in _BUILTIN_TYPE_NAMES:
                    return self.save_reduce(
                        _builtin_type, (_BUILTIN_TYPE_NAMES[obj],), obj=obj)

            typ = type(obj)
            if typ is not obj and isinstance(obj, (type, types.ClassType)):
                return self.save_dynamic_class(obj)

            raise

    dispatch[type] = save_global
    dispatch[types.ClassType] = save_global

    def save_instancemethod(self, obj):
        # Memoization rarely is ever useful due to python bounding
        if obj.__self__ is None:
            self.save_reduce(getattr, (obj.im_class, obj.__name__))
        else:
            if PY3:
                self.save_reduce(types.MethodType, (obj.__func__, obj.__self__), obj=obj)
            else:
                self.save_reduce(types.MethodType, (obj.__func__, obj.__self__, obj.__self__.__class__),
                                 obj=obj)

    dispatch[types.MethodType] = save_instancemethod

    def save_inst(self, obj):
        """Inner logic to save instance. Based off pickle.save_inst"""
        cls = obj.__class__

        # Try the dispatch table (pickle module doesn't do it)
        f = self.dispatch.get(cls)
        if f:
            f(self, obj)  # Call unbound method with explicit self
            return

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
            items = (items,)
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

    def save_partial(self, obj):
        """Partial objects do not serialize correctly in python2.x -- this fixes the bugs"""
        self.save_reduce(_genpartial, (obj.func, obj.args, obj.keywords))

    if sys.version_info < (2, 7):  # 2.7 supports partial pickling
        dispatch[partial] = save_partial

    def save_file(self, obj):
        """Save a file"""
        try:
            import StringIO as pystringIO  # we can't use cStringIO as it lacks the name attribute
        except ImportError:
            import io as pystringIO

        if not hasattr(obj, 'name') or not hasattr(obj, 'mode'):
            raise pickle.PicklingError("Cannot pickle files that do not map to an actual file")
        if obj is sys.stdout:
            return self.save_reduce(getattr, (sys, 'stdout'), obj=obj)
        if obj is sys.stderr:
            return self.save_reduce(getattr, (sys, 'stderr'), obj=obj)
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

    def save_ellipsis(self, obj):
        self.save_reduce(_gen_ellipsis, ())

    def save_not_implemented(self, obj):
        self.save_reduce(_gen_not_implemented, ())

    try:               # Python 2
        dispatch[file] = save_file
    except NameError:  # Python 3
        dispatch[io.TextIOWrapper] = save_file

    dispatch[type(Ellipsis)] = save_ellipsis
    dispatch[type(NotImplemented)] = save_not_implemented

    if hasattr(weakref, 'WeakSet'):
        def save_weakset(self, obj):
            self.save_reduce(weakref.WeakSet, (list(obj),))

        dispatch[weakref.WeakSet] = save_weakset

    def save_logger(self, obj):
        self.save_reduce(logging.getLogger, (obj.name,), obj=obj)

    dispatch[logging.Logger] = save_logger

    def save_root_logger(self, obj):
        self.save_reduce(logging.getLogger, (), obj=obj)

    dispatch[logging.RootLogger] = save_root_logger

    """Special functions for Add-on libraries"""
    def inject_addons(self):
        """Plug in system. Register additional pickling functions if modules already loaded"""
        pass


# Tornado support

def is_tornado_coroutine(func):
    """
    Return whether *func* is a Tornado coroutine function.
    Running coroutines are not supported.
    """
    if 'tornado.gen' not in sys.modules:
        return False
    gen = sys.modules['tornado.gen']
    if not hasattr(gen, "is_coroutine_function"):
        # Tornado version is too old
        return False
    return gen.is_coroutine_function(func)


def _rebuild_tornado_coroutine(func):
    from tornado import gen
    return gen.coroutine(func)


# Shorthands for legacy support

def dump(obj, file, protocol=None, dump_code=False):
    """Serialize obj as bytes streamed into file

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication speed
    between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python.
    """
    CloudPickler(file, protocol=protocol, dump_code=dump_code).dump(obj)


def dumps(obj, protocol=None, dump_code=False):
    """Serialize obj as a string of bytes allocated in memory

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication speed
    between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python.
    """
    file = StringIO()
    try:
        cp = CloudPickler(file, protocol=protocol, dump_code=dump_code)
        cp.dump(obj)
        return file.getvalue()
    finally:
        file.close()


# here we use a customized unpickler version instead of the original one
# to do code translation as well as dealing with sandbox in MaxCompute
class CloudUnpickler(Unpickler):

    dispatch = Unpickler.dispatch.copy()

    def __init__(self, *args, **kwargs):
        self._src_major, self._src_minor, self._src_impl = kwargs.pop('impl', None) or (None, None, None)
        self._src_version = (self._src_major, self._src_minor) if self._src_major is not None else None

        self._dump_code = kwargs.pop('dump_code', False)
        Unpickler.__init__(self, *args, **kwargs)

    def find_class(self, module, name):
        # Subclasses may override this
        try:
            if PY3 and _compat_pickle and self.proto < 3 and self.fix_imports:
                if (module, name) in _compat_pickle.NAME_MAPPING:
                    module, name = _compat_pickle.NAME_MAPPING[(module, name)]
                elif module in _compat_pickle.IMPORT_MAPPING:
                    module = _compat_pickle.IMPORT_MAPPING[module]

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
        # cause `marshal.loads` has been blocked by MaxCompute python sandbox.
        self.append(struct.unpack('<i', self.read(4))[0])
    dispatch[pickle.BININT] = load_binint

    def load_binint2(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by MaxCompute python sandbox.
        self.append(struct.unpack('<i', self.read(2) + '\000\000')[0])
    dispatch[pickle.BININT2] = load_binint2

    def load_long4(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by MaxCompute python sandbox.
        n = struct.unpack('<i', self.read(4))[0]
        bytes = self.read(n)
        self.append(pickle.decode_long(bytes))
    dispatch[pickle.LONG4] = load_long4

    def load_binstring(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by MaxCompute python sandbox.
        len = struct.unpack('<i', self.read(4))[0]
        self.append(self.read(len))
    dispatch[pickle.BINSTRING] = load_binstring

    def load_binunicode(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by MaxCompute python sandbox.
        len = struct.unpack('<i', self.read(4))[0]
        self.append(unicode(self.read(len), 'utf-8'))
    dispatch[pickle.BINUNICODE] = load_binunicode

    def load_ext2(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by MaxCompute python sandbox.
        code = struct.unpack('<i', self.read(2) + '\000\000')[0]
        self.get_extension(code)
    dispatch[pickle.EXT2] = load_ext2

    def load_ext4(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by MaxCompute python sandbox.
        code = struct.unpack('<i', self.read(4))[0]
        self.get_extension(code)
    dispatch[pickle.EXT4] = load_ext4

    def load_long_binget(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by MaxCompute python sandbox.
        i = struct.unpack('<i', self.read(4))[0]
        self.append(self.memo[repr(i)])
    dispatch[pickle.LONG_BINGET] = load_long_binget

    def load_long_binput(self):
        # Replace the internal implementation of pickle
        # cause `marshal.loads` has been blocked by MaxCompute python sandbox.
        i = struct.unpack('<i', self.read(4))[0]
        self.memo[repr(i)] = self.stack[-1]
    dispatch[pickle.LONG_BINPUT] = load_long_binput

    def load_reduce(self):
        # Replace the internal implementation of pickle
        # cause code representation in Python 3 differs from that in Python 2
        stack = self.stack
        args = stack.pop()
        func = stack[-1]
        if self._src_version is not None:
            if func.__name__ == 'code':
                if sys.version_info[:2] == (2, 7):
                    if self._src_version >= (3, 6):  # src >= PY36, dest PY27
                        args = Cp36_Cp35(args).translate_code()
                        args = Cp35_Cp27(args).translate_code()
                    elif self._src_major == 3 and self._src_version <= (3, 5):  # src PY3 && src <= PY35, dest PY27
                        args = Cp35_Cp27(args).translate_code()
                    elif self._src_version == (2, 6):  # src PY26, dest PY27
                        args = Cp26_Cp27(args).translate_code()
                    elif not hasattr(sys, "pypy_version_info") and self._src_impl == 'pypy':
                        args = Pypy2_Cp27(args).translate_code()
                elif sys.version_info[:2] != self._src_version:
                    raise NotImplementedError('Code conversion from Python %r to %r is not supported yet.'
                                              % (self._src_version, sys.version_info[:2]))

                if self._dump_code:
                    print(args[9 if not PY3 else 10])
                    dis.dis(args[4 if not PY3 else 5])
                    sys.stdout.flush()
            elif func.__name__ == 'type' or func.__name__ == 'classobj' or (
                    isinstance(func, type) and issubclass(func, type)):
                if not PY3:
                    args = list(args)
                    args[0] = args[0].encode('utf-8') if isinstance(args[0], unicode) else args[0]
        try:
            value = func(*args)
        except Exception as exc:
            traceback.print_exc()
            raise Exception('Failed to unpickle reduce. func=%s mod=%s args=%s msg="%s"' % (
            func.__name__, func.__module__, repr(args), str(exc)))
        stack[-1] = value
    dispatch[pickle.REDUCE] = load_reduce


def load(file, impl=None, dump_code=False):
    return CloudUnpickler(file, impl=impl, dump_code=dump_code).load()


def loads(str, impl=None, dump_code=False):
    file = StringIO(str)
    return CloudUnpickler(file, impl=impl, dump_code=dump_code).load()


# hack for __import__ not working as desired
def subimport(name):
    __import__(name)
    return sys.modules[name]


def dynamic_subimport(name, vars):
    mod = types.ModuleType(name)
    mod.__dict__.update(vars)
    return mod


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
            except Exception:
                sys.stderr.write('warning: could not import %s\n.  '
                                 'Your function may unexpectedly error due to this import failing;'
                                 'A version mismatch is likely.  Specific error was:\n' % modname)
                print_exec(sys.stderr)
            else:
                setattr(main, mod.__name__, mod)


# object generators:
def _genpartial(func, args, kwds):
    if not args:
        args = ()
    if not kwds:
        kwds = {}
    return partial(func, *args, **kwds)


def _gen_ellipsis():
    return Ellipsis


def _gen_not_implemented():
    return NotImplemented


def _get_cell_contents(cell):
    try:
        return cell.cell_contents
    except ValueError:
        # sentinel used by ``_fill_function`` which will leave the cell empty
        return _empty_cell_value


def instance(cls):
    """Create a new instance of a class.

    Parameters
    ----------
    cls : type
        The class to create an instance of.

    Returns
    -------
    instance : cls
        A new instance of ``cls``.
    """
    return cls()


@instance
class _empty_cell_value(object):
    """sentinel for empty closures
    """
    @classmethod
    def __reduce__(cls):
        return cls.__name__


def _fill_function(*args):
    """Fills in the rest of function data into the skeleton function object

    The skeleton itself is create by _make_skel_func().
    """
    if len(args) == 2:
        func = args[0]
        state = args[1]
    elif len(args) == 5:
        # Backwards compat for cloudpickle v0.4.0, after which the `module`
        # argument was introduced
        func = args[0]
        keys = ['globals', 'defaults', 'dict', 'closure_values']
        state = dict(zip(keys, args[1:]))
    elif len(args) == 6:
        # Backwards compat for cloudpickle v0.4.1, after which the function
        # state was passed as a dict to the _fill_function it-self.
        func = args[0]
        keys = ['globals', 'defaults', 'dict', 'module', 'closure_values']
        state = dict(zip(keys, args[1:]))
    else:
        raise ValueError('Unexpected _fill_value arguments: %r' % (args,))

    # Only set global variables that do not exist.
    for k, v in state['globals'].items():
        if k not in func.__globals__:
            func.__globals__[k] = v

    func.__defaults__ = state['defaults']
    func.__dict__ = state['dict']
    if 'annotations' in state:
        func.__annotations__ = state['annotations']
    if 'doc' in state:
        func.__doc__  = state['doc']
    if 'name' in state:
        func.__name__ = str(state['name'])
    if 'module' in state:
        func.__module__ = state['module']
    if 'qualname' in state:
        func.__qualname__ = state['qualname']

    cells = func.__closure__
    if cells is not None:
        for cell, value in zip(cells, state['closure_values']):
            if value is not _empty_cell_value:
                cell_set(cell, value)

    return func


def _make_empty_cell():
    if False:
        # trick the compiler into creating an empty cell in our lambda
        cell = None
        raise AssertionError('this route should not be executed')

    return (lambda: cell).__closure__[0]


def _make_skel_func(code, cell_count, base_globals=None):
    """ Creates a skeleton function object that contains just the provided
        code and the correct number of cells in func_closure.  All other
        func attributes (e.g. func_globals) are empty.
    """
    if base_globals is None:
        base_globals = {}
    elif isinstance(base_globals, string_types):
        base_globals_name = base_globals
        try:
            # First try to reuse the globals from the module containing the
            # function. If it is not possible to retrieve it, fallback to an
            # empty dictionary.
            if importlib is not None:
                base_globals = vars(importlib.import_module(base_globals))
            elif sys.modules.get(base_globals, None) is not None:
                base_globals = vars(sys.modules[base_globals])
            else:
                raise ImportError
        except ImportError:
            base_globals = _dynamic_modules_globals.get(
                    base_globals_name, None)
            if base_globals is None:
                base_globals = _DynamicModuleFuncGlobals()
            _dynamic_modules_globals[base_globals_name] = base_globals

    base_globals['__builtins__'] = __builtins__

    closure = (
        tuple(_make_empty_cell() for _ in range(cell_count))
        if cell_count >= 0 else
        None
    )
    return types.FunctionType(code, base_globals, None, None, closure)


def _rehydrate_skeleton_class(skeleton_class, class_dict):
    """Put attributes from `class_dict` back on `skeleton_class`.

    See CloudPickler.save_dynamic_class for more info.
    """
    registry = None
    for attrname, attr in class_dict.items():
        if attrname == "_abc_impl":
            registry = attr
        else:
            setattr(skeleton_class, attrname, attr)
    if registry is not None:
        for subclass in registry:
            skeleton_class.register(subclass)

    return skeleton_class


def _is_dynamic(module):
    """
    Return True if the module is special module that cannot be imported by its
    name.
    """
    # Quick check: module that have __file__ attribute are not dynamic modules.
    if hasattr(module, '__file__'):
        return False

    if hasattr(module, '__spec__'):
        return module.__spec__ is None
    else:
        # Backward compat for Python 2
        import imp
        try:
            path = None
            for part in module.__name__.split('.'):
                if path is not None:
                    path = [path]
                f, path, description = imp.find_module(part, path)
                if f is not None:
                    f.close()
        except ImportError:
            return True
        return False


"""Constructors for 3rd party libraries
Note: These can never be renamed due to client compatibility issues"""


def _getobject(modname, attribute):
    mod = __import__(modname, fromlist=[attribute])
    return mod.__dict__[attribute]


""" Use copy_reg to extend global pickle definitions """

if sys.version_info < (3, 4):
    method_descriptor = type(str.upper)

    def _reduce_method_descriptor(obj):
        return (getattr, (obj.__objclass__, obj.__name__))

    try:
        import copy_reg as copyreg
    except ImportError:
        import copyreg
    copyreg.pickle(method_descriptor, _reduce_method_descriptor)


"""
Code blow resolves bytecode translation between py26 / py3 and py27
"""
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
CALL_METHOD_PY37 = 161
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
LOAD_METHOD_PY37 = 160
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


class CodeRewriter(_with_metaclass(CodeRewriterMeta)):
    _translator = dict()

    CO_NLOCALS_POS = None
    CO_CODE_POS = None
    CO_CONSTS_POS = None
    CO_NAMES_POS = None
    CO_VARNAMES_POS = None
    CO_FILENAME_POS = None
    CO_NAME_POS = None
    CO_FREEVARS_POS = None
    CO_CELLVARS_POS = None
    CO_LNOTAB_POS = None
    OP_EXTENDED_ARG = EXTENDED_ARG

    def __init__(self, code_args):
        self.code_args = list(code_args)
        self._const_poses = dict()
        self._name_poses = dict()
        self._varname_poses = dict()
        self.code_writer = StringIO()

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
        cur_line, last_line = 0, 0
        cur_new_pc, last_new_pc = 0, 0
        for old_pc_delta, line_delta in zip(lnotab[::2], lnotab[1::2]):
            old_pc_delta = ord(old_pc_delta)

            line_delta = ord(line_delta)
            if line_delta >= 0x80:
                line_delta -= 0x100
            cur_line += line_delta
            cur_old_pc += old_pc_delta

            cur_new_pc = old_to_new[cur_old_pc]

            line_delta = cur_line - last_line
            pc_delta = cur_new_pc - last_new_pc

            while pc_delta >= 0x100:
                sio.write(bytes(bytearray([0xff, 0])))
                pc_delta -= 0xff
            while line_delta >= 0x80:
                sio.write(bytes(bytearray([0x7f, pc_delta])))
                pc_delta = 0
                line_delta -= 0x7f
            while line_delta < -128:
                sio.write(bytes(bytearray([0xff, pc_delta])))
                pc_delta = 0
                line_delta += 128

            if pc_delta or line_delta:
                sio.write(bytes(bytearray([pc_delta, (line_delta + 0x100) % 0x100])))

            last_new_pc = cur_new_pc
            last_line = cur_line
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
    CO_FILENAME_POS = 8
    CO_NAME_POS = 9
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
    CO_FILENAME_POS = 9
    CO_NAME_POS = 10
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
        code_args = list(CloudPickler._extract_code_args(func.__code__))

        # start translating code
        consts = code_args[5]
        n_consts = len(consts)
        name_cid, base_cid, metaclass_cid = irange(n_consts, n_consts + 3)
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
    def _conv_import_from_list(args):
        v = args[0]
        if not v:
            return v
        return tuple(to_ascii(a) for a in v)

    @staticmethod
    def _matmul(args):
        try:
            import numpy as np
        except ImportError:
            raise
        return np.dot(args[0], args[1])

    @staticmethod
    def _imatmul(args):
        try:
            import numpy as np
        except ImportError:
            raise
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

    @op_translator(IMPORT_NAME)
    def handle_import_name(self, arg):
        return sum([
            self.write_replacement_call(self._conv_import_from_list, 1),
            self.write_instruction(IMPORT_NAME, arg),
        ])

    @staticmethod
    def _conv_string_tuples(tp):
        if not PY3:
            return tuple(s.encode('utf-8') if isinstance(s, unicode) else s for s in tp)
        else:
            return tuple(s.decode('utf-8') if isinstance(s, bytes) else s for s in tp)

    def translate_code(self):
        code_args = super(Cp35_Cp27, self).translate_code()

        for col in (self.CO_CONSTS_POS, self.CO_NAMES_POS, self.CO_VARNAMES_POS,
                    self.CO_FREEVARS_POS, self.CO_CELLVARS_POS):
            code_args[col] = self._conv_string_tuples(code_args[col])

        for col in (self.CO_FILENAME_POS, self.CO_NAME_POS):
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

    @op_translator(LOAD_METHOD_PY37)
    def handle_load_method(self, arg):
        none_id = self.patch_consts(None)
        return sum([
            self.write_instruction(LOAD_ATTR, arg),
            self.write_instruction(LOAD_CONST, none_id),
            self.write_instruction(ROT_TWO),
        ])

    @op_translator(CALL_METHOD_PY37)
    def handle_call_method(self, arg):
        return sum([
            self.write_instruction(CALL_FUNCTION, arg),
            self.write_instruction(ROT_TWO),
            self.write_instruction(POP_TOP),
        ])
