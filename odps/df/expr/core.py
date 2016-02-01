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

import inspect

import six

from ... import compat
from . import utils


class NodeMetaclass(type):
    def __new__(mcs, name, bases, kv):
        if kv.get('_add_args_slots', True):
            slots = list(kv.get('__slots__', []))
            args = kv.get('_args', [])
            slots.extend(args)
            slots = compat.OrderedDict.fromkeys(slots)

            kv['__slots__'] = tuple(slot for slot in slots if slot not in kv)

        return type.__new__(mcs, name, bases, kv)


class Node(six.with_metaclass(NodeMetaclass)):
    __slots__ = '_cached_args', '__weakref__'
    _args = ()

    def __init__(self, *args, **kwargs):
        for arg, value in zip(self._args, args):
            setattr(self, arg, value)

        for key, value in six.iteritems(kwargs):
            setattr(self, key, value)

        self._cached_args = None

    @property
    def args(self):
        if self._cached_args is not None:
            return self._cached_args
        self._cached_args = self._get_args()
        return self._cached_args

    def _get_args(self):
        return tuple(getattr(self, arg, None) for arg in self._args)

    def iter_args(self):
        for arg in self._args:
            yield arg, getattr(self, arg, None)

    def data_source(self):
        for arg in self.children():
            for source in arg.data_source():
                yield source

    def output_type(self):
        if hasattr(self, '_validator'):
            return self.validator.output_type()

    def substitute(self, old_arg, new_arg, parent_cache=None):
        if hasattr(old_arg, '_name') and old_arg._name is not None and \
                        new_arg._name is None:
            new_arg = new_arg.rename(old_arg._name)

        cached_args = []
        for arg in self.args:
            if not isinstance(arg, (list, tuple)):
                if arg is old_arg:
                    cached_args.append(new_arg)
                else:
                    cached_args.append(arg)
            else:
                subs = list(arg)
                for i in range(len(subs)):
                    if subs[i] is old_arg:
                        subs[i] = new_arg
                cached_args.append(type(arg)(subs))
        self._cached_args = tuple(cached_args)

        if parent_cache is None:
            return
        if id(old_arg) in parent_cache:
            try:
                parent_cache[id(old_arg)].remove(self)
            except KeyError:
                pass
        if id(new_arg) not in parent_cache:
            parent_cache[id(new_arg)] = set()
        parent_cache[id(new_arg)].add(self)

    def children(self):
        args = []

        for arg in self.args:
            if isinstance(arg, (list, tuple)):
                args.extend(arg)
            else:
                args.append(arg)

        return [arg for arg in args if arg is not None]

    def leaves(self):
        args = self.children()

        for arg in args:
            if arg is None:
                continue
            if len(arg.args) == 0:
                yield arg
            else:
                for leave in arg.leaves():
                    yield leave

        if len(args) == 0:
            yield self

    def traverse(self, top_down=False, parent_cache=None, unique=False):
        def _traverse(node):
            args = node.children()

            if top_down:
                yield node

            for arg in args:
                if arg is None:
                    continue

                if parent_cache is not None:
                    if id(arg) not in parent_cache:
                        parent_cache[id(arg)] = set([node, ])
                    else:
                        parent_cache[id(arg)].add(node)
                if len(arg.args) == 0:
                    yield arg
                else:
                    for item in _traverse(arg):
                        yield item

            if not top_down:
                yield node

        if unique:
            traversed = set()
            for arg in _traverse(self):
                if id(arg) in traversed:
                    continue
                else:
                    yield arg
                    traversed.add(id(arg))
        else:
            for n in _traverse(self):
                yield n

    def _slot_values(self):
        return [getattr(self, slot, None) for slot in utils.get_attrs(self)
                if slot != '_cached_args']

    def __eq__(self, other):
        return self.equals(other)

    def equals(self, other):
        if other is None:
            return False

        # Hack here, we cannot just check the type here,
        # many types like Column is created dynamically,
        # so we check the class name and the sub classes.
        if self.__class__.__name__ !=  other.__class__.__name__ or \
                inspect.getmro(type(self))[1:] != inspect.getmro(type(self))[1:]:
            return False

        def cmp(x, y):
            if isinstance(x, Node):
                res = x.equals(y)
            elif isinstance(y, Node):
                res = y.equals(y)
            elif isinstance(x, (tuple, list)) and \
                    isinstance(y, (tuple, list)):
                res = all(map(cmp, x, y))
            else:
                res = x == y
            return res
        return all(map(cmp, self._slot_values(), other._slot_values()))

    def __hash__(self):
        to_tuple = lambda it: tuple(it) if isinstance(it, list) else it
        return hash((type(self), tuple(to_tuple(arg) for arg in self.args)))

    def is_ancestor(self, other):
        for node in self.traverse(unique=True):
            if node is other:
                return True

        return False

    def path(self, other):
        if not self.is_ancestor(other):
            if not other.is_ancestor(self):
                return
            else:
                expr, other = other, self
        else:
            expr = self

        while not expr.equals(other):
            yield expr

            for child in expr.children():
                if child.is_ancestor(other):
                    expr = child
                    break

        yield expr

    def all_path(self, other, strict=False):
        if not self.is_ancestor(other):
            if strict:
                return

            if not other.is_ancestor(self):
                return
            else:
                expr, other = other, self
        else:
            expr = self

        if self is other:
            yield [other, ]

        for child in expr.children():
            if child.is_ancestor(other):
                for path in child.all_path(other):
                    yield [expr, ] + path

    def __getstate__(self):
        slots = utils.get_attrs(self)

        return tuple((slot, object.__getattribute__(self, slot)) for slot in slots
                     if not slot.startswith('__'))

    def __setstate__(self, state):
        self.__init__(**dict(state))
