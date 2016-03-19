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

from __future__ import absolute_import

import inspect
import itertools
from collections import deque, defaultdict

import six

from ... import compat
from . import utils


class NodeMetaclass(type):
    def __new__(mcs, name, bases, kv):
        if kv.get('_add_args_slots', True):
            slots = list(kv.get('__slots__', [])) + list(kv.get('_slots', []))
            args = kv.get('_args', [])
            slots.extend(args)
            slots = compat.OrderedDict.fromkeys(slots)

            kv['__slots__'] = tuple(slot for slot in slots if slot not in kv)

        return type.__new__(mcs, name, bases, kv)


class Node(six.with_metaclass(NodeMetaclass)):
    __slots__ = '_cached_args', '__weakref__', '__parents'
    _args = ()

    def __init__(self, *args, **kwargs):
        self._init(*args, **kwargs)
        self._fill_parents()

    def _init(self, *args, **kwargs):
        for arg, value in zip(self._args, args):
            setattr(self, arg, value)

        for key, value in six.iteritems(kwargs):
            setattr(self, key, value)

        self._cached_args = None
        self.__parents = set()

    def _fill_parents(self):
        for child in self.children():
            child.__parents.add((id(self), self))
        self._cached_args = None

    def _remove_parent(self, parent):
        key = id(parent), parent
        if key in self.__parents:
            self.__parents.remove(key)

    def _add_parent(self, parent):
        key = id(parent), parent
        self.__parents.add(key)

    @property
    def parents(self):
        return [p[1] for p in self.__parents]

    @property
    def args(self):
        if self._cached_args is not None:
            return self._cached_args
        self._cached_args = self._get_args()
        return self._cached_args

    def _get_args(self):
        return tuple(getattr(self, arg, None) for arg in self._args)

    def iter_args(self):
        for name, arg in zip(self._args, self.args):
            yield name, arg

    def _data_source(self):
        yield None

    def data_source(self):
        for n in self.traverse(top_down=True, unique=True):
            for ds in n._data_source():
                if ds is not None:
                    yield ds

    def output_type(self):
        if hasattr(self, '_validator'):
            return self.validator.output_type()

    def substitute(self, old_arg, new_arg):
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

        old_arg._remove_parent(self)
        new_arg._add_parent(self)

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
            for leave in arg.leaves():
                yield leave

        if len(args) == 0:
            yield self

    def traverse(self, top_down=False, unique=False, traversed=None):
        traversed = traversed if traversed is not None else set()

        def is_trav(n):
            if not unique:
                return False
            if id(n) in traversed:
                return True
            traversed.add(id(n))
            return False

        q = deque()
        q.append(self)
        if is_trav(self):
            return

        checked = set()
        while len(q) > 0:
            curr = q.popleft()
            if top_down:
                yield curr
                children = [c for c in curr.children() if not is_trav(c)]
                q.extendleft(children[::-1])
            else:
                if id(curr) not in checked:
                    children = curr.children()
                    if len(children) == 0:
                        yield curr
                    else:
                        q.appendleft(curr)
                        q.extendleft([c for c in children if not is_trav(c)][::-1])
                    checked.add(id(curr))
                else:
                    yield curr

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

    def _is_ancestor_bottom_up(self, other):
        traversed = set()
        def is_trav(n):
            if id(n) in traversed:
                return True
            traversed.add(id(n))
            return False

        q = deque()
        q.append(other)
        is_trav(other)

        while len(q) > 0:
            curr = q.popleft()
            if curr is self:
                return True
            q.extendleft(
                [p for p in curr.parents if not is_trav(p)])
        return False

    def is_ancestor(self, other, updown=True):
        if updown:
            for n in self.traverse(top_down=True, unique=True):
                if n is other:
                    return True
            return False
        else:
            return self._is_ancestor_bottom_up(other)

    def path(self, other, strict=False):
        all_apaths = self.all_path(other, strict=strict)
        try:
            return next(all_apaths)
        except StopIteration:
            return

    def _all_path(self, other):
        if self is other:
            yield [self, ]

        node_poses = defaultdict(lambda: 0)

        q = deque()
        q.append(self)

        while len(q) > 0:
            curr = q[-1]
            children = curr.children()

            pos = node_poses[id(curr)]
            if len(children) == 0 or pos >= len(children):
                q.pop()
                continue

            n = children[pos]
            q.append(n)
            if n is other:
                yield list(q)
                q.pop()

            node_poses[id(curr)] += 1

    def all_path(self, other, strict=False):
        i = 0
        for i, path in zip(itertools.count(1), self._all_path(other)):
            yield path

        if i == 0 and not strict:
            for path in other._all_path(self):
                yield path

    def __getstate__(self):
        slots = utils.get_attrs(self)

        return tuple((slot, object.__getattribute__(self, slot)) for slot in slots
                     if not slot.startswith('__'))

    def __setstate__(self, state):
        self.__init__(**dict(state))
