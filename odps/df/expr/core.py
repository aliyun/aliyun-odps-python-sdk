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

import itertools
import weakref
from collections import deque, defaultdict

from ... import compat
from ...compat import six
from . import utils


class NodeMetaclass(type):
    def __new__(mcs, name, bases, kv):
        if kv.get('_add_args_slots', True):
            slots = list(kv.get('__slots__', [])) + list(kv.get('_slots', []))
            args = kv.get('_args', [])
            slots.extend(args)
            slots = compat.OrderedDict.fromkeys(slots)

            kv['__slots__'] = tuple(slot for slot in slots if slot not in kv)

        if '__slots__' not in kv:
            kv['__slots__'] = ()
        return type.__new__(mcs, name, bases, kv)


class Node(six.with_metaclass(NodeMetaclass)):
    __slots__ = '_cached_args', '_args_indexes', '__weakref__', '__parents', \
                '__siblings', '__tmp_cached_args',
    _args = ()

    def __init__(self, *args, **kwargs):
        self._args_indexes = dict((name, i) for i, name in enumerate(self._args))
        self._init(*args, **kwargs)
        self._fill_parents()
        self.__tmp_cached_args = None

    def _init(self, *args, **kwargs):
        for arg, value in zip(self._args, args):
            setattr(self, arg, value)

        for key, value in six.iteritems(kwargs):
            setattr(self, key, value)

        # When we substitute an arg with another,
        # we do not change the original arg,
        # instead, we store the new arg in the _cached_args.
        # When we traverse the node,
        # we will try to fetch args from the _cached_args first.
        self._init_attr('_cached_args', None)

    def _init_attr(self, attr, val):
        try:
            object.__getattribute__(self, attr)
        except AttributeError:
            setattr(self, attr, val)

    def _fill_parents(self):
        if not hasattr(self, '_Node__parents') or self.__parents is None:
            self.__parents = weakref.WeakValueDictionary()

        for child in self.children():
            child._add_parent(self)

    def _remove_parent(self, parent):
        if id(parent) in self.__parents:
            del self.__parents[id(parent)]

    def _add_parent(self, parent):
        self.__parents[id(parent)] = parent

    def _extend_parents(self, parents):
        for parent in parents:
            self._add_parent(parent)

    @property
    def parents(self):
        return compat.lvalues(self.__parents)

    @property
    def _tmp_cached_args(self):
        return self.__tmp_cached_args

    @_tmp_cached_args.setter
    def _tmp_cached_args(self, args):
        self.__tmp_cached_args = args

    def _get_attr(self, attr, silent=False):
        if silent:
            try:
                return object.__getattribute__(self, attr)
            except AttributeError:
                return
        return object.__getattribute__(self, attr)

    def _init_cached_args(self):
        if self._cached_args is not None:
            return self._cached_args
        self._cached_args = tuple(self._get_attr(arg_name, True)
                                  for arg_name in self._args)

    def _get_arg(self, arg_name):
        self._init_cached_args()

        idx = self._args_indexes[arg_name]
        return self._cached_args[idx]

    @property
    def args(self):
        self._init_cached_args()
        return self._cached_args

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
        for n in self.traverse(unique=True):
            if len(n.children()) == 0:
                yield n

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

        if not isinstance(other, type(self)):
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
        return hash((type(self), tuple(self.children())))

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
        # remember, if the node has been changed into another one during traversing
        # the modification may not be applied to the paths

        i = 0
        for i, path in zip(itertools.count(1), self._all_path(other)):
            yield path

        if i == 0 and not strict:
            for path in other._all_path(self):
                yield path

    def copy(self):
        slots = utils.get_attrs(self)

        attr_dict = dict((attr, getattr(self, attr, None)) for attr in slots)
        copied = type(self)(**attr_dict)
        copied._extend_parents(self.parents)
        return copied

    def __getstate__(self):
        slots = utils.get_attrs(self)

        return tuple((slot, object.__getattribute__(self, slot)) for slot in slots
                     if not slot.startswith('__'))

    def __setstate__(self, state):
        self.__init__(**dict(state))
