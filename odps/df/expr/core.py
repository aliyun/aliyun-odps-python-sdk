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

from __future__ import absolute_import

import itertools
import weakref
from collections import deque, defaultdict, Iterable

from ... import compat
from ...compat import six
from ...dag import DAG, Queue
from . import utils


class NodeMetaclass(type):
    def __new__(mcs, name, bases, kv):
        mixin_slots = []
        for b in bases:
            mixin_slots.extend(list(getattr(b, '_mixin_slots', [])))

        if kv.pop('_add_args_slots', True):
            slots = mixin_slots + list(kv.get('__slots__', [])) + list(kv.get('_mixin_slots', [])) + \
                    list(kv.get('_slots', []))
            args = kv.get('_args', [])
            slots.extend(args)
            slots = compat.OrderedDict.fromkeys(slots)

            kv['__slots__'] = tuple(slot for slot in slots if slot not in kv)

        if '__slots__' not in kv:
            kv['__slots__'] = ()
        if mixin_slots:
            kv['_mixin_slots'] = []
        return type.__new__(mcs, name, bases, kv)

    def __instancecheck__(self, instance):
        from .expressions import CollectionExpr

        if issubclass(type(instance), CollectionExpr) and \
                hasattr(instance, '_proxy') and instance._proxy is not None:
            return super(NodeMetaclass, self).__instancecheck__(instance._proxy)

        return super(NodeMetaclass, self).__instancecheck__(instance)


class Node(six.with_metaclass(NodeMetaclass)):
    __slots__ = '_args_indexes', '__weakref__'
    _args = ()  # the child(ren) of the current node
    _extra_args = ()
    _non_table = False

    def __init__(self, *args, **kwargs):
        self._args_indexes = dict((name, i) for i, name in enumerate(self._args))
        self._init(*args, **kwargs)

    def _init(self, *args, **kwargs):
        for arg, value in zip(self._args, args):
            setattr(self, arg, value)

        for key, value in six.iteritems(kwargs):
            setattr(self, key, value)

    def _init_attr(self, attr, val):
        if not hasattr(self, attr):
            setattr(self, attr, val)

    @property
    def args(self):
        return tuple(getattr(self, arg, None) for arg in self._args)

    def iter_args(self):
        for name, arg in zip(self._args, self.args):
            yield name, arg

    @property
    def extra_args(self):
        return tuple(getattr(self, extra_arg, None)
                     for extra_arg in self._extra_args)

    def arg_name_values(self, extra=False):
        arg_names = self._args if not extra else self._args + self._extra_args
        arg_values = self.args if not extra else self.args + self.extra_args
        for arg_name, arg in zip(arg_names, arg_values):
            yield arg_name, arg

    def _data_source(self):
        yield None

    def data_source(self):
        stop_cond = lambda x: hasattr(x, '_source_data') and x._source_data is not None
        for n in self.traverse(top_down=True, unique=True, stop_cond=stop_cond):
            for ds in n._data_source():
                if ds is not None:
                    yield ds

    def substitute(self, old_arg, new_arg, dag=None):
        if dag is not None:
            dag.substitute(old_arg, new_arg, parents=[self])
            return

        if hasattr(old_arg, '_name') and old_arg._name is not None and \
                        new_arg._name is None:
            new_arg._name = old_arg._name

        for arg_name, arg in self.arg_name_values(extra=True):
            if not isinstance(arg, (list, tuple)):
                if arg is old_arg:
                    setattr(self, arg_name, new_arg)
            else:
                subs = list(arg)
                for i in range(len(subs)):
                    if subs[i] is old_arg:
                        subs[i] = new_arg
                setattr(self, arg_name, type(arg)(subs))

    def children(self, extra=False):
        args = []

        all_args = self.args if not extra else self.args + self.extra_args
        for arg in all_args:
            if isinstance(arg, (list, tuple)):
                args.extend(arg)
            else:
                args.append(arg)

        return [arg for arg in args if arg is not None]

    def leaves(self):
        for n in self.traverse(unique=True):
            if len(n.children()) == 0:
                yield n

    def traverse(self, top_down=False, unique=False, traversed=None,
                 extra=False, stop_cond=None):
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
        yields = set()
        while len(q) > 0:
            curr = q.popleft()
            if top_down:
                yield curr
                if stop_cond is None or not stop_cond(curr):
                    children = [c for c in curr.children(extra=extra)
                                if not is_trav(c)]
                    q.extendleft(children[::-1])
            else:
                if id(curr) not in checked:
                    children = curr.children(extra=extra)
                    if len(children) == 0:
                        yield curr
                    else:
                        q.appendleft(curr)
                        if stop_cond is None or not stop_cond(curr):
                            q.extendleft([c for c in children
                                          if not is_trav(c) or id(c) not in checked][::-1])
                    checked.add(id(curr))
                else:
                    if id(curr) not in yields:
                        yield curr
                        if unique:
                            yields.add(id(curr))

    def __eq__(self, other):
        return self.equals(other)

    def equals(self, other):
        if other is None:
            return False

        if not isinstance(other, type(self)):
            return False

        def slot_values(obj):
            return [getattr(obj, slot, None) for slot in utils.get_attrs(obj)]

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
        return all(map(cmp, slot_values(self), slot_values(other)))

    def __hash__(self):
        return hash((type(self), tuple(self.children())))

    def is_ancestor(self, other):
        other = utils.get_proxied_expr(other)
        for n in self.traverse(top_down=True, unique=True):
            if n is other:
                return True
        return False

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
                # TODO: add this in the future
                # if pos >= len(children):
                #     node_poses[id(curr)] = 0
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

    def _attr_dict(self):
        slots = utils.get_attrs(self)

        return dict((attr, getattr(self, attr, None)) for attr in slots)

    def _copy_type(self):
        return type(self)

    def copy(self, clear_keys=None, **kw):
        proxied = utils.get_proxied_expr(self)
        attr_dict = proxied._attr_dict()
        if clear_keys is not None:
            for k in clear_keys:
                attr_dict.pop(k, None)
        attr_dict.update(kw)
        copied = type(proxied)(**attr_dict)
        return copied

    def copy_to(self, target):
        slots = utils.get_attrs(self)
        for attr in slots:
            if hasattr(self, attr):
                setattr(target, attr, getattr(self, attr, None))

    def copy_tree(self, on_copy=None, extra=False):
        if on_copy is not None and not isinstance(on_copy, Iterable):
            on_copy = [on_copy, ]

        expr_id_to_copied = dict()

        def get(n):
            if n is None:
                return n
            try:
                return expr_id_to_copied[id(n)]
            except KeyError:
                raise

        for node in self.traverse(unique=True, extra=extra):
            node = utils.get_proxied_expr(node)
            attr_dict = node._attr_dict()

            for arg_name, arg in node.arg_name_values(extra=extra):
                if isinstance(arg, (tuple, list)):
                    attr_dict[arg_name] = type(arg)(get(it) for it in arg)
                else:
                    attr_dict[arg_name] = get(arg)

            copied_node = type(node)(**attr_dict)
            expr_id_to_copied[id(node)] = copied_node
            if on_copy is not None:
                [func(node, copied_node) for func in on_copy]

        return expr_id_to_copied[id(self)]

    def to_dag(self, copy=True, on_copy=None, dag=None, validate=True):
        if copy:
            expr = self.copy_tree(on_copy=on_copy, extra=True)
        else:
            expr = self

        dag = dag or ExprDAG(expr)
        queue = Queue()

        if not dag.contains_node(expr):
            dag.add_node(expr)
            queue.put(expr)

        traversed = set()
        traversed.add(id(expr))

        while not queue.empty():
            node = queue.get()

            for child in node.children(extra=True):
                if not dag.contains_node(child):
                    dag.add_node(child)
                if not dag.contains_edge(child, node):
                    dag.add_edge(child, node, validate=False)

                if id(child) in traversed:
                    continue
                traversed.add(id(child))
                queue.put(child)

        if validate:
            dag._validate()  # validate the DAG
        return dag

    def __getstate__(self):
        slots = utils.get_attrs(self)

        return tuple((slot, object.__getattribute__(self, slot)) for slot in slots
                     if not slot.startswith('__'))

    def __setstate__(self, state):
        self.__init__(**dict(state))


def _extract_df_inputs(o):
    if isinstance(o, Node):
        yield o
    elif isinstance(o, dict):
        for v in itertools.chain(*(_extract_df_inputs(dv) for dv in six.itervalues(o))):
            if v is not None:
                yield v
    elif isinstance(o, (list, set, tuple)):
        for v in itertools.chain(*(_extract_df_inputs(dv) for dv in o)):
            if v is not None:
                yield v
    else:
        yield None


class ExprProxy(object):
    def __init__(self, expr, d=None, compare=False):
        if d is not None:
            def callback(_):
                if self in d:
                    del d[self]
        else:
            callback = None
        self._ref = weakref.ref(expr, callback)
        self._cmp = compare
        self._hash = hash(expr)
        self._expr_id = id(expr)

    def __call__(self):
        return self._ref()

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, ExprProxy):
            if self._ref() is not None and other() is not None:
                return self._ref() is other()
            return self._expr_id == other._expr_id

        obj = self._ref()
        if obj is not None and self._cmp:
            return obj.equals(other)

        return self._expr_id == id(other)


class ExprDAG(DAG):
    def __init__(self, root, dag=None):
        self._root = weakref.ref(root)
        super(ExprDAG, self).__init__()
        if dag is not None:
            self._graph = dag._graph
            self._map = dag._map

    @property
    def root(self):
        return self._root()

    @root.setter
    def root(self, root):
        self._root = weakref.ref(root)

    def ensure_all_nodes_in_dag(self):
        self.root.to_dag(copy=False, dag=self, validate=False)

    def traverse(self, root=None, top_down=False, traversed=None, stop_cond=None):
        root = root or self.root
        return root.traverse(top_down=top_down, unique=True,
                             traversed=traversed, stop_cond=stop_cond)

    def substitute(self, expr, new_expr, parents=None):
        if expr is self.root:
            self.root = new_expr

        parents = self.successors(expr) if parents is None else parents

        if expr._need_cache:
            new_expr.cache()

        q = Queue()
        q.put(new_expr)

        while not q.empty():
            node = q.get()
            if not self.contains_node(node):
                self.add_node(node)

            for child in node.children():
                if not self.contains_node(child):
                    q.put(child)
                    self.add_node(child)
                if not self.contains_edge(child, node):
                    self.add_edge(child, node, validate=False)

        for parent in parents:
            parent.substitute(expr, new_expr)
            self.add_edge(new_expr, parent, validate=False)
            try:
                self.remove_edge(expr, parent)
            except KeyError:
                pass

    def prune(self):
        while True:
            nodes = [n for n, succ in six.iteritems(self._graph) if len(succ) == 0]
            if len(nodes) == 1 and nodes[0] is self.root:
                break
            for node in nodes:
                if node is not self.root:
                    self.remove_node(node)

    def closest_ancestors(self, node, cond):
        collected = set()
        stop_cond = lambda n: not collected.intersection(self.successors(n))
        for n in self.bfs(node, self.predecessors, stop_cond):
            if cond(n):
                collected.add(n)
                yield n


class ExprDictionary(dict):
    def _ref(self, obj, ref_self=False):
        r = self if ref_self else None
        return obj if isinstance(obj, ExprProxy) else ExprProxy(obj, d=r)

    def __getitem__(self, item):
        if item is None:
            raise KeyError
        return dict.__getitem__(self, self._ref(item))

    def __setitem__(self, key, value):
        if key is None:
            raise KeyError
        return dict.__setitem__(self, self._ref(key, True), value)

    def __iter__(self):
        for k in dict.__iter__(self):
            yield k()

    def __delitem__(self, key):
        if key is None:
            raise KeyError
        return dict.__delitem__(self, self._ref(key))

    def __contains__(self, item):
        if item is None:
            return False
        return dict.__contains__(self, self._ref(item))

    def get(self, k, d=None):
        if k is None:
            return d
        return dict.get(self, self._ref(k), d)

    def has_key(self, k):
        if k is None:
            return False
        return dict.has_key(self, self._ref(k))

    def pop(self, k, d=None):
        if k is None:
            return False
        return dict.pop(self, self._ref(k), d)

    def popitem(self):
        k, v = dict.popitem(self)
        return k(), v

    def setdefault(self, k, d=None):
        if k is None:
            raise KeyError
        return dict.setdefault(self, self._ref(k, True), d)

    def update(self, E=None, **F):
        if hasattr(E, 'keys'):
            for k in E.keys():
                self[k] = E[k]
        elif E is not None:
            for k, v in E:
                self[k] = v
        else:
            for k, v in six.iteritems(F):
                self[k] = v

    if six.PY2:
        def items(self):
            return [(k(), v) for k, v in dict.items(self)]

        def keys(self):
            return [k() for k in dict.keys(self)]

        def iteritems(self):
            for k, v in dict.iteritems(self):
                yield k(), v

        def iterkeys(self):
            for k in dict.iterkeys(self):
                yield k()
    else:
        def items(self):
            for k, v in dict.items(self):
                yield k(), v

        def keys(self):
            for k in dict.keys(self):
                yield k()
