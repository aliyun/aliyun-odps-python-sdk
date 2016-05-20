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

import re
import string
import random

from ..core import Backend
from ...expr.arithmetic import *
from ...expr.math import *
from ...expr.datetimes import *
from ...expr.strings import *
from ...expr.element import *
from ...expr.reduction import *
from ...expr.window import *
from ...expr.collections import *
from ...expr.merge import *
from ..errors import CompileError
from ... import types
from ....models import Schema


class Analyzer(Backend):
    def __init__(self, expr, traversed=None):
        self._expr = expr
        self._indexer = itertools.count(0)
        self._traversed = traversed or set()

        self._iters = []

    def analyze(self):
        for node in self._iter():
            self._visit_node(node)

        return self._expr

    def _iter(self):
        for node in self._expr.traverse(top_down=True, unique=True,
                                        traversed=self._traversed):
            yield node

        for node in itertools.chain(
                *(it.traverse(top_down=True, unique=True, traversed=self._traversed)
                  for it in self._iters)):
            yield node

    def _visit_node(self, node):
        try:
            node.accept(self)
        except NotImplementedError:
            return

    def _sub(self, expr, to_sub, parents):
        if not parents and expr is self._expr:
            self._expr = to_sub
        else:
            [p.substitute(expr, to_sub) for p in parents]
        self._iters.append(to_sub)

    def visit_project_collection(self, expr):
        # FIXME how to handle nested reduction?
        if isinstance(expr, Summary):
            return

        collection = expr.input

        sink_selects = []
        columns = set()
        to_replace = []

        windows_rewrite = False
        for field in expr.fields:
            has_window = False
            traversed = set()
            for path in field.all_path(collection, strict=True):
                for node in path:
                    if id(node) in traversed:
                        continue
                    else:
                        traversed.add(id(node))
                    if isinstance(node, SequenceReduction):
                        windows_rewrite = True
                        has_window = True

                        win = self._reduction_to_window(node)
                        window_name = '%s_%s' % (win.name, next(self._indexer))
                        sink_selects.append(win.rename(window_name))
                        to_replace.append((node, window_name))
                        break
                    elif isinstance(node, Column):
                        if node.input is not collection:
                            continue

                        if node.name in columns:
                            to_replace.append((node, node.name))
                            continue
                        columns.add(node.name)
                        select_field = collection[node.source_name]
                        if node.is_renamed():
                            select_field = select_field.rename(node.name)
                        sink_selects.append(select_field)
                        to_replace.append((node, node.name))

            if has_window:
                field._name = field.name

        if not windows_rewrite:
            return

        get = lambda x: x.name if not isinstance(x, six.string_types) else x
        projected = collection[sorted(sink_selects, key=get)]
        projected.optimize_banned = True  # TO prevent from optimizing
        expr.substitute(collection, projected)

        for col, col_name in to_replace:
            parents = col.parents
            self._sub(col, projected[col_name], parents)

    def visit_filter_collection(self, expr):
        # FIXME how to handle nested reduction?
        collection = expr.input

        sink_selects = []
        columns = set()
        to_replace = []

        windows_rewrite = False
        traversed = set()
        for path in expr.predicate.all_path(collection, strict=True):
            for node in path:
                if id(node) in traversed:
                    continue
                else:
                    traversed.add(id(node))
                if isinstance(node, SequenceReduction):
                    windows_rewrite = True

                    win = self._reduction_to_window(node)
                    window_name = '%s_%s' % (win.name, next(self._indexer))
                    sink_selects.append(win.rename(window_name))
                    to_replace.append((node, window_name))
                    break
                elif isinstance(node, Column):
                    if node.input is not collection:
                        continue
                    if node.name in columns:
                        to_replace.append((node, node.name))
                        continue
                    columns.add(node.name)
                    select_field = collection[node.source_name]
                    if node.is_renamed():
                        select_field = select_field.rename(node.name)
                    sink_selects.append(select_field)
                    to_replace.append((node, node.name))

        for column_name in expr.schema.names:
            if column_name in columns:
                continue
            columns.add(column_name)
            sink_selects.append(column_name)

        if not windows_rewrite:
            return

        get = lambda x: x.name if not isinstance(x, six.string_types) else x
        projected = collection[sorted(sink_selects, key=get)]
        projected.optimize_banned = True  # TO prevent from optimizing
        expr.substitute(collection, projected)

        for col, col_name in to_replace:
            parents = col.parents
            self._sub(col, projected[col_name], parents)

        parents = expr.parents
        to_sub = expr[expr.schema.names]
        self._sub(expr, to_sub, parents)

    def _reduction_to_window(self, expr):
        clazz = 'Cum' + expr.node_name
        return globals()[clazz](_input=expr.input, _data_type=expr.dtype)

    def visit_join(self, expr):
        for node in (expr.rhs, ):
            parents = node.parents
            if isinstance(node, JoinCollectionExpr):
                projection = JoinProjectCollectionExpr(
                    _input=node, _schema=node.schema,
                    _fields=node._fetch_fields())
                self._sub(node, projection, parents)
            elif isinstance(node, JoinProjectCollectionExpr):
                self._sub(node.input, node, parents)

        need_project = [False, ]

        def walk(node):
            if isinstance(node, JoinCollectionExpr) and \
                    node.column_conflict:
                need_project[0] = True
                return

            if isinstance(node, JoinCollectionExpr):
                walk(node.lhs)

        walk(expr)

        if need_project[0]:
            parents = expr.parents
            if not parents or \
                    not any(isinstance(parent, (ProjectCollectionExpr, JoinCollectionExpr))
                            for parent in parents):
                to_sub = expr[expr]
                self._sub(expr, to_sub, parents)

    def _handle_function(self, expr, raw_inputs):
        # Since Python UDF cannot support decimal field,
        # We will try to replace the decimal input with string.
        # If the output is decimal, we will also try to replace it with string,
        # and then cast back to decimal
        def no_output_decimal():
            if isinstance(expr, (SequenceExpr, Scalar)):
                return expr.dtype != types.decimal
            else:
                return all(t != types.decimal for t in expr.schema.types)

        if all(input.dtype != types.decimal for input in raw_inputs) and \
                no_output_decimal():
            return

        inputs = []
        for input in raw_inputs:
            if input.dtype != types.decimal:
                inputs.append(input)
            else:
                inputs.append(input.astype('string'))

        attrs = get_attrs(expr)
        attr_values = dict((attr, getattr(expr, attr, None)) for attr in attrs)
        if isinstance(expr, (SequenceExpr, Scalar)):
            if expr.dtype == types.decimal:
                if isinstance(expr, SequenceExpr):
                    attr_values['_data_type'] = types.string
                    attr_values['_source_data_type'] = types.string
                else:
                    attr_values['_value_type'] = types.string
                    attr_values['_source_value_type'] = types.string
            sub = type(expr)._new(**attr_values)

            if expr.dtype == types.decimal:
                sub = sub.astype('decimal')
        else:
            names = expr.schema.names
            tps = expr.schema.types
            cast_names = set()
            if any(tp == types.decimal for tp in tps):
                new_tps = []
                for name, tp in zip(names, tps):
                    if tp != types.decimal:
                        new_tps.append(tp)
                        continue
                    new_tps.append(types.string)
                    cast_names.add(name)
                if len(cast_names) > 0:
                    attr_values['_schema'] = Schema.from_lists(names, new_tps)
            sub = type(expr)(**attr_values)

            if len(cast_names) > 0:
                fields = []
                for name in names:
                    if name in cast_names:
                        fields.append(sub[name].astype('decimal'))
                    else:
                        fields.append(name)
                sub = sub[fields]

        for src_input, input in zip(raw_inputs, inputs):
            sub.substitute(src_input, input)

        parents = expr.parents
        self._sub(expr, sub, parents=parents)

    def visit_function(self, expr):
        self._handle_function(expr, expr._inputs)

    def visit_apply_collection(self, expr):
        self._handle_function(expr, expr._fields)

    def visit_user_defined_aggregator(self, expr):
        self._handle_function(expr, [expr.input, ])

    def _visit_cut(self, expr):
        is_seq = isinstance(expr, SequenceExpr)
        kw = dict()
        if is_seq:
            kw['_data_type'] = expr.dtype
        else:
            kw['_value_type'] = expr.dtype

        conditions = []
        thens = []

        if expr.include_under:
            bin = expr.bins[0]
            if expr.right and not expr.include_lowest:
                conditions.append(expr.input <= bin)
            else:
                conditions.append(expr.input < bin)
            thens.append(expr.labels[0])
        for i, bin in enumerate(expr.bins[1:]):
            lower_bin = expr.bins[i]
            if not expr.right or (i == 0 and expr.include_lowest):
                condition = lower_bin <= expr.input
            else:
                condition = lower_bin < expr.input

            if expr.right:
                condition = (condition & (expr.input <= bin))
            else:
                condition = (condition & (expr.input < bin))

            conditions.append(condition)
            if expr.include_under:
                thens.append(expr.labels[i + 1])
            else:
                thens.append(expr.labels[i])
        if expr.include_over:
            bin = expr.bins[-1]
            if expr.right:
                conditions.append(bin < expr.input)
            else:
                conditions.append(bin <= expr.input)
            thens.append(expr.labels[-1])

        parents = expr.parents
        to_sub = Switch(_conditions=conditions, _thens=thens, **kw)
        self._sub(expr, to_sub, parents)

    def _find(self, expr, tp):
        res = next(it for it in expr.traverse(top_down=True, unique=True)
                   if isinstance(it, tp))
        return res

    def _find_table(self, seq_expr):
        return self._find(seq_expr, CollectionExpr)

    def _gen_name(self, expr):
        if expr.name is None:
            rand = ''.join(random.choice(string.ascii_lowercase)
                           for _ in range(random.randint(5, 10)))
            return '%s_%s' % (rand, next(self._indexer))
        return '%s_%s' % (expr.name, next(self._indexer))

    def visit_element_op(self, expr):
        if isinstance(expr, Between):
            parents = expr.parents
            if expr.inclusive:
                to_sub = ((expr.left <= expr.input) & (expr.input <= expr.right))
            else:
                to_sub = ((expr.left < expr.input) & (expr.input < expr.right))
            self._sub(expr, to_sub.rename(expr.name), parents)
        elif isinstance(expr, Cut):
            self._visit_cut(expr)
        else:
            raise NotImplementedError

    def visit_sample(self, expr):
        if expr._parts is None:
            raise CompileError('ODPS SQL only support sampling by specifying `parts` arg')

        idxes = [None, ] if expr._i is None else expr._i

        condition = None
        for idx in idxes:
            inputs = [expr._parts]
            if idx is not None:
                new_val = idx.value + 1
                inputs.append(Scalar(_value=new_val, _value_type=idx.value_type))
            if expr._sampled_fields:
                inputs.extend(expr._sampled_fields)
            cond = MappedExpr(_inputs=inputs, _func='SAMPLE', _data_type=types.boolean)
            if condition is None:
                condition = cond
            else:
                condition |= cond
        to_sub = FilterCollectionExpr(_input=expr.input, _predicate=condition,
                                      _schema=expr.schema)
        expr.input.optimize_banned = True

        parents = expr.parents
        self._sub(expr, to_sub, parents)

    def visit_value_counts(self, expr):
        collection = expr.input
        by = expr._by
        sort = expr._sort.value

        parents = expr.parents
        to_sub = collection.groupby(by).agg(count=by.count())
        if sort:
            to_sub = to_sub.sort('count', ascending=False)
        self._sub(expr, to_sub, parents)

    def _gen_mapped_expr(self, expr, inputs, func, name,
                         args=None, kwargs=None, multiple=False):
        kwargs = dict(_inputs=inputs, _func=func, _name=name,
                      _func_args=args, _func_kwargs=kwargs,
                      _multiple=multiple)
        if isinstance(expr, SequenceExpr):
            kwargs['_data_type'] = expr.dtype
        else:
            kwargs['_value_type'] = expr.dtype
        return MappedExpr(**kwargs)

    def visit_binary_op(self, expr):
        if not options.df.analyze:
            raise NotImplementedError

        if isinstance(expr, FloorDivide):
            func = lambda l, r: l // r
            # multiple False will pass *args instead of namedtuple
            to_sub = self._gen_mapped_expr(expr, (expr.lhs, expr.rhs),
                                           func, expr.name, multiple=False)
            parents = expr.parents
            self._sub(expr, to_sub, parents)
            return
        if isinstance(expr, Add) and \
                all(child.dtype == types.datetime for child in (expr.lhs, expr.rhs)):
            return
        elif isinstance(expr, (Add, Substract)):
            if expr.lhs.dtype == types.datetime and expr.rhs.dtype == types.datetime:
                pass
            elif any(isinstance(child, MilliSecondScalar) for child in (expr.lhs, expr.rhs)):
                pass
            else:
                return

            def func(l, r, method):
                from datetime import datetime, timedelta
                if not isinstance(l, datetime):
                    l = timedelta(milliseconds=l)
                if not isinstance(r, datetime):
                    r = timedelta(milliseconds=r)

                if method == '+':
                    res = l + r
                else:
                    res = l - r
                if isinstance(res, timedelta):
                    return int(res.microseconds / 1000)
                return res

            inputs = expr.lhs, expr.rhs, Scalar('+') if isinstance(expr, Add) else Scalar('-')
            to_sub = self._gen_mapped_expr(expr, inputs, func, expr.name, multiple=False)
            parents = expr.parents
            self._sub(expr, to_sub, parents)

        raise NotImplementedError

    def visit_unary_op(self, expr):
        if not options.df.analyze:
            raise NotImplementedError

        if isinstance(expr, Invert) and isinstance(expr.input.dtype, types.Integer):
            parents = expr.parents
            to_sub = expr.input.map(lambda x: ~x)
            self._sub(expr, to_sub, parents)
            return

        raise NotImplementedError

    def visit_math(self, expr):
        if not options.df.analyze:
            raise NotImplementedError

        if expr.dtype != types.decimal:
            if isinstance(expr, Arccosh):
                def func(x):
                    import numpy as np
                    return float(np.arccosh(x))
            elif isinstance(expr, Arcsinh):
                def func(x):
                    import numpy as np
                    return float(np.arcsinh(x))
            elif isinstance(expr, Arctanh):
                def func(x):
                    import numpy as np
                    return float(np.arctanh(x))
            elif isinstance(expr, Radians):
                def func(x):
                    import numpy as np
                    return float(np.radians(x))
            elif isinstance(expr, Degrees):
                def func(x):
                    import numpy as np
                    return float(np.degrees(x))
            else:
                raise NotImplementedError

            parents = expr.parents
            to_sub = expr.input.map(func, expr.dtype)
            self._sub(expr, to_sub, parents)
            return

        raise NotImplementedError

    def visit_datetime_op(self, expr):
        if isinstance(expr, Strftime):
            if not options.df.analyze:
                raise NotImplementedError

            date_format = expr.date_format

            def func(x):
                return x.strftime(date_format)

            parents = expr.parents
            to_sub = expr.input.map(func, expr.dtype)
            self._sub(expr, to_sub, parents)
            return

        raise NotImplementedError

    def visit_string_op(self, expr):
        parents = expr.parents
        if isinstance(expr, Ljust):
            rest = expr.width - expr.input.len()
            to_sub = expr.input + \
                     (rest >= 0).ifelse(expr._fillchar.repeat(rest), '')
            self._sub(expr, to_sub.rename(expr.name), parents)
            return
        elif isinstance(expr, Rjust):
            rest = expr.width - expr.input.len()
            to_sub = (rest >= 0).ifelse(expr._fillchar.repeat(rest), '') + expr.input
            self._sub(expr, to_sub.rename(expr.name), parents)
            return
        elif isinstance(expr, Zfill):
            fillchar = Scalar('0')
            rest = expr.width - expr.input.len()
            to_sub = (rest >= 0).ifelse(fillchar.repeat(rest), '') + expr.input
            self._sub(expr, to_sub.rename(expr.name), parents)
            return

        if not options.df.analyze:
            raise NotImplementedError

        func = None
        if isinstance(expr, Contains) and (not expr.case or expr.flags > 0):
            flags = 0
            if not expr.case:
                flags = re.I
            if expr.flags > 0:
                flags = flags | expr.flags
            pat = expr.pat

            def func(x):
                r = re.compile(pat, flags)
                return r.search(x) is not None
        elif isinstance(expr, Extract) and expr.flags > 0:
            pat = expr.pat
            flags = expr.flags
            group = expr.group

            def func(x):
                r = re.compile(pat, flags)
                if group is None:
                    match = r.search(x)
                    if match:
                        return match.group()
                else:
                    match = r.search(x)
                    if match:
                        return match.group(group)
        elif isinstance(expr, Find) and expr.end is not None:
            start = expr.start
            end = expr.end
            sub = expr.sub

            def func(x):
                return x.find(sub, start, end)
        elif isinstance(expr, RFind):
            start = expr.start
            end = expr.end
            sub = expr.sub

            def func(x):
                return x.rfind(sub, start, end)
        elif isinstance(expr, (Lstrip, Strip, Rstrip)) and expr.to_strip != ' ':
            to_strip = expr.to_strip

            if isinstance(expr, Lstrip):
                def func(x):
                    return x.lstrip(to_strip)
            elif isinstance(expr, Strip):
                def func(x):
                    return x.strip(to_strip)
            elif isinstance(expr, Rstrip):
                def func(x):
                    return x.rstrip(to_strip)
        elif isinstance(expr, Pad):
            side = expr.side
            fillchar = expr.fillchar
            width = expr.width

            if side == 'left':
                func = lambda x: x.rjust(width, fillchar)
            elif side == 'right':
                func = lambda x: x.ljust(width, fillchar)
            elif side == 'both':
                func = lambda x: x.center(width, fillchar)
            else:
                raise NotImplementedError
        elif isinstance(expr, Slice):
            start, end, step = expr.start, expr.end, expr.step

            if end is None and step is None:
                raise NotImplementedError
            if isinstance(start, six.integer_types) and \
                    isinstance(end, six.integer_types) and step is None:
                if start >= 0 and end >= 0:
                    raise NotImplementedError

            has_start = start is not None
            has_end = end is not None
            has_step = step is not None

            def func(x, *args):
                idx = 0
                s, e, t = None, None, None
                for i in range(3):
                    if i == 0 and has_start:
                        s = args[idx]
                        idx += 1
                    if i == 1 and has_end:
                        e = args[idx]
                        idx += 1
                    if i == 2 and has_step:
                        t = args[idx]
                        idx += 1
                return x[s: e: t]

            inputs = expr.input, expr._start, expr._end, expr._step
            to_sub = self._gen_mapped_expr(expr, tuple(i for i in inputs if i is not None),
                                           func, expr.name, multiple=False)
            self._sub(expr, to_sub, parents)
            return
        elif isinstance(expr, Swapcase):
            func = lambda x: x.swapcase()
        elif isinstance(expr, Title):
            func = lambda x: x.title()
        elif isinstance(expr, Strptime):
            date_format = expr.date_format

            def func(x):
                from datetime import datetime
                return datetime.strptime(x, date_format)
        else:
            if isinstance(expr, Isalnum):
                func = lambda x: x.isalnum()
            elif isinstance(expr, Isalpha):
                func = lambda x: x.isalpha()
            elif isinstance(expr, Isdigit):
                func = lambda x: x.isdigit()
            elif isinstance(expr, Isspace):
                func = lambda x: x.isspace()
            elif isinstance(expr, Islower):
                func = lambda x: x.islower()
            elif isinstance(expr, Isupper):
                func = lambda x: x.isupper()
            elif isinstance(expr, Istitle):
                func = lambda x: x.istitle()
            elif isinstance(expr, (Isnumeric, Isdecimal)):
                def u_safe(s):
                    try:
                        return unicode(s, "unicode_escape")
                    except:
                        return s

                if isinstance(expr, Isnumeric):
                    func = lambda x: u_safe(x).isnumeric()
                else:
                    func = lambda x: u_safe(x).isdecimal()

        if func is not None:
            to_sub = expr.input.map(func, expr.dtype)
            self._sub(expr, to_sub, parents)
            return

        raise NotImplementedError

    def visit_reduction(self, expr):
        parents = expr.parents
        if isinstance(expr, (Var, GroupedVar)):
            to_sub = expr.input.std(ddof=expr._ddof) ** 2
            self._sub(expr, to_sub, parents)

        raise NotImplementedError
