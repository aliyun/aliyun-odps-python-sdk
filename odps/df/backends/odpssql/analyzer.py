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

import re
import sys
import math as pymath

from ..analyzer import BaseAnalyzer
from ...expr.arithmetic import *
from ...expr.composites import ListDictGetItem
from ...expr.math import *
from ...expr.datetimes import *
from ...expr.strings import *
from ...expr.strings import Count as StrCount
from ...expr.element import *
from ...expr.reduction import *
from ...expr.collections import *
from ...expr.merge import *
from ...expr.window import QCut
from ...utils import output
from ..errors import CompileError
from ..utils import refresh_dynamic
from ... import types
from .... import compat
from ....utils import to_text

_NAN = float('nan')


class Analyzer(BaseAnalyzer):
    def _parents(self, expr):
        return self._dag.successors(expr)

    def visit_composite_op(self, expr):
        if isinstance(expr, ListDictGetItem) and isinstance(expr.input.dtype, types.List):
            if is_constant_scalar(expr._negative_handled) and expr._negative_handled.value:
                return

            key_expr = expr._key
            if is_constant_scalar(key_expr):
                if key_expr.value >= 0:
                    return
                sub = expr.input[expr.input.len() - (-key_expr.value)]
                sub._negative_handled = Scalar(True)
            else:
                expr._negative_handled = Scalar(True)
                neg_expr = expr.input[expr.input.len() + key_expr]
                neg_expr._negative_handled = Scalar(True)
                sub = (key_expr >= 0).ifelse(expr, neg_expr).rename(expr.name)
            self._sub(expr, sub)
        else:
            raise NotImplementedError

    def visit_element_op(self, expr):
        if isinstance(expr, Between):
            if expr.inclusive:
                sub = ((expr.left <= expr.input) & (expr.input.copy() <= expr.right))
            else:
                sub = ((expr.left < expr.input) & (expr.input.copy() < expr.right))
            self._sub(expr, sub.rename(expr.name))
        elif isinstance(expr, Cut):
            sub = self._get_cut_sub_expr(expr)
            self._sub(expr, sub)
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
        sub = FilterCollectionExpr(_input=expr.input, _predicate=condition,
                                   _schema=expr.schema)
        expr.input.optimize_banned = True

        self._sub(expr, sub)

    def _visit_pivot(self, expr):
        sub = self._get_pivot_sub_expr(expr)
        self._sub(expr, sub)

    def _visit_pivot_table(self, expr):
        sub = self._get_pivot_table_sub_expr(expr)
        self._sub(expr, sub)

    def visit_pivot(self, expr):
        if isinstance(expr, PivotCollectionExpr):
            self._visit_pivot(expr)
        else:
            self._visit_pivot_table(expr)

    def visit_extract_kv(self, expr):
        kv_delimiter = expr._kv_delimiter.value
        item_delimiter = expr._item_delimiter.value
        default = expr._default.value if expr._default else None

        class KeyAgg(object):
            def buffer(self):
                return set()

            def __call__(self, buf, val):
                if not val:
                    return

                def validate_kv(v):
                    parts = v.split(kv_delimiter)
                    if len(parts) != 2:
                        raise ValueError('Malformed KV pair: %s' % v)
                    return parts[0]

                buf.update([validate_kv(item) for item in val.split(item_delimiter)])

            def merge(self, buf, pbuffer):
                buf.update(pbuffer)

            def getvalue(self, buf):
                return item_delimiter.join(sorted(buf))

        columns_expr = expr.input.exclude(expr._intact).apply(KeyAgg, names=[c.name for c in expr._columns])

        intact_names = [g.name for g in expr._intact]
        intact_types = [g.dtype for g in expr._intact]
        exprs = [expr]

        def callback(result, new_expr):
            expr = exprs[0]

            names = list(intact_names)
            tps = list(intact_types)
            kv_slot_map = dict()

            for col, key_str in compat.izip(result.columns, result[0]):
                kv_slot_map[col.name] = dict()
                for k in key_str.split(item_delimiter):
                    names.append('%s_%s' % (col.name, k))
                    tps.append(expr._column_type)
                    kv_slot_map[col.name][k] = len(names) - 1
            kv_slot_names = list(kv_slot_map.keys())

            type_adapter = None
            if isinstance(expr._column_type, types.Float):
                type_adapter = float
            elif isinstance(expr._column_type, types.Integer):
                type_adapter = int

            @output(names, tps)
            def mapper(row):
                ret = [default, ] * len(names)
                ret[:len(intact_names)] = [getattr(row, col) for col in intact_names]
                for col in kv_slot_names:
                    kv_val = getattr(row, col)
                    if not kv_val:
                        continue
                    for kv_item in kv_val.split(item_delimiter):
                        k, v = kv_item.split(kv_delimiter)
                        if type_adapter:
                            v = type_adapter(v)
                        ret[kv_slot_map[col][k]] = v
                return tuple(ret)

            new_expr._schema = Schema.from_lists(names, tps)

            extracted = expr.input.map_reduce(mapper)
            self._sub(new_expr, extracted)

            # trigger refresh of dynamic operations
            refresh_dynamic(extracted, self._dag)

        sub = CollectionExpr(_schema=DynamicSchema.from_lists(intact_names, intact_types),
                             _deps=[(columns_expr, callback)])
        self._sub(expr, sub)

    def visit_value_counts(self, expr):
        self._sub(expr, self._get_value_counts_sub_expr(expr))

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
            sub = self._gen_mapped_expr(expr, (expr.lhs, expr.rhs),
                                        func, expr.name, multiple=False)
            self._sub(expr, sub)
            return
        if isinstance(expr, Mod):
            func = lambda l, r: l % r
            sub = self._gen_mapped_expr(expr, (expr.lhs, expr.rhs),
                                        func, expr.name, multiple=False)
            self._sub(expr, sub)
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

            if sys.version_info[:2] <= (2, 6):
                def total_seconds(self):
                    return self.days * 86400.0 + self.seconds + self.microseconds * 1.0e-6
            else:
                from datetime import timedelta

                def total_seconds(self):
                    return self.total_seconds()

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
                    return int(total_seconds(res) * 1000)
                return res

            inputs = expr.lhs, expr.rhs, Scalar('+') if isinstance(expr, Add) else Scalar('-')
            sub = self._gen_mapped_expr(expr, inputs, func, expr.name, multiple=False)
            self._sub(expr, sub)

        raise NotImplementedError

    def visit_unary_op(self, expr):
        if not options.df.analyze:
            raise NotImplementedError

        if isinstance(expr, Invert) and isinstance(expr.input.dtype, types.Integer):
            sub = expr.input.map(lambda x: ~x)
            self._sub(expr, sub)
            return

        raise NotImplementedError

    def visit_math(self, expr):
        if not options.df.analyze:
            raise NotImplementedError

        if expr.dtype != types.decimal:
            if isinstance(expr, Arccosh):
                def func(x):
                    return pymath.acosh(x)
            elif isinstance(expr, Arcsinh):
                def func(x):
                    return pymath.asinh(x)
            elif isinstance(expr, Arctanh):
                def func(x):
                    try:
                        return pymath.atanh(x)
                    except ValueError:
                        return _NAN
            elif isinstance(expr, Radians):
                def func(x):
                    return pymath.radians(x)
            elif isinstance(expr, Degrees):
                def func(x):
                    return pymath.degrees(x)
            else:
                raise NotImplementedError

            sub = expr.input.map(func, expr.dtype)
            self._sub(expr, sub)
            return

        raise NotImplementedError

    def visit_datetime_op(self, expr):
        if isinstance(expr, Strftime):
            if not options.df.analyze:
                raise NotImplementedError

            date_format = expr.date_format

            def func(x):
                return x.strftime(date_format)

            sub = expr.input.map(func, expr.dtype)
            self._sub(expr, sub)
            return

        raise NotImplementedError

    def visit_string_op(self, expr):
        if isinstance(expr, Ljust):
            rest = expr.width - expr.input.len()
            sub = expr.input + \
                     (rest >= 0).ifelse(expr._fillchar.repeat(rest), '')
            self._sub(expr, sub.rename(expr.name))
            return
        elif isinstance(expr, Rjust):
            rest = expr.width - expr.input.len()
            sub = (rest >= 0).ifelse(expr._fillchar.repeat(rest), '') + expr.input
            self._sub(expr, sub.rename(expr.name))
            return
        elif isinstance(expr, Zfill):
            fillchar = Scalar('0')
            rest = expr.width - expr.input.len()
            sub = (rest >= 0).ifelse(fillchar.repeat(rest), '') + expr.input
            self._sub(expr, sub.rename(expr.name))
            return
        elif isinstance(expr, CatStr):
            input = expr.input
            others = expr._others if isinstance(expr._others, Iterable) else (expr._others, )
            for other in others:
                if expr.na_rep is not None:
                    for e in (input, ) + tuple(others):
                        self._sub(e, e.fillna(expr.na_rep), parents=(expr, ))
                    return
                else:
                    if expr._sep is not None:
                        input = other.isnull().ifelse(input, input + expr._sep + other)
                    else:
                        input = other.isnull().ifelse(input, input + other)
            self._sub(expr, input.rename(expr.name))
            return

        if not options.df.analyze:
            raise NotImplementedError

        func = None
        if isinstance(expr, Contains) and expr.regex:
            def func(x, pat, case, flags):
                if x is None:
                    return None

                flgs = 0
                if not case:
                    flgs = re.I
                if flags > 0:
                    flgs = flgs | flags
                r = re.compile(pat, flgs)
                return r.search(x) is not None

            pat = expr._pat if not isinstance(expr._pat, StringScalar) or expr._pat._value is None \
                else Scalar(re.escape(to_text(expr.pat)))
            inputs = expr.input, pat, expr._case, expr._flags
            sub = self._gen_mapped_expr(expr, inputs, func,
                                        expr.name, multiple=False)
            self._sub(expr, sub)
            return
        elif isinstance(expr, StrCount):
            def func(x, pat, flags):
                if x is None:
                    return None

                regex = re.compile(pat, flags=flags)
                return len(regex.findall(x))

            pat = expr._pat if not isinstance(expr._pat, StringScalar) or expr._pat._value is None \
                else Scalar(re.escape(to_text(expr.pat)))
            inputs = expr.input, pat, expr._flags
            sub = self._gen_mapped_expr(expr, inputs, func,
                                        expr.name, multiple=False)
            self._sub(expr, sub)
            return
        elif isinstance(expr, Find) and expr.end is not None:
            start = expr.start
            end = expr.end
            substr = expr.sub

            def func(x):
                if x is None:
                    return None

                return x.find(substr, start, end)
        elif isinstance(expr, RFind):
            start = expr.start
            end = expr.end
            substr = expr.sub

            def func(x):
                if x is None:
                    return None

                return x.rfind(substr, start, end)
        elif isinstance(expr, Extract):
            def func(x, pat, flags, group):
                if x is None:
                    return None

                regex = re.compile(pat, flags=flags)
                m = regex.search(x)
                if m:
                    if group is None:
                        return m.group()
                    return m.group(group)

            pat = expr._pat if not isinstance(expr._pat, StringScalar) or expr._pat._value is None \
                else Scalar(re.escape(to_text(expr.pat)))
            inputs = expr.input, pat, expr._flags, expr._group
            sub = self._gen_mapped_expr(expr, inputs, func,
                                        expr.name, multiple=False)
            self._sub(expr, sub)
            return
        elif isinstance(expr, Replace):
            use_regex = [expr.regex]
            def func(x, pat, repl, n, case, flags):
                if x is None:
                    return None

                use_re = use_regex[0] and (not case or len(pat) > 1 or flags)

                if use_re:
                    if not case:
                        flags |= re.IGNORECASE
                    regex = re.compile(pat, flags=flags)
                    n = n if n >= 0 else 0

                    return regex.sub(repl, x, count=n)
                else:
                    return x.replace(pat, repl, n)

            pat = expr._pat if not isinstance(expr._pat, StringScalar) or expr._value is None \
                else Scalar(re.escape(to_text(expr.pat)))
            inputs = expr.input, pat, expr._repl, expr._n, \
                     expr._case, expr._flags
            sub = self._gen_mapped_expr(expr, inputs, func,
                                        expr.name, multiple=False)
            self._sub(expr, sub)
            return
        elif isinstance(expr, (Lstrip, Strip, Rstrip)) and expr.to_strip != ' ':
            to_strip = expr.to_strip

            if isinstance(expr, Lstrip):
                def func(x):
                    if x is None:
                        return None

                    return x.lstrip(to_strip)
            elif isinstance(expr, Strip):
                def func(x):
                    if x is None:
                        return None

                    return x.strip(to_strip)
            elif isinstance(expr, Rstrip):
                def func(x):
                    if x is None:
                        return None

                    return x.rstrip(to_strip)
        elif isinstance(expr, Pad):
            side = expr.side
            fillchar = expr.fillchar
            width = expr.width

            if side == 'left':
                func = lambda x: x.rjust(width, fillchar) if x is not None else None
            elif side == 'right':
                func = lambda x: x.ljust(width, fillchar) if x is not None else None
            elif side == 'both':
                func = lambda x: x.center(width, fillchar) if x is not None else None
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
                if x is None:
                    return None

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
            sub = self._gen_mapped_expr(expr, tuple(i for i in inputs if i is not None),
                                        func, expr.name, multiple=False)
            self._sub(expr, sub)
            return
        elif isinstance(expr, Swapcase):
            func = lambda x: x.swapcase() if x is not None else None
        elif isinstance(expr, Title):
            func = lambda x: x.title() if x is not None else None
        elif isinstance(expr, Strptime):
            date_format = expr.date_format

            def func(x):
                from datetime import datetime
                return datetime.strptime(x, date_format) if x is not None else None
        else:
            if isinstance(expr, Isalnum):
                func = lambda x: x.isalnum() if x is not None else None
            elif isinstance(expr, Isalpha):
                func = lambda x: x.isalpha() if x is not None else None
            elif isinstance(expr, Isdigit):
                func = lambda x: x.isdigit() if x is not None else None
            elif isinstance(expr, Isspace):
                func = lambda x: x.isspace() if x is not None else None
            elif isinstance(expr, Islower):
                func = lambda x: x.islower() if x is not None else None
            elif isinstance(expr, Isupper):
                func = lambda x: x.isupper() if x is not None else None
            elif isinstance(expr, Istitle):
                func = lambda x: x.istitle() if x is not None else None
            elif isinstance(expr, (Isnumeric, Isdecimal)):
                def u_safe(s):
                    try:
                        return unicode(s, "unicode_escape")
                    except:
                        return s

                if isinstance(expr, Isnumeric):
                    func = lambda x: u_safe(x).isnumeric() if x is not None else None
                else:
                    func = lambda x: u_safe(x).isdecimal() if x is not None else None

        if func is not None:
            sub = expr.input.map(func, expr.dtype)
            self._sub(expr, sub)
            return

        raise NotImplementedError

    def visit_rank_window(self, expr):
        if isinstance(expr, QCut):
            self._sub(expr, expr - 1)
            return

        raise NotImplementedError

    def visit_reduction(self, expr):
        expr_input = expr.input
        if getattr(expr, '_unique', False):
            expr_input = expr_input.unique()

        if isinstance(expr, (Var, GroupedVar)):
            std = expr_input.std(ddof=expr._ddof)
            if isinstance(expr, GroupedVar):
                std = std.to_grouped_reduction(expr._grouped)
            sub = (std ** 2).rename(expr.name)
            self._sub(expr, sub)
            return
        elif isinstance(expr, (Moment, GroupedMoment)):
            order = expr._order
            center = expr._center

            sub = self._get_moment_sub_expr(expr, expr_input, order, center)
            sub = sub.rename(expr.name)
            self._sub(expr, sub)
            return
        elif isinstance(expr, (Skewness, GroupedSkewness)):
            std = expr_input.std(ddof=1)
            if isinstance(expr, GroupedSequenceReduction):
                std = std.to_grouped_reduction(expr._grouped)
            cnt = expr_input.count()
            if isinstance(expr, GroupedSequenceReduction):
                cnt = cnt.to_grouped_reduction(expr._grouped)
            sub = self._get_moment_sub_expr(expr, expr_input, 3, True) / (std ** 3)
            sub *= (cnt ** 2) / (cnt - 1) / (cnt - 2)
            sub = sub.rename(expr.name)
            self._sub(expr, sub)
        elif isinstance(expr, (Kurtosis, GroupedKurtosis)):
            std = expr_input.std(ddof=0)
            if isinstance(expr, GroupedSequenceReduction):
                std = std.to_grouped_reduction(expr._grouped)
            cnt = expr_input.count()
            if isinstance(expr, GroupedSequenceReduction):
                cnt = cnt.to_grouped_reduction(expr._grouped)
            m4 = self._get_moment_sub_expr(expr, expr_input, 4, True)
            sub = 1.0 / (cnt - 2) / (cnt - 3) * ((cnt * cnt - 1) * m4 / (std ** 4) - 3 * (cnt - 1) ** 2)
            sub = sub.rename(expr.name)
            self._sub(expr, sub)

        raise NotImplementedError
