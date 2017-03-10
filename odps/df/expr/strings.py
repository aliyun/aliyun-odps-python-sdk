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


from .element import ElementWise
from .expressions import Expr, StringSequenceExpr, Scalar, StringScalar, SequenceExpr
from . import utils
from .. import types
from ...compat import six


class StringOp(ElementWise):
    __slots__ = ()

    def _init(self, *args, **kwargs):
        for arg in self._args[1:]:
            self._init_attr(arg, None)

        super(StringOp, self)._init(*args, **kwargs)

        for attr in self._args[1:]:
            val = getattr(self, attr)
            if val is not None and not isinstance(val, (Expr, list, tuple)):
                setattr(self, attr, Scalar(_value=val))

    def __getattribute__(self, attr):
        if attr in ('input', '_input'):
            return super(StringOp, self).__getattribute__(attr)
        else:
            try:
                return object.__getattribute__(self, attr)
            except AttributeError as e:
                err = e
            if not attr.startswith('_'):
                private_attr = '_%s' % attr
                try:
                    scalar = object.__getattribute__(self, private_attr)
                    if isinstance(scalar, Scalar):
                        return scalar.value
                    return scalar
                except AttributeError:
                    raise err

    def accept(self, visitor):
        return visitor.visit_string_op(self)


class Capitalize(StringOp):
    __slots__ = ()


class CatStr(StringOp):
    _args = '_input', '_others', '_sep', '_na_rep'
    _add_args_slots = False

    @property
    def node_name(self):
        return 'Cat'


class Contains(StringOp):
    _args = '_input', '_pat', '_case', '_flags', '_regex'
    _add_args_slots = False


class Count(StringOp):
    _args = '_input', '_pat', '_flags'
    _add_args_slots = False


class Endswith(StringOp):
    _args = '_input', '_pat'
    _add_args_slots = False


class Startswith(StringOp):
    _args = '_input', '_pat'
    _add_args_slots = False


class Extract(StringOp):
    _args = '_input', '_pat', '_flags', '_group'
    _add_args_slots = False


class Find(StringOp):
    _args = '_input', '_sub', '_start', '_end'
    _add_args_slots = False


class RFind(StringOp):
    _args = '_input', '_sub', '_start', '_end'
    _add_args_slots = False


class Replace(StringOp):
    _args = '_input', '_pat', '_repl', '_n', '_case', '_flags', '_regex'
    _add_args_slots = False


class Get(StringOp):
    _args = '_input', '_index'
    _add_args_slots = False


class Join(StringOp):
    _args = '_input', '_sep'
    _add_args_slots = False


class Len(StringOp):
    _args = '_input',
    _add_args_slots = False


class Ljust(StringOp):
    _args = '_input', '_width', '_fillchar'
    _add_args_slots = False


class Rjust(StringOp):
    _args = '_input', '_width', '_fillchar'
    _add_args_slots = False


class Lower(StringOp):
    _args = '_input',
    _add_args_slots = False


class Upper(StringOp):
    _args = '_input',
    _add_args_slots = False


class Lstrip(StringOp):
    _args = '_input', '_to_strip'
    _add_args_slots = False


class Rstrip(StringOp):
    _args = '_input', '_to_strip'
    _add_args_slots = False


class Strip(StringOp):
    _args = '_input', '_to_strip'
    _add_args_slots = False


class Pad(StringOp):
    _args = '_input', '_width', '_side', '_fillchar'
    _add_args_slots = False

    def _init(self, *args, **kwargs):
        super(Pad, self)._init(*args, **kwargs)

        if self.side not in ('left', 'right', 'both'):
            raise ValueError('Side should be left, right or both')


class Repeat(StringOp):
    _args = '_input', '_repeats'
    _add_args_slots = False


class Split(StringOp):
    _args = '_input', '_pat', '_n'
    _add_args_slots = False


class RSplit(StringOp):
    _args = '_input', '_pat', '_n'
    _add_args_slots = False


class Slice(StringOp):
    _args = '_input', '_start', '_end', '_step'
    _add_args_slots = False


class Swapcase(StringOp):
    _args = '_input',
    _add_args_slots = False


class Title(StringOp):
    _args = '_input',
    _add_args_slots = False


class Zfill(StringOp):
    _args = '_input', '_width'
    _add_args_slots = False


class Strptime(StringOp):
    _args = '_input', '_date_format'
    _add_args_slots = False


class Isalnum(StringOp):
    _args = '_input',
    _add_args_slots = False


class Isalpha(StringOp):
    _args = '_input',
    _add_args_slots = False


class Isdigit(StringOp):
    _args = '_input',
    _add_args_slots = False


class Isspace(StringOp):
    _args = '_input',
    _add_args_slots = False


class Islower(StringOp):
    _args = '_input',
    _add_args_slots = False


class Isupper(StringOp):
    _args = '_input',
    _add_args_slots = False


class Istitle(StringOp):
    _args = '_input',
    _add_args_slots = False


class Isnumeric(StringOp):
    _args = '_input',
    _add_args_slots = False


class Isdecimal(StringOp):
    _args = '_input',
    _add_args_slots = False


def _string_op(expr, output_expr_cls, output_type=None, **kwargs):
    output_type = output_type or types.string
    if isinstance(expr, (StringSequenceExpr, StringScalar)):
        is_sequence = isinstance(expr, StringSequenceExpr)

        if is_sequence:
            return output_expr_cls(_data_type=output_type, _input=expr, **kwargs)
        else:
            return output_expr_cls(_value_type=output_type, _input=expr, **kwargs)


def _capitalize(expr):
    """
    Convert strings in the Sequence or string scalar to be capitalized. Equivalent to str.capitalize().

    :param expr:
    :return: sequence or scalar
    """

    return _string_op(expr, Capitalize)


def _cat(expr, others, sep=None, na_rep=None):
    if isinstance(others, six.string_types):
        raise ValueError('Did you mean to supply a `sep` keyword?')
    return _string_op(expr, CatStr, _others=others, _sep=sep, _na_rep=na_rep)


def _contains(expr, pat, case=True, flags=0, regex=True):
    """
    Return boolean sequence whether given pattern/regex is contained in each string in the sequence

    :param expr: sequence or scalar
    :param pat: Character sequence or regular expression
    :param case: If True, case sensitive
    :type case: bool
    :param flags: re module flags, e.g. re.IGNORECASE
    :param regex: If True use regex, otherwise use string finder
    :return: sequence or scalar
    """

    return _string_op(expr, Contains, output_type=types.boolean,
                      _pat=pat, _case=case, _flags=flags, _regex=regex)


def _count(expr, pat, flags=0):
    """
    Count occurrences of pattern in each string of the sequence or scalar

    :param expr: sequence or scalar
    :param pat: valid regular expression
    :param flags: re module flags, e.g. re.IGNORECASE
    :return:
    """
    return _string_op(expr, Count, output_type=types.int64,
                      _pat=pat, _flags=flags)


def _endswith(expr, pat):
    """
    Return boolean sequence or scalar indicating whether each string in the sequence or scalar
    ends with passed pattern. Equivalent to str.endswith().

    :param expr:
    :param pat: Character sequence
    :return: sequence or scalar
    """

    return _string_op(expr, Endswith, output_type=types.boolean, _pat=pat)


def _startswith(expr, pat):
    """
    Return boolean sequence or scalar indicating whether each string in the sequence or scalar
    starts with passed pattern. Equivalent to str.startswith().

    :param expr:
    :param pat: Character sequence
    :return: sequence or scalar
    """

    return _string_op(expr, Startswith, output_type=types.boolean, _pat=pat)


def _extract(expr, pat, flags=0, group=0):
    """
    Find group in each string in the Series using passed regular expression.

    :param expr:
    :param pat: Pattern or regular expression
    :param flags: re module, e.g. re.IGNORECASE
    :param group: if None as group 0
    :return: sequence or scalar
    """

    return _string_op(expr, Extract, _pat=pat, _flags=flags, _group=group)


def _find(expr, sub, start=0, end=None):
    """
    Return lowest indexes in each strings in the sequence or scalar
    where the substring is fully contained between [start:end]. Return -1 on failure.
    Equivalent to standard str.find().

    :param expr:
    :param sub: substring being searched
    :param start: left edge index
    :param end: right edge index
    :return: sequence or scalar
    """

    return _string_op(expr, Find, output_type=types.int64,
                      _sub=sub, _start=start, _end=end)


def _rfind(expr, sub, start=0, end=None):
    """
    Return highest indexes in each strings in the sequence or scalar
    where the substring is fully contained between [start:end]. Return -1 on failure.
    Equivalent to standard str.rfind().

    :param expr:
    :param sub:
    :param start:
    :param end:
    :return: sequence or scalar
    """

    return _string_op(expr, RFind, output_type=types.int64,
                      _sub=sub, _start=start, _end=end)


def _replace(expr, pat, repl, n=-1, case=True, flags=0, regex=True):
    """
    Replace occurrence of pattern/regex in the sequence or scalar with some other string.
    Equivalent to str.replace()

    :param expr:
    :param pat: Character sequence or regular expression
    :param repl: Replacement
    :param n: Number of replacements to make from start
    :param case: if True, case sensitive
    :param flags: re module flag, e.g. re.IGNORECASE
    :return: sequence or scalar
    """

    return _string_op(expr, Replace, _pat=pat, _repl=repl,
                      _n=n, _case=case, _flags=flags, _regex=regex)


def _get(expr, index):
    """
    Extract element from lists, tuples, or strings in each element in the sequence or scalar

    :param expr:
    :param index: Integer index(location)
    :return: sequence or scalar
    """

    return _string_op(expr, Get, _index=index)


def _join(expr, sep):
    """
    Join lists contained as elements in the Series/Index with passed delimiter.
    Equivalent to str.join().

    :param expr:
    :param sep: Delimiter
    :return: sequence or scalar
    """

    return _string_op(expr, Join, _sep=sep)


def _len(expr):
    """
    Compute length of each string in the sequence or scalar

    :param expr:
    :return: lengths
    """
    return _string_op(expr, Len, output_type=types.int64)


def _ljust(expr, width, fillchar=' '):
    """
    Filling right side of strings in the sequence or scalar with an additional character.
    Equivalent to str.ljust().

    :param expr:
    :param width: Minimum width of resulting string; additional characters will be filled with `fillchar`
    :param fillchar: Additional character for filling, default is whitespace.
    :return: sequence or scalar
    """
    return _string_op(expr, Ljust, _width=width, _fillchar=fillchar)


def _rjust(expr, width, fillchar=' '):
    """
    Filling left side of strings in the sequence or scalar with an additional character.
    Equivalent to str.rjust().

    :param expr:
    :param width: Minimum width of resulting string; additional characters will be filled with `fillchar`
    :param fillchar: Additional character for filling, default is whitespace.
    :return: sequence or scalar
    """

    return _string_op(expr, Rjust, _width=width, _fillchar=fillchar)


def _lower(expr):
    """
    Convert strings in the sequence or scalar lowercase. Equivalent to str.lower().

    :param expr:
    :return: sequence or scalar
    """
    return _string_op(expr, Lower)


def _upper(expr):
    """
    Convert strings in the sequence or scalar uppercase. Equivalent to str.upper().

    :param expr:
    :return: sequence or scalar
    """

    return _string_op(expr, Upper)


def _lstrip(expr, to_strip=None):
    """
    Strip whitespace (including newlines) from each string in the sequence or scalar from left side.
    Equivalent to str.lstrip().

    :param expr:
    :param to_strip:
    :return: sequence or sclaar
    """

    return _string_op(expr, Lstrip, _to_strip=to_strip)


def _rstrip(expr, to_strip=None):
    """
    Strip whitespace (including newlines) from each string in the sequence or scalar from right side.
    Equivalent to str.rstrip().

    :param expr:
    :param to_strip:
    :return: sequence or scalar
    """

    return _string_op(expr, Rstrip, _to_strip=to_strip)


def _split(expr, pat=None, n=-1):
    """
    Split each string (a la re.split) in the Series/Index by given pattern, propagating NA values.
    Equivalent to str.split().

    :param expr:
    :param pat: Separator to split on. If None, splits on whitespace
    :param n: None, 0 and -1 will be interpreted as return all splits
    :return: sequence or scalar
    """

    return _string_op(expr, Split, output_type=types.List(types.string),
                      _pat=pat, _n=n)


def _rsplit(expr, pat=None, n=-1):
    """
    Split each string in the Series/Index by the given delimiter string,
    starting at the end of the string and working to the front.
    Equivalent to str.rsplit().

    :param expr:
    :param pat: Separator to split on. If None, splits on whitespace
    :param n: None, 0 and -1 will be interpreted as return all splits
    :return: sequence or scalar
    """

    return _string_op(expr, RSplit, output_type=types.List(types.string),
                      _pat=pat, _n=n)


def _strip(expr, to_strip=None):
    """
    Strip whitespace (including newlines) from each string in the sequence or scalar from left and right sides.
    Equivalent to str.strip().

    :param expr:
    :param to_strip:
    :return: sequence or scalar
    """

    return _string_op(expr, Strip, _to_strip=to_strip)


def _pad(expr, width, side='left', fillchar=' '):
    """
    Pad strings in the sequence or scalar with an additional character to specified side.

    :param expr:
    :param width: Minimum width of resulting string; additional characters will be filled with spaces
    :param side: {‘left’, ‘right’, ‘both’}, default ‘left’
    :param fillchar: Additional character for filling, default is whitespace
    :return: sequence or scalar
    """

    if not isinstance(fillchar, six.string_types):
        msg = 'fillchar must be a character, not {0}'
        raise TypeError(msg.format(type(fillchar).__name__))

    if len(fillchar) != 1:
        raise TypeError('fillchar must be a character, not str')

    if side not in ('left', 'right', 'both'):
        raise ValueError('Invalid side')

    return _string_op(expr, Pad, _width=width, _side=side, _fillchar=fillchar)


def _repeat(expr, repeats):
    """
    Duplicate each string in the sequence or scalar by indicated number of times.

    :param expr:
    :param repeats: times
    :return: sequence or scalar
    """

    return _string_op(expr, Repeat, _repeats=repeats)


def _slice(expr, start=None, stop=None, step=None):
    """
    Slice substrings from each element in the sequence or scalar

    :param expr:
    :param start: int or None
    :param stop: int or None
    :param step: int or None
    :return: sliced
    """

    return _string_op(expr, Slice, _start=start, _end=stop, _step=step)


def _getitem(expr, item):
    if isinstance(item, six.integer_types) or \
            (isinstance(item, (SequenceExpr, Scalar)) and isinstance(item.dtype, types.Integer)):
        return _get(expr, item)
    elif isinstance(item, slice):
        return _slice(expr, start=item.start, stop=item.stop, step=item.step)
    else:
        raise TypeError('Unknown argument: %r' % item)


def _swapcase(expr):
    """
    Convert strings in the sequence or scalar to be swapcased. Equivalent to str.swapcase().

    :param expr:
    :return: converted
    """

    return _string_op(expr, Swapcase)


def _title(expr):
    """
    Convert strings in the sequence or scalar to titlecase. Equivalent to str.title().


    :param expr:
    :return: converted
    """

    return _string_op(expr, Title)


def _zfill(expr, width):
    """
    Filling left side of strings in the sequence or scalar with 0. Equivalent to str.zfill().

    :param expr:
    :param width: Minimum width of resulting string; additional characters will be filled with 0
    :return: filled
    """

    return _string_op(expr, Zfill, _width=width)


def _strptime(expr, date_format):
    """
    Return datetimes specified by date_format,
    which supports the same string format as the python standard library.
    Details of the string format can be found in python string format doc

    :param expr:
    :param date_format: date format string (e.g. “%Y-%m-%d”)
    :type date_format: str
    :return:
    """

    return _string_op(expr, Strptime, _date_format=date_format,
                      output_type=types.datetime)


def _isalnum(expr):
    """
    Check whether all characters in each string in the sequence or scalar are alphanumeric.
    Equivalent to str.isalnum().

    :param expr:
    :return: boolean sequence or scalar
    """

    return _string_op(expr, Isalnum, output_type=types.boolean)


def _isalpha(expr):
    """
    Check whether all characters in each string in the sequence or scalar are alphabetic.
    Equivalent to str.isalpha().

    :param expr:
    :return: boolean sequence or scalar
    """

    return _string_op(expr, Isalpha, output_type=types.boolean)


def _isdigit(expr):
    """
    Check whether all characters in each string in the sequence or scalar are digits.
    Equivalent to str.isdigit().

    :param expr:
    :return: boolean sequence or scalar
    """

    return _string_op(expr, Isdigit, output_type=types.boolean)


def _isspace(expr):
    """
    Check whether all characters in each string in the sequence or scalar are whitespace.
    Equivalent to str.isspace().

    :param expr:
    :return: boolean sequence or scalar
    """

    return _string_op(expr, Isspace, output_type=types.boolean)


def _islower(expr):
    """
    Check whether all characters in each string in the sequence or scalar are lowercase.
    Equivalent to str.islower().

    :param expr:
    :return: boolean sequence or scalar
    """

    return _string_op(expr, Islower, output_type=types.boolean)


def _isupper(expr):
    """
    Check whether all characters in each string in the sequence or scalar are uppercase.
    Equivalent to str.isupper().

    :param expr:
    :return: boolean sequence or scalar
    """

    return _string_op(expr, Isupper, output_type=types.boolean)


def _istitle(expr):
    """
    Check whether all characters in each string in the sequence or scalar are titlecase.
    Equivalent to str.istitle().

    :param expr:
    :return: boolean sequence or scalar
    """

    return _string_op(expr, Istitle, output_type=types.boolean)


def _isnumeric(expr):
    """
    Check whether all characters in each string in the sequence or scalar are numeric.
    Equivalent to str.isnumeric().


    :param expr:
    :return: boolean sequence or scalar
    """

    return _string_op(expr, Isnumeric, output_type=types.boolean)


def _isdecimal(expr):
    """
    Check whether all characters in each string in the sequence or scalar are decimal.
    Equivalent to str.isdecimal().

    :param expr:
    :return: boolean sequence or scalar
    """

    return _string_op(expr, Isdecimal, output_type=types.boolean)


_string_methods = dict(
    capitalize=_capitalize,
    contains=_contains,
    count=_count,
    endswith=_endswith,
    startswith=_startswith,
    extract=_extract,
    find=_find,
    rfind=_rfind,
    replace=_replace,
    get=_get,
    len=_len,
    ljust=_ljust,
    rjust=_rjust,
    lower=_lower,
    upper=_upper,
    lstrip=_lstrip,
    rstrip=_rstrip,
    strip=_strip,
    pad=_pad,
    repeat=_repeat,
    slice=_slice,
    __getitem__=_getitem,
    swapcase=_swapcase,
    title=_title,
    zfill=_zfill,
    strptime=_strptime,
    isalnum=_isalnum,
    isalpha=_isalpha,
    isdigit=_isdigit,
    isspace=_isspace,
    islower=_islower,
    isupper=_isupper,
    istitle=_istitle,
    isnumeric=_isnumeric,
    isdecimal=_isdecimal
)


utils.add_method(StringSequenceExpr, _string_methods)
utils.add_method(StringScalar, _string_methods)
utils.add_method(StringScalar, {'cat': _cat})