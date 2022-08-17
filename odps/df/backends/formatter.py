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

from distutils.version import LooseVersion
import itertools

from ... import compat
from ...config import options
from ...compat import cgi, u, izip, Iterable
from ...console import get_terminal_size
from ...utils import to_text, to_str, indent, require_package
from ...models import Table
from ..types import *
from ..expr.expressions import CollectionExpr, Scalar
from ..utils import is_source_collection, traverse_until_source


def is_integer(val):
    return isinstance(val, six.integer_types)


def is_sequence(x):
    try:
        iter(x)
        len(x)  # it has a length
        return not isinstance(x, six.string_types) and \
               not isinstance(x, six.binary_type)
    except (TypeError, AttributeError):
        return False


def _pprint_seq(seq, _nest_lvl=0, max_seq_items=None, **kwds):
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather then calling this directly.

    bounds length of printed sequence, depending on options
    """
    if isinstance(seq, set):
        fmt = u("set([%s])")
    else:
        fmt = u("[%s]") if hasattr(seq, '__setitem__') else u("(%s)")

    if max_seq_items is False:
        nitems = len(seq)
    else:
        nitems = max_seq_items or options.display.max_seq_items or len(seq)

    s = iter(seq)
    r = []
    for i in range(min(nitems, len(seq))):  # handle sets, no slicing
        r.append(pprint_thing(next(s), _nest_lvl + 1, max_seq_items=max_seq_items, **kwds))
    body = ", ".join(r)

    if nitems < len(seq):
        body += ", ..."
    elif isinstance(seq, tuple) and len(seq) == 1:
        body += ','

    return fmt % body


def _pprint_dict(seq, _nest_lvl=0, max_seq_items=None, **kwds):
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather then calling this directly.
    """
    fmt = u("{%s}")
    pairs = []

    pfmt = u("%s: %s")

    if max_seq_items is False:
        nitems = len(seq)
    else:
        nitems = max_seq_items or options.display.max_seq_items or len(seq)

    for k, v in list(seq.items())[:nitems]:
        pairs.append(pfmt % (pprint_thing(k, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds),
                             pprint_thing(v, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds)))

    if nitems < len(seq):
        return fmt % (", ".join(pairs) + ", ...")
    else:
        return fmt % ", ".join(pairs)


def pprint_thing(thing, _nest_lvl=0, escape_chars=None, default_escapes=False,
                 quote_strings=False, max_seq_items=None):
    """
    This function is the sanctioned way of converting objects
    to a unicode representation.

    properly handles nested sequences containing unicode strings
    (unicode(object) does not)

    Parameters
    ----------
    thing : anything to be formatted
    _nest_lvl : internal use only. pprint_thing() is mutually-recursive
        with pprint_sequence, this argument is used to keep track of the
        current nesting level, and limit it.
    escape_chars : list or dict, optional
        Characters to escape. If a dict is passed the values are the
        replacements
    default_escapes : bool, default False
        Whether the input escape characters replaces or adds to the defaults
    max_seq_items : False, int, default None
        Pass thru to other pretty printers to limit sequence printing

    Returns
    -------
    result - unicode object on py2, str on py3. Always Unicode.

    """
    def as_escaped_unicode(thing, escape_chars=escape_chars):
        # Unicode is fine, else we try to decode using utf-8 and 'replace'
        # if that's not it either, we have no way of knowing and the user
        # should deal with it himself.

        try:
            result = six.text_type(thing)  # we should try this first
        except UnicodeDecodeError:
            # either utf-8 or we replace errors
            result = str(thing).decode('utf-8', "replace")

        translate = {'\t': r'\t',
                     '\n': r'\n',
                     '\r': r'\r',
                     }
        if isinstance(escape_chars, dict):
            if default_escapes:
                translate.update(escape_chars)
            else:
                translate = escape_chars
            escape_chars = list(escape_chars.keys())
        else:
            escape_chars = escape_chars or tuple()
        for c in escape_chars:
            result = result.replace(c, translate[c])

        return six.text_type(result)

    if (six.PY3 and hasattr(thing, '__next__')) or hasattr(thing, 'next'):
        return six.text_type(thing)
    elif (isinstance(thing, dict) and
          _nest_lvl < options.display.pprint_nest_depth):
        result = _pprint_dict(thing, _nest_lvl, quote_strings=True, max_seq_items=max_seq_items)
    elif is_sequence(thing) and _nest_lvl < \
            options.display.pprint_nest_depth:
        result = _pprint_seq(thing, _nest_lvl, escape_chars=escape_chars,
                             quote_strings=quote_strings, max_seq_items=max_seq_items)
    elif isinstance(thing, six.string_types) and quote_strings:
        if six.PY3:
            fmt = "'%s'"
        else:
            fmt = "u'%s'"
        result = fmt % as_escaped_unicode(thing)
    else:
        result = as_escaped_unicode(thing)

    return six.text_type(result)  # always unicode


def _justify(texts, max_len, mode='right'):
    """
    Perform ljust, center, rjust against string or list-like
    """
    if mode == 'left':
        return [x.ljust(max_len) for x in texts]
    elif mode == 'center':
        return [x.center(max_len) for x in texts]
    else:
        return [x.rjust(max_len) for x in texts]


def _join_unicode(lines, sep=''):
    try:
        return sep.join(lines)
    except UnicodeDecodeError:
        sep = six.text_type(sep)
        return sep.join([x.decode('utf-8') if isinstance(x, str) else x
                         for x in lines])


def adjoin(space, *lists, **kwargs):
    """
    Glues together two sets of strings using the amount of space requested.
    The idea is to prettify.

    ----------
    space : int
        number of spaces for padding
    lists : str
        list of str which being joined
    strlen : callable
        function used to calculate the length of each str. Needed for unicode
        handling.
    justfunc : callable
        function used to justify str. Needed for unicode handling.
    """
    strlen = kwargs.pop('strlen', len)
    justfunc = kwargs.pop('justfunc', _justify)

    out_lines = []
    newLists = []
    lengths = [max(map(strlen, x)) + space for x in lists[:-1]]
    # not the last one
    lengths.append(max(map(len, lists[-1])))
    maxLen = max(map(len, lists))
    for i, lst in enumerate(lists):
        nl = justfunc(lst, lengths[i], mode='left')
        nl.extend([' ' * lengths[i]] * (maxLen - len(lst)))
        newLists.append(nl)
    toJoin = zip(*newLists)
    for lines in toJoin:
        out_lines.append(_join_unicode(lines))
    return _join_unicode(out_lines, sep='\n')


def _make_fixed_width(strings, justify='right', minimum=None,
                      adj=None):

    if len(strings) == 0 or justify == 'all':
        return strings

    if adj is None:
        adj = _get_adjustment()

    max_len = max([adj.len(x) for x in strings])

    if minimum is not None:
        max_len = max(minimum, max_len)

    conf_max = options.display.max_colwidth
    if conf_max is not None and max_len > conf_max:
        max_len = conf_max

    def just(x):
        if conf_max is not None:
            if (conf_max > 3) & (adj.len(x) > max_len):
                x = x[:max_len - 3] + '...'
        return x

    strings = [just(x) for x in strings]
    result = adj.justify(strings, max_len, mode=justify)
    return result


def _binify(cols, line_width):
    adjoin_width = 1
    bins = []
    curr_width = 0
    i_last_column = len(cols) - 1
    for i, w in enumerate(cols):
        w_adjoined = w + adjoin_width
        curr_width += w_adjoined
        if i_last_column == i:
            wrap = curr_width + 1 > line_width and i > 0
        else:
            wrap = curr_width + 2 > line_width and i > 0
        if wrap:
            bins.append(i)
            curr_width = w_adjoined

    bins.append(len(cols))
    return bins


class TextAdjustment(object):

    def __init__(self):
        self.encoding = options.display.encoding

    def len(self, text):
        return compat.strlen(text, encoding=self.encoding)

    def justify(self, texts, max_len, mode='right'):
        return _justify(texts, max_len, mode=mode)

    def adjoin(self, space, *lists, **kwargs):
        return adjoin(space, *lists, strlen=self.len,
                      justfunc=self.justify, **kwargs)


class EastAsianTextAdjustment(TextAdjustment):

    def __init__(self):
        super(EastAsianTextAdjustment, self).__init__()
        if options.display.unicode.ambiguous_as_wide:
            self.ambiguous_width = 2
        else:
            self.ambiguous_width = 1

    def len(self, text):
        return compat.east_asian_len(to_text(text), encoding=self.encoding,
                                     ambiguous_width=self.ambiguous_width)

    def justify(self, texts, max_len, mode='right'):
        # re-calculate padding space per str considering East Asian Width
        def _get_pad(t):
            return max_len - self.len(t) + len(t)

        if mode == 'left':
            return [x.ljust(_get_pad(x)) for x in texts]
        elif mode == 'center':
            return [x.center(_get_pad(x)) for x in texts]
        else:
            return [x.rjust(_get_pad(x)) for x in texts]


def _get_adjustment():
    use_east_asian_width = options.display.unicode.east_asian_width
    if use_east_asian_width:
        return EastAsianTextAdjustment()
    else:
        return TextAdjustment()


class TableFormatter(object):
    is_truncated = False
    show_dimensions = None

    @property
    def should_show_dimensions(self):
        return self.show_dimensions is True or (self.show_dimensions == 'truncate' and
                                                self.is_truncated)

    def _get_formatter(self, i):
        if isinstance(self.formatters, (list, tuple)):
            if is_integer(i):
                return self.formatters[i]
            else:
                return None
        else:
            if is_integer(i) and i not in self.columns:
                i = self.columns[i]
            return self.formatters.get(i, None)


class ResultFrameFormatter(TableFormatter):

    """
    Render a Expr result

    self.to_string() : console-friendly tabular output
    self.to_html()   : html table
    self.to_latex()   : LaTeX tabular environment table

    """

    def __init__(self, frame, buf=None, columns=None, col_space=None,
                 header=True, index=True, na_rep='NaN', formatters=None,
                 justify=None, float_format=None, sparsify=None,
                 index_names=True, line_width=None, max_rows=None,
                 max_cols=None, show_dimensions=False, **kwds):
        self.frame = frame
        self.buf = buf if buf is not None else six.StringIO()
        self.show_index_names = index_names

        if sparsify is None:
            sparsify = options.display.multi_sparse

        self.sparsify = sparsify

        self.float_format = float_format
        self.formatters = formatters if formatters is not None else {}
        self.na_rep = na_rep
        self.col_space = col_space
        self.header = header
        self.index = index
        self.line_width = line_width
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_rows_displayed = min(max_rows or len(self.frame),
                                      len(self.frame))
        self.show_dimensions = show_dimensions

        if justify is None:
            self.justify = options.display.colheader_justify
        else:
            self.justify = justify

        self.kwds = kwds

        if columns is not None:
            self.columns = columns
            self.frame = self.frame[self.columns]
        else:
            self.columns = frame.columns

        self._chk_truncate()
        self.adj = _get_adjustment()

    def _chk_truncate(self):
        '''
        Checks whether the frame should be truncated. If so, slices
        the frame up.
        '''

        # Column of which first element is used to determine width of a dot col
        self.tr_size_col = -1

        # Cut the data to the information actually printed
        max_cols = self.max_cols
        max_rows = self.max_rows

        if max_cols == 0 or max_rows == 0:  # assume we are in the terminal (why else = 0)
            (w, h) = get_terminal_size()
            self.w = w
            self.h = h
            if self.max_rows == 0:
                dot_row = 1
                prompt_row = 1
                if self.show_dimensions:
                    show_dimension_rows = 3
                n_add_rows = self.header + dot_row + show_dimension_rows + prompt_row
                max_rows_adj = self.h - n_add_rows  # rows available to fill with actual data
                self.max_rows_adj = max_rows_adj

            # Format only rows and columns that could potentially fit the screen
            if max_cols == 0 and len(self.frame.columns) > w:
                max_cols = w
            if max_rows == 0 and len(self.frame) > h:
                max_rows = h

        if not hasattr(self, 'max_rows_adj'):
            self.max_rows_adj = max_rows
        if not hasattr(self, 'max_cols_adj'):
            self.max_cols_adj = max_cols

        max_cols_adj = self.max_cols_adj
        max_rows_adj = self.max_rows_adj

        truncate_h = max_cols_adj and (len(self.columns) > max_cols_adj)
        truncate_v = max_rows_adj and (len(self.frame) > max_rows_adj)

        frame = self.frame
        if truncate_h:
            if max_cols_adj == 0:
                col_num = len(frame.columns)
            elif max_cols_adj == 1:
                frame = frame[:, :max_cols]
                col_num = max_cols
            else:
                col_num = (max_cols_adj // 2)
                frame = frame[:, :col_num].concat(frame[:, -col_num:], axis=1)
            self.tr_col_num = col_num
        if truncate_v:
            if max_rows_adj == 0:
                row_num = len(frame)
            if max_rows_adj == 1:
                row_num = max_rows
                frame = frame[:max_rows, :]
            else:
                row_num = max_rows_adj // 2
                frame = frame[:row_num, :].concat(frame[-row_num:, :])
            self.tr_row_num = row_num

        self.tr_frame = frame
        self.truncate_h = truncate_h
        self.truncate_v = truncate_v
        self.is_truncated = self.truncate_h or self.truncate_v

    def _to_str_columns(self):
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
        frame = self.tr_frame

        # may include levels names also

        str_index = self._get_formatted_index(frame)
        str_columns = self._get_formatted_column_labels(frame)

        if self.header:
            stringified = []
            for i, c in enumerate(frame.columns):
                cheader = str_columns[i]
                max_colwidth = max(self.col_space or 0,
                                   *(self.adj.len(x) for x in cheader))
                fmt_values = self._format_col(i)
                fmt_values = _make_fixed_width(fmt_values, self.justify,
                                               minimum=max_colwidth,
                                               adj=self.adj)

                max_len = max(max([self.adj.len(x) for x in fmt_values]),
                              max_colwidth)
                cheader = self.adj.justify(cheader, max_len, mode=self.justify)
                stringified.append(cheader + fmt_values)
        else:
            stringified = []
            for i, c in enumerate(frame):
                fmt_values = self._format_col(i)
                fmt_values = _make_fixed_width(fmt_values, self.justify,
                                               minimum=(self.col_space or 0),
                                               adj=self.adj)

                stringified.append(fmt_values)

        strcols = stringified
        if self.index:
            strcols.insert(0, str_index)

        # Add ... to signal truncated
        truncate_h = self.truncate_h
        truncate_v = self.truncate_v

        if truncate_h:
            col_num = self.tr_col_num
            col_width = self.adj.len(strcols[self.tr_size_col][0])  # infer from column header
            strcols.insert(self.tr_col_num + 1, ['...'.center(col_width)] * (len(str_index)))
        if truncate_v:
            n_header_rows = len(str_index) - len(frame)
            row_num = self.tr_row_num
            for ix, col in enumerate(strcols):
                cwidth = self.adj.len(strcols[ix][row_num])  # infer from above row
                is_dot_col = False
                if truncate_h:
                    is_dot_col = ix == col_num + 1
                if cwidth > 3 or is_dot_col:
                    my_str = '...'
                else:
                    my_str = '..'

                if ix == 0:
                    dot_mode = 'left'
                elif is_dot_col:
                    cwidth = self.adj.len(strcols[self.tr_size_col][0])
                    dot_mode = 'center'
                else:
                    dot_mode = 'right'
                dot_str = self.adj.justify([my_str], cwidth, mode=dot_mode)[0]
                strcols[ix].insert(row_num + n_header_rows, dot_str)
        return strcols

    def to_string(self):
        """
        Render a DataFrame to a console-friendly tabular output.
        """
        frame = self.frame

        if len(frame.columns) == 0 or len(frame.index) == 0:
            info_line = (u('Empty %s\nColumns: %s\nIndex: %s')
                         % (type(self.frame).__name__,
                            pprint_thing(frame.columns),
                            pprint_thing(frame.index)))
            text = info_line
        else:
            strcols = self._to_str_columns()
            if self.line_width is None:  # no need to wrap around just print the whole frame
                text = self.adj.adjoin(1, *strcols)
            elif not isinstance(self.max_cols, int) or self.max_cols > 0:  # need to wrap around
                text = self._join_multiline(*strcols)
            else:  # max_cols == 0. Try to fit frame to terminal
                text = self.adj.adjoin(1, *strcols).split('\n')
                row_lens = [len(it) for it in text]
                max_len_col_ix = row_lens.index(max(row_lens))
                max_len = row_lens[max_len_col_ix]
                headers = [ele[0] for ele in strcols]
                # Size of last col determines dot col size. See `self._to_str_columns
                size_tr_col = len(headers[self.tr_size_col])
                max_len += size_tr_col  # Need to make space for largest row plus truncate dot col
                dif = max_len - self.w
                adj_dif = dif
                col_lens = [max(len(it) for it in ele) for ele in strcols]
                n_cols = len(col_lens)
                counter = 0
                while adj_dif > 0 and n_cols > 1:
                    counter += 1
                    mid = int(round(n_cols / 2.))
                    # mid_ix = col_lens.index[mid]
                    col_len = col_lens[mid]
                    adj_dif -= (col_len + 1)  # adjoin adds one
                    col_lens = col_lens[:mid] + col_len[mid+1: ]
                    n_cols = len(col_lens)
                max_cols_adj = n_cols - self.index  # subtract index column
                self.max_cols_adj = max_cols_adj

                # Call again _chk_truncate to cut frame appropriately
                # and then generate string representation
                self._chk_truncate()
                strcols = self._to_str_columns()
                text = self.adj.adjoin(1, *strcols)

        self.buf.writelines(text)

        if self.should_show_dimensions:
            self.buf.write("\n\n[%d rows x %d columns]"
                           % (len(frame), len(frame.columns)))

    def _join_multiline(self, *strcols):
        lwidth = self.line_width
        adjoin_width = 1
        strcols = list(strcols)
        if self.index:
            idx = strcols.pop(0)
            lwidth -= max(self.adj.len(x) for x in idx) + adjoin_width

        col_widths = [max(self.adj.len(x) for x in col)
                      if len(col) > 0 else 0
                      for col in strcols]
        col_bins = _binify(col_widths, lwidth)
        nbins = len(col_bins)

        if self.truncate_v:
            nrows = self.max_rows_adj + 1
        else:
            nrows = len(self.frame)

        str_lst = []
        st = 0
        for i, ed in enumerate(col_bins):
            row = strcols[st:ed]
            row.insert(0, idx)
            if nbins > 1:
                if ed <= len(strcols) and i < nbins - 1:
                    row.append([' \\'] + ['  '] * (nrows - 1))
                else:
                    row.append([' '] * nrows)
            str_lst.append(self.adj.adjoin(adjoin_width, *row))
            st = ed
        return '\n\n'.join(str_lst)

    def _format_col(self, i):
        frame = self.tr_frame
        formatter = self._get_formatter(i)
        return format_array(
            frame[:, i],
            frame.dtypes[i],
            formatter, float_format=self.float_format, na_rep=self.na_rep,
            space=self.col_space
        )

    def to_html(self, classes=None, notebook=False):
        """
        Render a DataFrame to a html table.

        Parameters
        ----------
        notebook : {True, False}, optional, default False
            Whether the generated HTML is for IPython Notebook.

        """
        html_renderer = HTMLFormatter(self, classes=classes,
                                      max_rows=self.max_rows,
                                      max_cols=self.max_cols,
                                      notebook=notebook)
        if hasattr(self.buf, 'write'):
            html_renderer.write_result(self.buf)
        elif isinstance(self.buf, six.string_types):
            with open(self.buf, 'w') as f:
                html_renderer.write_result(f)
        else:
            raise TypeError('buf is not a file name and it has no write '
                            ' method')

    def _get_formatted_column_labels(self, frame):

        def is_numeric_dtype(dtype):
            return is_number(dtype)

        columns = frame.columns

        fmt_columns = [col.name for col in columns]
        dtypes = self.frame.dtypes
        need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))
        str_columns = [[' ' + x
                        if not self._get_formatter(i) and need_leadsp[x]
                        else x]
                       for i, (col, x) in
                       enumerate(zip(columns, fmt_columns))]

        if self.show_index_names and self.has_index_names:
            for x in str_columns:
                x.append('')

        # self.str_columns = str_columns
        return str_columns

    @property
    def has_index_names(self):
        return _has_names(self.frame.index)

    @property
    def has_column_names(self):
        return _has_names(self.frame.columns)

    def _get_formatted_index(self, frame):
        # Note: this is only used by to_string() and to_latex(), not by to_html().
        index = frame.index

        show_index_names = self.show_index_names and self.has_index_names
        show_col_names = (self.show_index_names and self.has_column_names)

        fmt = self._get_formatter('__index__')

        fmt_index = [[str(i) for i in index]]
        fmt_index = [tuple(_make_fixed_width(list(x), justify='left',
                                             minimum=(self.col_space or 0),
                                             adj=self.adj))
                     for x in fmt_index]

        adjoined = self.adj.adjoin(1, *fmt_index).split('\n')

        # empty space for columns
        if show_col_names:
            col_header = ['%s' % x for x in self._get_column_name_list()]
        else:
            col_header = ['']

        if self.header:
            return col_header + adjoined
        else:
            return adjoined

    def _get_column_name_list(self):
        names = []
        columns = self.frame.columns
        names.append('' if columns.name is None else columns.name)
        return names


class HTMLFormatter(TableFormatter):

    indent_delta = 2

    def __init__(self, formatter, classes=None, max_rows=None, max_cols=None,
                 notebook=False):
        self.fmt = formatter
        self.classes = classes

        self.frame = self.fmt.frame
        self.columns = self.fmt.tr_frame.columns
        self.elements = []
        self.bold_rows = self.fmt.kwds.get('bold_rows', False)
        self.escape = self.fmt.kwds.get('escape', True)

        self.max_rows = max_rows or len(self.fmt.frame)
        self.max_cols = max_cols or len(self.fmt.columns)
        self.show_dimensions = self.fmt.show_dimensions
        self.is_truncated = (self.max_rows < len(self.fmt.frame) or
                             self.max_cols < len(self.fmt.columns))
        self.notebook = notebook

    def write(self, s, indent=0):
        rs = pprint_thing(s)
        self.elements.append(' ' * indent + rs)

    def write_th(self, s, indent=0, tags=None):
        if (self.fmt.col_space is not None
                and self.fmt.col_space > 0):
            tags = (tags or "")
            tags += 'style="min-width: %s;"' % self.fmt.col_space

        return self._write_cell(s, kind='th', indent=indent, tags=tags)

    def write_td(self, s, indent=0, tags=None):
        return self._write_cell(s, kind='td', indent=indent, tags=tags)

    def _write_cell(self, s, kind='td', indent=0, tags=None):
        if tags is not None:
            start_tag = '<%s %s>' % (kind, tags)
        else:
            start_tag = '<%s>' % kind

        if self.escape:
            # escape & first to prevent double escaping of &
            esc = OrderedDict(
                [('&', r'&amp;'), ('<', r'&lt;'), ('>', r'&gt;')]
            )
        else:
            esc = {}
        rs = pprint_thing(s, escape_chars=esc).strip()
        self.write(
            '%s%s</%s>' % (start_tag, rs, kind), indent)

    def write_tr(self, line, indent=0, indent_delta=4, header=False,
                 align=None, tags=None, nindex_levels=0):
        if tags is None:
            tags = {}

        if align is None:
            self.write('<tr>', indent)
        else:
            self.write('<tr style="text-align: %s;">' % align, indent)
        indent += indent_delta

        for i, s in enumerate(line):
            val_tag = tags.get(i, None)
            if header or (self.bold_rows and i < nindex_levels):
                self.write_th(s, indent, tags=val_tag)
            else:
                self.write_td(s, indent, tags=val_tag)

        indent -= indent_delta
        self.write('</tr>', indent)

    def write_result(self, buf):
        indent = 0
        frame = self.frame

        _classes = ['dataframe']  # Default class.
        if self.classes is not None:
            if isinstance(self.classes, str):
                self.classes = self.classes.split()
            if not isinstance(self.classes, (list, tuple)):
                raise AssertionError(('classes must be list or tuple, '
                                      'not %s') % type(self.classes))
            _classes.extend(self.classes)

        if self.notebook:
            div_style = ''
            try:
                import IPython
                if IPython.__version__ < LooseVersion('3.0.0'):
                    div_style = ' style="max-width:1500px;overflow:auto;"'
            except ImportError:
                pass

            self.write('<div{0}>'.format(div_style))

        self.write('<table border="1" class="%s">' % ' '.join(_classes),
                   indent)

        indent += self.indent_delta
        indent = self._write_header(indent)
        indent = self._write_body(indent)

        self.write('</table>', indent)
        if self.should_show_dimensions:
            by = chr(215) if six.PY3 else unichr(215)  # ×
            self.write(u('<p>%d rows %s %d columns</p>') %
                       (len(frame), by, len(frame.columns)))

        if self.notebook:
            self.write('</div>')

        _put_lines(buf, self.elements)

    def _write_header(self, indent):
        truncate_h = self.fmt.truncate_h
        row_levels = 1
        if not self.fmt.header:
            # write nothing
            return indent

        def _column_header():
            row = []

            if self.fmt.index:
                row.append('')
            row.extend([col.name for col in self.columns])
            return row

        self.write('<thead>', indent)
        row = []

        indent += self.indent_delta

        col_row = _column_header()
        align = self.fmt.justify

        if truncate_h:
            ins_col = row_levels + self.fmt.tr_col_num
            col_row.insert(ins_col, '...')

        self.write_tr(col_row, indent, self.indent_delta, header=True,
                      align=align)

        if self.fmt.has_index_names and self.fmt.index:
            row = [
                x if x is not None else '' for x in self.frame.index.names
            ] + [''] * min(len(self.columns), self.max_cols)
            if truncate_h:
                ins_col = row_levels + self.fmt.tr_col_num
                row.insert(ins_col, '')
            self.write_tr(row, indent, self.indent_delta, header=True)

        indent -= self.indent_delta
        self.write('</thead>', indent)

        return indent

    def _write_body(self, indent):
        self.write('<tbody>', indent)
        indent += self.indent_delta

        fmt_values = {}
        for i in range(min(len(self.columns), self.max_cols)):
            fmt_values[i] = self.fmt._format_col(i)

        # write values
        if self.fmt.index:
            self._write_regular_rows(fmt_values, indent)
        else:
            for i in range(len(self.frame)):
                row = [fmt_values[j][i] for j in range(len(self.columns))]
                self.write_tr(row, indent, self.indent_delta, tags=None)

        indent -= self.indent_delta
        self.write('</tbody>', indent)
        indent -= self.indent_delta

        return indent

    def _write_regular_rows(self, fmt_values, indent):
        truncate_h = self.fmt.truncate_h
        truncate_v = self.fmt.truncate_v

        ncols = len(self.fmt.tr_frame.columns)
        nrows = len(self.fmt.tr_frame)
        fmt = self.fmt._get_formatter('__index__')
        if fmt is not None:
            index_values = self.fmt.tr_frame.index.map(fmt)
        else:
            index_values = [str(i) for i in self.fmt.tr_frame.index]

        row = []
        for i in range(nrows):

            if truncate_v and i == (self.fmt.tr_row_num):
                str_sep_row = ['...' for ele in row]
                self.write_tr(str_sep_row, indent, self.indent_delta, tags=None,
                              nindex_levels=1)

            row = []
            row.append(index_values[i])
            row.extend(fmt_values[j][i] for j in range(ncols))

            if truncate_h:
                dot_col_ix = self.fmt.tr_col_num + 1
                row.insert(dot_col_ix, '...')
            self.write_tr(row, indent, self.indent_delta, tags=None,
                          nindex_levels=1)

    def _write_hierarchical_rows(self, fmt_values, indent):
        template = 'rowspan="%d" valign="top"'

        truncate_h = self.fmt.truncate_h
        truncate_v = self.fmt.truncate_v
        frame = self.fmt.tr_frame
        ncols = len(frame.columns)
        nrows = len(frame)
        row_levels = self.frame.index.nlevels

        idx_values = frame.index.format(sparsify=False, adjoin=False, names=False)
        idx_values = compat.lzip(*idx_values)

        if self.fmt.sparsify:
            # GH3547
            sentinel = sentinel_factory()
            levels = frame.index.format(sparsify=sentinel, adjoin=False, names=False)

            level_lengths = _get_level_lengths(levels, sentinel)
            inner_lvl = len(level_lengths) - 1
            if truncate_v:
                # Insert ... row and adjust idx_values and
                # level_lengths to take this into account.
                ins_row = self.fmt.tr_row_num
                for lnum, records in enumerate(level_lengths):
                    rec_new = {}
                    for tag, span in list(records.items()):
                        if tag >= ins_row:
                            rec_new[tag + 1] = span
                        elif tag + span > ins_row:
                            rec_new[tag] = span + 1
                            dot_row = list(idx_values[ins_row - 1])
                            dot_row[-1] = u('...')
                            idx_values.insert(ins_row, tuple(dot_row))
                        else:
                            rec_new[tag] = span
                        # If ins_row lies between tags, all cols idx cols receive ...
                        if tag + span == ins_row:
                            rec_new[ins_row] = 1
                            if lnum == 0:
                                idx_values.insert(ins_row, tuple([u('...')]*len(level_lengths)))
                    level_lengths[lnum] = rec_new

                level_lengths[inner_lvl][ins_row] = 1
                for ix_col in range(len(fmt_values)):
                    fmt_values[ix_col].insert(ins_row, '...')
                nrows += 1

            for i in range(nrows):
                row = []
                tags = {}

                sparse_offset = 0
                j = 0
                for records, v in zip(level_lengths, idx_values[i]):
                    if i in records:
                        if records[i] > 1:
                            tags[j] = template % records[i]
                    else:
                        sparse_offset += 1
                        continue

                    j += 1
                    row.append(v)

                row.extend(fmt_values[j][i] for j in range(ncols))
                if truncate_h:
                    row.insert(row_levels - sparse_offset + self.fmt.tr_col_num, '...')
                self.write_tr(row, indent, self.indent_delta, tags=tags,
                              nindex_levels=len(levels) - sparse_offset)
        else:
            for i in range(len(frame)):
                idx_values = list(zip(*frame.index.format(sparsify=False,
                                                          adjoin=False,
                                                          names=False)))
                row = []
                row.extend(idx_values[i])
                row.extend(fmt_values[j][i] for j in range(ncols))
                if truncate_h:
                    row.insert(row_levels + self.fmt.tr_col_num, '...')
                self.write_tr(row, indent, self.indent_delta, tags=None,
                              nindex_levels=frame.index.nlevels)


# ----------------------------------------------------------------------
# Array formatters


def is_float_dtype(t):
    return isinstance(t, Float)


def is_integer_dtype(t):
    return isinstance(t, Integer)


def is_datetime_dtype(t):
    return isinstance(t, Datetime)


def format_array(values, dtype, formatter, float_format=None, na_rep='NaN',
                 digits=None, space=None, justify='right'):

    if is_float_dtype(dtype):
        fmt_klass = FloatArrayFormatter
    elif is_integer_dtype(dtype):
        fmt_klass = IntArrayFormatter
    elif is_datetime_dtype(dtype):
        fmt_klass = Datetime64Formatter
    else:
        fmt_klass = GenericArrayFormatter

    if space is None:
        space = options.display.column_space

    if float_format is None:
        float_format = options.display.float_format

    if digits is None:
        digits = options.display.precision

    fmt_obj = fmt_klass(values, digits=digits, na_rep=na_rep,
                        float_format=float_format,
                        formatter=formatter, space=space,
                        justify=justify)

    return fmt_obj.get_result()


class GenericArrayFormatter(object):

    def __init__(self, values, digits=7, formatter=None, na_rep='NaN',
                 space=12, float_format=None, justify='right'):
        self.values = values
        self.digits = digits
        self.na_rep = na_rep
        self.space = space
        self.formatter = formatter
        self.float_format = float_format
        self.justify = justify

    def get_result(self):
        fmt_values = [v if v is not None else self.na_rep for v in self._format_strings()]
        return _make_fixed_width(fmt_values, self.justify)

    def _format_strings(self):
        if self.float_format is None:
            float_format = options.display.float_format
            if float_format is None:
                fmt_str = '%% .%dg' % options.display.precision
                float_format = lambda x: fmt_str % x
        else:
            float_format = self.float_format

        formatter = self.formatter if self.formatter is not None else \
            (lambda x: pprint_thing(x, escape_chars=('\t', '\r', '\n')))

        def _format(x):
            if self.na_rep is not None and x is None:
                if x is None:
                    return 'None'
                return self.na_rep
            else:
                # object dtype
                return '%s' % formatter(x)

        vals = self.values

        fmt_values = []
        for i, v in enumerate(vals):
            fmt_values.append(' %s' % _format(v))

        return fmt_values


class FloatArrayFormatter(GenericArrayFormatter):

    """

    """

    def __init__(self, *args, **kwargs):
        GenericArrayFormatter.__init__(self, *args, **kwargs)

        if self.float_format is not None and self.formatter is None:
            self.formatter = self.float_format

    def _format_with(self, fmt_str):
        def _val(x, threshold):
            if x is not None:
                if (threshold is None or
                        abs(x) > options.display.chop_threshold):
                    return fmt_str % x
                else:
                    if fmt_str.endswith("e"):  # engineering format
                        return "0"
                    else:
                        return fmt_str % 0
            else:

                return self.na_rep

        threshold = options.display.chop_threshold
        fmt_values = [_val(x, threshold) for x in self.values]
        return _trim_zeros(fmt_values, self.na_rep)

    def _format_strings(self):
        if self.formatter is not None:
            fmt_values = [self.formatter(x) for x in self.values]
        else:
            fmt_str = '%% .%df' % self.digits
            fmt_values = self._format_with(fmt_str)

            if len(fmt_values) > 0:
                maxlen = max(len(x) for x in fmt_values)
            else:
                maxlen = 0

            too_long = maxlen > self.digits + 6

            abs_vals = [abs(val) if val is not None else float('nan') for val in self.values]

            # this is pretty arbitrary for now
            has_large_values = any(abs_val > 1e8 for abs_val in abs_vals)
            has_small_values = any((abs_val < 10 ** (-self.digits)) &
                                   (abs_val > 0) for abs_val in abs_vals)

            if too_long and has_large_values:
                fmt_str = '%% .%de' % self.digits
                fmt_values = self._format_with(fmt_str)
            elif has_small_values:
                fmt_str = '%% .%de' % self.digits
                fmt_values = self._format_with(fmt_str)

        return fmt_values


class IntArrayFormatter(GenericArrayFormatter):

    def _format_strings(self):
        formatter = self.formatter or (lambda x: '% d' % x if x is not None else self.na_rep)
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values


class Datetime64Formatter(GenericArrayFormatter):
    def __init__(self, values, nat_rep='NaT', date_format=None, **kwargs):
        super(Datetime64Formatter, self).__init__(values, **kwargs)
        self.nat_rep = nat_rep
        self.date_format = date_format

    def _format_strings(self):
        """ we by definition have DO NOT have a TZ """

        values = self.values
        return [val.strftime('%Y-%m-%d %H:%M:%S') if val is not None else self.nat_rep for val in values]


def _has_names(index):
    return hasattr(index, 'name') and index.name is not None


def _put_lines(buf, lines):
    if any(isinstance(x, six.text_type) for x in lines):
        lines = [six.text_type(x) for x in lines]
    buf.write('\n'.join(lines))


def sentinel_factory():

    class Sentinel(object):
        pass

    return Sentinel()


def _get_level_lengths(levels, sentinel=''):
    from itertools import groupby

    def _make_grouper():
        record = {'count': 0}

        def grouper(x):
            if x != sentinel:
                record['count'] += 1
            return record['count']
        return grouper

    result = []
    for lev in levels:
        i = 0
        f = _make_grouper()
        recs = {}
        for key, gpr in groupby(lev, f):
            values = list(gpr)
            recs[i] = len(values)
            i += len(values)

        result.append(recs)

    return result



def _trim_zeros(str_floats, na_rep='NaN'):
    """
    Trims zeros and decimal points.
    """
    trimmed = str_floats

    def _cond(values):
        non_na = [x for x in values if x != na_rep]
        return (len(non_na) > 0 and all([x.endswith('0') for x in non_na]) and
                not(any([('e' in x) or ('E' in x) for x in non_na])))

    while _cond(trimmed):
        trimmed = [x[:-1] if x != na_rep else x for x in trimmed]

    # trim decimal points
    return [x[:-1] if x.endswith('.') and x != na_rep else x for x in trimmed]



class ExprExecutionGraphFormatter(object):
    def __init__(self, dag):
        self._dag = dag

    @require_package('graphviz')
    def _repr_svg_(self):
        from graphviz import Source
        return Source(self._to_dot())._repr_svg_()

    def _format_expr(self, expr):
        if is_source_collection(expr):
            if isinstance(expr._source_data, Table):
                return 'Collection: %s' % expr._source_data.name
            else:
                return 'Collection: pandas.DataFrame'
        elif isinstance(expr, Scalar) and expr._value is not None:
            return 'Scalar: %r' % expr._value
        else:
            node_name = getattr(expr, 'node_name', expr.__class__.__name__)
            if isinstance(expr, CollectionExpr):
                return '%s[Collection]' % node_name
            else:
                t = 'Scalar' if isinstance(expr, Scalar) else 'Sequence'
                return '{%s[%s]|name: %s|type: %s}' % (
                    node_name.capitalize(), t, expr.name, expr.dtype)

    def _to_str(self):
        buffer = six.StringIO()

        nodes = self._dag.topological_sort()
        for i, node in enumerate(nodes):
            sid = i + 1

            buffer.write('Stage {0}: \n\n'.format(sid))
            buffer.write(repr(node))
            if i < len(nodes) - 1:
                buffer.write('\n\n')

        return to_str(buffer.getvalue())

    def _to_html(self):
        buffer = six.StringIO()

        for i, node in enumerate(self._dag.topological_sort()):
            sid = i + 1

            buffer.write('<h3>Stage {0}</h3>'.format(sid))
            buffer.write(node._repr_html_())

        return to_str(buffer.getvalue())

    def _to_dot(self):
        buffer = six.StringIO()
        write = lambda x: buffer.write(to_text(x))
        write_newline = lambda x: write(x if x.endswith('\n') else x + '\n')
        write_indent_newline = lambda x, ind=1: write_newline(indent(x, 2 * ind))

        nid = itertools.count(1)

        write_newline('digraph DataFrameDAG {')
        write_indent_newline('START [shape=ellipse, label="start", style=filled, fillcolor=Pink];')

        nodes = self._dag.topological_sort()
        traversed = dict()
        for sid, node in izip(itertools.count(1), nodes):
            expr_node = node.expr
            traversed[id(node)] = sid

            pres = self._dag.predecessors(node)
            write_indent_newline('subgraph clusterSTAGE{0} {{'.format(sid))
            write_indent_newline('label = "Stage {0}"'.format(sid), ind=2)

            compiled = str(node._sql()) if hasattr(node, '_sql') else None

            for expr in traverse_until_source(expr_node, unique=True):
                if id(expr) not in traversed:
                    eid = next(nid)
                    traversed[id(expr)] = eid
                else:
                    eid = traversed[id(expr)]

                name_args = list(expr.iter_args())
                labels = [self._format_expr(expr), ]
                for i, name_arg in enumerate(name_args):
                    if name_arg[1] is None:
                        continue
                    labels.append('<f{0}>{1}'.format(i, name_arg[0].strip('_')))

                attr = ', style=filled, fillcolor=LightGrey' if isinstance(expr, CollectionExpr) else ''
                write_indent_newline(
                    'EXPR{0} [shape=record, label="{1}"{2}];'.format(eid, '|'.join(labels), attr), ind=2)

                no_child = True
                for i, name_arg in enumerate(name_args):
                    name, args = name_arg
                    if args is None:
                        continue

                    def get_arg(arg):
                        if id(arg) not in traversed:
                            arg_id = next(nid)
                            traversed[id(arg)] = arg_id
                        return 'EXPR{0} -> EXPR{1}:f{2};'.format(traversed[id(arg)], eid, i)
                    if isinstance(args, Iterable):
                        for arg in args:
                            write_indent_newline(get_arg(arg), ind=2)
                    else:
                        write_indent_newline(get_arg(args), ind=2)
                    no_child = False

                if no_child:
                    if len(pres) == 0:
                        if isinstance(expr, CollectionExpr):
                            write_indent_newline('START -> EXPR{0};'.format(eid), ind=2)
                    else:
                        for pre in pres:
                            pre_expr = pre.expr
                            pid = traversed[id(pre_expr)]
                            if (isinstance(pre_expr, Scalar) and isinstance(expr, Scalar)) or \
                                    (isinstance(pre_expr, CollectionExpr) and isinstance(expr, CollectionExpr)):
                                write_indent_newline('EXPR{0} -> EXPR{1};'.format(pid, eid), ind=2)

            if compiled:
                eid = traversed[id(expr_node)]
                compiled = '<TABLE ALIGN="LEFT" BORDER="0">%s</TABLE>' % ''.join(
                    '<TR><TD ALIGN="LEFT">%s</TD></TR>' % cgi.escape(l) for l in compiled.split('\n'))

                write_indent_newline(
                    'COMPILED{0} [shape=record, style="filled", fillcolor="SkyBlue", label=<\n'
                        .format(eid), ind=2)
                write_indent_newline(compiled, ind=3)
                write_indent_newline('>];', ind=2)
                write_indent_newline(
                    'EXPR{0} -> COMPILED{0} [arrowhead = none, style = dashed];'.format(eid), ind=2)

            write_indent_newline('}')

        write('}')

        return buffer.getvalue()
