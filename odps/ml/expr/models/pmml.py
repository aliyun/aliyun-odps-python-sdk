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

import re
import sys
import logging
from collections import namedtuple

from ....compat import six, ElementTree as ET, reduce
from ....utils import require_package

PMML_VOLUME = 'pyodps_volume'
logger = logging.getLogger(__name__)


class PmmlRepr(object):
    def _repr(self):
        return None

    def __repr__(self):
        r = self._repr()
        if r is None:
            return ''
        elif not six.PY2:
            return r
        else:
            return r.encode('utf-8') if isinstance(r, six.text_type) else r


class PmmlPredictor(PmmlRepr):
    def __init__(self, element):
        self._element = element

    @property
    def coefficient(self):
        return self._element.attrib.get('coefficient')

    @property
    def name(self):
        return self._element.attrib.get('name')


class PmmlNumericPredictor(PmmlPredictor):
    @property
    def exponent(self):
        exp = self._element.attrib.get('exponent')
        return None if exp is None or float(exp) == 1.0 else exp

    def _repr(self):
        if float(self.coefficient) == 0.0:
            return None
        if self.exponent:
            return '%s * %s ** %s' % (self.coefficient, self.name, self.exponent)
        else:
            return '%s * %s' % (self.coefficient, self.name)

    def _repr_html_(self):
        if float(self.coefficient) == 0.0:
            return None
        if self.exponent:
            return '%s * %s<sup>%s</sup>' % (self.coefficient, self.name, self.exponent)
        else:
            return '%s * %s' % (self.coefficient, self.name)


class PmmlCategoricalPredictor(PmmlPredictor):
    @property
    def value(self):
        return self._element.attrib.get('value')

    def _repr(self):
        return '%s * I(%s = %s)' % (self.coefficient, self.name, self.value)

    def _repr_html_(self):
        return self._repr()


class PmmlRegressionTable(PmmlRepr):
    def __init__(self, element, target_field='y'):
        self._element = element
        self._target_field = target_field

    @property
    def intercept(self):
        intercept = self._element.attrib.get('intercept')
        return intercept if float(intercept) != 0.0 else None

    @property
    def target_category(self):
        return self._element.attrib.get('targetCategory')

    def _numeric_predictors(self):
        return [PmmlNumericPredictor(el) for el in self._element.findall('NumericPredictor')]

    def _categorical_predictors(self):
        return [PmmlCategoricalPredictor(el) for el in self._element.findall('CategoricalPredictor')]

    def predictors(self):
        return self._numeric_predictors() + self._categorical_predictors()

    def _build_expr(self, rf=repr):
        expr_parts = []

        if self.intercept:
            expr_parts.append(self.intercept)

        predictors = self.predictors()
        preds = [r for r in (rf(pred) for pred in predictors) if r]
        expr_parts.extend([pred for pred in preds if pred])
        if len(predictors) == 0:
            return ''

        value_part = re.sub(' *\+ *- *', ' - ', ' + '.join(expr_parts))
        return self._target_field + ' = ' + value_part

    def _repr(self):
        expr = self._build_expr()
        if self.target_category and expr:
            return 'Target: %s\n    ' % self.target_category + expr
        else:
            return expr

    def _repr_html_(self):
        expr = self._build_expr(rf=lambda v: v._repr_html_())
        if not expr:
            return ''
        elif self.target_category:
            return '<div style="font-weight: bold">Target: %s</div><div style="text-indent: 30px">%s</div>' % (self.target_category, expr)
        else:
            return '<div>%s</div>' % expr


class PmmlExpr(PmmlRepr):
    def __init__(self, s, func):
        self._str = s
        self._func = func

    def __str__(self):
        return repr(self)

    def _repr(self):
        return self._str

    def __call__(self, *args):
        return self._func(*args)


EXPR_DICT = {
    'equal': PmmlExpr(u'=', lambda a, b: a == b),
    'notEqual': PmmlExpr(u'≠', lambda a, b: a != b),
    'lessThan': PmmlExpr(u'<', lambda a, b: a < b),
    'lessOrEqual': PmmlExpr(u'≤', lambda a, b: a <= b),
    'greaterThan': PmmlExpr(u'>', lambda a, b: a > b),
    'greaterOrEqual': PmmlExpr(u'≥', lambda a, b: a >= b),
    'isIn': PmmlExpr('in', lambda a, b: a in b),
    'isNotIn': PmmlExpr('not in', lambda a, b: a not in b),
}
SCORE_SIZE = 16
DETAIL_SIZE = 11


def escape_graphviz(src):
    target = src
    for cfr, cto in GV_ESCAPES:
        target = target.replace(cfr, cto)
    return target

TEXT_TREE_CORNER = '└'
TEXT_TREE_FILL = '   '
TEXT_TREE_FORK = '├'
TEXT_TREE_HLINE = '─'
TEXT_TREE_VLINE = '│'


def build_text_tree(root):

    def write_node_step(writer, node, header, is_last):
        front_line = (TEXT_TREE_CORNER if is_last else TEXT_TREE_FORK) + TEXT_TREE_HLINE * 2 + ' '
        writer.write(header + front_line + node.text + '\n')

        sub_nodes = list(node.children())
        if sub_nodes:
            if is_last:
                append_header = TEXT_TREE_FILL * 2 + ' '
            else:
                append_header = TEXT_TREE_VLINE + TEXT_TREE_FILL + ' '
            for node in sub_nodes[:-1]:
                write_node_step(writer, node, header + append_header, False)
            write_node_step(writer, sub_nodes[-1], header + append_header, True)

    sio = six.StringIO()
    sio.write(root.text + '\n')
    root_children = list(root.children())
    for snode in root_children[:-1]:
        write_node_step(sio, snode, '', False)
    write_node_step(sio, root_children[-1], '', True)
    return sio.getvalue()

GV_ESCAPES = [
    ('<', '&lt;'),
    ('>', '&gt;'),
]


def build_gv_tree(root):
    counter = [0, ]

    def write_gv_step(writer, node, parent_key):
        writer.write(u'{0} {1};\n'.format(parent_key, node.gv_text))

        sub_nodes = list(node.children())
        if sub_nodes:
            for node in sub_nodes:
                counter[0] += 1
                struct_str = u'struct%d' % counter[0]
                write_gv_step(writer, node, struct_str)
                writer.write(u'%s -> %s;\n' % (parent_key,  struct_str))

    sio = six.StringIO()
    sio.write(u'digraph {{\nroot {0};\n'.format(root.gv_text))
    root_children = list(root.children())
    for snode in root_children:
        counter[0] += 1
        struct_str = u'struct%d' % counter[0]
        write_gv_step(sio, snode, struct_str)
        sio.write('root -> %s;\n' % struct_str)
    sio.write('}\n')
    return six.text_type(sio.getvalue())


if six.PY2:
    def unescape_string(s):
        return s.decode('string-escape')
else:
    def unescape_string(s):
        return s.encode('utf-8').decode('unicode-escape')


def parse_pmml_array(element):
    if element is None:
        return None
    parts = []
    sio = six.StringIO()
    quoted = False
    last_ch = None
    for ch in element.text:
        if ch == '\"':
            if last_ch == '\\':
                sio.write('\"')
            else:
                quoted = not quoted
        elif not quoted and (ch == ' ' or ch == '\t'):
            s = sio.getvalue()
            if s:
                parts.append(unescape_string(s))
                sio = six.StringIO()
        else:
            sio.write(ch)
        last_ch = ch
    s = sio.getvalue()
    if s:
        parts.append(unescape_string(s))
    if 'n' in element.attrib:
        assert int(element.attrib['n']) == len(parts)
    array_type = element.attrib['type'].lower()
    if array_type == 'int':
        wrapper = int
    elif array_type == 'real':
        wrapper = float
    else:
        wrapper = lambda v: v
    return [wrapper(p) for p in parts]


def pmml_predicate(name):
    def _decorator(cls):
        PmmlPredicate._subclasses[name] = cls
        return cls

    return _decorator


class PmmlPredicate(PmmlRepr):
    _subclasses = dict()

    def __new__(cls, *args, **kwargs):
        if 'element' in kwargs:
            element = kwargs['element']
        else:
            element = args[0]
        if element.tag in cls._subclasses:
            return object.__new__(cls._subclasses[element.tag])

    def __init__(self, element):
        self._element = element

    @classmethod
    def exists(cls, element):
        return any(element.find(s) is not None for s in six.iterkeys(cls._subclasses))

    @classmethod
    def iterate(cls, element):
        return (cls(el) for el in reduce(lambda a, b: a + b, (element.findall('./' + c) for c in six.iterkeys(cls._subclasses)), []))

    @property
    def field(self):
        return self._element.attrib.get('field')


@pmml_predicate('SimplePredicate')
class PmmlSimplePredicate(PmmlPredicate):
    @property
    def operator(self):
        return self._element.attrib.get('operator')

    @property
    def value(self):
        return self._element.attrib.get('value')

    def _repr(self):
        return '`%s` %s %s' % (self.field, EXPR_DICT.get(self.operator, self.operator), self.value)


@pmml_predicate('SimpleSetPredicate')
class PmmlSimpleSetPredicate(PmmlPredicate):
    def __init__(self, element):
        super(PmmlSimpleSetPredicate, self).__init__(element)

    @property
    def operator(self):
        return self._element.attrib.get('booleanOperator')

    @property
    def array(self):
        return parse_pmml_array(self._element.find('Array'))

    def _repr(self):
        return '`%s` %s (%s)' % (self.field, EXPR_DICT.get(self.operator, self.operator),
                                 ', '.join(repr(v) for v in self.array))


@pmml_predicate('CompoundPredicate')
class PmmlCompoundPredicate(PmmlPredicate):
    @property
    def operator(self):
        return self._element.attrib.get('booleanOperator')

    def predicates(self):
        return PmmlPredicate.iterate(self._element)

    def _repr(self):
        pds = []
        for sub_pd in self.predicates():
            pd = repr(sub_pd)
            if isinstance(sub_pd, PmmlCompoundPredicate):
                pd = '({0})'.format(pd)
            pds.append(pd)
        return ' {0} '.format(self.operator).join(pds)


PmmlSegmentSummary = namedtuple('PmmlSegmentSummary', 'type id weight')


class PmmlSegment(PmmlRepr):
    def __init__(self, element):
        if element.tag == 'Segment':
            self._segment_element = element
        else:
            self._segment_element = None

    @property
    def segment_id(self):
        return self._segment_element.attrib.get('id') if self._segment_element is not None else None

    @property
    def segment_weight(self):
        return self._segment_element.attrib.get('weight') if self._segment_element is not None else None

    @property
    def segment_summary(self):
        return PmmlSegmentSummary(type=self.__class__.__name__, id=self.segment_id, weight=self.segment_weight)


class PmmlTreeNode(PmmlRepr):
    def __init__(self, element, text=None):
        self._element = element
        self._text = text

    @property
    def score(self):
        return self._element.attrib.get('score')

    @property
    def predicate(self):
        try:
            return next(PmmlPredicate.iterate(self._element))
        except StopIteration:
            return None

    @property
    def text(self):
        if self._text:
            return self._text
        content = ''
        if 'score' in self._element.attrib:
            content += 'SCORE = ' + self._element.attrib['score'] + ' '
        if PmmlPredicate.exists(self._element):
            dists = ['%s:%s' % (e.attrib['value'], e.attrib['recordCount'])
                     for e in self._element.findall('./ScoreDistribution')]
            pstr = repr(self.predicate)
            expr = 'WHEN %s' % pstr
            if dists:
                expr += ' (COUNTS: %s)' % ', '.join(dists)
            content += expr
        return content

    @property
    def gv_text(self):
        if self._text:
            return u'[shape=record,label=<\n    {0}\n>]'.format(self._text)
        label_lines = []
        extra_style = ''
        if 'score' in self._element.attrib:
            label_lines.append(u'<FONT POINT-SIZE="{0}">{1}</FONT>'.format(SCORE_SIZE, self._element.attrib['score']))
            extra_style = u'style=filled,fillcolor=azure2,'
        if PmmlPredicate.exists(self._element):
            dists = [u'%s:%s' % (e.attrib['value'], e.attrib['recordCount'])
                     for e in self._element.findall('./ScoreDistribution')]
            expr = repr(self.predicate)
            if six.PY2:
                expr = expr.decode('utf-8')
            label_lines.append(u'<FONT POINT-SIZE="{0}">{1}</FONT>'.format(DETAIL_SIZE, escape_graphviz(expr)))
            if dists:
                label_lines.append(u'<FONT POINT-SIZE="{0}">LABELS: {1}</FONT>'.format(DETAIL_SIZE, ', '.join(dists)))
        label = u'<br />'.join(label_lines)
        return u'[shape=record,{0}label=<\n    {1}\n>]'.format(extra_style, label)

    def children(self):
        return [PmmlTreeNode(el) for el in self._element.findall('./Node')]

    def _repr(self):
        return build_text_tree(self)

    def _repr_gv_(self):
        return build_gv_tree(self)

    @require_package('graphviz')
    def _repr_svg_(self):
        from graphviz import Source
        return Source(self._repr_gv_(), encoding='utf-8')._repr_svg_()


class PmmlTree(PmmlSegment):
    def __init__(self, element):
        super(PmmlTree, self).__init__(element)
        if element.tag == 'Segment':
            self._element = element.find('TreeModel')
        else:
            self._element = element

    @property
    def root(self):
        return PmmlTreeNode(self._element.find('./Node'), 'ROOT')

    def _repr(self):
        return build_text_tree(self.root)

    def _repr_gv_(self):
        return build_gv_tree(self.root)

    @require_package('graphviz')
    def _repr_svg_(self):
        from graphviz import Source
        return Source(self._repr_gv_(), encoding='utf-8')._repr_svg_()


class PmmlResult(object):
    _result_types = []

    def __new__(cls, pmml):
        et = ET.fromstring(re.sub(' xmlns="[^"]+"', '', pmml, count=1))
        obj = None
        if cls is not PmmlResult:
            obj = object.__new__(cls)
        else:
            if not cls._result_types:
                for c in six.itervalues(globals()):
                    if c is not PmmlResult and isinstance(c, type) and issubclass(c, PmmlResult):
                        cls._result_types.append(c)

            for c in cls._result_types:
                if c.adaptable(et):
                    obj = object.__new__(c)
                    break
        if obj is None:
            obj = object.__new__(cls)
        obj.pmml = pmml
        obj._pmml_element = et
        return obj

    @classmethod
    def adaptable(cls, et):
        raise NotImplementedError


class PmmlRegressionResult(PmmlResult, PmmlRepr):
    def __init__(self, *_, **__):
        if self._pmml_element.find('RegressionModel') is not None:
            self._reg_element = self._pmml_element.find('RegressionModel')
        elif self._pmml_element.find('MiningModel/Regression') is not None:
            self._reg_element = self._pmml_element.find('MiningModel/Regression')

    @classmethod
    def adaptable(cls, et):
        if et.find('RegressionModel') is not None:
            return True
        elif et.find('MiningModel/Regression') is not None:
            return True
        return False

    @property
    def target_field(self):
        v = self._reg_element.attrib.get('targetFieldName')
        return v if v else 'y'

    @property
    def normalization(self):
        v = self._reg_element.attrib.get('normalizationMethod')
        return v if v != 'none' else None

    @property
    def function(self):
        return self._reg_element.attrib.get('functionName')

    def __iter__(self):
        return (PmmlRegressionTable(e) for e in self._reg_element.findall('RegressionTable'))

    def _repr(self):
        sio = six.StringIO()
        sio.write('Function: %s\n' % self.function)
        sio.write('Target Field: %s\n' % self.target_field)
        if self.normalization:
            sio.write('Normalization: %s\n' % self.normalization)
        reprs = (repr(v) for v in self)
        sio.write('\n'.join(v for v in reprs if v))
        return sio.getvalue()

    def _repr_html_(self):
        sio = six.StringIO()
        sio.write('<div><span style="font-weight: bold">Function</span>: %s</div>\n' % self.function)
        sio.write('<div><span style="font-weight: bold">Target Field</span>: %s</div>\n' % self.target_field)
        if self.normalization:
            sio.write('<div><span style="font-weight: bold">Normalization</span>: %s</div>\n' % self.normalization)
        reprs = (v._repr_html_() for v in self)
        sio.write('\n'.join(v for v in reprs if v))
        return sio.getvalue()


class PmmlSegmentsResult(PmmlResult):
    def __init__(self, *_, **__):
        self._seg_element = self._pmml_element.find('MiningModel/Segmentation')

    @classmethod
    def adaptable(cls, et):
        return et.find('MiningModel/Segmentation') is not None

    @staticmethod
    def _segment_to_object(xsegment):
        if xsegment.find('./TreeModel') is not None:
            return PmmlTree(xsegment)
        else:
            raise ValueError('Unrecognized PMML node.')

    def __getitem__(self, item):
        if sys.version_info[:2] < (2, 7):
            xsegment = None
            for seg in self._seg_element.findall('Segment'):
                if seg.attrib.get('id') == str(item):
                    xsegment = seg
                    break
        else:
            xsegment = self._seg_element.find('Segment[@id="{0}"]'.format(item))

        if xsegment is None:
            raise KeyError('No segments found in PMML result.')
        return self._segment_to_object(xsegment)

    def __iter__(self):
        return (self._segment_to_object(xseg) for xseg in self._seg_element.findall('./Segment'))

    def _get_segments_summary(self):
        return [seg.segment_summary for seg in self if seg.segment_id is not None]

    def _repr(self):
        return repr(self._get_segments_summary())

    def _repr_html_(self):
        html_writer = six.StringIO()
        html_writer.write('<table><tr><th>ID</th><th>Type</th><th>Weight</th></tr>')
        for seg in self._get_segments_summary():
            html_writer.write('<tr><td>{0}</td><td>{1}</td><td>{2}</td></tr>'.format(seg.id, seg.type, seg.weight))
        html_writer.write('</table>')
        return html_writer.getvalue()
