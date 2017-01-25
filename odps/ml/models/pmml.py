# encoding: utf-8
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

import hashlib
import re
import logging
from collections import namedtuple

from ... import options, tempobj, utils
from ...compat import six, ElementTree as ET, reduce
from ...models.ml import OfflineModel
from ...utils import require_package
from ...df.core import DataFrame, CollectionExpr
from ...df.expr.expressions import SequenceExpr
from ...runner import RunnerContext, ObjectDescription, DFAdapter, adapter_from_df
from .base import MLModel
from ..adapter import StaticFieldChangeOperation
from ..algorithms.nodes import PmmlPredictNode
from ..nodes.io_nodes import PmmlModelInputNode, PmmlModelOutputNode
from ..enums import FieldContinuity, FieldRole
from ..utils import MLField, gen_model_name

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


class PmmlRegression(PmmlRepr):
    def __init__(self, element):
        self._element = element

    @property
    def target_field(self):
        v = self._element.attrib.get('targetFieldName')
        return v if v else 'y'

    @property
    def normalization(self):
        v = self._element.attrib.get('normalizationMethod')
        return v if v != 'none' else None

    @property
    def function(self):
        return self._element.attrib.get('functionName')

    def __iter__(self):
        return (PmmlRegressionTable(e) for e in self._element.findall('RegressionTable'))

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


class PmmlSegments(PmmlRepr):
    def __init__(self, element):
        self._element = element

    @staticmethod
    def _segment_to_object(xsegment):
        if xsegment.find('./TreeModel') is not None:
            return PmmlTree(xsegment)
        else:
            raise ValueError('Unrecognized PMML node.')

    def __getitem__(self, item):
        xsegment = self._element.find('Segment[@id="{0}"]'.format(item))
        if xsegment is None:
            raise KeyError
        return self._segment_to_object(xsegment)

    def __iter__(self):
        return (self._segment_to_object(xseg) for xseg in self._element.findall('./Segment'))

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


class PmmlModel(MLModel):
    """
    PMML model reference for PyODPS ML.

    ``predict`` method can be used to predict a data set using the referenced model.
    ``persist`` method can be used to store models with given names.

    :param model_obj: OfflineModel object, can be created with odps.get_offline_model
    :type model_obj: OfflineModel
    """
    def __init__(self, model_obj, **kw):
        if isinstance(model_obj, OfflineModel):
            project_name = None if model_obj.project.name == model_obj.odps.project else model_obj.project.name
            source_node = PmmlModelInputNode(model_obj.name, project=project_name)

            super(PmmlModel, self).__init__(model_obj.odps, port=source_node.outputs.get('model'))
        else:
            super(PmmlModel, self).__init__(model_obj, kw.get('port'))

        self._model_name = None
        self._pmml = None
        self._pmml_element = None

    def gen_temp_names(self):
        if not self._model_name:
            self._model_name = gen_model_name(self._bind_node.code_name, node_id=self._bind_node.node_id)
            return ObjectDescription(offline_models=self._model_name)
        else:
            return None

    def describe(self):
        return ObjectDescription(offline_models=self._model_name)

    def fill(self, desc):
        if desc.offline_models:
            self._model_name = desc.offline_models[0]

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    def persist(self, model_name, project=None, **kwargs):
        """
        Store offline model into ODPS

        Note that this method will trigger the defined flow to execute.

        :param model_name: name of the offline model.
        :param project: name of project, if you want to store into a new one.
        :param execute: whether to execute the whole flow now. True for default.
        """
        if self.model_name:
            raise RuntimeError('Model already persisted.')

        # if _output_node == true, return node for testing purpose
        _output_node = kwargs.pop('_output_node', False)
        delay = kwargs.pop('delay', False)

        model_node = PmmlModelOutputNode(model_name, project=project)
        self._link_node(model_node, 'model')
        if kwargs.get('execute', True):
            RunnerContext.instance()._run(self._bind_node)
        return None if not _output_node else model_node

    def load_pmml(self):
        """
        Load PMML of the model

        Note that
        1. As ODPS restricts download size, downloading too large models will raise exceptions.
        2. This method will trigger the defined flow to execute.

        :return: PMML string
        :rtype: str
        """
        if not self._pmml:
            RunnerContext.instance()._run(self._bind_node)

            model = self._odps.get_offline_model(self._model_name)
            self._pmml = model.get_model()
            if not self._pmml and options.ml.auto_transfer_pmml:
                if not self._odps.exist_volume(PMML_VOLUME):
                    self._odps.create_parted_volume(PMML_VOLUME)
                vol_part = hashlib.md5(utils.to_binary(self._model_name)).hexdigest()
                tempobj.register_temp_volume_partition(self._odps, (PMML_VOLUME, vol_part))
                try:
                    self._odps.execute_xflow('modeltransfer', options.ml.xflow_project, {
                        'modelName': self._model_name,
                        'volumeName': PMML_VOLUME,
                        'partition': vol_part,
                        'format': 'pmml'
                    })
                    part_path = '/{0}/{1}'.format(PMML_VOLUME, vol_part)
                    self._pmml = self._odps.open_volume_reader(part_path).read()
                    self._odps.delete_volume_partition(part_path)
                except:
                    return None
            self._pmml_element = ET.fromstring(re.sub(' xmlns="[^"]+"', '', self._pmml, count=1))
        return self._pmml

    @property
    def segments(self):
        """
        Get random forests model representation. This can be slow as models are required
        to be uploaded to servers.

        @rtype: PmmlSegments
        """
        self.load_pmml()
        if self._pmml_element.find('MiningModel/Segmentation') is not None:
            return PmmlSegments(self._pmml_element.find('MiningModel/Segmentation'))
        raise ValueError('`Segmentation` element not found in PMML.')

    @property
    def regression(self):
        """
        Get regression model representation. This can be slow as models are required
        to be uploaded to servers.

        @rtype: PmmlRegression
        """
        self.load_pmml()
        if self._pmml_element.find('RegressionModel') is not None:
            return PmmlRegression(self._pmml_element.find('RegressionModel'))
        elif self._pmml_element.find('MiningModel/Regression') is not None:
            return PmmlRegression(self._pmml_element.find('MiningModel/Regression'))
        raise ValueError('`RegressionModel` element not found in PMML.')

    def predict(self, df):
        """
        Predict given data set using the given model. Actual prediction steps will not
        be executed till an operational step is called.

        After execution, three columns will be appended to the table:

        +-------------------+--------+----------------------------------------------------+
        | Field name        | Type   | Comments                                           |
        +===================+========+====================================================+
        | prediction_result | string | field indicating the predicted label, absent if    |
        |                   |        | the model is a regression model                    |
        +-------------------+--------+----------------------------------------------------+
        | prediction_score  | double | field indicating the score value if the model is   |
        |                   |        | a classification model, or the predicted value if  |
        |                   |        | the model is a regression model.                   |
        +-------------------+--------+----------------------------------------------------+
        | prediction_detail | string | field in JSON format indicating the score for      |
        |                   |        | every class.                                       |
        +-------------------+--------+----------------------------------------------------+

        :type df: DataFrame
        :rtype: DataFrame

        :Example:

        >>> model = PmmlModel(odps.get_offline_model('model_name'))
        >>> data = DataFrame(odps.get_table('table_name'))
        >>> # prediction below will not be executed till predicted.persist is called
        >>> predicted = model.predict(data)
        >>> predicted.persist('predicted')
        """
        if not isinstance(df, (CollectionExpr, SequenceExpr)):
            raise TypeError("Cannot predict on objects other than a DataFrame object.")

        predict_node = PmmlPredictNode()

        self._link_node(predict_node, 'model')
        src_adapter = adapter_from_df(df)
        src_adapter._link_node(predict_node, 'input')

        output_adapter = DFAdapter(self._odps, predict_node.outputs.get("output"), None, uplink=[src_adapter, ],
                                   fields=src_adapter._fields)
        op = StaticFieldChangeOperation([
            MLField('prediction_result', 'string', FieldRole.PREDICTED_CLASS, continuity=FieldContinuity.DISCRETE,
                    is_append=True),
            MLField('prediction_score', 'double', set((FieldRole.PREDICTED_SCORE, FieldRole.PREDICTED_VALUE)),
                    continuity=FieldContinuity.CONTINUOUS, is_append=True),
            MLField('prediction_detail', 'string', FieldRole.PREDICTED_DETAIL, is_append=True),
        ], is_append=True)
        output_adapter.perform_operation(op)
        return output_adapter.df_from_fields(force_create=True)
