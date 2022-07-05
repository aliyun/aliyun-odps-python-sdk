# encoding: utf-8
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
import json
from collections import namedtuple

from ... import serializers, DataFrame
from ...compat import six, OrderedDict
from ..utils import camel_to_underline, parse_hist_repr
from ..algolib.loader import load_defined_algorithms

INF_PATTERN = re.compile(r'(: *)(inf|-inf)( *,| *\})')
load_defined_algorithms(sys.modules[__name__], 'metrics')


def replace_json_infs(s):
    def repl(m):
        mid = m.group(2)
        if mid == 'inf':
            mid = 'Infinity'
        elif mid == '-inf':
            mid = '-Infinity'
        return m.group(1) + mid + m.group(3)

    return INF_PATTERN.sub(repl, s)


def get_confusion_matrix_result(expr, odps):
    records = list(DataFrame(odps.get_table(expr.tables[0])).execute())
    # skip the first row
    col_data = json.loads(records[0][0])
    col_data = map(lambda p: p[1], sorted(six.iteritems(col_data), key=lambda a: int(a[0][1:])))

    mat = [list(rec[2:]) for rec in records[1:]]
    try:
        import numpy as np
        mat = np.matrix(mat)
    except ImportError:
        pass
    return mat, col_data

RocResult = namedtuple('RocResult', 'thresh tp fn tn fp')


def get_roc_result(expr, odps):
    mat = [list(rec) for rec in DataFrame(odps.get_table(expr.tables[0])).execute()]
    thresh, tp, fn, tn, fp = [[r[ridx] for r in mat] for ridx in range(0, 5)]
    try:
        import numpy as np
        thresh, tp, fn, tn, fp = [np.asarray(l) for l in (thresh, tp, fn, tn, fp)]
    except ImportError:
        pass

    return RocResult(thresh, tp, fn, tn, fp)


def get_regression_eval_result(expr, odps):
    records = list(DataFrame(odps.get_table(expr.tables.index)).execute())
    indices = dict((camel_to_underline(k), v) for k, v in six.iteritems(json.loads(replace_json_infs(records[0].values[0]))))
    records = list(DataFrame(odps.get_table(expr.tables.residual)).execute())
    indices['hist'] = parse_hist_repr(records[0].values[1])
    return indices


class MultiClassEvaluationResult(serializers.JSONSerializableModel):
    class LabelMeasure(serializers.JSONSerializableModel):
        accuracy = serializers.JSONNodeField('Accuracy')
        f1 = serializers.JSONNodeField('F1')
        false_discovery_rate = serializers.JSONNodeField('FalseDiscoveryRate')
        kappa = serializers.JSONNodeField('Kappa')
        npv = serializers.JSONNodeField('NegativePredictiveValue')
        precision = serializers.JSONNodeField('Precision')
        sensitivity = serializers.JSONNodeField('Sensitivity')
        specificity = serializers.JSONNodeField('Specificity')
        fn = serializers.JSONNodeField('FalseNegative')
        fnr = serializers.JSONNodeField('FalseNegativeRate')
        fp = serializers.JSONNodeField('FalsePositive')
        fpr = serializers.JSONNodeField('FalsePositiveRate')
        tn = serializers.JSONNodeField('TrueNegative')
        tp = serializers.JSONNodeField('TruePositive')

        @property
        def tpr(self):
            return self.sensitivity

        @tpr.setter
        def tpr(self, value):
            self.sensitivity = value

        @property
        def fpr(self):
            return self.specificity

        @fpr.setter
        def fpr(self, value):
            self.specificity = value

    class OverallMeasures(serializers.JSONSerializableModel):
        accuracy = serializers.JSONNodeField('Accuracy')
        kappa = serializers.JSONNodeField('Kappa')
        frequency_micro = serializers.JSONNodeReferenceField('MultiClassEvaluationResult.LabelMeasure',
                                                             'LabelFrequencyBasedMicro')
        macro_average = serializers.JSONNodeReferenceField('MultiClassEvaluationResult.LabelMeasure',
                                                           'MacroAveraged')
        micro_average = serializers.JSONNodeReferenceField('MultiClassEvaluationResult.LabelMeasure',
                                                           'MicroAveraged')

    actual_label_counts = serializers.JSONNodeField('ActualLabelFrequencyList')
    actual_label_ratios = serializers.JSONNodeField('ActualLabelProportionList')

    predicted_label_counts = serializers.JSONNodeField('PredictedLabelFrequencyList')
    predicted_label_ratios = serializers.JSONNodeField('PredictedLabelProportionList')

    confusion_matrix = serializers.JSONNodeField('ConfusionMatrix', type='ndarray')
    ratio_matrix = serializers.JSONNodeField('ProportionMatrix', type='ndarray')

    labels = serializers.JSONNodeField('LabelList')
    label_measures = serializers.JSONNodesReferencesField(LabelMeasure, 'LabelMeasureList')


def get_multi_class_eval_result(expr, odps):
    rec = DataFrame(odps.get_table(expr.tables[0])).execute()[0]
    return MultiClassEvaluationResult.parse(rec[0])


class BinaryClassEvaluationResult(object):
    def __init__(self):
        self.records = []
        self.positive = 0
        self.negative = 0

    @staticmethod
    def _to_numpy(v):
        try:
            import numpy as np
            return np.asarray(v)
        except ImportError:
            return v

    @property
    def thresh(self):
        return self._to_numpy([rec['data_range'][1] for rec in self.records])

    @property
    def fp(self):
        return self._to_numpy([self.negative * rec['fpr'] for rec in self.records])

    @property
    def tn(self):
        return self._to_numpy([self.negative * (1 - rec['fpr']) for rec in self.records])

    @property
    def tp(self):
        return self._to_numpy([self.positive * rec['recall'] for rec in self.records])

    @property
    def fn(self):
        return self._to_numpy([self.positive * (1 - rec['recall']) for rec in self.records])


def get_binary_class_eval_result(expr, odps):
    result = BinaryClassEvaluationResult()

    metric_recs = DataFrame(odps.get_table(expr.tables.metric)).execute()

    targets = {
        'Total Samples': 'size',
        'Positive Samples': 'positive',
        'Negative Samples': 'negative',
        'AUC': 'auc',
        'KS': 'ks',
        'F1 Score': 'f1',
    }
    for rec in metric_recs:
        if rec['name'] not in targets:
            continue
        setattr(result, targets[rec['name']], rec['value'])

    detail_recs = DataFrame(odps.get_table(expr.tables.detail)).execute()
    new_recs = []
    for rec in detail_recs:
        if rec['total'] == 0:
            continue
        new_rec = OrderedDict(rec.iteritems())
        new_rec['data_range'] = tuple(float(v) for v in rec['data_range'].strip('[()]').split(','))
        new_recs.append(new_rec)
    result.records = new_recs
    return result


def get_clustering_eval_result(expr, odps):
    return json.loads(DataFrame(odps.get_table(expr.tables[0])).execute()[0][0])
