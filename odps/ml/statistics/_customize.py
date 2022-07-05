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

import json
from collections import namedtuple
from functools import partial

from ..expr.exporters import get_ml_input
from ..utils import parse_hist_repr
from ... import utils
from ...compat import StringIO, six
from ...df import DataFrame

try:
    import numpy as np
except ImportError:
    np = None

from ..enums import FieldRole
from odps.ml.expr.exporters import get_input_table_name, get_input_partitions, get_input_field_names


get_predicted_value_column = partial(get_input_field_names, field_role=FieldRole.PREDICTED_VALUE)


def get_y_table_name(expr, input_name):
    if not expr._params['yColName']:
        return None
    if not get_ml_input(expr, input_name):
        return get_input_table_name(expr, 'x')
    return get_input_table_name(expr, input_name)


def get_y_partitions(expr, input_name):
    if not expr._params['yColName']:
        return None
    if not get_ml_input(expr, input_name):
        return get_input_partitions(expr, 'x')
    return get_input_partitions(expr, input_name)


def get_histograms(expr, odps):
    hist_dict = dict()
    for rec in DataFrame(odps.get_table(expr.tables[0])).execute():
        col, bins = rec
        hist_dict[col] = parse_hist_repr(bins)
    return hist_dict


def get_pearson_result(expr, odps):
    return DataFrame(odps.get_table(expr.tables[0])).execute()[0][-1]


class TTestResult(object):
    def __init__(self, result):
        convs = dict(level='ConfidenceLevel', alpha='alpha', df='df', mean='mean', alternative='alternative', mu='mu',
                     p='p', t='t', stddev='stdDeviation', diff_mean='mean of the differences', x_mean='mean of x',
                     y_mean='mean of y', alter_hyp='AlternativeHypthesis')
        convs_comment = dict(x_mean='xMean', y_mean='yMean', x_dev='xDeviation', y_dev='yDeviation', cov='cov')

        self._items = set()

        for k, v in six.iteritems(convs):
            if v in result:
                setattr(self, k, result[v])
                self._items.add(k)

        if result.get('comment'):
            comm_values = json.loads('{' + result.get('comment') + '}')
            for k, v in six.iteritems(convs_comment):
                if v in comm_values:
                    setattr(self, k, comm_values[v])
                    self._items.add(k)

        self._items = sorted(self._items)
        convs.update(convs_comment)
        self._convs = convs

    def __repr__(self):
        buf = StringIO()
        space = 2 * max(len(it) for it in self._items)
        for name in self._items:
            buf.write('\n{0}{1}'.format(name.ljust(space), repr(getattr(self, name))))
        return 'TTestResult {{{0}\n}}'.format(utils.indent(buf.getvalue(), 2))

    def _repr_html_(self):
        buf = StringIO()
        buf.write('<table><tr><th>Field</th><th>Name</th><th>Value</th></tr>')
        for name in self._items:
            buf.write('<tr><td>{0}</td><td>{1}</td><td>{2}</td></tr>'.format(name, self._convs[name], getattr(self, name)))
        buf.write('</table>')
        return buf.getvalue()


def get_t_test_result(expr, odps):
    return TTestResult(json.loads(DataFrame(odps.get_table(expr.tables[0])).execute()[0][0]))


class ChiSquareTestResult(object):
    def __init__(self, result, correct_result, details):
        self.result = result
        self.correct_result = correct_result
        self.detail = details


def get_chisq_test_result(expr, odps):
    ChiSquareData = namedtuple('ChiSquareData', 'comment df p_value value')
    json_result = json.loads(DataFrame(odps.get_table(expr.tables.summary)).execute()[0][0].replace('p-value', 'p_value'))
    result = ChiSquareData(**json_result.get('Chi-Square'))
    correct_result = ChiSquareData(**json_result.get('Continity Correct Chi-Square'))

    fields = 'f0 f1 observed expected residuals'.split()
    details = []
    for rec in DataFrame(odps.get_table(expr.tables.detail)).execute():
        details.append(dict(zip(fields, rec.values)))

    return ChiSquareTestResult(result, correct_result, details)


def get_mat_result(expr, odps):
    MatResult = namedtuple('MatResult', 'matrix cols')
    cols = []
    mat = []
    for rec in DataFrame(odps.get_table(expr.tables[0])).execute():
        if not cols:
            cols = list(rec.keys()[1:])
        mat.append(rec.values[1:])
    if np:
        mat = np.matrix(mat)
    return MatResult(mat, cols)
