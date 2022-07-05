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

from collections import namedtuple

from ...df import DataFrame
from ...compat import six
from ...utils import underline_to_capitalized


class ColImportanceResult(dict):
    def __init__(self, stats_type, **kwargs):
        super(ColImportanceResult, self).__init__(**kwargs)
        self._titles = [(fn, underline_to_capitalized(fn)) for fn in stats_type._fields]

    def _repr_html_(self):
        sio = six.StringIO()
        sio.write('<table><tr><th>Column</th>')
        for _, ft in self._titles:
            sio.write('<th>{0}</th>'.format(ft))
        sio.write('</tr>')
        for col, stats in six.iteritems(self):
            sio.write('<tr><td>{0}</td>'.format(col))
            for fn, _ in self._titles:
                sio.write('<td>{0}</td>'.format(getattr(stats, fn)))
            sio.write('</tr>')
        sio.write('</table>')
        return sio.getvalue()


def get_rf_importance(expr, odps):
    stats_type = namedtuple('RFFeature', 'gini entropy')
    mapping = dict((row[0], stats_type(gini=row[1], entropy=row[2]))
                   for row in DataFrame(odps.get_table(expr.tables[0])).execute())
    return ColImportanceResult(stats_type, **mapping)


def get_gbdt_importance(expr, odps):
    stats_type = namedtuple('GBDTFeature', 'importance')
    mapping = dict((row[0], stats_type(importance=row[1]))
                   for row in DataFrame(odps.get_table(expr.tables[0])).execute())
    return ColImportanceResult(stats_type, **mapping)


def get_regression_importance(expr, odps):
    stats_type = namedtuple('RegressionFeature', 'weight importance')
    mapping = dict((row[0], stats_type(weight=row[1], importance=row[2]))
                   for row in DataFrame(odps.get_table(expr.tables[0])).execute())
    return ColImportanceResult(stats_type, **mapping)
