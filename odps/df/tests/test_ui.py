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

import decimal
import json
import random
from collections import namedtuple

import pytest

from .. import DataFrame
from ..backends.tests.core import pandas_case, NumGenerators
from ..types import validate_data_type
from ..expr.expressions import *
from ..backends.odpssql.types import df_schema_to_odps_schema
from ..ui import DFViewMixin, MAX_TABLE_FETCH_SIZE, _rv


@pytest.fixture
@pandas_case
def setup(odps):
    def gen_data(rows=None, data=None, nullable_field=None, value_range=None):
        if data is None:
            data = []
            for _ in range(rows):
                record = []
                for col in schema:
                    method = getattr(NumGenerators, 'gen_random_%s' % col.type.name)
                    if col.name == 'category':
                        record.append(u'鬼'.encode('utf-8') if random.random() > 0.5 else u'人')
                    elif col.type.name == 'bigint':
                        record.append(method(value_range=value_range))
                    else:
                        record.append(method())
                data.append(record)

            if nullable_field is not None:
                j = schema._name_indexes[nullable_field.lower()]
                for i, l in enumerate(data):
                    if i % 2 == 0:
                        data[i][j] = None
        return data

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    pd_schema = TableSchema.from_lists(
        ['name', 'category', 'id', 'fid', 'isMale', 'scale', 'birth'],
        datatypes('string', 'string', 'int64', 'float64', 'boolean', 'decimal', 'datetime')
    )
    schema = df_schema_to_odps_schema(pd_schema)

    import pandas as pd
    data = gen_data(20, value_range=(-1000, 1000))
    df = pd.DataFrame(data, columns=pd_schema.names)
    expr = DataFrame(df, schema=pd_schema)

    nt = namedtuple("NT", "data df expr schema gen_data")
    return nt(data, df, expr, schema, gen_data)


@pytest.mark.skipif(DFViewMixin is None, reason="Need jupyter to run the test")
def test_fetch_table(odps, setup):
    df_widget = DFViewMixin()
    df_widget.df = setup.expr

    df_widget._handle_fetch_table({}, None)
    rendered_data = [[_rv(v) for v in r] for r in setup.data[:MAX_TABLE_FETCH_SIZE]]
    assert rendered_data == df_widget.table_records['data']

    df_widget._handle_fetch_table({'page': 1}, None)
    rendered_data = [[_rv(v) for v in r] for r in setup.data[MAX_TABLE_FETCH_SIZE:MAX_TABLE_FETCH_SIZE * 2]]
    assert json.dumps(rendered_data) == json.dumps(df_widget.table_records['data'])


@pytest.mark.skipif(DFViewMixin is None, reason="Need jupyter to run the test")
def test_aggregate_graph(odps, setup):
    df_widget = DFViewMixin()
    df_widget.df = setup.expr

    df_widget._handle_aggregate_graph(dict(groups=['isMale'], keys=['category'],
                                           values={'scale': ['sum']}, target='test_case_target'), None)

    sum_v = dict()
    cats, genders = set(), set()
    for r in setup.data:
        k = (_rv(r[4]), _rv(r[1]))
        if k not in sum_v:
            sum_v[k] = decimal.Decimal(0)
        sum_v[k] += r[5]
        cats.add(_rv(r[1]))
        genders.add(_rv(r[4]))

    agg_result = getattr(df_widget, 'test_case_target')
    assert set(v[0] for v in agg_result['keys']) == cats
    assert set(v[0] for v in agg_result['groups']) == genders

    for g, dr in zip(agg_result['groups'], agg_result['data']):
        for cat, s in zip(dr['category'], dr['scale__sum']):
            assert _rv(sum_v[(g[0], cat)]) == s
