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

from __future__ import print_function

import os
import uuid
import warnings
from collections import Iterable

from odps.compat import six
from odps.config import options
from odps.df import CollectionExpr
from odps.df.core import DataFrame
from odps.examples.tables import TestDataMixIn
from odps.ml.enums import PortType
from odps.ml.expr.core import AlgoExprMixin, AlgoCollectionExpr
from odps.ml.expr.models import ODPSModelExpr
from odps.ml.runners import BaseNodeRunner
from odps.ml.utils import TEMP_TABLE_PREFIX, TABLE_MODEL_PREFIX, TABLE_MODEL_SEPARATOR, ML_ARG_PREFIX
from odps.tests.core import tn, ci_skip_case, TestBase

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')


class BaseMockAlgoCollectionExpr(AlgoCollectionExpr):
    __slots__ = 'message', 'action'
    node_name = 'MockAlgoCollection'
    _algo = 'mock_algo'
    algo_meta = dict(engine='mock')

    def _init(self, *args, **kwargs):
        super(BaseMockAlgoCollectionExpr, self)._init(*args, **kwargs)

        self._init_attr('_params', dict())
        self._init_attr('_engine_kw', dict())


class BaseMockModelExpr(ODPSModelExpr):
    __slots__ = 'message', 'action'
    node_name = 'MockModel'
    _algo = 'mock_algo'
    algo_meta = dict(engine='mock')

    def _init(self, *args, **kwargs):
        super(BaseMockModelExpr, self)._init(*args, **kwargs)

        self._init_attr('_params', dict())
        self._init_attr('_engine_kw', dict())


def build_mock_expr_class(input_types, output_id, output_type):
    class_exprs = dict(
        _args=[ML_ARG_PREFIX + 'input%d' % idx for idx in range(len(input_types))],
        _output_name='output%d' % output_id
    )
    if output_type == PortType.DATA:
        return type(BaseMockAlgoCollectionExpr)('MockAlgoCollectionExpr', (BaseMockAlgoCollectionExpr, ), class_exprs)
    else:
        return type(BaseMockModelExpr)('MockModelExpr', (BaseMockModelExpr, ), class_exprs)


def tmp_otm(s):
    return TEMP_TABLE_PREFIX + TABLE_MODEL_PREFIX + s


class MLTestBase(TestDataMixIn, TestBase):
    def setUp(self):
        super(MLTestBase, self).setUp()

        # Force to false
        options.ml.dry_run = False
        options.lifecycle = 3
        options.verbose = 'CI_MODE' not in os.environ
        options.interactive = False
        # Disable warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'.*widget\.py.*')

    def mock_action(self, sources, output_desc=1, msg='', action=None):
        exec_id = uuid.uuid4()

        if not isinstance(sources, Iterable):
            sources = [sources, ]

        input_types = [PortType.DATA if isinstance(o, CollectionExpr) else PortType.MODEL for o in sources]

        if isinstance(output_desc, six.integer_types):
            output_types = [PortType.DATA for _ in range(output_desc)]
        else:
            output_types = [PortType.DATA if ch == 'd' else PortType.MODEL for ch in output_desc]

        outputs = []
        for idx, ot in enumerate(output_types):
            expr_cls = build_mock_expr_class(input_types, idx, ot)
            init_args = dict(register_expr=True, _exec_id=exec_id, _params=dict(message=msg), action=action)
            if ot == PortType.DATA:
                init_args['_schema'] = six.next(s for s in sources if isinstance(s, CollectionExpr)).schema
            outputs.append(expr_cls(**init_args))
        return tuple(outputs) if len(outputs) != 1 else outputs[0]

    def after_create_test_data(self, table_name):
        if options.lifecycle:
            self.odps.run_sql('alter table %s set lifecycle %d' % (table_name, options.lifecycle))

    @staticmethod
    def gen_print_params_case():
        def _case(_, gen_params):
            gen_params = dict([(k, BaseNodeRunner._format_value(v))
                               for k, v in six.iteritems(gen_params) if v])
            print(repr(gen_params))

        return _case

    def gen_check_params_case(self, target_params, ignores=None):
        ignores = set(ignores or [])

        def _case(_, gen_params):
            gen_params = dict([(k, BaseNodeRunner._format_value(v))
                               for k, v in six.iteritems(gen_params) if k not in ignores and v])
            targets = dict((k, v) for k, v in six.iteritems(target_params) if k not in ignores)
            self.assertDictEqual(targets, gen_params)

        return _case

    def create_test_pmml_model(self, model_name):
        if self.odps.exist_offline_model(model_name):
            return

        old_dry_run = options.ml.dry_run
        options.ml.dry_run = False

        self.create_ionosphere(IONOSPHERE_TABLE)

        from odps.ml import classifiers

        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
        lr = classifiers.LogisticRegression(epsilon=0.001).set_max_iter(50)
        lr.train(df).persist(model_name)

        options.ml.dry_run = old_dry_run

    def delete_table(self, table_name):
        self.odps.delete_table(table_name, if_exists=True)

    def delete_offline_model(self, model_name):
        try:
            self.odps.delete_offline_model(model_name)
        except Exception:
            pass


def otm(model_name, key):
    prefix = TABLE_MODEL_PREFIX
    if model_name.startswith(TEMP_TABLE_PREFIX):
        prefix = TEMP_TABLE_PREFIX + TABLE_MODEL_PREFIX
        model_name = model_name[len(TEMP_TABLE_PREFIX):]
    return prefix + model_name + TABLE_MODEL_SEPARATOR + key


def _add_case(self, case):
    self._init_attr('_cases', [])
    self._cases.append(case)
    return self

AlgoExprMixin._add_case = _add_case


__all__ = ['tn', 'otm', 'MLTestBase', 'ci_skip_case']
