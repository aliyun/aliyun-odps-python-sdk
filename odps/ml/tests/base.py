# encoding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function

from odps.compat import six
from odps.config import options
from odps.df.core import DataFrame
from odps.runner import adapter_from_df, BaseNodeEngine, RunnerContext
from odps.runner.tests.base import RunnerTestBase
from odps.ml.adapter import ml_collection_mixin
from odps.ml.utils import TABLE_MODEL_PREFIX, TABLE_MODEL_SEPARATOR
from odps.tests.core import tn, ci_skip_case

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')


class MLTestBase(RunnerTestBase):
    @staticmethod
    def gen_print_params_case():
        def _case(_, gen_params):
            gen_params = dict([(k, BaseNodeEngine._format_value(v))
                               for k, v in six.iteritems(gen_params) if v])
            print(repr(gen_params))

        return _case

    def gen_check_params_case(self, target_params):
        def _case(_, gen_params):
            gen_params = dict([(k, BaseNodeEngine._format_value(v))
                               for k, v in six.iteritems(gen_params) if v])
            self.assertDictEqual(target_params, gen_params)

        return _case

    def create_test_pmml_model(self, model_name):
        if self.odps.exist_offline_model(model_name):
            return

        old_node_id = RunnerContext.instance()._dag._node_seq_id
        old_dry_run = options.runner.dry_run
        options.runner.dry_run = False

        self.create_ionosphere(IONOSPHERE_TABLE)

        from odps.ml import classifiers

        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
        lr = classifiers.LogisticRegression(epsilon=0.001).set_max_iter(50)
        lr.train(df).persist(model_name)

        options.runner.dry_run = old_dry_run
        RunnerContext.instance()._dag._node_seq_id = old_node_id

    def delete_table(self, table_name):
        self.odps.delete_table(table_name, if_exists=True)

    def delete_offline_model(self, model_name):
        try:
            self.odps.delete_offline_model(model_name)
        except Exception:
            pass


def otm(model_name, key):
    return TABLE_MODEL_PREFIX + model_name + TABLE_MODEL_SEPARATOR + key


@ml_collection_mixin
class MLCasesMixIn(object):
    def _add_case(self, case):
        adapter = adapter_from_df(self)
        adapter._bind_node.cases.append(case)
        return self


__all__ = ['tn', 'otm', 'MLTestBase', 'ci_skip_case']
