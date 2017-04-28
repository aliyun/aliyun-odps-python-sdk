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

import textwrap

from odps import options, errors
from odps.compat import six
from odps.examples.tables import TestDataMixIn
from odps.models import Schema
from odps.models.ml.onlinemodel import ModelPredictor, OnlineModel, PREDICT_TYPE_CODES
from odps.tests.core import TestBase, tn, ci_skip_case
from odps.types import Record

IRIS_TABLE = tn('pyodps_test_ml_iris')
TEST_LR_MODEL_NAME = tn('pyodps_test_irislr', 32)
TEST_OFFLINE_ONLINE_MODEL_NAME = tn('pyodps_test_irislr', 32)
TEST_OFFLINE_ONLINE_MODEL_NAME_2 = tn('pyodps_2_test_irislr', 32)
TEST_PIPELINE_ONLINE_MODEL_NAME = tn('pyodps_test_pl_lr', 32)

PMML_CONTENT = """<?xml version="1.0"?>
<PMML version="3.2" xmlns="http://www.dmg.org/PMML-3_2" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.dmg.org/PMML-3_2 http://www.dmg.org/v3-2/pmml-3-2.xsd">
 <Header copyright="Copyright (c) 2012 DMG" description="Linear Regression Model">
  <Extension name="user" value="DMG" extender="Rattle/PMML"/>
  <Application name="Rattle/PMML" version="1.2.29"/>
  <Timestamp>2012-09-27 12:34:14</Timestamp>
 </Header>
 <DataDictionary numberOfFields="5">
  <DataField name="sepal_length" optype="continuous" dataType="double"/>
  <DataField name="sepal_width" optype="continuous" dataType="double"/>
  <DataField name="petal_length" optype="continuous" dataType="double"/>
  <DataField name="petal_width" optype="continuous" dataType="double"/>
  <DataField name="class" optype="categorical" dataType="string">
   <Value value="Iris-setosa"/>
   <Value value="Iris-versicolor"/>
   <Value value="Iris-virginica"/>
  </DataField>
 </DataDictionary>
 <RegressionModel modelName="Linear_Regression_Model" functionName="regression" algorithmName="least squares" targetFieldName="sepal_length">
  <MiningSchema>
   <MiningField name="sepal_length" usageType="predicted"/>
   <MiningField name="sepal_width" usageType="active"/>
   <MiningField name="petal_length" usageType="active"/>
   <MiningField name="petal_width" usageType="active"/>
   <MiningField name="class" usageType="active"/>
  </MiningSchema>
  <RegressionTable intercept="2.17126629215507">
   <NumericPredictor name="sepal_width" exponent="1" coefficient="0.495888938388551"/>
   <NumericPredictor name="petal_length" exponent="1" coefficient="0.829243912234806"/>
   <NumericPredictor name="petal_width" exponent="1" coefficient="-0.315155173326474"/>
   <CategoricalPredictor name="class" value="Iris-setosa" coefficient="0"/>
   <CategoricalPredictor name="class" value="Iris-versicolor" coefficient="-0.723561957780729"/>
   <CategoricalPredictor name="class" value="Iris-virginica" coefficient="-1.02349781449083"/>
  </RegressionTable>
 </RegressionModel>
</PMML>
"""


class Test(TestDataMixIn, TestBase):
    def create_test_pmml_model(self, model_name):
        if self.odps.exist_offline_model(model_name):
            return

        old_dry_run = options.ml.dry_run
        options.ml.dry_run = False

        self.create_iris(IRIS_TABLE)

        from odps.df import DataFrame
        from odps.ml import classifiers

        df = DataFrame(self.odps.get_table(IRIS_TABLE)).roles(label='category')
        lr = classifiers.LogisticRegression(epsilon=0.001).set_max_iter(50)
        lr.train(df).persist(model_name)

        options.ml.dry_run = old_dry_run

    def testSerializePipeline(self):
        test_json = """
        {
            "target": {
                "name": "label"
            },
            "pipeline": {
                "processor": [
                    {
                        "offlinemodelProject": "online_test",
                        "offlinemodelName": "sample_offlinemodel"
                    },
                    {
                        "pmml": "data_preprocess.xml",
                        "refResource": "online_test/resources/data_preprocess.xml",
                        "runMode": "Converter"
                    },
                    {
                        "className": "SampleProcessor",
                        "libName": "libsample_processor.so",
                        "refResource": "online_test/resources/sample_processor.tar.gz"
                    }
                ]
            }
        }
        """.strip()
        expect_xml = textwrap.dedent("""
        <?xml version="1.0" encoding="utf-8"?>
        <PredictDesc>
          <Pipeline>
            <BuiltinProcessor>
              <OfflinemodelProject>online_test</OfflinemodelProject>
              <OfflinemodelName>sample_offlinemodel</OfflinemodelName>
            </BuiltinProcessor>
            <PmmlProcessor>
              <Pmml>data_preprocess.xml</Pmml>
              <RefResource>online_test/resources/data_preprocess.xml</RefResource>
              <RunMode>Converter</RunMode>
            </PmmlProcessor>
            <Processor>
              <LibName>libsample_processor.so</LibName>
              <RefResource>online_test/resources/sample_processor.tar.gz</RefResource>
            </Processor>
          </Pipeline>
          <Target>
            <Name>label</Name>
          </Target>
        </PredictDesc>
        """).strip()
        predictor_xml = ModelPredictor.parse(test_json).serialize().strip()
        expect_xml = ModelPredictor.parse(expect_xml).serialize().strip()
        self.assertEqual(predictor_xml, expect_xml)

    def testMakePredictRequest(self):
        import decimal

        def typed(v, type_code=None):
            if isinstance(v, dict):
                return dict([(key, typed(value, type_code)) for key, value in six.iteritems(v)])
            elif isinstance(v, list):
                return [typed(it, type_code) for it in v]
            return {
                'dataType': type_code or PREDICT_TYPE_CODES[type(v).__name__],
                'dataValue': v,
            }

        self.assertRaises(ValueError, lambda: OnlineModel._build_predict_request(()))
        self.assertRaises(ValueError, lambda: OnlineModel._build_predict_request([()]))
        self.assertRaises(ValueError, lambda: OnlineModel._build_predict_request([[]]))
        self.assertRaises(ValueError, lambda: OnlineModel._build_predict_request(['malformed']))
        self.assertEqual(
            OnlineModel._build_predict_request([[1, 2], [3, 4]], schema=['a', 'b']),
            dict(inputs=typed([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]))
        )
        self.assertEqual(
            OnlineModel._build_predict_request([1, 2], schema=['a', 'b']),
            dict(inputs=[typed({'a': 1, 'b': 2})])
        )
        self.assertEqual(
            OnlineModel._build_predict_request({'a': 1, 'b': 2}),
            dict(inputs=[typed({'a': 1, 'b': 2})])
        )
        self.assertEqual(
            OnlineModel._build_predict_request(Record(schema=Schema.from_lists(['a', 'b'], ['string', 'string']), values=['1', '2'])),
            dict(inputs=[typed({'a': '1', 'b': '2'})])
        )
        self.assertEqual(
            OnlineModel._build_predict_request([1, 2], schema=Schema.from_lists(['a', 'b'], ['bigint', 'bigint'])),
            dict(inputs=[typed({'a': 1, 'b': 2}, PREDICT_TYPE_CODES['long'])])
        )
        self.assertEqual(
            OnlineModel._build_predict_request([('a', 1), ('b', 2)]),
            dict(inputs=[typed({'a': 1, 'b': 2})])
        )
        self.assertEqual(
            OnlineModel._build_predict_request([decimal.Decimal('1.23'), '2'], schema=Schema.from_lists(['a', 'b'], ['decimal', 'string'])),
            dict(inputs=[typed({'a': 1.23, 'b': '2'})])
        )
        self.assertEqual(
            OnlineModel._build_predict_request([('a', decimal.Decimal("1.23")), ('b', 2)]),
            dict(inputs=[typed({'a': 1.23, 'b': 2})])
        )

    @ci_skip_case
    def testPublishOfflineModel(self):
        self.create_test_pmml_model(TEST_LR_MODEL_NAME)
        try:
            self.odps.delete_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME)
        except errors.NoSuchObject:
            pass
        try:
            self.odps.delete_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME_2)
        except errors.NoSuchObject:
            pass

        try:
            model = self.odps.create_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME, TEST_LR_MODEL_NAME, async=True)
            self.assertEqual(model.name, TEST_OFFLINE_ONLINE_MODEL_NAME)
            self.assertEqual(model.status, OnlineModel.Status.DEPLOYING)
            model.wait_for_service()
            self.assertEqual(model.status, OnlineModel.Status.SERVING)

            model2 = self.odps.create_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME_2, TEST_LR_MODEL_NAME)
            self.assertEqual(model2.name, TEST_OFFLINE_ONLINE_MODEL_NAME_2)
            self.assertEqual(model2.status, OnlineModel.Status.SERVING)

            model = self.odps.get_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME)
            self.assertEqual(model.name, TEST_OFFLINE_ONLINE_MODEL_NAME)

            model.update()

            self.odps.config_online_model_ab_test(TEST_OFFLINE_ONLINE_MODEL_NAME, model2, 50)

            self.assertTrue(self.odps.exist_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME))
            self.assertFalse(self.odps.exist_online_model('non_exist_online_model'))

            exist_model_list = False
            for mod in self.odps.list_online_models():
                self.assertIsNotNone(mod.name)
                if mod.name == TEST_OFFLINE_ONLINE_MODEL_NAME:
                    exist_model_list = True
            self.assertTrue(exist_model_list)

            predicted = self.odps.predict_online_model(
                TEST_OFFLINE_ONLINE_MODEL_NAME,
                [4.0, 3.0, 2.0, 1.0],
                schema=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            )
            self.assertEqual(len(predicted), 1)
        finally:
            # ensure that models are deleted
            if self.odps.exist_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME):
                model = self.odps.get_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME)
                model.drop(async=True)
                self.assertEqual(model.status, OnlineModel.Status.DELETING)
                model.wait_for_deletion()
                self.assertFalse(self.odps.exist_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME))
            if self.odps.exist_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME_2):
                self.odps.delete_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME_2)
                self.assertFalse(self.odps.exist_online_model(TEST_OFFLINE_ONLINE_MODEL_NAME_2))

    @ci_skip_case
    def testPublishPmmlPipelineModel(self):
        from odps.models.ml.onlinemodel import ModelPredictor, PmmlProcessor, PmmlRunMode

        resource_name = tn('pyodps_test_online_model_pmml') + '.xml'
        if self.odps.exist_resource(resource_name):
            self.odps.delete_resource(resource_name)
        self.odps.create_resource(resource_name, 'file', file_obj=PMML_CONTENT)

        resource_path = '/'.join([self.odps.project, 'resources', resource_name])
        pmml_processor = PmmlProcessor(pmml=resource_name, resources=resource_path, run_mode=PmmlRunMode.Evaluator)

        predictor = ModelPredictor(target_name='category')
        predictor.pipeline.append(pmml_processor)

        try:
            self.odps.delete_online_model(TEST_PIPELINE_ONLINE_MODEL_NAME)
        except errors.NoSuchObject:
            pass

        try:
            model = self.odps.create_online_model(TEST_PIPELINE_ONLINE_MODEL_NAME, predictor)
            self.assertEqual(model.name, TEST_PIPELINE_ONLINE_MODEL_NAME)

            model = self.odps.get_online_model(TEST_PIPELINE_ONLINE_MODEL_NAME)
            self.assertEqual(model.name, TEST_PIPELINE_ONLINE_MODEL_NAME)

            model.update(async=True)
            model.wait_for_service()

            self.assertTrue(self.odps.exist_online_model(TEST_PIPELINE_ONLINE_MODEL_NAME))
            self.assertFalse(self.odps.exist_online_model('non_exist_online_model'))

            predicted = self.odps.predict_online_model(
                TEST_PIPELINE_ONLINE_MODEL_NAME,
                [[0.3, 0.7, 0.9], [0.7, 0.3, 0.9]],
                schema=['sepal_width', 'petal_length', 'petal_width']
            )
            assert len(predicted) == 2
        finally:
            # ensure that models are deleted
            if self.odps.exist_online_model(TEST_PIPELINE_ONLINE_MODEL_NAME):
                self.odps.delete_online_model(TEST_PIPELINE_ONLINE_MODEL_NAME)
                self.assertFalse(self.odps.exist_online_model(TEST_PIPELINE_ONLINE_MODEL_NAME))
