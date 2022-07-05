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

from __future__ import print_function

from collections import namedtuple

from odps.df import DataFrame
from odps.ml.text import *
from odps.ml.classifiers import *
from odps.ml.pipeline import Pipeline, FeatureUnion
from odps.ml.pipeline.core import PipelineStep
from odps.ml.tests.base import MLTestBase, tn, ci_skip_case

CORPUS_TABLE = tn('pyodps_test_ml_corpus')
W2V_TABLE = tn('pyodps_test_ml_w2v')
TFIDF_TABLE = tn('pyodps_test_ml_tf_idf')
LDA_TABLE = tn('pyodps_test_ml_plda')

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
IONOSPHERE_LR_MODEL = tn('pyodps_test_out_model')


class MockTransformStep(PipelineStep):
    def __init__(self, test_cls, step_name, action=None, params=None, outputs=None):
        super(MockTransformStep, self).__init__(step_name, params, outputs)

        self.action = action
        self.test_cls = test_cls
        for pn in (params or []):
            setattr(self, pn, None)

    def transform(self, ds):
        output_ds = self.test_cls.mock_action(ds, len(self._output_names), action=self.action)
        if len(self._output_names) == 1:
            return output_ds
        out_type = namedtuple('OutType', self._output_names)
        return out_type(**dict(zip(self._output_names,  output_ds)))


class Test(MLTestBase):
    def test_pipeline_param(self):
        step1 = MockTransformStep(self, 'Step1', params=['p1', 'p2'], outputs=['o1', ])
        step2 = MockTransformStep(self, 'Step2', params=['p1', 'p2'], outputs=['o1', 'o2'])
        step2.p2 = 'val22'
        step3 = MockTransformStep(self, 'Step3', params=['p1', 'p2'], outputs=['o1', ])

        pl1 = Pipeline(step1, (step2, 'o2'), name='pl1')
        pl = Pipeline(pl1, ('s3', step3))
        pl.pl1__step1__p1, pl.pl1__step2__p1, pl.s3__p1 = 'val1', 'val2', 'val3'

        self.assertRaises(AttributeError, lambda: setattr(pl, 'step3__p1', 1))
        self.assertTrue(step1 == pl['pl1'][0])
        self.assertTrue(step2 == pl['pl1']['Step2'])
        self.assertTrue(step3 == pl['s3'])
        self.assertEqual(pl.pl1__step1__p1, 'val1')
        self.assertEqual(pl.pl1__step2__p1, 'val2')
        self.assertEqual(pl.pl1__step2__p2, 'val22')
        self.assertEqual(pl1.step1__p1, 'val1')
        self.assertEqual(pl1.step2__p1, 'val2')
        self.assertEqual(pl1.step2__p2, 'val22')
        self.assertEqual(pl.s3__p1, 'val3')
        self.assertEqual(step1.p1, 'val1')
        self.assertEqual(step2.p1, 'val2')
        self.assertEqual(step2.p2, 'val22')
        self.assertEqual(step3.p1, 'val3')

        fu = FeatureUnion(pl1, step3)
        self.assertTrue(step1 == fu['pl1'][0])
        self.assertTrue(step2 == fu['pl1']['Step2'])
        self.assertTrue(step3 == fu['step3'])

    @ci_skip_case
    def test_tfidf_array(self):
        self.delete_table(W2V_TABLE)
        self.create_corpus(CORPUS_TABLE)
        df = DataFrame(self.odps.get_table(CORPUS_TABLE)).doc_content_field('content')
        pl = Pipeline(Pipeline(SplitWord(), (DocWordStat(), 'multi'), Word2Vec()))
        word_feature, _ = pl.transform(df)
        word_feature.persist(W2V_TABLE)

    @ci_skip_case
    def test_tfidf_code(self):
        self.delete_table(TFIDF_TABLE)
        self.create_corpus(CORPUS_TABLE)
        df = DataFrame(self.odps.get_table(CORPUS_TABLE)).doc_content_field('content')
        pl = Pipeline(SplitWord())
        TFIDF().link(DocWordStat().link(pl).triple)
        ret_df = pl.transform(df)
        ret_df.persist(TFIDF_TABLE)
