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

import pytest

from ....df import DataFrame
from ...text import *
from ...classifiers import *
from ...pipeline import Pipeline, FeatureUnion
from ...pipeline.core import PipelineStep
from ...tests.base import MLTestUtil, tn, ci_skip_case

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


@pytest.fixture
def utils(odps, tunnel):
    return MLTestUtil(odps, tunnel)


def test_pipeline_param(utils):
    step1 = MockTransformStep(utils, 'Step1', params=['p1', 'p2'], outputs=['o1', ])
    step2 = MockTransformStep(utils, 'Step2', params=['p1', 'p2'], outputs=['o1', 'o2'])
    step2.p2 = 'val22'
    step3 = MockTransformStep(utils, 'Step3', params=['p1', 'p2'], outputs=['o1', ])

    pl1 = Pipeline(step1, (step2, 'o2'), name='pl1')
    pl = Pipeline(pl1, ('s3', step3))
    pl.pl1__step1__p1, pl.pl1__step2__p1, pl.s3__p1 = 'val1', 'val2', 'val3'

    pytest.raises(AttributeError, lambda: setattr(pl, 'step3__p1', 1))
    assert step1 == pl['pl1'][0]
    assert step2 == pl['pl1']['Step2']
    assert step3 == pl['s3']
    assert pl.pl1__step1__p1 == 'val1'
    assert pl.pl1__step2__p1 == 'val2'
    assert pl.pl1__step2__p2 == 'val22'
    assert pl1.step1__p1 == 'val1'
    assert pl1.step2__p1 == 'val2'
    assert pl1.step2__p2 == 'val22'
    assert pl.s3__p1 == 'val3'
    assert step1.p1 == 'val1'
    assert step2.p1 == 'val2'
    assert step2.p2 == 'val22'
    assert step3.p1 == 'val3'

    fu = FeatureUnion(pl1, step3)
    assert step1 == fu['pl1'][0]
    assert step2 == fu['pl1']['Step2']
    assert step3 == fu['step3']


@ci_skip_case
def test_tfidf_array(odps, utils):
    utils.delete_table(W2V_TABLE)
    utils.create_corpus(CORPUS_TABLE)
    df = DataFrame(odps.get_table(CORPUS_TABLE)).doc_content_field('content')
    pl = Pipeline(Pipeline(SplitWord(), (DocWordStat(), 'multi'), Word2Vec()))
    word_feature, _ = pl.transform(df)
    word_feature.persist(W2V_TABLE)


@ci_skip_case
def test_tfidf_code(odps, utils):
    utils.delete_table(TFIDF_TABLE)
    utils.create_corpus(CORPUS_TABLE)
    df = DataFrame(odps.get_table(CORPUS_TABLE)).doc_content_field('content')
    pl = Pipeline(SplitWord())
    TFIDF().link(DocWordStat().link(pl).triple)
    ret_df = pl.transform(df)
    ret_df.persist(TFIDF_TABLE)
