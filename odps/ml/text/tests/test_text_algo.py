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

import pytest

from ....df import DataFrame
from ....config import options
from ...utils import TEMP_TABLE_PREFIX
from ...text import *
from ...tests.base import MLTestUtil, tn

CORPUS_TABLE = tn('pyodps_test_ml_corpus')
WORD_TRIPLE_TABLE = tn('pyodps_test_ml_word_triple')
SPLITED_TABLE = tn('pyodps_test_ml_splited_text')
NOISE_TABLE = tn('pyodps_test_ml_noises')
W2V_TABLE = tn('pyodps_test_ml_w2v')
TFIDF_TABLE = tn('pyodps_test_ml_tf_idf')
LDA_TABLE = tn('pyodps_test_ml_plda')
STR_COMP_TABLE = tn('pyodps_test_ml_str_comp')
COMP_RESULT_TABLE = tn('pyodps_test_ml_str_comp_result')
TOP_N_TABLE = tn('pyodps_test_ml_top_n_result')
FILTERED_WORDS_TABLE = tn('pyodps_test_ml_filtered_words_result')
KW_EXTRACTED_TABLE = tn('pyodps_test_ml_kw_extracted_result')
TEXT_SUMMARIZED_TABLE = tn('pyodps_test_ml_text_summarized_result')
COUNT_NGRAM_TABLE = tn('pyodps_test_ml_count_ngram_result')
DOC2VEC_DOC_TABLE = tn('pyodps_test_ml_doc2vec_doc_result')
SEMANTIC_DIST_TABLE = tn('pyodps_test_ml_semantic_dist_result')


@pytest.fixture
def utils(odps, tunnel):
    util = MLTestUtil(odps, tunnel)
    util.create_corpus(CORPUS_TABLE)
    util.df = DataFrame(odps.get_table(CORPUS_TABLE)).roles(doc_id='id', doc_content='content')
    options.ml.dry_run = True
    return util


def _create_str_compare_table(odps, table_name):
    data_rows = [
        ['inputTableName', 'inputTableName'], ['outputTableName', 'mapTableName'],
        ['inputSelectedColName1', 'outputTableName'], ['inputSelectedColName2', 'inputSelectedColName'],
        ['inputAppendColNames', 'mapSelectedColName'], ['inputTablePartitions', 'inputAppendColNames'],
        ['outputColName', 'inputAppendRenameColNames'], ['method', 'mapAppendColNames'],
        ['lambda', 'mapAppendRenameColNames'], ['k', 'inputTablePartitions'],
        ['lifecycle', 'mapTablePartitions'], ['coreNum', 'outputColName'], ['memSizePerCore', 'method'],
    ]
    for idx, r in enumerate(data_rows):
        data_rows[idx] = [idx] + r
    odps.execute_sql('drop table if exists ' + table_name)
    odps.execute_sql('create table %s (str_id bigint, col1 string, col2 string)' % table_name)
    odps.write_table(table_name, data_rows)


def _create_noise_table(odps, table_name):
    data_rows = (u'，', u'。', u'《', u'》', u'的', u'是')
    data_rows = [[v] for v in data_rows]
    odps.execute_sql('drop table if exists ' + table_name)
    odps.execute_sql('create table %s (noise_col string)' % table_name)
    odps.write_table(table_name, data_rows)


def test_tf_idf(utils):
    splited = SplitWord().transform(utils.df)
    freq, _ = DocWordStat().transform(splited)
    tf_set = TFIDF().transform(freq)
    tf_set._add_case(utils.gen_check_params_case({
        'docIdCol': 'id', 'inputTableName': TEMP_TABLE_PREFIX + '_doc_word_stat', 'countCol': 'count',
        'outputTableName': TFIDF_TABLE, 'wordCol': 'word'}))
    tf_set.persist(TFIDF_TABLE)


def test_str_diff(odps, utils):
    _create_str_compare_table(odps, STR_COMP_TABLE)
    df = DataFrame(odps.get_table(STR_COMP_TABLE))
    diff_df = str_diff(df, col1='col1', col2='col2')
    diff_df._add_case(utils.gen_check_params_case({
        'inputTableName': STR_COMP_TABLE, 'k': '2', 'outputTableName': COMP_RESULT_TABLE,
        'inputSelectedColName2': 'col2', 'inputSelectedColName1': 'col1', 'method': 'levenshtein_sim',
        'lambda': '0.5', 'outputColName': 'output'}))
    diff_df.persist(COMP_RESULT_TABLE)


def test_top_n(odps, utils):
    _create_str_compare_table(odps, STR_COMP_TABLE)
    df = DataFrame(odps.get_table(STR_COMP_TABLE))
    top_n_df = top_n_similarity(df, df, col='col1', map_col='col1')
    top_n_df._add_case(utils.gen_check_params_case({
        'inputTableName': STR_COMP_TABLE, 'k': '2', 'outputColName': 'output',
        'mapSelectedColName': 'col1', 'topN': '10', 'inputSelectedColName': 'col1',
        'outputTableName': TOP_N_TABLE, 'mapTableName': odps.project + '.' + STR_COMP_TABLE,
        'method': 'levenshtein_sim', 'lambda': '0.5'}))
    top_n_df.persist(TOP_N_TABLE)


def test_filter_noises(odps, utils):
    odps.delete_table(FILTERED_WORDS_TABLE, if_exists=True)

    utils.create_splited_words(SPLITED_TABLE)
    _create_noise_table(odps, NOISE_TABLE)
    df = DataFrame(odps.get_table(SPLITED_TABLE)).roles(doc_content='content')
    ndf = DataFrame(odps.get_table(NOISE_TABLE))
    filtered = filter_noises(df, ndf)
    filtered._add_case(utils.gen_check_params_case({
        'noiseTableName': odps.project + '.' + NOISE_TABLE, 'outputTableName': FILTERED_WORDS_TABLE,
        'selectedColNames': 'content', 'inputTableName': SPLITED_TABLE}))
    filtered.persist(FILTERED_WORDS_TABLE)


def test_keywords_extraction(odps, utils):
    odps.delete_table(KW_EXTRACTED_TABLE, if_exists=True)
    utils.create_splited_words(SPLITED_TABLE)
    df = DataFrame(odps.get_table(SPLITED_TABLE)).roles(doc_id='doc_id', doc_content='content')
    extracted = extract_keywords(df)
    extracted._add_case(utils.gen_check_params_case(
        {'dumpingFactor': '0.85', 'inputTableName': SPLITED_TABLE, 'epsilon': '0.000001', 'windowSize': '2',
         'topN': '5', 'outputTableName': KW_EXTRACTED_TABLE, 'docIdCol': 'doc_id', 'maxIter': '100',
         'docContent': 'content'}))
    extracted.persist(KW_EXTRACTED_TABLE)


def test_summarize_text(odps, utils):
    utils.create_corpus(CORPUS_TABLE)
    summarized = summarize_text(utils.df.roles(sentence='content'))
    summarized._add_case(utils.gen_check_params_case(
        {'dumpingFactor': '0.85', 'inputTableName': CORPUS_TABLE, 'sentenceCol': 'content',
         'epsilon': '0.000001', 'k': '2', 'topN': '3', 'outputTableName': TEXT_SUMMARIZED_TABLE,
         'docIdCol': 'id', 'maxIter': '100', 'similarityType': 'lcs_sim', 'lambda': '0.5'}))
    summarized.persist(TEXT_SUMMARIZED_TABLE)


def test_count_ngram(odps, utils):
    utils.create_word_triple(WORD_TRIPLE_TABLE)
    word_triple_df = DataFrame(odps.get_table(WORD_TRIPLE_TABLE)).select_features('word')
    counted = count_ngram(word_triple_df)
    counted._add_case(utils.gen_check_params_case({
        'outputTableName': COUNT_NGRAM_TABLE, 'inputSelectedColNames': 'word', 'order': '3',
        'inputTableName': WORD_TRIPLE_TABLE}))
    counted.persist(COUNT_NGRAM_TABLE)


def test_doc2vec(odps, utils):
    word_df, doc_df, _ = Doc2Vec().transform(utils.df)
    doc_df._add_case(utils.gen_check_params_case(
        {'minCount': '5', 'docColName': 'content', 'hs': '1', 'inputTableName': tn('pyodps_test_ml_corpus'),
         'negative': '0', 'layerSize': '100', 'sample': '0', 'randomWindow': '1', 'window': '5',
         'docIdColName': 'id', 'iterTrain': '1', 'alpha': '0.025', 'cbow': '0',
         'outVocabularyTableName': 'tmp_pyodps__doc2_vec', 'outputWordTableName': 'tmp_pyodps__doc2_vec',
         'outputDocTableName': tn('pyodps_test_ml_doc2vec_doc_result')}))
    doc_df.persist(DOC2VEC_DOC_TABLE)


def test_semantic_vector_distance(odps, utils):
    result_df = semantic_vector_distance(utils.df)
    result_df._add_case(utils.gen_check_params_case(
        {'topN': '5', 'outputTableName': tn('pyodps_test_ml_semantic_dist_result'), 'distanceType': 'euclidean',
         'inputTableName': tn('pyodps_test_ml_corpus')}))
    result_df.persist(SEMANTIC_DIST_TABLE)
