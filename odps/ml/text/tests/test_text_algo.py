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

from odps.df import DataFrame
from odps.config import options
from odps.ml.utils import TEMP_TABLE_PREFIX
from odps.ml.text import *
from odps.ml.tests.base import MLTestBase, tn

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


class Test(MLTestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.create_corpus(CORPUS_TABLE)
        self.df = DataFrame(self.odps.get_table(CORPUS_TABLE)).roles(doc_id='id', doc_content='content')

        options.runner.dry_run = True

    def _create_str_compare_table(self, table_name):
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
        self.odps.execute_sql('drop table if exists ' + table_name)
        self.odps.execute_sql('create table %s (str_id bigint, col1 string, col2 string)' % table_name)
        self.odps.write_table(table_name, data_rows)

    def _create_noise_table(self, table_name):
        data_rows = (u'，', u'。', u'《', u'》', u'的', u'是')
        data_rows = [[v] for v in data_rows]
        self.odps.execute_sql('drop table if exists ' + table_name)
        self.odps.execute_sql('create table %s (noise_col string)' % table_name)
        self.odps.write_table(table_name, data_rows)

    def test_tf_idf(self):
        splited = SplitWord().transform(self.df)
        freq, _ = DocWordStat().transform(splited)
        tf_set = TFIDF().transform(freq)
        tf_set._add_case(self.gen_check_params_case({
            'docIdCol': 'id', 'inputTableName': TEMP_TABLE_PREFIX + '0_doc_word_stat_3_1', 'countCol': 'count',
            'outputTableName': TFIDF_TABLE, 'wordCol': 'word'}))
        tf_set.persist(TFIDF_TABLE)

    def test_str_diff(self):
        self._create_str_compare_table(STR_COMP_TABLE)
        df = DataFrame(self.odps.get_table(STR_COMP_TABLE))
        diff_df = str_diff(df, col1='col1', col2='col2')
        diff_df._add_case(self.gen_check_params_case({
            'inputTableName': STR_COMP_TABLE, 'k': '2', 'outputTableName': COMP_RESULT_TABLE,
            'inputSelectedColName2': 'col2', 'inputSelectedColName1': 'col1', 'method': 'levenshtein_sim',
            'lambda': '0.5', 'outputColName': 'output'}))
        diff_df.persist(COMP_RESULT_TABLE)

    def test_top_n(self):
        self._create_str_compare_table(STR_COMP_TABLE)
        df = DataFrame(self.odps.get_table(STR_COMP_TABLE))
        top_n_df = top_n_similarity(df, df, col='col1', map_col='col1')
        top_n_df._add_case(self.gen_check_params_case({
            'inputTableName': STR_COMP_TABLE, 'k': '2', 'outputColName': 'output',
            'mapSelectedColName': 'col1', 'topN': '10', 'inputSelectedColName': 'col1',
            'outputTableName': TOP_N_TABLE, 'mapTableName': STR_COMP_TABLE,
            'method': 'levenshtein_sim', 'lambda': '0.5'}))
        top_n_df.persist(TOP_N_TABLE)

    def test_filter_noises(self):
        self.odps.delete_table(FILTERED_WORDS_TABLE, if_exists=True)

        self.create_splited_words(SPLITED_TABLE)
        self._create_noise_table(NOISE_TABLE)
        df = DataFrame(self.odps.get_table(SPLITED_TABLE)).roles(doc_content='content')
        ndf = DataFrame(self.odps.get_table(NOISE_TABLE))
        filtered = filter_noises(df, ndf)
        filtered._add_case(self.gen_check_params_case({
            'noiseTableName': NOISE_TABLE, 'outputTableName': FILTERED_WORDS_TABLE,
            'selectedColNames': 'content', 'inputTableName': SPLITED_TABLE}))
        filtered.persist(FILTERED_WORDS_TABLE)

    def test_keywords_extraction(self):
        self.odps.delete_table(KW_EXTRACTED_TABLE, if_exists=True)
        self.create_splited_words(SPLITED_TABLE)
        df = DataFrame(self.odps.get_table(SPLITED_TABLE)).roles(doc_id='doc_id', doc_content='content')
        extracted = extract_keywords(df)
        extracted._add_case(self.gen_check_params_case(
            {'dumpingFactor': '0.85', 'inputTableName': SPLITED_TABLE, 'epsilon': '0.000001', 'windowSize': '2',
             'topN': '5', 'outputTableName': KW_EXTRACTED_TABLE, 'docIdCol': 'doc_id', 'maxIter': '100',
             'docContent': 'content'}))
        extracted.persist(KW_EXTRACTED_TABLE)

    def test_summarize_text(self):
        self.create_corpus(CORPUS_TABLE)
        summarized = summarize_text(self.df.roles(sentence='content'))
        summarized._add_case(self.gen_check_params_case(
            {'dumpingFactor': '0.85', 'inputTableName': CORPUS_TABLE, 'sentenceCol': 'content',
             'epsilon': '0.000001', 'k': '2', 'topN': '3', 'outputTableName': TEXT_SUMMARIZED_TABLE,
             'docIdCol': 'id', 'maxIter': '100', 'similarityType': 'lcs_sim', 'lambda': '0.5'}))
        summarized.persist(TEXT_SUMMARIZED_TABLE)

    def test_count_ngram(self):
        self.create_word_triple(WORD_TRIPLE_TABLE)
        word_triple_df = DataFrame(self.odps.get_table(WORD_TRIPLE_TABLE)).select_features('word')
        counted = count_ngram(word_triple_df)
        counted._add_case(self.gen_check_params_case({
            'outputTableName': COUNT_NGRAM_TABLE, 'inputSelectedColNames': 'word', 'order': '3',
            'inputTableName': WORD_TRIPLE_TABLE}))
        counted.persist(COUNT_NGRAM_TABLE)
