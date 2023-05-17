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

import logging
import textwrap

from .....compat import ElementTree as ET
from ..pmml import PmmlResult, PmmlRegressionResult, PmmlSegmentsResult, \
    parse_pmml_array

logger = logging.getLogger(__name__)

TREE_XML = """
<?xml version="1.0" encoding="utf-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_2" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="4.2" xsi:schemaLocation="http://www.dmg.org/PMML-4_2 http://www.dmg.org/v4-2/pmml-4-2.xsd">
  <Header copyright="Copyright (c) 2014, Alibaba Inc." description="">
    <Application name="ODPS/PMML" version="0.1.0"/>
    <Timestamp>Wed, 22 Mar 2017 09:35:38 GMT</Timestamp>
  </Header>
  <DataDictionary numberOfFields="35">
    <DataField name="a01" optype="continuous" dataType="double"/>
    <DataField name="a02" optype="continuous" dataType="double"/>
    <DataField name="a03" optype="continuous" dataType="double"/>
    <DataField name="a04" optype="continuous" dataType="double"/>
    <DataField name="a05" optype="continuous" dataType="double"/>
    <DataField name="a06" optype="continuous" dataType="double"/>
    <DataField name="a07" optype="continuous" dataType="double"/>
    <DataField name="a08" optype="continuous" dataType="double"/>
    <DataField name="a09" optype="continuous" dataType="double"/>
    <DataField name="a10" optype="continuous" dataType="double"/>
    <DataField name="a11" optype="continuous" dataType="double"/>
    <DataField name="a12" optype="continuous" dataType="double"/>
    <DataField name="a13" optype="continuous" dataType="double"/>
    <DataField name="a14" optype="continuous" dataType="double"/>
    <DataField name="a15" optype="continuous" dataType="double"/>
    <DataField name="a16" optype="continuous" dataType="double"/>
    <DataField name="a17" optype="continuous" dataType="double"/>
    <DataField name="a18" optype="continuous" dataType="double"/>
    <DataField name="a19" optype="continuous" dataType="double"/>
    <DataField name="a20" optype="continuous" dataType="double"/>
    <DataField name="a21" optype="continuous" dataType="double"/>
    <DataField name="a22" optype="continuous" dataType="double"/>
    <DataField name="a23" optype="continuous" dataType="double"/>
    <DataField name="a24" optype="continuous" dataType="double"/>
    <DataField name="a25" optype="continuous" dataType="double"/>
    <DataField name="a26" optype="continuous" dataType="double"/>
    <DataField name="a27" optype="continuous" dataType="double"/>
    <DataField name="a28" optype="continuous" dataType="double"/>
    <DataField name="a29" optype="continuous" dataType="double"/>
    <DataField name="a30" optype="continuous" dataType="double"/>
    <DataField name="a31" optype="continuous" dataType="double"/>
    <DataField name="a32" optype="continuous" dataType="double"/>
    <DataField name="a33" optype="continuous" dataType="double"/>
    <DataField name="a34" optype="continuous" dataType="double"/>
    <DataField name="class" optype="categorical" dataType="integer">
      <Value value="0"/>
      <Value value="1"/>
    </DataField>
  </DataDictionary>
  <MiningModel modelName="pyodps_test_iris_model" functionName="classification" algorithmName="RandomForests">
    <MiningSchema>
      <MiningField name="sepal_length" usageType="active"/>
      <MiningField name="sepal_width" usageType="active"/>
      <MiningField name="petal_length" usageType="active"/>
      <MiningField name="petal_width" usageType="active"/>
      <MiningField name="category" usageType="target"/>
    </MiningSchema>
    <Segmentation multipleModelMethod="majorityVote">
      <Segment id="0">
        <True/>
        <TreeModel modelName="pm_random_forests_46724_3" functionName="classification" algorithmName="RandomForests">
          <Node id="4">
            <Node id="5">
              <SimplePredicate field="a03" operator="lessOrEqual" value="0.4404"/>
              <ScoreDistribution value="0" recordCount="8"/>
              <ScoreDistribution value="1" recordCount="9"/>
              <Node id="6">
                <SimplePredicate field="a14" operator="lessOrEqual" value="0.261825"/>
                <ScoreDistribution value="0" recordCount="3"/>
                <ScoreDistribution value="1" recordCount="9"/>
                <Node id="7" score="0">
                  <SimplePredicate field="a26" operator="lessOrEqual" value="-0.327175"/>
                  <ScoreDistribution value="0" recordCount="2"/>
                </Node>
                <Node id="8" score="1">
                  <SimplePredicate field="a26" operator="greaterThan" value="-0.327175"/>
                  <ScoreDistribution value="0" recordCount="1"/>
                  <ScoreDistribution value="1" recordCount="9"/>
                </Node>
              </Node>
              <Node id="9" score="0">
                <SimplePredicate field="a14" operator="greaterThan" value="0.261825"/>
                <ScoreDistribution value="0" recordCount="5"/>
              </Node>
            </Node>
            <Node id="10">
              <SimplePredicate field="a03" operator="greaterThan" value="0.4404"/>
              <ScoreDistribution value="0" recordCount="5"/>
              <ScoreDistribution value="1" recordCount="105"/>
              <Node id="11" score="1">
                <SimplePredicate field="a24" operator="lessOrEqual" value="0.233095"/>
                <ScoreDistribution value="0" recordCount="1"/>
                <ScoreDistribution value="1" recordCount="89"/>
              </Node>
              <Node id="12">
                <SimplePredicate field="a24" operator="greaterThan" value="0.233095"/>
                <ScoreDistribution value="0" recordCount="4"/>
                <ScoreDistribution value="1" recordCount="16"/>
                <Node id="13">
                  <SimplePredicate field="a16" operator="lessOrEqual" value="0.424775"/>
                  <ScoreDistribution value="0" recordCount="4"/>
                  <ScoreDistribution value="1" recordCount="1"/>
                  <Node id="14" score="0">
                    <SimplePredicate field="a20" operator="lessOrEqual" value="-0.059785"/>
                    <ScoreDistribution value="0" recordCount="1"/>
                    <ScoreDistribution value="1" recordCount="1"/>
                  </Node>
                  <Node id="15" score="0">
                    <SimplePredicate field="a20" operator="greaterThan" value="-0.059785"/>
                    <ScoreDistribution value="0" recordCount="3"/>
                  </Node>
                </Node>
                <Node id="16" score="1">
                  <SimplePredicate field="a16" operator="greaterThan" value="0.424775"/>
                  <ScoreDistribution value="1" recordCount="15"/>
                </Node>
              </Node>
            </Node>
          </Node>
        </TreeModel>
      </Segment>
    </Segmentation>
  </MiningModel>
</PMML>
""".strip()


REGRESSION_XML = """
<?xml version="1.0" encoding="utf-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_2" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="4.2" xsi:schemaLocation="http://www.dmg.org/PMML-4_2 http://www.dmg.org/v4-2/pmml-4-2.xsd">
  <Header copyright="Copyright (c) 2014, Alibaba Inc." description="">
    <Application name="ODPS/PMML" version="0.1.0"/>
    <Timestamp>Wed, 22 Mar 2017 09:35:38 GMT</Timestamp>
  </Header>
  <DataDictionary numberOfFields="3">
    <DataField name="x1" optype="continuous" dataType="double"/>
    <DataField name="x2" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="integer">
      <Value value="0"/>
      <Value value="1"/>
    </DataField>
  </DataDictionary>
  <RegressionModel functionName="regression" modelName="Sample for stepwise polynomial regression" algorithmName="stepwisePolynomialRegression" normalizationMethod="softmax" targetFieldName="y">
    <MiningSchema>
      <MiningField name="x1"/>
      <MiningField name="x2"/>
      <MiningField name="y" usageType="target"/>
    </MiningSchema>
    <RegressionTable targetCategory="no" intercept="125.56601826">
      <NumericPredictor name="x1" coefficient="-28.6617384"/>
      <NumericPredictor name="x2" coefficient="-20.42027426"/>
    </RegressionTable>
    <RegressionTable targetCategory="yes" intercept="0"/>
  </RegressionModel>
</PMML>
""".strip()


def test_classes():
    assert isinstance(PmmlResult(TREE_XML), PmmlSegmentsResult)
    assert isinstance(PmmlResult(REGRESSION_XML), PmmlRegressionResult)


def test_tree_text_format():

    expected = textwrap.dedent("""
    ROOT
    ├── WHEN `a03` ≤ 0.4404 (COUNTS: 0:8, 1:9)
    │    ├── WHEN `a14` ≤ 0.261825 (COUNTS: 0:3, 1:9)
    │    │    ├── SCORE = 0 WHEN `a26` ≤ -0.327175 (COUNTS: 0:2)
    │    │    └── SCORE = 1 WHEN `a26` > -0.327175 (COUNTS: 0:1, 1:9)
    │    └── SCORE = 0 WHEN `a14` > 0.261825 (COUNTS: 0:5)
    └── WHEN `a03` > 0.4404 (COUNTS: 0:5, 1:105)
           ├── SCORE = 1 WHEN `a24` ≤ 0.233095 (COUNTS: 0:1, 1:89)
           └── WHEN `a24` > 0.233095 (COUNTS: 0:4, 1:16)
                  ├── WHEN `a16` ≤ 0.424775 (COUNTS: 0:4, 1:1)
                  │    ├── SCORE = 0 WHEN `a20` ≤ -0.059785 (COUNTS: 0:1, 1:1)
                  │    └── SCORE = 0 WHEN `a20` > -0.059785 (COUNTS: 0:3)
                  └── SCORE = 1 WHEN `a16` > 0.424775 (COUNTS: 1:15)
    """).strip()
    result = PmmlSegmentsResult(TREE_XML)
    assert(repr(result[0]).strip() == expected)


def test_tree_gv_format():
    expected = textwrap.dedent(u"""
    digraph {
    root [shape=record,label=<
        ROOT
    >];
    struct1 [shape=record,label=<
        <FONT POINT-SIZE="11">`a03` ≤ 0.4404</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:8, 1:9</FONT>
    >];
    struct2 [shape=record,label=<
        <FONT POINT-SIZE="11">`a14` ≤ 0.261825</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:3, 1:9</FONT>
    >];
    struct3 [shape=record,style=filled,fillcolor=azure2,label=<
        <FONT POINT-SIZE="16">0</FONT><br /><FONT POINT-SIZE="11">`a26` ≤ -0.327175</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:2</FONT>
    >];
    struct2 -> struct3;
    struct4 [shape=record,style=filled,fillcolor=azure2,label=<
        <FONT POINT-SIZE="16">1</FONT><br /><FONT POINT-SIZE="11">`a26` &gt; -0.327175</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:1, 1:9</FONT>
    >];
    struct2 -> struct4;
    struct1 -> struct2;
    struct5 [shape=record,style=filled,fillcolor=azure2,label=<
        <FONT POINT-SIZE="16">0</FONT><br /><FONT POINT-SIZE="11">`a14` &gt; 0.261825</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:5</FONT>
    >];
    struct1 -> struct5;
    root -> struct1;
    struct6 [shape=record,label=<
        <FONT POINT-SIZE="11">`a03` &gt; 0.4404</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:5, 1:105</FONT>
    >];
    struct7 [shape=record,style=filled,fillcolor=azure2,label=<
        <FONT POINT-SIZE="16">1</FONT><br /><FONT POINT-SIZE="11">`a24` ≤ 0.233095</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:1, 1:89</FONT>
    >];
    struct6 -> struct7;
    struct8 [shape=record,label=<
        <FONT POINT-SIZE="11">`a24` &gt; 0.233095</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:4, 1:16</FONT>
    >];
    struct9 [shape=record,label=<
        <FONT POINT-SIZE="11">`a16` ≤ 0.424775</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:4, 1:1</FONT>
    >];
    struct10 [shape=record,style=filled,fillcolor=azure2,label=<
        <FONT POINT-SIZE="16">0</FONT><br /><FONT POINT-SIZE="11">`a20` ≤ -0.059785</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:1, 1:1</FONT>
    >];
    struct9 -> struct10;
    struct11 [shape=record,style=filled,fillcolor=azure2,label=<
        <FONT POINT-SIZE="16">0</FONT><br /><FONT POINT-SIZE="11">`a20` &gt; -0.059785</FONT><br /><FONT POINT-SIZE="11">LABELS: 0:3</FONT>
    >];
    struct9 -> struct11;
    struct8 -> struct9;
    struct12 [shape=record,style=filled,fillcolor=azure2,label=<
        <FONT POINT-SIZE="16">1</FONT><br /><FONT POINT-SIZE="11">`a16` &gt; 0.424775</FONT><br /><FONT POINT-SIZE="11">LABELS: 1:15</FONT>
    >];
    struct8 -> struct12;
    struct6 -> struct8;
    root -> struct6;
    }
    """).strip()
    result = PmmlSegmentsResult(TREE_XML)
    assert result[0]._repr_gv_().strip() == expected


def test_regression_text_format():
    expected = textwrap.dedent("""
    Function: regression
    Target Field: y
    Normalization: softmax
    Target: no
        y = 125.56601826 - 28.6617384 * x1 - 20.42027426 * x2
    """).strip()
    regr = PmmlRegressionResult(REGRESSION_XML)
    assert repr(regr).strip() == expected


def test_array_parse():
    elem = ET.fromstring(r'<Array n="3" type="string">ab  "a b"   "with \"quotes\" "</Array>')
    assert parse_pmml_array(elem) == ['ab', 'a b', 'with "quotes" ']
    elem = ET.fromstring(r'<Array n="3" type="int">1  22 3</Array>')
    assert parse_pmml_array(elem) == [1, 22, 3]
