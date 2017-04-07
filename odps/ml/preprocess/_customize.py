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

import json
from collections import Iterable

from ...compat import six
from ...serializers import JSONSerializableModel, JSONNodeField
from ..utils import ML_ARG_PREFIX

"""
Exporter
"""


def get_append_id_selected_cols(expr, param_value):
    if param_value:
        if isinstance(param_value, (list, tuple, set)):
            return ','.join(param_value)
        return param_value
    else:
        return ','.join(f.name for f in getattr(expr, ML_ARG_PREFIX + 'input')._ml_fields)


def get_modify_abnormal_json_string(expr, param_value):
    if isinstance(param_value, JSONSerializableModel):
        return json.dumps([param_value.serial(), ], separators=(',', ':'))
    if isinstance(param_value, Iterable) and not isinstance(param_value, six.string_types):
        return json.dumps([obj.serial() if isinstance(obj, JSONSerializableModel) else obj for obj in param_value],
                          separators=(',', ':'))
    else:
        return param_value


"""
Output Schemas
"""


def binning_predict_output(params, fields):
    out_cols = params['metaColNames'].split(',') if 'metaColNames' in params else []
    input_ml_fields = fields['feature']
    if not out_cols:
        out_cols = list(six.iterkeys(fields['feature']))
    return ','.join('%s: %s' % (col, input_ml_fields[col]) for col in out_cols) + ', prediction_score: double, ' +\
           'prediction_prob: double, prediction_detail: string'


"""
Modify Abnormal JSON classes
"""


class ReplaceNull(JSONSerializableModel):
    col = JSONNodeField('col')
    type = JSONNodeField('type')
    replace = JSONNodeField('replace')

    def __init__(self, col, replace, **kwargs):
        kwargs.update(dict(col=col, type='null', replace=replace))
        super(ReplaceNull, self).__init__(**kwargs)


class ReplaceEmpty(JSONSerializableModel):
    col = JSONNodeField('col')
    type = JSONNodeField('type')
    replace = JSONNodeField('replace')

    def __init__(self, col, replace, **kwargs):
        kwargs.update(dict(col=col, type='empty', replace=replace))
        super(ReplaceEmpty, self).__init__(**kwargs)


class ReplaceNullEmpty(JSONSerializableModel):
    col = JSONNodeField('col')
    type = JSONNodeField('type')
    replace = JSONNodeField('replace')

    def __init__(self, col, replace, **kwargs):
        kwargs.update(dict(col=col, type='null-empty', replace=replace))
        super(ReplaceNullEmpty, self).__init__(**kwargs)


class ReplaceCustom(JSONSerializableModel):
    col = JSONNodeField('col')
    type = JSONNodeField('type')
    original = JSONNodeField('original')
    replace = JSONNodeField('replace')

    def __init__(self, col, original, replace, **kwargs):
        kwargs.update(dict(col=col, type='user-defined', original=original, replace=replace))
        super(ReplaceCustom, self).__init__(**kwargs)


class ReplacePercentile(JSONSerializableModel):
    col = JSONNodeField('col')
    type = JSONNodeField('type')
    low_range = JSONNodeField('low_range')
    high_range = JSONNodeField('high_range')
    low_replace = JSONNodeField('low_replace')
    high_replace = JSONNodeField('high_replace')

    def __init__(self, col, low_range=None, low_replace=None, high_range=None, high_replace=None, **kwargs):
        kwargs.update(dict(col=col, type='percentile', low_range=low_range, low_replace=low_replace,
                           high_range=high_range, high_replace=high_replace))
        super(ReplacePercentile, self).__init__(**kwargs)


class ReplaceConfidence(JSONSerializableModel):
    col = JSONNodeField('col')
    type = JSONNodeField('type')
    confidence = JSONNodeField('confidence')
    low_replace = JSONNodeField('low_replace')
    high_replace = JSONNodeField('high_replace')

    def __init__(self, col, confidence, low_replace=None, high_replace=None, **kwargs):
        kwargs.update(dict(col=col, type='zscore', confidence=confidence, low_replace=low_replace,
                           high_replace=high_replace))
        super(ReplaceConfidence, self).__init__(**kwargs)


class ReplaceZScore(JSONSerializableModel):
    col = JSONNodeField('col')
    type = JSONNodeField('type')
    low_range = JSONNodeField('low_range')
    high_range = JSONNodeField('high_range')
    low_replace = JSONNodeField('low_replace')
    high_replace = JSONNodeField('high_replace')

    def __init__(self, col, low_range=None, low_replace=None, high_range=None, high_replace=None, **kwargs):
        kwargs.update(dict(col=col, type='zscore', low_range=low_range, low_replace=low_replace,
                           high_range=high_range, high_replace=high_replace))
        super(ReplaceZScore, self).__init__(**kwargs)
