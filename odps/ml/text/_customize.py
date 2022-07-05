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

from functools import partial

from ..expr.exporters import get_ml_input, get_input_field_names, get_input_field_ids, get_input_field_id
from ..expr.mixin import ml_collection_mixin
from ...compat import Enum


class TextFieldRole(Enum):
    DOC_ID = 'DOC_ID'
    DOC_CONTENT = 'DOC_CONTENT'
    WORD = 'WORD'
    SENTENCE = 'SENTENCE'
    WORD_COUNT = 'WORD_COUNT'


@ml_collection_mixin
class TextDFMixIn(object):
    __slots__ = ()

    field_role_enum = TextFieldRole
    non_feature_roles = set([TextFieldRole.DOC_ID, ])

"""
Common NLP exporters
"""
get_doc_id_column = partial(get_input_field_names, field_role=TextFieldRole.DOC_ID)
get_doc_content_column = partial(get_input_field_names, field_role=TextFieldRole.DOC_CONTENT)
get_sentence_column = partial(get_input_field_names, field_role=TextFieldRole.SENTENCE)
get_word_column = partial(get_input_field_names, field_role=TextFieldRole.WORD)
get_word_count_column = partial(get_input_field_names, field_role=TextFieldRole.WORD_COUNT)

get_doc_content_column_ids = partial(get_input_field_ids, field_role=TextFieldRole.DOC_CONTENT)
get_doc_content_column_id = partial(get_input_field_id, field_role=TextFieldRole.DOC_CONTENT)


def normalize_get_append_col_names(expr, param_name, input_name):
    if expr._params[param_name]:
        return expr._params[param_name]
    data_obj = get_ml_input(expr, input_name)
    if data_obj is None:
        return None
    fields = data_obj._ml_fields
    sel_cols = set(expr._params['selectedColNames'].split(',')) if expr._params['selectedColNames']\
        else set(get_doc_content_column(expr, param_name, input_name))
    return [f.name for f in fields if f.name not in sel_cols]
