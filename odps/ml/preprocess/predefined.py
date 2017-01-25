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

from ..nodes.transform_nodes import SQLNode
from ..nodes.exporters import *
from ..pipeline.steps import SimpleFunctionStep
from ...utils import deprecated


class NormalizeNode(SQLNode):
    def __init__(self, keep_original=False, fields=None):
        super(NormalizeNode, self).__init__(code_name='normalize')
        self.parameters.update(dict(keep_original=keep_original,
                                    fields=','.join(fields) if fields is not None else None))
        self.keep_original = keep_original
        self.fields = set(fields) if fields is not None else None
        self.reload_on_finish = False

    def generate_sql(self):
        summary_ep = self.inputs['input2'].obj
        summary_data = summary_ep._bind_node.sink

        ep = self.inputs['input1'].obj
        mappings = dict()

        app_fields = [f for f in ep.fields if f.name in self.fields] if self.fields is not None else ep.fields
        for field in app_fields:
            if FieldRole.FEATURE in field.role:
                field_summary = summary_data[field.name]
                if field_summary['min'] != field_summary['max']:
                    mappings[field.name] = '(%s - (%f)) / ((%f) - (%f))' % \
                                           (field.name, field_summary['min'], field_summary['max'], field_summary['min'])
                else:
                    mappings[field.name] = 1 if field_summary['min'] != 0.0 else 0
        mapping_parts = ['%s as %s' % (mappings[f.name], f.name) if f.name in mappings else f.name for f in app_fields]
        if self.keep_original:
            mapping_parts.extend('%s as %s_orig' % (fn, fn) for fn in six.iterkeys(mappings))
        select_stmt = 'select %s from %s' % (', '.join(mapping_parts), ep.table)
        if ep._partitions:
            select_stmt += ' where ' + ep._partitions.to_sql_condition()
        return self.wrap_create_sql(1, select_stmt)


class StandardizeNode(SQLNode):
    def __init__(self, with_means=True, with_std=True, keep_original=False, fields=None):
        super(StandardizeNode, self).__init__(code_name='standardize')
        self.parameters.update(dict(with_means=with_means, with_std=with_std, keep_original=keep_original,
                                    fields=','.join(fields) if fields is not None else None))
        self.with_means = with_means
        self.with_std = with_std
        self.keep_original = keep_original
        self.fields = set(fields) if fields is not None else None
        self.reload_on_finish = False

    def generate_sql(self):
        summary_ep = self.inputs['input2'].obj
        summary_data = summary_ep._bind_node.sink

        ep = self.inputs['input1'].obj
        mappings = dict()

        app_fields = [f for f in ep.fields if f.name in self.fields] if self.fields is not None else ep.fields
        for field in app_fields:
            if FieldRole.FEATURE in field.role:
                field_summary = summary_data[field.name]
                mappings[field.name] = field.name
                if self.with_means:
                    mappings[field.name] = '%s - (%f)' % (mappings[field.name], field_summary['mean'])
                if self.with_std and field_summary['standard_deviation'] != 0:
                    mappings[field.name] = '(%s) / (%f)' % (mappings[field.name], field_summary['standard_deviation'])

        mapping_parts = ['%s as %s' % (mappings[f.name], f.name) if f.name in mappings else f.name for f in app_fields]
        if self.keep_original:
            mapping_parts.extend('%s as %s_orig' % (fn, fn) for fn in six.iterkeys(mappings))
        select_stmt = 'select %s from %s' % (', '.join(mapping_parts), ep.table)
        if ep.partitions:
            select_stmt += ' where ' + ep.partitions.to_sql_condition()
        return self.wrap_create_sql(1, select_stmt)


@deprecated('Normalize deprecated. Please use df.min_max_scale instead.')
def normalize(df, keep_original=False, fields=None):
    """
    Normalize a data set, i.e., (X - min(X)) / (max(X) - min(X))

    :param df: Input data set
    :type df: DataFrame
    :param keep_original: Determine whether input data should be kept. If True, original input data will be appended to the data set with suffix "_orig"
    :type keep_original: bool
    :param fields: Field names to be normalized. If set to None, fields marked as features will be normalized.
    :type fields: list[str]

    :return: Normalized data frame
    :rtype: DataFrame
    """
    # do summary first
    summary_adapter = df._create_summary_adapter(fields)
    normalize_node = NormalizeNode(keep_original, fields)
    adapter_from_df(df)._link_node(normalize_node, 'input1')
    summary_adapter._link_node(normalize_node, 'input2')
    # todo add support for missing odps
    src_adapter = adapter_from_df(df)._duplicate_df_adapter(normalize_node.outputs.get('output1'))
    return src_adapter.df_from_fields(force_create=True)


class Normalize(SimpleFunctionStep):
    def __init__(self):
        super(Normalize, self).__init__(normalize)


@deprecated('Standardize deprecated. Please use df.std_scale instead.')
def standardize(df, with_means=True, with_std=True, keep_original=False, fields=None):
    """
    Standardize a data set.

    :param df: Input data set
    :type df: DataFrame
    :param with_means: Determine whether the output will be subtracted by means
    :type with_means: bool
    :param with_std: Determine whether the output will be divided by standard deviations
    :type with_std: bool
    :param keep_original: Determine whether input data should be kept. If True, original input data will be appended to the data set with suffix "_orig"
    :type keep_original: bool
    :param fields: Field names to be normalized. If set to None, fields marked as features will be normalized.
    :type fields: list[str]

    :return: Standardized data frame
    :rtype: DataFrame
    """
    # do summary first
    summary_adapter = df._create_summary_adapter(fields)
    standardize_node = StandardizeNode(with_means, with_std, keep_original, fields)
    adapter_from_df(df)._link_node(standardize_node, 'input1')
    summary_adapter._link_node(standardize_node, 'input2')
    # todo add support for missing odps
    src_adapter = adapter_from_df(df)._duplicate_df_adapter(standardize_node.outputs.get('output1'))
    return src_adapter.df_from_fields(force_create=True)


class Standardize(SimpleFunctionStep):
    def __init__(self):
        super(Standardize, self).__init__(standardize)
