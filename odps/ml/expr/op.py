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

import copy

from ...compat import six, reduce
from ...df.expr.collections import CollectionExpr
from ..utils import MLField, FieldRole, FieldContinuity


"""
Base Operation
"""


class DFOperation(object):
    def execute(self, sources, target):
        """
        :type sources: list[DFAdapter]
        :type target: DFAdapter
        """
        fields = self._get_fields_list_from_eps(sources)
        target._ml_fields = copy.deepcopy(fields[0])

    @staticmethod
    def _get_fields_list_from_eps(dfs):
        def _filter_ml_fields(df):
            if isinstance(df, CollectionExpr):
                fields = set(c.name for c in df.schema.columns)
            else:
                fields = set([df.name])
            return [f for f in df._ml_fields if f.name in fields]

        return copy.deepcopy([_filter_ml_fields(ep) for ep in dfs])

    @staticmethod
    def _norm_name_set(name_set):
        if isinstance(name_set, six.string_types):
            return set(v.strip() for v in name_set.split(','))
        elif isinstance(name_set, (list, tuple)):
            return set(name_set)
        else:
            return name_set

    @staticmethod
    def _set_singleton_role(fields, role_mapping):
        roles = set(six.itervalues(role_mapping))
        for f in fields:
            if f.name in role_mapping:
                yield f.copy(role_mapping[f.name])
            elif roles & f.role:
                ret_field = f.copy()
                ret_field.role -= roles
                yield ret_field
            else:
                yield copy.deepcopy(f)
        for fname in set(six.iterkeys(role_mapping)) - set(f.name for f in fields):
            yield MLField(fname, 'EXPECTED', role_mapping[fname])

    @classmethod
    def _remove_field_roles(cls, fields, name_set, role):
        name_set = cls._norm_name_set(name_set)
        for f in fields:
            if f.name in name_set:
                ret_field = f.copy()
                ret_field.role -= set([role, ])
                yield ret_field
            else:
                yield copy.deepcopy(f)
        # if name does not appear in existing fields, we assume that it will be generated
        for name in name_set - set(f.name for f in fields):
            yield MLField(name, 'EXPECTED', role)

    @classmethod
    def _clear_field_roles(cls, fields, name_set):
        name_set = cls._norm_name_set(name_set)
        for f in fields:
            if f.name in name_set:
                ret_field = f.copy()
                ret_field.role = set()
                yield ret_field
            else:
                yield copy.deepcopy(f)
        # if name does not appear in existing fields, we assume that it will be generated
        for name in name_set - set(f.name for f in fields):
            yield MLField(name, 'EXPECTED', None)

    @classmethod
    def _add_field_roles(cls, fields, name_set, role, augment):
        name_set = cls._norm_name_set(name_set)
        for f in fields:
            if f.name in name_set:
                yield f.copy(role)
            elif role in f.role:
                if augment:
                    yield f.copy()
                else:
                    ret_field = f.copy()
                    ret_field.role.remove(role)
                    yield ret_field
            else:
                yield copy.deepcopy(f)
        # if name does not appear in existing fields, we assume that it will be generated
        for name in name_set - set(f.name for f in fields):
            yield MLField(name, 'EXPECTED', role)


"""
Attribution Operations
"""


class BatchRoleOperation(DFOperation):
    def __init__(self, fields, role, augment):
        self.feature_names = fields
        self.augment = augment
        self.role = role

    def execute(self, sources, target):
        """
        :type sources: list[DFAdapter]
        :type target: DFAdapter
        """
        fields = self._get_fields_list_from_eps(sources)
        feature_set = set(self.feature_names)
        target._ml_fields = list(self._add_field_roles(fields[0], feature_set, self.role, self.augment))


class ExcludeFieldsOperation(DFOperation):
    def __init__(self, fields):
        self.exclude_names = fields

    def execute(self, sources, target):
        """
        :type sources: list[DFAdapter]
        :type target: DFAdapter
        """
        fields = self._get_fields_list_from_eps(sources)
        feature_set = set(self.exclude_names)
        target._ml_fields = list(self._clear_field_roles(fields[0], feature_set))


class SingletonRoleOperation(DFOperation):
    def __init__(self, field_mapping, clear_feature=False):
        self.field_mapping = field_mapping
        self.clear_feature = clear_feature

    def execute(self, sources, target):
        """
        :type sources: list[DFAdapter]
        :type target: DFAdapter
        """
        fields = self._get_fields_list_from_eps(sources)
        ret_fields = fields[0]
        if self.clear_feature:
            ret_fields = list(self._remove_field_roles(ret_fields, set(six.iterkeys(self.field_mapping)),
                                                       FieldRole.FEATURE))
        target._ml_fields = list(self._set_singleton_role(ret_fields, self.field_mapping))


class FieldContinuityOperation(DFOperation):
    def __init__(self, continuity):
        self.continuity = dict((k, FieldContinuity.CONTINUOUS if v else FieldContinuity.DISCRETE)
                               for k, v in six.iteritems(continuity))

    def execute(self, sources, target):
        """
        :type sources: list[DFAdapter]
        :type target: DFAdapter
        """
        fields = self._get_fields_list_from_eps(sources)
        for f in fields[0]:
            if f.name in self.continuity:
                f.continuity = self.continuity[f.name]
        for field_name in set(six.iterkeys(self.continuity)) - set(f.name for f in fields[0]):
            fields[0].append(MLField(field_name, 'EXPECTED', None, continuity=self.continuity[field_name]))
        target._ml_fields = fields[0]


class FieldKVConfigOperation(DFOperation):
    def __init__(self, kv_config):
        self.kv_config = kv_config

    def execute(self, sources, target):
        """
        :type sources: list[DFAdapter]
        :type target: DFAdapter
        """
        fields = self._get_fields_list_from_eps(sources)
        for f in fields[0]:
            if f.name in self.kv_config:
                f.kv_config = self.kv_config[f.name]
        for field_name in set(six.iterkeys(self.kv_config)) - set(f.name for f in fields[0]):
            fields[0].append(MLField(field_name, 'EXPECTED', None, kv_config=self.kv_config[field_name]))
        target._ml_fields = fields[0]


"""
Modify Operations
"""


class StaticFieldChangeOperation(DFOperation):
    def __init__(self, fields, is_append=False):
        self.fields = [fields, ] if isinstance(fields, MLField) else fields
        self.is_append = is_append

    def execute(self, sources, target):
        """
        :type sources: list[DFAdapter]
        :type target: DFAdapter
        """
        existing_ml_fields = self._get_fields_list_from_eps(sources)
        dup_ml_fields = [f.copy() for f in self.fields]
        if self.is_append:
            src = list([copy.deepcopy(f) for f in existing_ml_fields[0]])
            src.extend(dup_ml_fields)
            target._ml_fields = src
        else:
            target._ml_fields = self.fields


class ProgrammaticFieldChangeOperation(DFOperation):
    def __init__(self, evaluator, is_append=False):
        self.is_append = is_append
        self.evaluator = evaluator

    def execute(self, sources, target):
        """
        :type sources: list[DFAdapter]
        :type target: DFAdapter
        """
        existing_ml_fields = self._get_fields_list_from_eps(sources)

        def gen_field(field_str):
            parts = field_str.split(':')
            if len(parts) > 2:
                roles = set(MLField.translate_role_name(rn) for rn in parts[2].strip().split('#')) - set([None])
            else:
                roles = set([FieldRole.FEATURE, ])
            return MLField(parts[0].strip(), parts[1].strip(), roles)

        fields = [gen_field(fstr) for fstr in self.evaluator().split(',')
                  if fstr.strip() != '']

        dup_ml_fields = [f.copy(None) for f in fields]
        if self.is_append:
            src = list([copy.deepcopy(f) for f in existing_ml_fields[0]])
            src.extend(dup_ml_fields)
            target._ml_fields = src
        else:
            target._ml_fields = fields


class MergeFieldsOperation(DFOperation):
    def __init__(self, auto_rename, selected_cols, excluded_cols):
        self._auto_rename = auto_rename
        self._selected_cols = selected_cols
        self._excluded_cols = excluded_cols

    def execute(self, sources, target):
        """
        :type sources: list[DFAdapter]
        :type target: DFAdapter
        """
        existing_fields = self._get_fields_list_from_eps(sources)

        def trans_columns(tid, fields):
            sel_cols = self._selected_cols.get(tid)
            exc_cols = self._excluded_cols.get(tid)
            for field in fields:
                if field.role is None:
                    continue
                if sel_cols and field.name not in sel_cols:
                    continue
                if exc_cols and field.name in exc_cols:
                    continue
                new_col = field.copy(field.role)
                if self._auto_rename:
                    new_col.name = 't%d_%s' % (tid, field.name)
                yield new_col

        new_fields = map(lambda tp: list(trans_columns(tp[0], tp[1])), enumerate(existing_fields))
        target._ml_fields = list(reduce(lambda fa, fb: fa + fb, new_fields, []))
        target._ml_uplink = sources
