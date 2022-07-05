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

from __future__ import absolute_import

import collections
import re
import logging
import time
from json import JSONEncoder
from types import FunctionType, MethodType
from copy import deepcopy
from functools import partial
from threading import Thread
from collections import namedtuple

from .. import options
from ..compat import six, getargspec, Iterable
from ..types import Partition
from ..utils import camel_to_underline, TEMP_TABLE_PREFIX
from .enums import FieldContinuity, FieldRole

ML_PACKAGE_ROOT = 'odps.ml'
ML_INTERNAL_PACKAGE_ROOT = 'odps.internal.ml'
TEMP_MODEL_PREFIX = 'pm_'
TABLE_MODEL_PREFIX = 'otm_'
TEMP_TABLE_MODEL_PREFIX = TEMP_TABLE_PREFIX + 'otm_'
TABLE_MODEL_SEPARATOR = '__'

ML_ARG_PREFIX = '_mlattr_'

logger = logging.getLogger(__name__)


class KVConfig(object):
    def __init__(self, kv=':', item=','):
        self.kv_delimiter = kv
        self.item_delimiter = item

    def __repr__(self):
        return 'KVConfig(kv={0}, item={1})'.format(self.kv_delimiter, self.item_delimiter)

    def __eq__(self, other):
        return (self.kv_delimiter, self.item_delimiter) == (other.kv_delimiter, other.item_delimiter)


class MLFields(list):
    def __getslice__(self, i, j):
        return MLFields(super(MLFields, self).__getslice__(i, j))

    def _repr_html_(self):
        html = '<table>'
        html += '<thead><tr><th>Field</th><th>Type</th><th>Roles</th><th>Continuity</th></tr></thead>'
        html += '<tbody>'
        for f in sorted(self):
            html += '<tr>'
            html += '<td>%s</td>' % f.name
            html += '<td>%s</td>' % f._repr_type_()
            html += '<td>%s</td>' % f._repr_role_()
            html += '<td>%s</td>' % ('Continuous' if f.continuity == FieldContinuity.CONTINUOUS else 'Discrete')
            html += '</tr>'
        html += '</tbody></table>'
        return html


class MLField(object):
    """
    Represent table field definition

    :type continuity: FieldContinuity | None
    """
    _role_types = dict((str(e.name).upper(), e) for e in FieldRole)

    def __init__(self, name, field_type, role, continuity=None, is_append=False, is_partition=False, kv_config=None):
        self.name = name
        # field data type
        self.type = field_type.lower()
        # enum in FieldRole
        if role is None:
            role = set()
        self.role = set(role) if isinstance(role, Iterable) else set([role, ])
        # enum in FieldContinuity
        if continuity is not None:
            self.continuity = continuity
        elif self.type == 'double' or self.type == 'bigint':
            self.continuity = FieldContinuity.CONTINUOUS
        else:
            self.continuity = FieldContinuity.DISCRETE
        self.is_append = is_append
        self.is_partition = is_partition
        self.kv_config = kv_config

    def copy(self, role=None):
        if role is None:
            role = set()
        ret = deepcopy(self)
        new_roles = role if isinstance(role, Iterable) else set([role, ])
        ret.role = ret.role | new_roles
        return ret

    @classmethod
    def from_column(cls, column, role=None):
        return cls(column.name, str(column.type).lower(), role)

    @classmethod
    def from_string(cls, col_str):
        sparts = col_str.split(':')
        role = cls.translate_role_name(sparts[2].strip()) if len(sparts) > 2 else None
        return cls(sparts[0].strip(), sparts[1].strip().lower(), role)

    @classmethod
    def translate_role_name(cls, role_name):
        role_name = role_name.strip()
        if not role_name:
            return None
        if isinstance(role_name, FieldRole):
            return role_name
        try:
            role_name = str(role_name).strip().upper()
            return cls._role_types[role_name]
        except KeyError:
            raise KeyError('Role name ' + role_name + ' not registered. ' +
                           'Consider fixing the name or importing the right algorithm packages.')

    @classmethod
    def register_roles(cls, new_enums):
        cls._role_types.update(dict((str(e.name).upper(), e) for e in new_enums))

    def _repr_type_(self):
        return "KV('%s', '%s')" % (self.kv_config.kv_delimiter, self.kv_config.item_delimiter) if self.kv_config else self.type

    def _repr_role_(self):
        return '+'.join(v.name for v in self.role) if self.role else 'None'

    def __repr__(self):
        return '[%s]%s (%s) -> %s' % (self.continuity.name[0] if self.continuity is not None else 'N', self.name,
                                      self._repr_type_(), self._repr_role_())


def is_temp_table(table_name):
    return table_name.startswith(TEMP_TABLE_PREFIX)


def get_function_args(func):
    if isinstance(func, partial):
        arg_names = get_function_args(func.func)[len(func.args):]
        return [arg for arg in arg_names if arg not in func.keywords]
    elif not isinstance(func, (MethodType, FunctionType)) and hasattr(func, '__call__'):
        return get_function_args(getattr(func, '__call__'))
    else:
        return list(getargspec(func).args)


def import_class_member(path):
    import importlib

    package, func = path.rsplit('.', 1)

    outer_package = package.replace('$package_root', '.')
    try:
        imported = importlib.import_module(outer_package, __name__)
        return getattr(imported, func)
    except (ImportError, AttributeError):
        inter_package = package.replace('$package_root', '...internal.ml')
        if inter_package == outer_package:
            raise
        imported = importlib.import_module(inter_package, __name__)
        return getattr(imported, func)

HistogramOutput = namedtuple('HistogramOutput', 'hist bin_edges')


def parse_hist_repr(bins):
    hist, edges = [], []
    try:
        import numpy as np
    except ImportError:
        np = None
    for bar_str in bins.split(';'):
        bin_str, hist_str = bar_str.split(':')
        bin_str = bin_str.strip()
        if bin_str.endswith(']'):
            left, right = bin_str.strip('[]').split(',')
            edges.extend([float(left.strip()), float(right.strip())])
        else:
            left, _ = bin_str.strip('[)').split(',')
            edges.append(float(left.strip()))
        hist.append(float(hist_str.strip()))
    if np is not None:
        hist = np.array(hist)
        edges = np.array(edges)
    return HistogramOutput(hist=hist, bin_edges=edges)


def build_model_table_name(model_name, item_name):
    if model_name.startswith(TEMP_TABLE_PREFIX):
        model_name = model_name[len(TEMP_TABLE_PREFIX):]
        return ''.join((TEMP_TABLE_MODEL_PREFIX, model_name, TABLE_MODEL_SEPARATOR, item_name))
    else:
        return ''.join((TABLE_MODEL_PREFIX, model_name, TABLE_MODEL_SEPARATOR, item_name))
