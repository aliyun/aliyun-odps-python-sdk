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

from __future__ import absolute_import

import re
import logging
import time
from json import JSONEncoder
from types import FunctionType, MethodType
from copy import deepcopy
from functools import partial
from threading import Thread
from collections import namedtuple, Iterable

from .. import options
from ..compat import six, getargspec
from ..tunnel import TableTunnel
from ..types import Partition
from ..utils import camel_to_underline, TEMP_TABLE_PREFIX as GLOBAL_TEMP_PREFIX
from .enums import FieldContinuity, FieldRole

ML_PACKAGE_ROOT = 'odps.ml'
ML_INTERNAL_PACKAGE_ROOT = 'odps.internal.ml'
TEMP_TABLE_PREFIX = GLOBAL_TEMP_PREFIX + 'ml_'
TEMP_MODEL_PREFIX = 'pm_'
TABLE_MODEL_PREFIX = 'otm_'
TABLE_MODEL_SEPARATOR = '__'
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

    @staticmethod
    def copy(src, role=None):
        if role is None:
            role = set()
        ret = deepcopy(src)
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


def split_projected_name(name):
    if '.' in name:
        project, name = name.split('.', 1)
    else:
        project = None
    return project, name


def fetch_table_fields(odps, table, list_partitions=False, project=None):
    if isinstance(table, six.string_types):
        if project is not None:
            _, table_name = split_projected_name(table)
        else:
            project, table_name = split_projected_name(table)
        table = odps.get_table(table_name, project=project)
    ret = [MLField(c.name, c.type.name, FieldRole.FEATURE,
                   continuity=FieldContinuity.CONTINUOUS if c.type.name == 'double' else FieldContinuity.DISCRETE,
                   is_partition=True if isinstance(c, Partition) else False)
           for c in table.schema.columns]
    if list_partitions:
        return ret
    else:
        return [pf for pf in ret if not pf.is_partition]


def is_table_exists(odps, table_name):
    project, table_name = split_projected_name(table_name)
    return odps.exist_table(table_name, project=project)


def drop_table(odps, table_name, async=True):
    project, table_name = split_projected_name(table_name)
    instance = odps.run_sql('drop table if exists ' + table_name, project=project)
    if not async:
        instance.wait_for_success()


def set_table_lifecycle(odps, table_name, lifecycle, async=True, use_threads=True, wait=False):
    def _setter(tables):
        if isinstance(tables, six.string_types):
            tables = [tables, ]
        for table in tables:
            if not is_table_exists(odps, table):
                return
            instance = odps.run_sql('alter table %s set lifecycle %s' % (table, str(lifecycle)))
            if not async:
                instance.wait_for_success()

    if use_threads:
        th = Thread(target=_setter, args=(table_name, ))
        th.start()
        if wait:
            th.join()
    else:
        _setter(table_name)


def gen_model_name(code_name, ts=None, node_id=None):
    if ts is None:
        ts = int(time.time()) if not options.runner.dry_run else 0
    if node_id:
        ts_str = '%d_%d' % (ts % 99991, node_id)
    else:
        ts_str = str(ts % 99991)
    code_name = camel_to_underline(code_name)
    m_name = TEMP_MODEL_PREFIX + code_name + '_' + ts_str
    exc_len = len(code_name) - 32
    if len(code_name) >= exc_len > 0:
        truncated = code_name[0, len(code_name) - exc_len]
        m_name = TEMP_MODEL_PREFIX + truncated + '_' + ts_str
    return m_name


def is_temp_table(table_name):
    return table_name.startswith(TEMP_TABLE_PREFIX)


def is_temp_model(model_name):
    return model_name.startswith(TEMP_MODEL_PREFIX)


def get_function_args(func):
    if isinstance(func, partial):
        arg_names = get_function_args(func.func)[len(func.args):]
        return [arg for arg in arg_names if arg not in func.keywords]
    elif not isinstance(func, (MethodType, FunctionType)) and hasattr(func, '__call__'):
        return get_function_args(getattr(func, '__call__'))
    else:
        return list(getargspec(func).args)


class JSONSerialClassEncoder(JSONEncoder):
    def default(self, o):
        if hasattr(o, 'serial'):
            return o.serial()
        else:
            super(JSONSerialClassEncoder, self).default(o)


INF_PATTERN = re.compile(r'(: *)(inf|-inf)( *,| *\})')


def replace_json_infs(s):
    def repl(m):
        mid = m.group(2)
        if mid == 'inf':
            mid = 'Infinity'
        elif mid == '-inf':
            mid = '-Infinity'
        return m.group(1) + mid + m.group(3)

    return INF_PATTERN.sub(repl, s)


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
    if np:
        hist = np.array(hist)
        edges = np.array(edges)
    return HistogramOutput(hist=hist, bin_edges=edges)
