#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import itertools
import json

from ..compat import six
from ..errors import NoSuchObject
from .utils import build_model_table_name, TABLE_MODEL_PREFIX, TABLE_MODEL_SEPARATOR, \
    TEMP_TABLE_PREFIX, TEMP_TABLE_MODEL_PREFIX


class TablesModelObject(object):
    __slots__ = '_odps', 'name', '_parent', '_tables', '_params'

    def __init__(self, **kwargs):
        self._odps = kwargs.get('_odps')
        self.name = kwargs.get('name')

        self._parent = kwargs.get('_parent')
        if self._parent is None:
            project_name = kwargs.get('project')
            self._parent = self._odps.get_project(project_name)

        self._tables = kwargs.get('_tables') or dict()
        if not self._tables:
            prefix = build_model_table_name(self.name, '')
            for tb in self._odps.list_tables(project=self.project.name, prefix=prefix):
                item_name = tb.name[len(prefix):]
                self._tables[item_name] = tb
        if not isinstance(self._tables, dict):
            tables_dict = dict()
            prefix = build_model_table_name(self.name, '')
            for table_name in self._tables:
                tables_dict[table_name] = self._odps.get_table(prefix + table_name, project=self.project.name)
            self._tables = tables_dict

        self._params = kwargs.get('_params', None)

    @property
    def odps(self):
        return self._odps

    @property
    def project(self):
        return self._parent

    @property
    def tables(self):
        return self._tables

    def exists(self):
        if not self._tables:
            return False
        try:
            for t in six.itervalues(self._tables):
                t.reload()
        except NoSuchObject:
            return False
        return True

    @property
    def params(self):
        if self._params is None:
            try:
                tb = next(t for t in six.itervalues(self._tables) if self._odps.exist_table(t.name))
                if tb.comment:
                    self._params = json.loads(tb.comment)
                else:
                    self._params = dict()
            except StopIteration:
                pass
        return self._params


def _list_tables_model(self, prefix='', project=None):
    """
    List all TablesModel in the given project.

    :param prefix: model prefix
    :param str project: project name, if you want to look up in another project
    :rtype: list[str]
    """
    tset = set()

    if prefix.startswith(TEMP_TABLE_PREFIX):
        prefix = TEMP_TABLE_MODEL_PREFIX + prefix[len(TEMP_TABLE_PREFIX):]
        it = self.list_tables(project=project, prefix=prefix)
    else:
        it = self.list_tables(project=project, prefix=TABLE_MODEL_PREFIX + prefix)
        if TEMP_TABLE_PREFIX.startswith(prefix):
            new_iter = self.list_tables(project=project, prefix=TEMP_TABLE_MODEL_PREFIX)
            it = itertools.chain(it, new_iter)

    for table in it:
        if TABLE_MODEL_SEPARATOR not in table.name:
            continue
        if not table.name.startswith(TEMP_TABLE_MODEL_PREFIX) and not table.name.startswith(TABLE_MODEL_PREFIX):
            continue
        model_name, _ = table.name.rsplit(TABLE_MODEL_SEPARATOR, 1)
        if model_name.startswith(TEMP_TABLE_MODEL_PREFIX):
            model_name = TEMP_TABLE_PREFIX + model_name[len(TEMP_TABLE_MODEL_PREFIX):]
        else:
            model_name = model_name[len(TABLE_MODEL_PREFIX):]
        if model_name not in tset:
            tset.add(model_name)
            yield TablesModelObject(_odps=self, name=model_name, project=project)


def _get_tables_model(self, name, tables=None, project=None):
    return TablesModelObject(_odps=self, name=name, _tables=tables, project=project)


def _exist_tables_model(self, name, project=None):
    return _get_tables_model(self, name, project=project).exists()


def _delete_tables_model(self, name, project=None, async=False, if_exists=False):
    prefix = build_model_table_name(name, '')
    for tb in self.list_tables(project=project, prefix=prefix):
        tb.drop(async=async, if_exists=if_exists)


def install_plugin():
    from .. import ODPS
    ODPS.list_tables_model = _list_tables_model
    ODPS.get_tables_model = _get_tables_model
    ODPS.exist_tables_model = _exist_tables_model
    ODPS.delete_tables_model = _delete_tables_model

install_plugin()
