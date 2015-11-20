# encoding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from copy import deepcopy
from xml.etree.ElementTree import Element
import logging
from six import iteritems

logger = logging.getLogger(__name__)


class FieldRole(object):
    FEATURE = 'FEATURE'
    LABEL = 'LABEL'
    WEIGHT = 'WEIGHT'


class FieldContinuity(object):
    CONTINUOUS = 'CONTINUOUS'
    DISCRETE = 'DISCRETE'


class FieldParam(object):
    def __init__(self, name, type, role, continuity=None, is_append=False):
        self.name = name
        # field data type
        self.type = type
        # enum in FieldRole
        self.role = role
        # enum in FieldContinuity
        if continuity is not None:
            self.continuity = continuity
        elif type == 'double':
            self.continuity = FieldContinuity.CONTINUOUS
        else:
            self.continuity = FieldContinuity.DISCRETE
        self.is_append = is_append

    @staticmethod
    def copy(src, role):
        ret = deepcopy(src)
        ret.role = role
        return ret

    def __repr__(self):
        return '[%s]%s (%s) -> %s' % (self.continuity[0] if self.continuity is not None else 'N', self.name,
                                      self.type, self.role)

    def to_xml(self):
        attrs = {'name': self.name, 'type': self.type, 'role': self.role, 'continuity': self.continuity,
                 'is_append': self.is_append}
        return Element("field", {k: v for k, v in iteritems(attrs) if v is not None})


def fetch_table_fields(odps, table_name):
    table = odps.get_table(table_name)
    return [FieldParam(c.name, c.type.name, FieldRole.FEATURE,
                       FieldContinuity.CONTINUOUS if c.type.name == 'double' else FieldContinuity.DISCRETE)
            for c in table.schema.columns]


def is_table_exists(odps, table_name):
    try:
        odps.get_table(table_name)
    except Exception as ex:
        logger.debug('Table not found: {0}'.format(ex))
        return False
    return True


def drop_table(odps, table_name, async=True):
    instance = odps.run_sql('drop table if exists ' + table_name)
    if not async:
        instance.wait_for_success()


def drop_table_partition(odps, table_name, part_name, async=True):
    if not is_table_exists(odps, table_name):
        return
    instance = odps.run_sql('alter table %s drop if exists partition (%s)' % (table_name, part_name))
    if not async:
        instance.wait_for_success()


def drop_offline_model(odps, model_name):
    if not odps.exist_offline_model(model_name):
        return
    odps.delete_offline_model(model_name)


def set_table_lifecycle(odps, table_name, lifecycle, async=True):
    if not is_table_exists(odps, table_name):
        return
    instance = odps.run_sql('alter table %s set lifecycle %s' % (table_name, str(lifecycle)))
    if not async:
        instance.wait_for_success()

