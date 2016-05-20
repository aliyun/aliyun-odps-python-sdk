# -*- coding: utf-8 -*-
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

import json
import time
import uuid
import logging

from .common import init_frontend_scripts, build_unicode_control
from ..compat import OrderedDict, six
from ..models.instance import Instance
from ..serializers import JSONSerializableModel, JSONNodeField, JSONNodesReferencesField

logger = logging.getLogger(__name__)

"""
Progress Storage
"""

PROGRESS_REPO = dict()


class _StageProgressJSON(JSONSerializableModel):
    name = JSONNodeField('name')
    backup_workers = JSONNodeField('backup_workers', parse_callback=int)
    terminated_workers = JSONNodeField('terminated_workers', parse_callback=int, default=0)
    running_workers = JSONNodeField('running_workers', parse_callback=int, default=0)
    total_workers = JSONNodeField('total_workers', parse_callback=int, default=0)
    input_records = JSONNodeField('input_records', parse_callback=int, default=0)
    output_records = JSONNodeField('output_records', parse_callback=int, default=0)
    finished_percentage = JSONNodeField('finished_percentage', parse_callback=int, default=0)

    def __init__(self, **kwargs):
        super(_StageProgressJSON, self).__init__(**kwargs)


class _TaskProgressJSON(JSONSerializableModel):
    name = JSONNodeField('name')
    status = JSONNodeField('status', parse_callback=lambda v: Instance.Task.TaskStatus(v.upper()),
                           serialize_callback=lambda v: v.value)
    stages = JSONNodesReferencesField(_StageProgressJSON, 'stages')


class _InstanceProgressJSON(JSONSerializableModel):
    id = JSONNodeField('id')
    logview = JSONNodeField('logview')
    status = JSONNodeField('status', parse_callback=lambda v: Instance.Status(v.upper()),
                           serialize_callback=lambda v: v.value)
    tasks = JSONNodeField('tasks', parse_callback=lambda v: _InstanceProgressJSON._parse_tasks(v),
                          serialize_callback=lambda v: [d.serial() for d in six.itervalues(v)])

    @staticmethod
    def _parse_tasks(obj):
        return OrderedDict([(o['name'], _TaskProgressJSON.parse(o)) for o in obj])


class _InstancesProgressJSON(JSONSerializableModel):
    name = JSONNodeField('name')
    key = JSONNodeField('key')
    gen_time = JSONNodeField('gen_time')
    logview = JSONNodeField('logview')
    instances = JSONNodeField('instances', parse_callback=lambda v: _InstancesProgressJSON._parse_instances(v),
                              serialize_callback=lambda v: [d.serial() for d in six.itervalues(v)])

    @staticmethod
    def _parse_instances(obj):
        return OrderedDict([(o['id'], _InstanceProgressJSON.parse(o)) for o in obj])

    def update_instance(self, inst):
        self.instances[inst.id] = inst


def create_instance_group(name):
    key = '%x_%s' % (int(time.time()), str(uuid.uuid4()).lower())
    group_json = _InstancesProgressJSON(name=name, key=key, instances=OrderedDict())
    group_json.gen_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    PROGRESS_REPO[key] = group_json
    return key


def reload_instance_status(odps, group_id, instance_id):
    if group_id not in PROGRESS_REPO:
        raise KeyError('Instance group ID not exist.')
    group_json = PROGRESS_REPO[group_id]

    if instance_id in group_json.instances:
        inst_json = group_json.instances[instance_id]
        if inst_json.status == Instance.Status.TERMINATED:
            return
    else:
        inst_json = _InstanceProgressJSON(id=instance_id, tasks=dict())
        group_json.instances[instance_id] = inst_json

    group_json.gen_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    old_status = inst_json.status

    sub_inst = odps.get_instance(instance_id)
    inst_json.status = sub_inst.status
    inst_json.logview = sub_inst.get_logview_address()

    if old_status != Instance.Status.TERMINATED:
        for task_name, task in six.iteritems(sub_inst.get_task_statuses()):
            if task_name in inst_json.tasks:
                task_json = inst_json.tasks[task_name]
                task_json.status = task.status
                if task.status not in set([Instance.Task.TaskStatus.RUNNING, Instance.Task.TaskStatus.WAITING]):
                    continue
            else:
                task_json = _TaskProgressJSON(name=task_name, status=task.status, stages=[])
            inst_json.tasks[task_name] = task_json
            task_json.stages = []

            try:
                task_prog = sub_inst.get_task_progress(task_name)
            except Exception:
                continue

            for stage in task_prog.stages:
                stage_json = _StageProgressJSON()
                for field_name in six.iterkeys(_StageProgressJSON.__fields):
                    if hasattr(stage, field_name):
                        val = getattr(stage, field_name)
                        if val is not None:
                            setattr(stage_json, field_name, val)
                task_json.stages.append(stage_json)


def fetch_instance_group(group_id):
    if group_id not in PROGRESS_REPO:
        raise KeyError('Instance group ID not exist.')
    return PROGRESS_REPO[group_id]


"""
User Interface
"""


try:
    from ..console import widgets, ipython_major_version, in_ipython_frontend
    if ipython_major_version < 4:
        from IPython.utils.traitlets import Unicode, List
    else:
        from traitlets import Unicode, List
    from IPython.display import display
except Exception:
    InstancesProgress = None
else:
    if in_ipython_frontend():
        class InstancesProgress(widgets.DOMWidget):
            _view_name = build_unicode_control('InstancesProgress', sync=True)
            _view_module = build_unicode_control('pyodps/progress', sync=True)
            text = build_unicode_control(sync=True)

            def __init__(self, **kwargs):
                """Constructor"""
                init_frontend_scripts()
                widgets.DOMWidget.__init__(self, **kwargs)  # Call the base.

                # Allow the user to register error callbacks with the following signatures:
                #    callback()
                #    callback(sender)
                self.errors = widgets.CallbackDispatcher(accepted_nargs=[0, 1])

            def update(self):
                self.send(json.dumps(dict(action='update', content=[])))

            def update_group(self, group_jsons):
                if isinstance(group_jsons, six.string_types):
                    group_jsons = [group_jsons, ]
                self.send(json.dumps(dict(action='update', content=group_jsons)))

            def delete_group(self, group_keys):
                if isinstance(group_keys, six.string_types):
                    group_keys = [group_keys, ]
                self.send(json.dumps(dict(action='delete', content=group_keys)))

            def clear_groups(self):
                self.send(json.dumps(dict(action='clear')))
    else:
        InstancesProgress = None


class ProgressGroupUI(object):
    def __init__(self, ipython_widget=False):
        self._ipython_widget = ipython_widget and InstancesProgress
        self._widget = None
        self._group_keys = set()
        self._text = ''

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value
        self._update_text()

    def add_keys(self, keys):
        if isinstance(keys, six.string_types):
            keys = [keys, ]
        self._group_keys.update(keys)
        self._update_group(keys)

    def remove_keys(self, keys):
        if isinstance(keys, six.string_types):
            keys = [keys, ]
        self._group_keys -= set(keys)
        self._widget.delete_group(keys)

    def clear_keys(self):
        self._group_keys = set()
        self._widget.clear_groups()

    def _update_text(self):
        if self._ipython_widget:
            if not self._widget:
                self._widget = InstancesProgress()
                display(self._widget)
            self._widget.text = self._text
        self._widget.update()

    def _update_group(self, keys):
        if self._ipython_widget:
            if not self._widget:
                self._widget = InstancesProgress()
                display(self._widget)
        if isinstance(keys, six.string_types):
            keys = [keys, ]
        data = [fetch_instance_group(key).serialize() for key in keys]
        self._widget.update_group(data)

    def update(self):
        self._update_text()
        data = [fetch_instance_group(key).serialize() for key in self._group_keys]
        self._widget.update_group(data)

    def close(self):
        if self._ipython_widget and self._widget:
            self._widget.close()

