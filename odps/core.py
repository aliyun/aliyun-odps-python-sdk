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

"""Entry point of pyodps.
"""
import json
from xml.etree import ElementTree
from collections import  Iterable

import six

from .rest import RestClient
from .config import options
from . import models
from . import accounts


DEFAULT_ENDPOINT = 'http://service.odps.aliyun.com/api'
LOG_VIEW_HOST_DEFAULT = 'http://logview.odps.aliyun.com'


class ODPS(object):
    """A database-connection-like object that implements the main
    functionalities of pyodps.
    Most of the methods of this class issue http request and return
    ODPS object.
    """

    def __init__(self, access_id, secret_access_key, project,
                 endpoint=DEFAULT_ENDPOINT, tunnel_endpoint=None):
        self.account = self._build_account(access_id, secret_access_key)
        self.endpoint = endpoint
        self.project = project
        self.rest = RestClient(self.account, endpoint)

        self._projects = models.Projects(client=self.rest)
        self._project = self.get_project()

        self._tunnel_endpoint = tunnel_endpoint
        options.tunnel_endpoint = self._tunnel_endpoint

    def get_project(self, name=None):
        if name is None:
            return self._projects[self.project]
        elif isinstance(name, models.Project):
            return name
        return self._projects[name]

    def exist_project(self, name):
        return name in self._projects

    def list_tables(self, project=None, prefix=None, owner=None):
        project = self.get_project(name=project)
        return project.tables.iterate(name=prefix, owner=owner)

    def get_table(self, name, project=None):
        project = self.get_project(name=project)
        return project.tables[name]

    def exist_table(self, name, project=None):
        project = self.get_project(name=project)
        return name in project.tables

    def create_table(self, name, schema, project=None, comment=None, if_not_exists=False,
                     lifecycle=None, shard_num=None, hub_lifecycle=None):
        project = self.get_project(name=project)
        return project.tables.create(name, schema, comment=comment, if_not_exists=if_not_exists,
                                     lifecycle=lifecycle, shard_num=shard_num,
                                     hub_lifecycle=hub_lifecycle)

    def delete_table(self, name, project=None, if_exists=False):
        project = self.get_project(name=project)
        return project.tables.delete(name, if_exists=if_exists)

    def read_table(self, name, limit, start=0, project=None, partition=None, **kw):
        if not isinstance(name, six.string_types):
            name = name.name
        project = self.get_project(name=project)
        table = project.tables[name]

        compress = kw.pop('compress', False)

        with table.open_reader(partition=partition, **kw) as reader:
            for record in reader.read(start, limit, compress=compress):
                yield record

    def write_table(self, name, *block_records, **kw):
        if not isinstance(name, six.string_types):
            name = name.name

        project = self.get_project(name=kw.pop('project', None))
        table = project.tables[name]
        partition = kw.pop('partition', None)

        if len(block_records) == 1 and isinstance(block_records[0], Iterable):
            blocks = [0, ]
            records_iterators = block_records
        else:
            blocks = block_records[::2]
            records_iterators = block_records[1::2]

            if len(blocks) != len(records_iterators):
                raise ValueError('Should invoke like '
                                 'odps.write_table(block_id, records, block_id2, records2, ..., **kw)')

        with table.open_writer(partition=partition, blocks=blocks, **kw) as writer:
            for block, records in zip(blocks, records_iterators):
                writer.write(block, records)

    def list_resources(self, project=None):
        project = self.get_project(name=project)
        for resource in project.resources:
            yield resource

    def get_resource(self, name, project=None):
        project = self.get_project(name=project)
        return project.resources[name]

    def exist_resource(self, name, project=None):
        project = self.get_project(name=project)
        return name in project.resources

    def open_resource(self, name, project=None, mode='r+', encoding='utf-8'):
        from .models import FileResource

        if isinstance(name, FileResource):
            return name.open(mode=mode)
        return self.get_resource(name, project=project).open(mode=mode, encoding=encoding)

    def create_resource(self, name, typo, project=None, **kwargs):
        project = self.get_project(name=project)
        return project.resources.create(name=name, type=typo, **kwargs)

    def delete_resource(self, name, project=None):
        project = self.get_project(name=project)
        return project.resources.delete(name)

    def list_functions(self, project=None):
        project = self.get_project(name=project)
        for function in project.functions:
            yield function

    def get_function(self, name, project=None):
        project = self.get_project(name=project)
        return project.functions[name]

    def exist_function(self, name, project=None):
        project = self.get_project(name=project)
        return name in project.functions

    def create_function(self, name, project=None, **kwargs):
        project = self.get_project(name=project)
        return project.functions.create(name=name, **kwargs)

    def delete_function(self, name, project=None):
        project = self.get_project(name=project)
        return project.functions.delete(name)

    def list_instances(self, project=None, from_time=None, end_time=None,
                       status=None, only_owner=None):
        project = self.get_project(name=project)
        return project.instances.iterate(
            from_time=from_time, end_time=end_time,
            status=status, only_owner=only_owner)

    def get_instance(self, id_, project=None):
        project = self.get_project(name=project)
        return project.instances[id_]

    def exist_instance(self, id_, project=None):
        project = self.get_project(name=project)
        return id_ in project.instances

    def stop_instance(self, id_, project=None):
        project = self.get_project(name=project)
        project.instances[id_].stop()

    stop_job = stop_instance  # to keep compatible

    def execute_sql(self, sql, project=None, priority=None, running_cluster=None):
        """Run the sql statement and wait for the instance to complete.
        """
        inst = self.run_sql(
            sql, project=project, priority=priority, running_cluster=running_cluster)
        inst.wait_for_success()
        return inst

    def run_sql(self, sql, project=None, priority=None, running_cluster=None):
        task = models.SQLTask(query=sql)

        project = self.get_project(name=project)
        return project.instances.create(task=task, priority=priority,
                                        running_cluster=running_cluster)

    def list_xflows(self, project=None, owner=None):
        project = self.get_project(name=project)
        return project.xflows.iterate(owner=owner)

    def get_xflow(self, name, project=None):
        project = self.get_project(name=project)
        return project.xflows[name]

    def exist_xflow(self, name, project=None):
        project = self.get_project(name=project)
        return name in project.xflows

    def run_xflow(self, xflow_name, xflow_project=None, parameters=None, project=None):
        project = self.get_project(name=project)
        xflow_project = xflow_project or project
        if isinstance(xflow_project, models.Project):
            xflow_project = xflow_project.name
        return project.xflows.execute_xflow(
            xflow_name=xflow_name, xflow_project=xflow_project, project=project, parameters=parameters)

    def execute_xflow(self, xflow_name, xflow_project=None, parameters=None, project=None):
        inst = self.run_xflow(
            xflow_name, xflow_project=xflow_project, parameters=parameters, project=project)
        inst.wait_for_success()
        return inst

    def get_xflow_results(self, instance, project=None):
        project = self.get_project(name=project)

        from .models import Instance
        if not isinstance(instance, Instance):
            instance = project.instances[instance]

        return project.xflows.get_xflow_results(instance)

    def delete_xflow(self, name, project=None):
        project = self.get_project(name=project)

        return project.xflows.delete(name)

    def list_offline_models(self, project=None, prefix=None, owner=None):
        project = self.get_project(name=project)
        return project.offline_models.iterate(name=prefix, owner=owner)

    def get_offline_model(self, name, project=None):
        project = self.get_project(name=project)
        return project.offline_models[name]

    def exist_offline_model(self, name, project=None):
        project = self.get_project(name=project)
        return name in project.offline_models

    def delete_offline_model(self, name, project=None):
        project = self.get_project(name=project)
        return project.offline_models.delete(name)

    def get_logview_address(self, instanceid, hours, project=None):
        project = self.get_project(name=project)
        url = '%s/authorization' % project.resource()

        policy = {
            'expires_in_hours': hours,
            'policy': {
                'Statement': [{
                    'Action': ['odps:Read'],
                    'Effect': 'Allow',
                    'Resource': 'acs:odps:*:projects/%s/instances/%s' % \
                       (self.project, instanceid)
                }],
                'Version': '1',
            }
        }
        headers = {'Content-Type': 'application/json'}
        params = {'sign_bearer_token': ''}
        data = json.dumps(policy)
        res = self.rest.post(url, data, headers=headers, params=params)

        root = ElementTree.fromstring(str(res.content))
        token = root.find('Result').text

        link = LOG_VIEW_HOST_DEFAULT + "/logview/?h=" + self.endpoint + "&p=" \
               + self.project + "&i=" + instanceid + "&token=" + token
        return link

    @classmethod
    def _build_account(cls, access_id, secret_access_key):
        return accounts.AliyunAccount(access_id, secret_access_key)


try:
    from odps.internal.core import *
except ImportError:
    pass