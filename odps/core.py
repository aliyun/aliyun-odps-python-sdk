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

from .rest import RestClient
from . import models
from . import tasks
from . import accounts


DEFAULT_ENDPOINT = "http://service.odps.aliyun.com/api"
LOG_VIEW_HOST_DEFAULT = "http://logview.odps.aliyun.com"



class ODPS(object):
    """A database-connection-like object that implements the main
    functionalities of pyodps.
    Most of the methods of this class issue http request and return
    ODPS object.
    """

    def __init__(self, access_id, secret_access_key, project,
                 endpoint=DEFAULT_ENDPOINT):
        self.account = self._build_account(access_id, secret_access_key)
        self.endpoint = endpoint
        self.project = project
        self.rest = RestClient(self.account, endpoint)

    def _get_projects(self, marker=None, maxitems=None, name=None):
        params = {}
        if marker:
            params["marker"] = marker
        if maxitems:
            params["maxitems"] = maxitems
        if name:
            params["name"] = name
        res = self.rest.projects.get(params=params)
        obj = models.Projects.create(self, res)
        return obj

    def _get_project_extended(self, name):
        res = self.rest.projects[name].get(params={"extended": "null"})
        obj = models.ProjectExtended.create(self, res)
        return obj

    def get_project(self, name):
        res = self.rest.projects[name].get()
        obj = models.Project.create(self, res)
        return obj

    def get_tables(self):
        pass

    def get_table(self, name):
        res = self._project.tables[name].get()
        return models.Table.create(self, res)

    def list_resources(self):
        pass

    def open_resource(self, name, mode='rw'):
        pass

    def get_instances(self):
        res = self._project.instances.get()

    def get_instance(self, id):
        res = self._project.instances[id].get()
        return models.Instance.create(self, res, id)

    def execute_sql(self, sql):
        """Run the sql statement and wait for the instance to complete.
        """
        inst = self.run_sql(sql)
        inst.wait_for_success()
        return inst

    def run_sql(self, sql):
        sqltask = tasks.SQLTask(sql)
        inst = self._run_task(sqltask)
        return inst

    def _run_task(self, task):
        job = models.Job('arbitrary_job', 9, "", [])
        job.add_task(task)
        return self.create_instance(job)

    def create_instance(self, job):
        xml = job.to_xml()
        headers = {'Content-Type': 'application/xml'}
        res = self._project.instances.post(data=xml, headers=headers)
        if res.headers.get('content-length', 0) != '0' and 'Instance' not in res.content:
            #TODO resolve the exception before raise
            raise Exception(res.content)
        location = res.headers.get('location')
        assert location
        noncare, instanceid = location.rsplit('/', 1)
        if 'Instance' in res.content:
            return models.Instance.create(self, res, instanceid)
        return self.get_instance(instanceid)

    def stop_job(self, instanceid):
        xml = '''
        <Instance>
          <Status>Terminated</Status>
        </Instance>'''
        headers = {'Content-Type': 'application/xml'}
        self._project.instances[instanceid].put(data = xml, headers=headers)
        return self.get_instance(instanceid)

    def get_logview_address(self, instanceid, hours):
        policy = {
            'expires_in_hours': str(hours),
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
        params = {'sign_bearer_token': 'null'}
        res = self._project.authorization.post(data = json.dumps(policy),
                                               headers = headers,
                                               params = params)
        root = ElementTree.fromstring(str(res.content))
        token = root.find('Result').text
        link = LOG_VIEW_HOST_DEFAULT + "/logview/?h=" + self.endpoint + "&p=" \
               + self.project + "&i=" + instanceid + "&token=" + token
        return link

    def read_table(self, name):
        """Read the table given by name and return a list of records as
        tuples.
        """
        table = self.get_table(name)
        cols = []
        for c in table.columns:
            cols.append(c['name'])
        params = {'data':'', 'cols': ','.join(cols)}
        res = self._project.tables[name].get(params=params)
        def skip_header(records):
            first = True
            for i in records:
                if first:
                    first = False
                else:
                    yield i
        return skip_header(_convert_to_tuples(res.text))

    @property
    def _project(self):
        return self.rest.projects[self.project]

    def _build_account(self, access_id, secret_access_key):
        return accounts.AliyunAccount(access_id, secret_access_key)


def _convert_to_tuples(data):
    for line in data.splitlines():
        cols = [eval(s) for s in line.split(',')]
        yield tuple(cols)
