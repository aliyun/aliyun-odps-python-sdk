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

"""ODPS Tasks
"""

import json


class Task(object):
    def __init__(self, name, comment='', properties={}):
        self.name = name
        self.comment = comment
        self.properties = properties


class SQLTask(Task):

    TEMPLATE = '''
    <SQL>
      <Name>task_1</Name>
      <Comment/>
      <Config>
        <Property>
            <Name>settings</Name>
            <Value>{"odps.sql.udf.strict.mode": "true"}</Value>
        </Property>
      </Config>
      <Query><![CDATA[%(sql)s;]]></Query>
    </SQL>
    '''
    
    def __init__(self, query):
        super(SQLTask, self).__init__('')
        self.query = query

    def to_xml(self):
        return self.TEMPLATE % {'sql': self.query}
        

class MoyeTask(Task):

    TEMPLATE = '''
    <MOYE>
      <Name>task_1</Name>
      <Comment/>
      <Plan><![CDATA[%(planstr)s]]></Plan>
    </MOYE>'''

    def __init__(self, plan):
        super(MoyeTask, self).__init__(self, '')
        self.plan = plan

    def to_xml(self):
        p = self.TEMPLATE % {'planstr':(self.plan)}
        return p


class CtrTask(Task):
    """Dreprecated.
    """
    TEMPLATE = '''
    <JOINTASK>
      <Name>task_ctr_join</Name>
      <plan><![CDATA[%s]]></plan>
    </JOINTASK>'''

    def __init__(self, plan):
        super(CtrTask, self).__init__(self, '')
        self.plan = plan
    
    def to_xml(self):
        return self.TEMPLATE % self.plan


class AdminTask(Task):

    TEMPLATE = '''
    <Admin>
       <Name>task_1</Name>
       <Comment/>
       <Config>
         %(params_nodes)s
       </Config>
       <Command><![CDATA[%(command)s]]></Command>
    </Admin>
    '''

    def __init__(self, command, properties):
        super(AdminTask, self).__init__('task_1', properties=properties)
        self.command = command

    def to_xml(self):
        command = self.command
        params_nodes = self._gen_params_nodes()
        return self.TEMPLATE % locals()

    def _gen_params_nodes(self):
        lines = []
        for key, value in self.properties.iteritems():
            lines.append('<Property><Name>%s</Name><Value>%s</Value></Property>' % (key, value))
        return '\n'.join(lines)
