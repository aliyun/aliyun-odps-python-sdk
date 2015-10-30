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

"""ODPS object models.
This module contains two things:
  1. Definitions of all ODPS objects.
  2. XML parsing code for translating http response to corresponding object.
"""

import json
import rfc822
# import xml.sax
# import xml.sax.handler
import time
import logging
from xml.etree import ElementTree
from datetime import datetime

from . import (_models, utils)
from errors import ODPSError

LOG = logging.getLogger(__name__)

STRING_MAX_LENGTH = 2097152
DATETIME_MAX_TICKS = 2534022719.99
DATETIME_MIN_TICKS = -6213579840.00
BIGINT_MAX = 9223372036854775807L
BIGINT_MIN = -9223372036854775808L

# def objectify(content):
#     """Parse xml and convert it to a python object.
#     """
#     handler = ODPSXmlHandler()
#     xml.sax.parseString(content, handler)
#     return handler.obj


def to_time_field(timestamp):
    if isinstance(timestamp, int):
        return datetime.fromtimestamp(timestamp)
    elif timestamp == '':
        return None
    else:
        tp = rfc822.parsedate_tz(timestamp)
        itp = rfc822.mktime_tz(tp)
        return datetime.fromtimestamp(itp)


class BaseModel(object):

    fields = ()

    def __init__(self, **kwargs):
        for i in self.fields:
            if i not in kwargs:
                raise AttributeError('no such attribute: ' + i)
        self.__dict__.update(kwargs)
        self.odps = None


class Projects(BaseModel):

    fields = ('marker', 'maxitems', 'projects')

    @classmethod
    def create(clz, o, res):
        root = ElementTree.fromstring(str(res.content))
        
        marker = root.find('Marker').text
        maxitems = root.find('MaxItems').text

        projects = []
        for project in root.findall('Project'):
            name = project.find('Name').text
            comment = project.find('Comment').text
            owner = project.find('Owner').text
            state = project.find('State').text
            last_modified_time = project.find('LastModifiedTime').text
            create_time = project.find('CreationTime').text
            obj = Project(
                name=name, comment=comment, owner=owner, state=state,
                last_modified_time=last_modified_time, create_time=create_time,
                project_group_name=None, properties=None)
            projects.append(obj)

        obj = clz(marker=marker, maxitems=maxitems, projects=projects)
        obj.odps = o
        return obj

class Project(BaseModel):

    fields = ('name', 'owner', 'comment', 'create_time', 'last_modified_time', 'state', 'project_group_name', 'properties')
    
    @classmethod
    def create(clz, o, res):
        owner = res.headers['x-odps-owner']
        create_time = res.headers['x-odps-creation-time']
        last_modified_time = res.headers['Last-Modified']

class ProjectExtended(BaseModel):

    fields = ('used_quota_logical_size', 'used_quota_physical_size', 'failover_logical_size', 'failover_physical_size',
              'fuxi_job_temp_logical_size', 'fuxi_job_temp_physical_size', 'resource_logical_size', 'resource_physical_size',
              'table_logical_size', 'table_physical_size', 'table_backup_logical_size', 'table_backup_physical_size',
              'temp_resource_logical_size', 'temp_resource_physical_size', 'volume_logical_size', 'volume_physical_size'
    )

    @classmethod
    def create(clz, o, res):
        root = ElementTree.fromstring(str(res.content))

        propertys = {}
        for property in root.findall('./ExtendedProperties/Property'):
            name = property.find('Name').text
            value = property.find('Value').text
            propertys[utils.camel_to_underscore(name)] = value

        obj = clz(**propertys)
        obj.odps = o
        return obj

class Instance(object):

    ST_RUNNING, ST_SUSPENDED, ST_TERMINATED = range(3)

    (ST_TASK_WAITING,
     ST_TASK_RUNNING,
     ST_TASK_SUCCESS,
     ST_TASK_FAILED,
     ST_TASK_SUSPENDED,
     ST_TASK_CANCELLED) = range(6)

    @classmethod
    def create(clz, o, res, id):
        cons = {'id': id}
        cons['owner'] = res.headers['x-odps-owner']
        cons['start_time'] = to_time_field(res.headers['x-odps-start-time'])
        cons['end_time'] = to_time_field(res.headers['x-odps-end-time'])
        obj = clz(**cons)
        obj.odps = o
        if res.content:
            root = ElementTree.fromstring(res.content)
            assert root.tag == 'Instance'
            for child in root:
                if child.tag == 'Tasks':
                    for task_node in child:
                        for _attr in task_node:
                            if _attr.tag == 'Name':
                                task_name = _attr.text
                            elif _attr.tag == 'Result':
                                task_result = _attr.text
                        obj._task_results[task_name] = task_result
        return obj

    def __init__(self, id, owner, start_time, end_time):
        self.id = id
        self.owner = owner
        self.start_time = start_time
        self.end_time = end_time
        self._status = None
        self._task_status = {}
        self._task_results = {}

    @classmethod
    def get_status_name(cls, status):
        if status < 0 or status > 2:
            raise ValueError('status must be RUNNING, SUSPENDED or TERMINATED')
        return ['Running', 'Suspended', 'Terminated'][status]

    def get_status(self, xml_content=None):
        # TODO: Status caching.
        if xml_content is None:
            res = self.odps._project.instances[self.id].get()
            xml_content = res.content
        status = self.__parse_status(xml_content)
        self._status = getattr(self, 'ST_' + status.upper())

    def get_task_status(self):
        # These is a bug in restful api. Some tasks (SQL to be known) return
        # a corrupted response like this:
        #  <Instance><Status>Running</Status><Tasks/></Instance>
        # Since requests does not support query param without value, so we fake
        # one.
        params = {'taskstatus': 1}
        retry = 10
        while self._status != self.ST_TERMINATED:
            resp = self.odps._project.instances[self.id].get(params=params)
            task_status = _models.TaskStatuses.parse(resp.content)
            if retry and not task_status.tasks:
                # WARN: Until response task status.
                time.sleep(0.5)
                retry -= 1
                continue
            for _task_status in task_status.tasks:
                status = getattr(self, 'ST_TASK_'+_task_status.status.upper())
                self._task_status[_task_status.name] = status
            self.get_status(resp.content)
            break
        return self._task_status.copy()

    def get_task_result(self, name):
        result = self._task_results.get(name)
        if result:
            return result
        params = {
            'result': 1,
            'taskname': name
        }
        resp = self.odps._project.instances[self.id].get(params=params)
        root = ElementTree.fromstring(resp.content)
        st = root.getchildren()[0].getchildren()[0].getchildren()[1]
        assert st.tag == 'Result' 
        return st.text

    def get_task_results(self, task_name=None):
        if self._task_results:
            return self._task_results.values()[0]
        id_result = "%s?result&taskname=task_1" % self.id  # xxx
        res = self.odps._project.instances[id_result].get()
        root = ElementTree.fromstring(res.content)
        st = root.getchildren()[0].getchildren()[0].getchildren()[1]
        assert st.tag == 'Result'
        return st.text

    def get_task_detail_results(self, tasl_name=None):
        id_detail = "%s?detail&taskname=task_1" % self.id  # xxx
        detail_re = self.odps._project.instances[id_detail].get().text
        return detail_re

    def get_all_task_results(self):
        pass

    def wait_for_completion(self, interval=1):
        while True:
            for name, status in self.get_task_status().iteritems():
                if self._task_is_running_or_waiting(status):
                    LOG.debug('Task %s is running or waiting: %d' %
                              (name, status))
                    time.sleep(interval)
                    break
            else:
                LOG.debug('Instance is terminated: %s' % self._task_status)
                break

    def wait_for_success(self, interval=1):
        self.wait_for_completion(interval)
        for name, status in self.get_task_status().iteritems():
            if status != self.ST_TASK_SUCCESS:
                result = self.get_task_result(name)
                raise ODPSError('Task %s failed: %s, status: %s' %
                                (name, result, status), None)

    def _task_is_running_or_waiting(self, status):
        return status in (self.ST_TASK_WAITING, self.ST_TASK_RUNNING)

    def __parse_status(self, xml):
        root = ElementTree.fromstring(xml)
        assert root.tag == 'Instance'
        st = root.getchildren()[0]
        assert st.tag == 'Status'
        return st.text


class Table(BaseModel):

    fields = ('name', 'comment', 'owner', 'create_time', 
              'last_ddl_time', 'last_modified_time', 
              'is_view', 'size', 'life', 'columns',)

    @classmethod
    def create(clz, o, res):
        root = ElementTree.fromstring(res.content)
        assert root.tag == 'Table'
        cons = {}
        for child in root:
            if child.tag == 'Name':
                cons['name'] = child.text
            elif child.tag == 'Schema':
                data = json.loads(child.text)
                cons['comment'] = data['comment']
                cons['owner'] = data['owner']
                cons['create_time'] = to_time_field(data['createTime'])
                cons['last_ddl_time'] = to_time_field(data['lastDDLTime'])
                cons['last_modified_time'] = to_time_field(data['lastModifiedTime'])
                cons['is_view'] = data['isVirtualView']
                cons['size'] = data['size']
                cons['life'] = data['lifecycle']
                cons['columns'] = data['columns']
        obj = clz(**cons)
        obj.odps = o
        return obj


class Function(BaseModel):

    @classmethod
    def create(clz, o, res):
        pass


class Resource(BaseModel):

    @classmethod
    def create(clz, o, res):
        pass


# class ImmediateObject(object):
#     pass


# class ODPSXmlHandler(xml.sax.handler.ContentHandler):
    
#     def startDocument(self):
#         self.obj = ImmediateObject()

#     def startElement(self, name, attrs):
#         print name, attrs

class Job(object):
    """Job description class.
    Use this class to construct an ODPS Job and eventually convert it to xml 
    description.
    """
    def __init__(self, name, priority, comment='', tasks=[], run_mode='Sequence'):
        self.name = name
        self.priority = priority
        self.comment = comment
        self.tasks = tasks
        self.run_mode = run_mode

    def add_task(self, task):
        self.tasks.append(task)

    # Hand-crafted xml conversion method.
    # FIXME: Think a elegant way to do this.
    def to_xml(self):
        task_xmls = []
        for t in self.tasks:
            task_xmls.append(t.to_xml())
        task_xml = '\n'.join(task_xmls)
        template = '''<?xml version="1.0" encoding="utf-8"?>
        <Instance>
        <Job>
          <Name>%(name)s</Name>
          <Comment>%(comment)s</Comment>
          <Priority>%(priority)s</Priority>
          <Tasks>
             %(task_xml)s
          </Tasks>
          <DAG>
            <RunMode>%(run_mode)s</RunMode>
          </DAG>
        </Job>
        </Instance>
        '''
        return template % {
            'name': self.name,
            'comment': self.comment,
            'priority': self.priority,
            'task_xml': task_xml,
            'run_mode': self.run_mode
        }


class Record(object):
    def __init__(self, columns):
        if columns is None:
            raise ValueError
        
        self.columns = columns
        self.values = [None for _ in range(len(self.columns))]
        self.map = dict([(col.name, i) for i, col in enumerate(self.columns)])
        
    def get_columns_count(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]
    
    def get(self, i):
        return self.values[i]
    
    def set(self, i, value):
        if isinstance(value, (int, long)) and \
                (value > BIGINT_MAX or value <= BIGINT_MIN):
            raise ValueError('InvalidData: Bigint out of range')
        elif isinstance(value, datetime):
            ts = time.mktime(value.timetuple())
            if ts > DATETIME_MAX_TICKS or ts < DATETIME_MIN_TICKS:
                raise ValueError('InvalidData: Datetime out of range')
        elif (isinstance(value, str) and len(bytearray(value)) > STRING_MAX_LENGTH)\
            or (isinstance(value, bytearray) and len(value) > STRING_MAX_LENGTH):
            raise ValueError('InvalidData: The string\'s length is more than 2M.') 
        
        if isinstance(value, bytearray):
            value = str(value)
        self.values[i] = value
        
    def get_by_name(self, column_name):
        if column_name not in self.map:
            raise ValueError('no such column')
        
        return self.get(self.map[column_name])
    
    def set_by_name(self, column_name, value):
        if column_name not in self.map:
            raise ValueError('no such column')
        
        self.set(self.map[column_name], value)


class Schema(object):
    def __init__(self):
        self._columns = []
        self.partition_keys = []
        self.map_loaded = False
        
    @property
    def columns(self):
        return self._columns + self.partition_keys
    
    def _load_map(self):
        if self.map_loaded:
            return
        
        self.columns_map = dict([(c.name, idx) for idx, c in enumerate(self._columns)])
        self.partitions_map = dict([(p.name, idx) for idx, p \
                                        in enumerate(self.partition_keys)])
        
        self.map_loaded = True
        
    def get_column(self, name):
        self._load_map()
        return self._columns[self.columns_map[name]]
    
    def get_partition(self, name):
        self._load_map()
        return self.partition_keys[self.partitions_map[name]]
    
    def is_partition(self, name):
        self._load_map()
        return name in self.partitions_map
    
    def add_column(self, column):
        self._load_map()
        if column is None:
            raise ValueError('Column is null.')
        
        if column.name in self.columns_map or \
            column.name in self.partitions_map:
            raise ValueError('Column %s duplicated' % column.name)
        
        self.columns_map[column.name] = len(self._columns)
        self._columns.append(column)
        
    def add_partition(self, partition_key):
        self._load_map()
        if partition_key is None:
            raise ValueError('Partition key is null.')
        
        if partition_key.name in self.columns_map or \
            partition_key.name in self.partitions_map:
            raise ValueError('PartitionKey %s duplicated' % partition_key.name)
        
        self.partitions_map[partition_key.name] = len(self.partition_keys)
        self.partition_keys.append(partition_key)
    
    @classmethod
    def parse(cls, json_str):
        if len(json_str.strip()) == 0:
            return Schema()
        
        schema = Schema()
        
        jsn = json.loads(json_str)
        cols = jsn.pop('columns', None)
        if cols:
            schema._columns = []
            for col in cols:
                column = Column()
                for k, v in col.iteritems():
                    setattr(column, k, v)
                schema.add_column(column)
        
        part_keys = jsn.pop('partitionKeys', None)
        if part_keys:
            schema.partition_keys = []
            for part_key in part_keys:
                partition_key = PartitionKey()
                for k, v in part_key.iteritems():
                    setattr(partition_key, k, v)
                schema.add_partition(partition_key)
                
        for k, v in jsn.iteritems():
            setattr(schema, utils.camel_to_underline(k), v)
        
        return schema


class PlainModel(object):
    pass

Column = PlainModel
PartitionKey = PlainModel


class Resource(BaseModel):
    pass
