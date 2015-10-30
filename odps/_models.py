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

"""Internal use only!
"""

from email.utils import parsedate_tz
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from odps.utils import timetuple_to_datetime


class InstanceTask(object):
    WAITING, RUNNING, SUCCESS, FAILED, SUSPENDED, CANCELLED = range(6)
    
    def __init__(self, task_status):
        self.name = task_status.name
        self.type = task_status.type
        self.status = getattr(self, task_status.status.upper())
        
    def get_status_str(self):
        return ['WAITING', 'RUNNING', 'SUCCESS', 
                'FAILED', 'SUSPENDED', 'CANCELLED'][self.status]


class TaskStatuses(object):
    def __init__(self):
        self.tasks = []

    class TaskStatus(object):
        def __init__(self):
            self.histories = []
        
        @classmethod
        def parse(cls, xml_ele):
            task_status = TaskStatuses.TaskStatus()
            
            task_status.name = xml_ele.find('./Name').text
            task_status.type = xml_ele.get('Type')
            task_status.start_time = timetuple_to_datetime(
                parsedate_tz(xml_ele.find('./StartTime').text))
            task_status.end_time = xml_ele.find('./EndTime').text
            if task_status.end_time:
                task_status.end_time = timetuple_to_datetime(
                    parsedate_tz(task_status.end_time))
            task_status.status = xml_ele.find('./Status').text
            
            for history_ele in xml_ele.findall('./Histories/History'):
                task_status.histories.append(
                    TaskStatuses.TaskStatus.parse(history_ele))
            
            return task_status
    
    @classmethod
    def parse(cls, xml):
        content = xml.read() if hasattr(xml, 'read') else xml
        if isinstance(content, basestring):
            root = ET.fromstring(content)
        else:
            root = content
        
        task_statuses = TaskStatuses()
        if root.find('./Name') is not None:
            task_statuses.name = root.find('./Name').text
        if root.find('./Owner') is not None:
            task_statuses.owner = root.find('./Owner').text
        if root.find('./StartTime') is not None:
            task_statuses.start_time = timetuple_to_datetime(
                parsedate_tz(root.find('./StartTime').text))
        if root.find('./EndTime') is not None:
            task_statuses.end_time = root.find('./EndTime').text
            if task_statuses.end_time:
                task_statuses.end_time = timetuple_to_datetime(
                    parsedate_tz(task_statuses.end_time))
        if root.find('./Status') is not None:
            task_statuses.status = root.find('./Status').text
        
        for task_status_ele in root.findall('./Tasks/Task'):
            task_statuses.tasks.append(
                TaskStatuses.TaskStatus.parse(task_status_ele))
        
        return task_statuses
