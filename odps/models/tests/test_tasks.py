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

import json
import sys

import pytest

from ...errors import ODPSError
from ...config import options
from ...tests.core import tn, wait_filled
from ...utils import get_zone_name, to_text
from .. import SQLTask, MergeTask, CupidTask, SQLCostTask, Task

try:
    import zoneinfo
except ImportError:
    zoneinfo = None
try:
    import pytz
except ImportError:
    pytz = None


sql_template = '''<?xml version="1.0" encoding="utf-8"?>
<SQL>
  <Name>AnonymousSQLTask</Name>
  <Config>
    <Property>
      <Name>settings</Name>
      <Value>{"odps.sql.udf.strict.mode": "true"}</Value>
    </Property>
  </Config>
  <Query><![CDATA[%(sql)s;]]></Query>
</SQL>
'''

sql_tz_template = '''<?xml version="1.0" encoding="utf-8"?>
<SQL>
  <Name>AnonymousSQLTask</Name>
  <Config>
    <Property>
      <Name>settings</Name>
      <Value>{"PYODPS_VERSION": "%(pyodps_version)s", "PYODPS_PYTHON_VERSION": %(python_version)s, "odps.sql.timezone": "%(tz)s"}</Value>
    </Property>
  </Config>
  <Query><![CDATA[%(sql)s;]]></Query>
</SQL>
'''

merge_template = '''<?xml version="1.0" encoding="utf-8"?>
<Merge>
  <Name>%(name)s</Name>
  <Config>
    <Property>
      <Name>settings</Name>
      <Value>{"odps.merge.cross.paths": "true"}</Value>
    </Property>
  </Config>
  <TableName>%(table)s</TableName>
</Merge>
'''

cupid_template = '''<?xml version="1.0" encoding="utf-8"?>
<CUPID>
  <Name>task_1</Name>
  <Config>
    <Property>
      <Name>type</Name>
      <Value>cupid</Value>
    </Property>
    <Property>
      <Name>settings</Name>
      <Value>{"odps.cupid.wait.am.start.time": 600}</Value>
    </Property>
  </Config>
  <Plan><![CDATA[plan_text]]></Plan>
</CUPID>
'''

sql_cost_template = '''<?xml version="1.0" encoding="utf-8"?>
<SQLCost>
  <Name>AnonymousSQLCostTask</Name>
  <Query><![CDATA[%(sql)s;]]></Query>
</SQLCost>
'''


def test_task_class_type():
    typed = Task(type='SQL', query='select * from dual')
    assert isinstance(typed, SQLTask)

    unknown_typed = Task(type='UnknownType')
    assert type(unknown_typed) is Task
    pytest.raises(ODPSError, lambda: unknown_typed.serialize())

    untyped = Task()
    assert type(untyped) is Task
    pytest.raises(ODPSError, lambda: untyped.serialize())


def test_sql_task_to_xml():
    query = 'select * from dual'

    task = SQLTask(query=query)
    to_xml = task.serialize()
    right_xml = sql_template % {'sql': query}

    assert to_text(to_xml) == to_text(right_xml)

    task = Task.parse(None, to_xml)
    assert isinstance(task, SQLTask)


@pytest.mark.skipif(pytz is None and zoneinfo is None, reason='pytz not installed')
def test_sql_task_to_xml_timezone():
    from ... import __version__
    from ...lib import tzlocal

    query = 'select * from dual'
    versions = {"pyodps_version": __version__, "python_version": json.dumps(sys.version)}

    def _format_template(**kwargs):
        kwargs.update(versions)
        return sql_tz_template % kwargs

    try:
        options.local_timezone = True
        local_zone = tzlocal.get_localzone()
        local_zone_name = get_zone_name(local_zone)
        task = SQLTask(query=query)
        task.update_sql_settings()
        to_xml = task.serialize()
        right_xml = _format_template(sql=query, tz=local_zone_name)

        assert to_text(to_xml) == to_text(right_xml)

        options.local_timezone = False
        task = SQLTask(query=query)
        task.update_sql_settings()
        to_xml = task.serialize()
        right_xml = _format_template(sql=query, tz='Etc/GMT')

        assert to_text(to_xml) == to_text(right_xml)

        if zoneinfo:
            options.local_timezone = zoneinfo.ZoneInfo('Asia/Shanghai')
            task = SQLTask(query=query)
            task.update_sql_settings()
            to_xml = task.serialize()
            right_xml = _format_template(sql=query, tz=options.local_timezone.key)

            assert to_text(to_xml) == to_text(right_xml)

        if pytz:
            options.local_timezone = pytz.timezone('Asia/Shanghai')
            task = SQLTask(query=query)
            task.update_sql_settings()
            to_xml = task.serialize()
            right_xml = _format_template(sql=query, tz=options.local_timezone.zone)

            assert to_text(to_xml) == to_text(right_xml)
    finally:
        options.local_timezone = None


def test_merge_task_to_xml():
    task = MergeTask('task_1', table='table_name')
    task.update_settings({'odps.merge.cross.paths': True})
    to_xml = task.serialize()
    right_xml = merge_template % dict(name='task_1', table='table_name')

    assert to_text(to_xml) == to_text(right_xml)

    task = Task.parse(None, to_xml)
    assert isinstance(task, MergeTask)


def test_run_merge_task(odps):
    table_name = tn('pyodps_test_merge_task_table')
    if odps.exist_table(table_name):
        odps.delete_table(table_name)

    table = odps.create_table(table_name, ('col string', 'part1 string, part2 string'))
    table.create_partition('part1=1,part2=1', if_not_exists=True)
    odps.write_table(table_name, [('col_name', )], partition='part1=1,part2=1')

    inst = odps.run_merge_files(table_name, 'part1=1, part2="1"')
    wait_filled(lambda: inst.tasks)
    task = inst.tasks[0]
    assert isinstance(task, MergeTask)

    try:
        inst.stop()
    except:
        pass

    inst = odps.run_sql(
        'alter table %s partition (part1=1,part2=1) merge smallfiles;' % table_name
    )
    wait_filled(lambda: inst.tasks)
    task = inst.tasks[0]
    assert isinstance(task, MergeTask)

    try:
        inst.stop()
    except:
        pass

    odps.delete_table(table_name)


def test_cupid_task_to_xml():
    task = CupidTask('task_1', 'plan_text', {'odps.cupid.wait.am.start.time': 600})
    to_xml = task.serialize()
    right_xml = cupid_template

    assert to_text(to_xml) == to_text(right_xml)

    task = Task.parse(None, to_xml)
    assert isinstance(task, CupidTask)


def test_sql_cost_task_to_xml():
    query = 'select * from dual'
    task = SQLCostTask(query=query)
    to_xml = task.serialize()
    right_xml = sql_cost_template % {'sql': query}

    assert to_text(to_xml) == to_text(right_xml)

    task = Task.parse(None, to_xml)
    assert isinstance(task, SQLCostTask)
