import json
import os

import pytest

from ....tests.core import tn, wait_filled
from ....utils import to_text
from .. import MergeTask, Task

merge_template = """<?xml version="1.0" encoding="utf-8"?>
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
"""


@pytest.fixture
def test_table(odps):
    _, table_suffix = os.environ["PYTEST_CURRENT_TEST"].rsplit("::", 1)
    table_suffix, _ = table_suffix.split(" ", 1)
    table_name = tn("pyodps_test_merge_task_table_" + table_suffix)
    if odps.exist_table(table_name):
        odps.delete_table(table_name)

    table = odps.create_table(table_name, ("col string", "part1 string, part2 string"))
    table.create_partition("part1=1,part2=1", if_not_exists=True)
    odps.write_table(table_name, [("col_name",)], partition="part1=1,part2=1")

    try:
        yield table_name
    finally:
        odps.delete_table(table_name)


def test_merge_task_to_xml():
    task = MergeTask("task_1", table="table_name")
    task.update_settings({"odps.merge.cross.paths": True})
    to_xml = task.serialize()
    right_xml = merge_template % dict(name="task_1", table="table_name")

    assert to_text(to_xml) == to_text(right_xml)

    task = Task.parse(None, to_xml)
    assert isinstance(task, MergeTask)


def test_run_merge(odps, test_table):
    inst = odps.run_merge_files(test_table, 'part1=1, part2="1"')
    wait_filled(lambda: inst.tasks)
    task = inst.tasks[0]
    assert isinstance(task, MergeTask)
    try:
        inst.stop()
    except:
        pass

    inst = odps.run_sql(
        "alter table %s partition (part1=1,part2=1) merge smallfiles;" % test_table
    )
    wait_filled(lambda: inst.tasks)
    task = inst.tasks[0]
    assert isinstance(task, MergeTask)
    try:
        inst.stop()
    except:
        pass


def test_run_compact(odps, test_table):
    inst = odps.run_sql(
        "alter table %s partition (part1=1,part2=1) compact major;" % test_table
    )
    wait_filled(lambda: inst.tasks)
    task = inst.tasks[0]
    assert isinstance(task, MergeTask)
    assert (
        json.loads(task.properties["settings"])["odps.merge.txn.table.compact"]
        == "major_compact"
    )
    try:
        inst.stop()
    except:
        pass

    inst = odps.run_sql(
        "alter table %s partition (part1=1,part2=1) compact minor -h 5 -f;" % test_table
    )
    wait_filled(lambda: inst.tasks)
    task = inst.tasks[0]
    assert isinstance(task, MergeTask)
    settings_dict = json.loads(task.properties["settings"])
    assert settings_dict["odps.merge.txn.table.compact"] == "minor_compact"
    assert settings_dict["odps.merge.txn.table.compact.txn.id"] == "5"
    try:
        inst.stop()
    except:
        pass


def test_run_archive(odps, test_table):
    inst = odps.run_sql(
        "alter table %s partition (part1=1,part2=1) archive;" % test_table
    )
    wait_filled(lambda: inst.tasks)
    task = inst.tasks[0]
    assert isinstance(task, MergeTask)
    assert "archiveSettings" in task.properties
    try:
        inst.stop()
    except:
        pass


def test_run_freeze(odps, test_table):
    inst = odps.run_sql(
        "alter table %s partition (part1=1,part2=1) freeze;" % test_table
    )
    wait_filled(lambda: inst.tasks)
    task = inst.tasks[0]
    assert isinstance(task, MergeTask)
    settings_dict = json.loads(task.properties["settings"])
    assert settings_dict["odps.merge.cold.storage.mode"] == "backup"
    try:
        inst.stop()
    except:
        pass
