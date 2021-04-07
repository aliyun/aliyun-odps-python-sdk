#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2019 Alibaba Group Holding Ltd.
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
import time
import random
import textwrap
from datetime import datetime, timedelta

import requests

from odps.tests.core import TestBase, to_str, tn, pandas_case
from odps.compat import unittest, six
from odps.models import Instance, SQLTask, Schema, session, Record
from odps.errors import ODPSError
from odps import errors, compat, types as odps_types, utils, options

TEST_SESSION_WORKERS = 4
TEST_SESSION_WORKER_MEMORY = 512

TEST_TABLE_NAME = "_pyodps__session_test_table"
TEST_CREATE_SCHEMA = Schema.from_lists(['id'], ['string'])
TEST_DATA = [['1'], ['2'], ['3'], ['4'], ['5']]
TEST_SELECT_STRING = "select * from %s" % TEST_TABLE_NAME


class Test(TestBase):

    def testCreateSession(self):
        sess_instance = self.odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
        self.assertTrue(sess_instance)
        # wait to running
        try:
            sess_instance.wait_for_startup()
        except ODPSError as ex:
            print("LOGVIEW: " + sess_instance.get_logview_address())
            print("Task results: " + str(sess_instance.get_task_results()))
            raise ex
        # the status should keep consistent
        self.assertTrue(sess_instance.status == Instance.Status.RUNNING)
        # finally stop it.
        sess_instance.stop()

    def testAttachSession(self):
        sess_instance = self.odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
        self.assertTrue(sess_instance)
        # wait to running
        try:
            sess_instance.wait_for_startup()
        except ODPSError as ex:
            print("LOGVIEW: " + sess_instance.get_logview_address())
            print("Task results: " + str(sess_instance.get_task_results()))
            raise ex
        # the status should keep consistent
        self.assertTrue(sess_instance.status == Instance.Status.RUNNING)
        att_instance = self.odps.attach_session(sess_instance._session_name)
        self.assertTrue(att_instance)
        # wait to running
        try:
            att_instance.wait_for_startup()
        except ODPSError as ex:
            print("LOGVIEW: " + att_instance.get_logview_address())
            print("Task results: " + str(att_instance.get_task_results()))
            raise ex
        # the status should keep consistent
        self.assertTrue(att_instance.status == Instance.Status.RUNNING)
        # finally stop it.
        sess_instance.stop()

    # note: this test case may fail if the test environment does not have any default
    #       sessions enabled.
    def testAttachDefaultSession(self):
        sess_instance = self.odps.default_session()
        self.assertTrue(sess_instance)
        # wait to running
        try:
            sess_instance.wait_for_startup()
        except ODPSError as ex:
            print("LOGVIEW: " + sess_instance.get_logview_address())
            print("Task results: " + str(sess_instance.get_task_results()))
            raise ex
        # the status should keep consistent
        self.assertTrue(sess_instance.status == Instance.Status.RUNNING)

    def testSessionFailingSQL(self):
        self.odps.delete_table(TEST_TABLE_NAME, if_exists=True)
        sess_instance = self.odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
        self.assertTrue(sess_instance)
        # wait to running
        try:
            sess_instance.wait_for_startup()
        except ODPSError as ex:
            print("LOGVIEW: " + sess_instance.get_logview_address())
            print("Task results: " + str(sess_instance.get_task_results()))
            raise ex
        # the status should keep consistent
        self.assertTrue(sess_instance.status == Instance.Status.RUNNING)
        select_inst = sess_instance.run_sql(TEST_SELECT_STRING)
        select_inst.wait_for_completion() # should return normally even the task is failed
        try:
            select_inst.wait_for_success()
            self.assertTrue(False) # should not reach here: wait_for_success should throw exception on failed instance
        except ODPSError as ex:
            pass # good
        sess_instance.stop()

    def testDirectExecuteFailingSQL(self):
        self.odps.delete_table(TEST_TABLE_NAME, if_exists=True)
        # the default public session may not exist, so we create one beforehand
        sess_instance = self.odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
        self.assertTrue(sess_instance)
        # wait to running
        try:
            sess_instance.wait_for_startup()
        except ODPSError as ex:
            print("LOGVIEW: " + sess_instance.get_logview_address())
            print("Task results: " + str(sess_instance.get_task_results()))
            raise ex
        # the status should keep consistent
        self.assertTrue(sess_instance.status == Instance.Status.RUNNING)
        select_inst = self.odps.run_sql_interactive(TEST_SELECT_STRING, service_name=sess_instance.name)
        select_inst.wait_for_completion() # should return normally even the task is failed
        try:
            select_inst.wait_for_success()
            self.assertTrue(False) # should not reach here: wait_for_success should throw exception on failed instance
        except ODPSError as ex:
            pass # good
        sess_instance.stop()

    def testSessionSQL(self):
        self.odps.delete_table(TEST_TABLE_NAME, if_exists=True)
        table = self.odps.create_table(TEST_TABLE_NAME, TEST_CREATE_SCHEMA)
        self.assertTrue(table)
        sess_instance = self.odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
        self.assertTrue(sess_instance)
        # wait to running
        try:
            sess_instance.wait_for_startup()
        except ODPSError as ex:
            print("LOGVIEW: " + sess_instance.get_logview_address())
            print("Task results: " + str(sess_instance.get_task_results()))
            raise ex
        # the status should keep consistent
        self.assertTrue(sess_instance.status == Instance.Status.RUNNING)
        records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
        self.odps.write_table(table, 0, records)
        select_inst = sess_instance.run_sql(TEST_SELECT_STRING)
        select_inst.wait_for_success()
        rows = []
        try:
            with select_inst.open_reader() as rd:
                for each_row in rd:
                    rows.append(each_row.values)
        except BaseException as ex:
            print("LOGVIEW: " + select_inst.get_logview_address())
            print("Task Result:" + str(select_inst.get_task_results()))
            raise ex
        self.assertTrue(len(rows) == len(TEST_DATA))
        self.assertTrue(len(rows[0]) == len(TEST_DATA[0]))
        for index in range(5):
            self.assertTrue(int(rows[index][0]) == int(TEST_DATA[index][0]))
        # OK, clear up
        self.odps.delete_table(TEST_TABLE_NAME, if_exists=True)

    def testDirectExecuteSQL(self):
        self.odps.delete_table(TEST_TABLE_NAME, if_exists=True)
        table = self.odps.create_table(TEST_TABLE_NAME, TEST_CREATE_SCHEMA)
        self.assertTrue(table)
        # the default public session may not exist, so we create one beforehand
        sess_instance = self.odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
        self.assertTrue(sess_instance)
        # wait to running
        try:
            sess_instance.wait_for_startup()
        except ODPSError as ex:
            print("LOGVIEW: " + sess_instance.get_logview_address())
            print("Task results: " + str(sess_instance.get_task_results()))
            raise ex
        # the status should keep consistent
        self.assertTrue(sess_instance.status == Instance.Status.RUNNING)
        records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
        self.odps.write_table(table, 0, records)
        select_inst = self.odps.run_sql_interactive(TEST_SELECT_STRING, service_name=sess_instance.name)
        select_inst.wait_for_success()
        rows = []
        try:
            with select_inst.open_reader() as rd:
                for each_row in rd:
                    rows.append(each_row.values)
        except BaseException as ex:
            print("LOGVIEW: " + select_inst.get_logview_address())
            print("Task Result:" + str(select_inst.get_task_results()))
            raise ex
        self.assertTrue(len(rows) == len(TEST_DATA))
        self.assertTrue(len(rows[0]) == len(TEST_DATA[0]))
        for index in range(5):
            self.assertTrue(int(rows[index][0]) == int(TEST_DATA[index][0]))
        # OK, clear up
        self.odps.delete_table(TEST_TABLE_NAME, if_exists=True)

    def testSessionSQLWithInstanceTunnel(self):
        self.odps.delete_table(TEST_TABLE_NAME, if_exists=True)
        table = self.odps.create_table(TEST_TABLE_NAME, TEST_CREATE_SCHEMA)
        self.assertTrue(table)
        sess_instance = self.odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
        self.assertTrue(sess_instance)
        # wait to running
        try:
            sess_instance.wait_for_startup()
        except ODPSError as ex:
            print("LOGVIEW: " + sess_instance.get_logview_address())
            print("Task results: " + str(sess_instance.get_task_results()))
            raise ex
        # the status should keep consistent
        self.assertTrue(sess_instance.status == Instance.Status.RUNNING)
        records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
        self.odps.write_table(table, 0, records)
        select_inst = sess_instance.run_sql(TEST_SELECT_STRING)
        select_inst.wait_for_success()
        rows = []
        try:
            with select_inst.open_reader(tunnel=True) as rd:
                for each_row in rd:
                    rows.append(each_row.values)
        except BaseException as ex:
            print("LOGVIEW: " + select_inst.get_logview_address())
            print("Task Result:" + str(select_inst.get_task_results()))
            raise ex
        self.assertTrue(len(rows) == len(TEST_DATA))
        self.assertTrue(len(rows[0]) == len(TEST_DATA[0]))
        for index in range(5):
            self.assertTrue(int(rows[index][0]) == int(TEST_DATA[index][0]))
        # OK, clear up
        self.odps.delete_table(TEST_TABLE_NAME, if_exists=True)

if __name__ == '__main__':
    unittest.main()
