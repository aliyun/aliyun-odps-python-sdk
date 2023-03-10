#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import uuid

from odps import ODPS, errors, options
from odps.compat import unittest
from odps.tests.core import TestBase, tn
from odps.accounts import SignServer, SignServerAccount, SignServerError, BearerTokenAccount

try:
    from cupid.runtime import context as cupid_context
except ImportError:
    cupid_context = None


class Test(TestBase):
    def testSignServerAccount(self):
        server = SignServer()
        server.accounts[self.odps.account.access_id] = self.odps.account.secret_access_key
        try:
            server.start(('127.0.0.1', 0))
            account = SignServerAccount(self.odps.account.access_id, server.server.server_address)
            odps = self.odps.as_account(account=account)
            odps.delete_table(tn('test_sign_account_table'), if_exists=True)
            t = odps.create_table(tn('test_sign_account_table'), 'col string', lifecycle=1)
            self.assertTrue(odps.exist_table(tn('test_sign_account_table')))
            t.drop(async_=True)
        finally:
            server.stop()
            options.account = options.default_project = options.endpoint = None

    def testTokenizedSignServerAccount(self):
        server = SignServer(token=str(uuid.uuid4()))
        server.accounts[self.odps.account.access_id] = self.odps.account.secret_access_key
        try:
            server.start(('127.0.0.1', 0))
            account = SignServerAccount(self.odps.account.access_id, server.server.server_address)
            odps = ODPS(None, None, self.odps.project, self.odps.endpoint, account=account)
            self.assertRaises(SignServerError,
                              lambda: odps.delete_table(tn('test_sign_account_table'), if_exists=True))

            account = SignServerAccount(self.odps.account.access_id, server.server.server_address, token=server.token)
            odps = ODPS(None, None, self.odps.project, self.odps.endpoint, account=account)
            odps.delete_table(tn('test_sign_account_table'), if_exists=True)
            t = odps.create_table(tn('test_sign_account_table'), 'col string', lifecycle=1)
            self.assertTrue(odps.exist_table(tn('test_sign_account_table')))
            t.drop(async_=True)
        finally:
            server.stop()

    @unittest.skipIf(cupid_context is None, "cannot import cupid context")
    def testBearerTokenAccount(self):
        self.odps.delete_table(tn('test_bearer_token_account_table'), if_exists=True)
        t = self.odps.create_table(tn('test_bearer_token_account_table'), 'col string', lifecycle=1)
        with t.open_writer() as writer:
            records = [['val1'], ['val2'], ['val3']]
            writer.write(records)

        inst = self.odps.execute_sql('select count(*) from {0}'.format(tn('test_bearer_token_account_table')), async_=True)
        inst.wait_for_success()
        task_name = inst.get_task_names()[0]

        logview_address = inst.get_logview_address()
        token = logview_address[logview_address.find('token=') + len('token='):]
        bearer_token_account = BearerTokenAccount(token=token)
        bearer_token_odps = ODPS(None, None, self.odps.project, self.odps.endpoint, account=bearer_token_account)
        bearer_token_instance = bearer_token_odps.get_instance(inst.id)

        self.assertEqual(inst.get_task_result(task_name),
                         bearer_token_instance.get_task_result(task_name))
        self.assertEqual(inst.get_task_summary(task_name),
                         bearer_token_instance.get_task_summary(task_name))

        with self.assertRaises(errors.NoPermission):
            bearer_token_odps.create_table(tn('test_bearer_token_account_table_test1'),
                                           'col string', lifecycle=1)

        fake_token_account = BearerTokenAccount(token='fake-token')
        bearer_token_odps = ODPS(None, None, self.odps.project, self.odps.endpoint, account=fake_token_account)

        with self.assertRaises(errors.ODPSError):
            bearer_token_odps.create_table(tn('test_bearer_token_account_table_test2'),
                                           'col string', lifecycle=1)
