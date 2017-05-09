#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from odps import ODPS
from odps.tests.core import TestBase, tn
from odps.accounts import SignServer, SignServerAccount, SignServerError


class Test(TestBase):
    def testSignServerAccount(self):
        server = SignServer()
        server.accounts[self.odps.account.access_id] = self.odps.account.secret_access_key
        try:
            server.start(('127.0.0.1', 0))
            account = SignServerAccount(self.odps.account.access_id, server.server.server_address)
            odps = ODPS(None, None, self.odps.project, self.odps.endpoint, account=account)
            odps.delete_table(tn('test_sign_account_table'), if_exists=True)
            t = odps.create_table(tn('test_sign_account_table'), 'col string', lifecycle=1)
            self.assertTrue(odps.exist_table(tn('test_sign_account_table')))
            t.drop(async=True)
        finally:
            server.stop()

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
            t = odps.create_table(tn('test_sign_account_table'), 'col string', lifecycle=1)
            self.assertTrue(odps.exist_table(tn('test_sign_account_table')))
            t.drop(async=True)
        finally:
            server.stop()
