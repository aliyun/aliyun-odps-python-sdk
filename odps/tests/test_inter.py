#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import tempfile

from odps.tests.core import TestBase, tn
from odps.compat import unittest, StringIO
from odps.inter import *
from odps.config import options
from odps.errors import InteractiveError
from odps.models import Schema
from odps.df.backends.odpssql import cloudpickle


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        # install CloudUnpickler
        cloudpickle.CloudUnpickler(StringIO('abcdefg'))

    def testRoom(self):
        access_id = 'test_access_id'
        access_key = 'test_access_key'
        project = 'test_default_project'
        endpoint = 'test_endpoint'

        test_room = '__test'

        teardown(test_room)

        setup(access_id, access_key, project, endpoint=endpoint, room=test_room)
        enter(test_room)

        self.assertEqual(access_id, options.access_id)
        self.assertEqual(access_key, options.access_key)
        self.assertEqual(project, options.default_project)
        self.assertEqual(endpoint, options.end_point)
        self.assertIsNone(options.tunnel_endpoint)

        self.assertRaises(
            InteractiveError,
            lambda: setup(access_id, access_key, project, room=test_room))

        self.assertIn(test_room, list_rooms())

        teardown(test_room)
        self.assertRaises(
            InteractiveError, lambda: enter(test_room)
        )

    def testRoomStores(self):
        class FakeRoom(Room):
            def _init(self):
                return

        room = FakeRoom('__test')
        room._room_dir = tempfile.mkdtemp()

        try:
            s = Schema.from_lists(['name', 'id'], ['string', 'bigint'])
            table_name = tn('pyodps_test_room_stores')
            self.odps.delete_table(table_name, if_exists=True)
            t = self.odps.create_table(table_name, s)
            data = [['name1', 1], ['name2', 2]]
            with t.open_writer() as writer:
                writer.write(data)

            del t

            t = self.odps.get_table(table_name)
            self.assertEqual(t.schema.names, ['name', 'id'])

            try:
                room.store('table', t)

                t2 = room['table']
                self.assertEqual(t2.name, table_name)

                with t2.open_reader() as reader:
                    values = [r.values for r in reader]
                    self.assertEqual(data, values)

                self.assertEqual(room.list_stores(), [['table', None]])
            finally:
                t.drop()
        finally:
            shutil.rmtree(room._room_dir)


if __name__ == '__main__':
    unittest.main()
