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

from __future__ import print_function
import time
import cProfile
from pstats import Stats

# switch on to run in pure Python mode
from odps import options
# options.force_py = True

from odps.compat import unittest, Decimal
from odps.tests.core import TestBase
from odps.models import Schema
from datetime import datetime

# remember to reset False before committing
ENABLE_PROFILE = False
DUMP_PROFILE = False


class Test(TestBase):
    COMPRESS_DATA = True
    BUFFER_SIZE = 1024*1024
    DATA_AMOUNT = 100000
    STRING_LITERAL = "Soft kitty, warm kitty, little ball of fur; happy kitty, sleepy kitty, purr, purr"

    def setUp(self):
        TestBase.setUp(self)
        if ENABLE_PROFILE:
            self.pr = cProfile.Profile()
            self.pr.enable()
        fields = ['a', 'b', 'c', 'd', 'e', 'f']
        types = ['bigint', 'double', 'datetime', 'boolean', 'string', 'decimal']
        self.SCHEMA = Schema.from_lists(fields, types)

    def tearDown(self):
        if ENABLE_PROFILE:
            if DUMP_PROFILE:
                self.pr.dump_stats('profile.out')
            p = Stats(self.pr)
            p.strip_dirs()
            p.sort_stats('time')
            p.print_stats(40)
            p.print_callees('types.py:846\(validate_value', 20)
            p.print_callees('types.py:828\(_validate_primitive_value', 20)
            p.print_callees('uploadsession.py:185\(write', 20)
        TestBase.teardown(self)

    def testWrite(self):
        table_name = 'pyodps_test_tunnel_write_performance'
        self.odps.create_table(table_name, schema=self.SCHEMA, if_not_exists=True)
        ss = self.tunnel.create_upload_session(table_name)
        r = ss.new_record()

        start = time.time()
        with ss.open_record_writer(0) as writer:
            for i in range(self.DATA_AMOUNT):
                r[0] = 2**63-1
                r[1] = 0.0001
                r[2] = datetime(2015, 11, 11)
                r[3] = True
                r[4] = self.STRING_LITERAL
                r[5] = Decimal('3.15')
                writer.write(r)
            n_bytes = writer.n_bytes
        print(n_bytes, 'bytes', float(n_bytes) / 1024 / 1024 / (time.time() - start), 'MiB/s')
        ss.commit([0])
        self.odps.delete_table(table_name, if_exists=True)

    def testRead(self):
        table_name = 'pyodps_test_tunnel_read_performance'
        self.odps.delete_table(table_name, if_exists=True)
        t = self.odps.create_table(table_name, schema=self.SCHEMA)

        def gen_data():
            for i in range(self.DATA_AMOUNT):
                r = t.new_record()
                r[0] = 2 ** 63 - 1
                r[1] = 0.0001
                r[2] = datetime(2015, 11, 11)
                r[3] = True
                r[4] = self.STRING_LITERAL
                r[5] = Decimal('3.15')
                yield r

        self.odps.write_table(t, gen_data())

        ds = self.tunnel.create_download_session(table_name)

        start = time.time()
        cnt = 0
        with ds.open_record_reader(0, ds.count) as reader:
            for _ in reader:
                cnt += 1
            n_bytes = reader.n_bytes
        print(n_bytes, 'bytes', float(n_bytes) / 1024 / 1024 / (time.time() - start), 'MiB/s')
        self.assertEqual(self.DATA_AMOUNT, cnt)
        self.odps.delete_table(table_name, if_exists=True)

    def testBufferedWrite(self):
        table_name = 'test_tunnel_bufferred_write'
        self.odps.create_table(table_name, schema=self.SCHEMA, if_not_exists=True)
        ss = self.tunnel.create_upload_session(table_name)
        r = ss.new_record()

        start = time.time()
        with ss.open_record_writer(buffer_size=self.BUFFER_SIZE, compress=self.COMPRESS_DATA) as writer:
            for i in range(self.DATA_AMOUNT):
                r[0] = 2**63-1
                r[1] = 0.0001
                r[2] = datetime(2015, 11, 11)
                r[3] = True
                r[4] = self.STRING_LITERAL
                r[5] = Decimal('3.15')
                writer.write(r)
            n_bytes = writer.n_bytes
        print(n_bytes, 'bytes', float(n_bytes) / 1024 / 1024 / (time.time() - start), 'MiB/s')
        ss.commit(writer.get_blocks_written())
        self.odps.delete_table(table_name, if_exists=True)

if __name__ == '__main__':
    unittest.main()
