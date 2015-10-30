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


import math
from datetime import datetime

import unittest
import ut

from odps.tunnel.conf import CompressOption


RECORDS = [
    ('hello world', 2**63-1, math.pi, datetime(2015, 9, 19, 2, 11, 00), True),
    ('goodbye', 222222, math.e, datetime(2020, 3, 10), False),
    ('c'*300, -2**63+1, -2.222, datetime(1999, 5, 25, 3, 10), True),
]

TEST_TABLE = 'pyodps_test_tunnel'


def convert_to_list(one_item_tuple_list):
    ret = []
    for i in one_item_tuple_list:
        ret.append(i[0])
    return ret


class TestTunnel(ut.TestBase):
    
    def setup(self):
        self.odps.execute_sql("drop table if exists " + TEST_TABLE)
        self.odps.execute_sql(
            "create table %s (id string, num bigint, math double, dt datetime, t boolean)" % TEST_TABLE)

    def _upload_data(self, compress=False, compress_option=None):
        upload_ss = self.tunnel.create_upload_session(
            TEST_TABLE, compress_option=compress_option)
        writer = upload_ss.open_record_writer(0, compress=compress)
        for r in RECORDS:
            record = upload_ss.new_record()
            for i, it in enumerate(r):
                record.set(i, it)
            writer.write(record)
        writer.close()
        upload_ss.commit([0,])

    def _download_data(self, compress=False, compress_option=None):
        download_ss = self.tunnel.create_download_session(
            TEST_TABLE, compresss_option=compress_option)
        reader = download_ss.open_record_reader(0, 3, compress=compress)
        records = [tuple(record.values) for record in reader.reads()]

        return records

    def test_updown(self):
        print 'Test raw upload'
        self._upload_data()

        print 'Test raw download'
        records = self._download_data()

        self.assertSequenceEqual(RECORDS, records)

    def test_zlib_updown(self):
        print 'Test zlib upload'
        self._upload_data(compress=True)

        print 'Test zlib download'
        records = self._download_data(compress=True)

        self.assertSequenceEqual(RECORDS, records)

    def test_snappy_updown(self):
        compress_option = CompressOption(CompressOption.CompressionAlgorithm.ODPS_SNAPPY)

        print 'Test snappy upload'
        self._upload_data(compress=True, compress_option=compress_option)

        print 'Test snappy download'
        records = self._download_data(compress=True, compress_option=compress_option)

        self.assertSequenceEqual(RECORDS, records)

    def teardown(self):
        self.odps.execute_sql("drop table if exists " + TEST_TABLE)

if __name__ == '__main__':
    unittest.main()
