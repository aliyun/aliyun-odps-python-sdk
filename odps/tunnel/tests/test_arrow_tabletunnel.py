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

import math

try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters  # noqa: F401
try:
    import pytz
except ImportError:
    pytz = None
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

try:
    import pyarrow as pa
except (ImportError, AttributeError):
    pa = None

from odps.tests.core import TestBase, tn
from odps.compat import unittest
from odps.config import options
from odps.models import Schema


@unittest.skipIf(pa is None, "need to install pyarrow")
class Test(TestBase):
    def _upload_data(self, test_table, data, compress=False, **kw):
        upload_ss = self.tunnel.create_upload_session(test_table, **kw)
        writer = upload_ss.open_arrow_writer(0, compress=compress)

        writer.write(data)
        writer.close()
        upload_ss.commit([0, ])

    def _download_data(self, test_table, columns=None, compress=False, **kw):
        count = kw.pop('count', 4)
        download_ss = self.tunnel.create_download_session(test_table, **kw)
        with download_ss.open_arrow_reader(0, count, compress=compress, columns=columns) as reader:
            return reader.to_pandas()

    def _gen_data(self, repeat=1):
        data = dict()
        data['id'] = ['hello \x00\x00 world', 'goodbye', 'c' * 2, 'c' * 20] * repeat
        data['int_num'] = [2**63-1, 222222, -2 ** 63 + 1, -2 ** 11 + 1] * repeat
        data['float_num'] = [math.pi, math.e, -2.222, 2.222] * repeat
        data['bool'] = [True, False, True, True] * repeat

        return pa.RecordBatch.from_pandas(pd.DataFrame(data))

    def _create_table(self, table_name):
        fields = ['id', 'int_num', 'float_num', 'bool']
        types = ['string', 'bigint', 'double', 'boolean']

        self.odps.delete_table(table_name, if_exists=True)
        return self.odps.create_table(table_name, schema=Schema.from_lists(fields, types), lifecycle=1)

    def _create_partitioned_table(self, table_name):
        fields = ['id', 'int_num', 'float_num', 'bool']
        types = ['string', 'bigint', 'double', 'boolean']

        self.odps.delete_table(table_name, if_exists=True)
        return self.odps.create_table(table_name,
                                      schema=Schema.from_lists(fields, types, ['ds'], ['string']))

    def _delete_table(self, table_name):
        self.odps.delete_table(table_name)

    def testUploadAndDownloadByRawTunnel(self):
        test_table_name = tn('pyodps_test_arrow_tunnel')
        self._create_table(test_table_name)
        pd_df = self._download_data(test_table_name)
        self.assertEqual(len(pd_df), 0)

        data = self._gen_data()
        self._upload_data(test_table_name, data)

        pd_df = self._download_data(test_table_name)
        pd.testing.assert_frame_equal(data.to_pandas(), pd_df)

    def testDownloadWithSpecifiedColumns(self):
        test_table_name = tn('pyodps_test_arrow_tunnel_columns')
        self._create_table(test_table_name)

        data = self._gen_data()
        self._upload_data(test_table_name, data)

        records = self._download_data(test_table_name, columns=['id'])
        pd.testing.assert_frame_equal(data.to_pandas()[['id']], records)
        self._delete_table(test_table_name)

    def testPartitionUploadAndDownloadByRawTunnel(self):
        test_table_name = tn('pyodps_test_arrow_partition_tunnel')
        test_table_partition = 'ds=test'
        self.odps.delete_table(test_table_name, if_exists=True)

        table = self._create_partitioned_table(test_table_name)
        table.create_partition(test_table_partition)
        data = self._gen_data()

        self._upload_data(test_table_name, data, partition_spec=test_table_partition)
        records = self._download_data(test_table_name, partition_spec=test_table_partition)
        pd.testing.assert_frame_equal(data.to_pandas(), records)

    def testPartitionDownloadWithSpecifiedColumns(self):
        test_table_name = tn('pyodps_test_arrow_tunnel_partition_columns')
        test_table_partition = 'ds=test'
        self.odps.delete_table(test_table_name, if_exists=True)

        table = self._create_partitioned_table(test_table_name)
        table.create_partition(test_table_partition)
        data = self._gen_data()

        self._upload_data(test_table_name, data, partition_spec=test_table_partition)
        records = self._download_data(test_table_name, partition_spec=test_table_partition,
                                      columns=['int_num'])
        pd.testing.assert_frame_equal(data.to_pandas()[['int_num']], records)

    def testUploadAndDownloadWithCompress(self):
        raw_chunk_size = options.chunk_size
        options.chunk_size = 16

        try:
            test_table_name = tn('pyodps_test_arrow_zlib_tunnel')
            self.odps.delete_table(test_table_name, if_exists=True)

            self._create_table(test_table_name)
            data = self._gen_data()

            self._upload_data(test_table_name, data, compress=True)
            records = self._download_data(test_table_name, compress=True)
            pd.testing.assert_frame_equal(data.to_pandas(), records)

            self._delete_table(test_table_name)
        finally:
            options.chunk_size = raw_chunk_size


if __name__ == '__main__':
    unittest.main()
