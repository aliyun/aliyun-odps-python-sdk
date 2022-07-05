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
from datetime import datetime
from decimal import Decimal

try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters  # noqa: F401

from odps.compat import reload_module
from odps.tests.core import TestBase, tn, snappy_case
from odps.compat import OrderedDict
from odps.models import Schema
from odps import options
from odps.tunnel import TableTunnel, InstanceTunnel


def bothPyAndC(func):
    def inner(self, *args, **kwargs):
        try:
            import cython  # noqa: F401

            ts = 'py', 'c'
        except ImportError:
            ts = 'py',
            import warnings
            warnings.warn('No c code tests for table tunnel')
        for t in ts:
            old_config = getattr(options, 'force_{0}'.format(t))
            setattr(options, 'force_{0}'.format(t), True)
            try:
                from odps.models import record
                reload_module(record)

                from odps import models
                reload_module(models)

                from odps.tunnel.io import writer
                reload_module(writer)

                from odps.tunnel.io import reader
                reload_module(reader)

                from odps.tunnel import tabletunnel
                reload_module(tabletunnel)

                from odps.tunnel import instancetunnel
                reload_module(instancetunnel)

                self.tunnel = TableTunnel(self.odps, endpoint=self.odps._tunnel_endpoint)
                self.mode = t

                func(self, *args, **kwargs)
            finally:
                setattr(options, 'force_{0}'.format(t), old_config)

    return inner


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.instance_tunnel = InstanceTunnel(self.odps)

    def _upload_data(self, test_table, records, compress=False, **kw):
        upload_ss = self.tunnel.create_upload_session(test_table, **kw)
        # make sure session reprs work well
        repr(upload_ss)
        writer = upload_ss.open_record_writer(0, compress=compress)

        # test use right py or c writer
        self.assertEqual(self.mode, writer._mode())

        for r in records:
            record = upload_ss.new_record()
            # test record
            self.assertEqual(self.mode, record._mode())
            for i, it in enumerate(r):
                record[i] = it
            writer.write(record)
        writer.close()
        upload_ss.commit([0, ])

    def _download_instance_data(self, test_instance, compress=False, columns=None, **kw):
        count = kw.pop('count', 3)
        download_ss = self.instance_tunnel.create_download_session(test_instance, **kw)
        # make sure session reprs work well
        repr(download_ss)
        with download_ss.open_record_reader(0, count, compress=compress, columns=columns) as reader:
            # test use right py or c writer
            self.assertEqual(self.mode, reader._mode())

            records = []

            for record in reader:
                records.append(tuple(record.values))
                self.assertEqual(self.mode, record._mode())

            return records

    def _gen_data(self):
        return [
            ('hello \x00\x00 world', 2**63-1, math.pi, datetime(2015, 9, 19, 2, 11, 25, 33000),
             True, Decimal('3.14'), ['simple', 'easy'], OrderedDict({'s': 1})),
            ('goodbye', 222222, math.e, datetime(2020, 3, 10), False, Decimal('2.555555'),
             ['true', None], OrderedDict({'true': 1})),
            ('c'*300, -2**63+1, -2.222, datetime(1990, 5, 25, 3, 10), True, Decimal(22222),
             ['false'], OrderedDict({'false': 0})),
        ]

    def _create_table(self, table_name):
        fields = ['id', 'int_num', 'float_num', 'dt', 'bool', 'dec', 'arr', 'm']
        types = ['string', 'bigint', 'double', 'datetime', 'boolean', 'decimal',
                 'array<string>', 'map<string,bigint>']

        self.odps.delete_table(table_name, if_exists=True)
        return self.odps.create_table(table_name, schema=Schema.from_lists(fields, types), lifecycle=1)

    def _create_partitioned_table(self, table_name):
        fields = ['id', 'int_num', 'float_num', 'dt', 'bool', 'dec', 'arr', 'm']
        types = ['string', 'bigint', 'double', 'datetime', 'boolean', 'decimal',
                 'array<string>', 'map<string,bigint>']

        self.odps.delete_table(table_name, if_exists=True)
        return self.odps.create_table(table_name,
                                      schema=Schema.from_lists(fields, types, ['ds'], ['string']),
                                      lifecycle=1)

    def _delete_table(self, table_name):
        self.odps.delete_table(table_name)

    @bothPyAndC
    def testDownloadByRawTunnel(self):
        test_table_name = tn('pyodps_test_raw_inst_tunnel')
        self._create_table(test_table_name)
        data = self._gen_data()

        self._upload_data(test_table_name, data)
        inst = self.odps.execute_sql('select * from %s' % test_table_name)
        records = self._download_instance_data(inst)
        self.assertSequenceEqual(data, records)

        self._delete_table(test_table_name)

    @bothPyAndC
    def testUploadAndDownloadByZlibTunnel(self):
        raw_chunk_size = options.chunk_size
        options.chunk_size = 16

        try:
            test_table_name = tn('pyodps_test_zlib_inst_tunnel')
            self._create_table(test_table_name)
            data = self._gen_data()

            self._upload_data(test_table_name, data, compress=True)
            inst = self.odps.execute_sql('select * from %s' % test_table_name)
            records = self._download_instance_data(inst, compress=True)
            self.assertSequenceEqual(data, records)

            self._delete_table(test_table_name)
        finally:
            options.chunk_size = raw_chunk_size

    @snappy_case
    @bothPyAndC
    def testUploadAndDownloadBySnappyTunnel(self):
        test_table_name = tn('pyodps_test_snappy_inst_tunnel')
        self._create_table(test_table_name)
        data = self._gen_data()

        self._upload_data(test_table_name, data, compress=True, compress_algo='snappy')
        inst = self.odps.execute_sql('select * from %s' % test_table_name)
        records = self._download_instance_data(inst, compress=True, compress_algo='snappy')
        self.assertSequenceEqual(data, records)

        self._delete_table(test_table_name)

    @bothPyAndC
    def testPartitionUploadAndDownloadByRawTunnel(self):
        test_table_name = tn('pyodps_test_raw_partition_tunnel')
        test_table_partition = 'ds=test'
        self.odps.delete_table(test_table_name, if_exists=True)

        table = self._create_partitioned_table(test_table_name)
        table.create_partition(test_table_partition)
        data = self._gen_data()

        self._upload_data(test_table_name, data, partition_spec=test_table_partition)
        inst = self.odps.execute_sql("select * from %s where ds='test'" % test_table_name)
        records = self._download_instance_data(inst)
        self.assertSequenceEqual(data, [r[:-1] for r in records])

        self._delete_table(test_table_name)
