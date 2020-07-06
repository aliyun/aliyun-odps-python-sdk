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

import math
import random
import time
from datetime import datetime, date
from decimal import Decimal
from multiprocessing.pool import ThreadPool

try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters
try:
    import pytz
except ImportError:
    pytz = None

from odps.compat import reload_module
from odps.tests.core import TestBase, to_str, tn, snappy_case, pandas_case, odps2_typed_case
from odps.compat import unittest, OrderedDict, Decimal, Monthdelta
from odps.models import Schema
from odps import types, options
from odps.tunnel import TableTunnel


def bothPyAndC(func):
    def inner(self, *args, **kwargs):
        try:
            import cython
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

                self.tunnel = TableTunnel(self.odps, endpoint=self.odps._tunnel_endpoint)
                self.mode = t

                func(self, *args, **kwargs)
            finally:
                setattr(options, 'force_{0}'.format(t), old_config)

    return inner


class Test(TestBase):
    def _upload_data(self, test_table, records, compress=False, **kw):
        upload_ss = self.tunnel.create_upload_session(test_table, **kw)
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

    def _stream_upload_data(self, test_table, records, compress=False, **kw):
        upload_ss = self.tunnel.create_stream_upload_session(test_table, **kw)
        writer = upload_ss.open_record_writer(compress=compress)

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

    def _buffered_upload_data(self, test_table, records, buffer_size=None, compress=False, **kw):
        upload_ss = self.tunnel.create_upload_session(test_table, **kw)
        writer = upload_ss.open_record_writer(buffer_size=buffer_size, compress=compress)
        for r in records:
            record = upload_ss.new_record()
            for i, it in enumerate(r):
                record[i] = it
            writer.write(record)
        writer.close()
        upload_ss.commit(writer.get_blocks_written())

    def _download_data(self, test_table, compress=False, columns=None, **kw):
        count = kw.pop('count', 4)
        download_ss = self.tunnel.create_download_session(test_table, **kw)
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
            ('c' * 300, -2 ** 63 + 1, -2.222, datetime(1999, 5, 25, 3, 10), True, Decimal(22222),
             ['false'], OrderedDict({'false': 0})),
            ('c' * 20, -2 ** 11 + 1, 2.222, datetime(1961, 10, 30, 11, 32), True, Decimal(33333),
             ['true'], OrderedDict({'false': 0})),
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
                                      schema=Schema.from_lists(fields, types, ['ds'], ['string']))

    def _delete_table(self, table_name):
        self.odps.delete_table(table_name)

    @bothPyAndC
    def testUploadAndDownloadByRawTunnel(self):
        test_table_name = tn('pyodps_test_raw_tunnel')
        self._create_table(test_table_name)
        data = self._gen_data()

        self._upload_data(test_table_name, data)
        records = self._download_data(test_table_name)
        self.assertSequenceEqual(data, records)

        self._delete_table(test_table_name)

    @bothPyAndC
    def testStreamUploadAndDownloadTunnel(self):
        test_table_name = tn('pyodps_test_stream_upload')
        self._create_table(test_table_name)
        data = self._gen_data()

        self._stream_upload_data(test_table_name, data)
        records = self._download_data(test_table_name)
        self.assertSequenceEqual(data, records)

        self._delete_table(test_table_name)

    @bothPyAndC
    def testBufferredUploadAndDownloadByRawTunnel(self):
        table, data = self._gen_table(size=10)
        self._buffered_upload_data(table, data)
        records = self._download_data(table)
        self._assert_reads_data_equal(records, data)
        self._delete_table(table)

        table, data = self._gen_table(size=10)
        self._buffered_upload_data(table, data, buffer_size=1024)
        records = self._download_data(table)
        self._assert_reads_data_equal(records, data)
        self._delete_table(table)

    @bothPyAndC
    @unittest.skipIf(pytz is None, 'pytz not installed')
    def testBufferredUploadAndDownloadWithTimezone(self):
        from odps.utils import _get_tz
        zones = (False, True, 'Asia/Shanghai', 'America/Los_Angeles')
        try:
            for zone in zones:
                tz = _get_tz(zone)
                options.local_timezone = zone
                table, data = self._gen_table(size=10)
                self._buffered_upload_data(table, data)
                records = self._download_data(table, count=10)

                if not isinstance(zone, bool):
                    for idx, rec in enumerate(records):
                        new_rec = []
                        for d in rec:
                            if not isinstance(d, datetime):
                                new_rec.append(d)
                                continue
                            self.assertEqual(d.tzinfo.zone, tz.zone)
                            new_rec.append(d.replace(tzinfo=None))
                        records[idx] = tuple(new_rec)

                self._assert_reads_data_equal(records, data)
                self._delete_table(table)
        finally:
            options.local_timezone = None

    @bothPyAndC
    def testDownloadWithSpecifiedColumns(self):
        test_table_name = tn('pyodps_test_raw_tunnel_columns')
        self._create_table(test_table_name)

        data = self._gen_data()
        self._upload_data(test_table_name, data)

        records = self._download_data(test_table_name, columns=['id'])
        self.assertSequenceEqual([r[0] for r in records], [r[0] for r in data])
        for r in records:
            for i in range(1, len(r)):
                self.assertIsNone(r[i])
        self._delete_table(test_table_name)

    @bothPyAndC
    def testDownloadLimitation(self):
        old_limit = options.table_read_limit

        test_table_name = tn('pyodps_test_tunnel_limit')
        self._create_table(test_table_name)
        data = self._gen_data()
        self._upload_data(test_table_name, data * 20)

        options.table_read_limit = 10
        records = self.assertWarns(lambda: self._download_data(test_table_name, compress=True, count=20))
        self.assertEqual(len(records), 10)

        options.table_read_limit = None
        records = self.assertNoWarns(lambda: self._download_data(test_table_name, compress=True, count=20))
        self.assertEqual(len(records), 20)

        options.table_read_limit = old_limit

    @bothPyAndC
    def testPartitionUploadAndDownloadByRawTunnel(self):
        test_table_name = tn('pyodps_test_raw_partition_tunnel')
        test_table_partition = 'ds=test'
        self.odps.delete_table(test_table_name, if_exists=True)

        table = self._create_partitioned_table(test_table_name)
        table.create_partition(test_table_partition)
        data = self._gen_data()

        self._upload_data(test_table_name, data, partition_spec=test_table_partition)
        records = self._download_data(test_table_name, partition_spec=test_table_partition)
        self.assertSequenceEqual(data, [r[:-1] for r in records])

        self._delete_table(test_table_name)

    @bothPyAndC
    def testPartitionDownloadWithSpecifiedColumns(self):
        test_table_name = tn('pyodps_test_raw_tunnel_partition_columns')
        test_table_partition = 'ds=test'
        self.odps.delete_table(test_table_name, if_exists=True)

        table = self._create_partitioned_table(test_table_name)
        table.create_partition(test_table_partition)
        data = self._gen_data()

        self._upload_data(test_table_name, data, partition_spec=test_table_partition)
        records = self._download_data(test_table_name, partition_spec=test_table_partition,
                                      columns=['int_num'])
        self.assertSequenceEqual([r[1] for r in data], [r[0] for r in records])

        self._delete_table(test_table_name)

    @bothPyAndC
    def testUploadAndDownloadByZlibTunnel(self):
        raw_chunk_size = options.chunk_size
        options.chunk_size = 16

        try:
            test_table_name = tn('pyodps_test_zlib_tunnel')
            self._create_table(test_table_name)
            data = self._gen_data()

            self._upload_data(test_table_name, data, compress=True)
            records = self._download_data(test_table_name, compress=True)
            self.assertSequenceEqual(data, records)

            self._delete_table(test_table_name)
        finally:
            options.chunk_size = raw_chunk_size

    @bothPyAndC
    def testBufferredUploadAndDownloadByZlibTunnel(self):
        table, data = self._gen_table(size=10)
        self._buffered_upload_data(table, data, compress=True)
        records = self._download_data(table, compress=True)
        self._assert_reads_data_equal(records, data)
        self._delete_table(table)

    @snappy_case
    @bothPyAndC
    def testUploadAndDownloadBySnappyTunnel(self):
        test_table_name = tn('pyodps_test_snappy_tunnel')
        self._create_table(test_table_name)
        data = self._gen_data()

        self._upload_data(test_table_name, data, compress=True, compress_algo='snappy')
        records = self._download_data(test_table_name, compress=True, compress_algo='snappy')
        self.assertSequenceEqual(data, records)

        self._delete_table(test_table_name)

    @snappy_case
    @bothPyAndC
    def testBufferredUploadAndDownloadBySnappyTunnel(self):
        table, data = self._gen_table(size=10)
        self._buffered_upload_data(table, data, compress=True, compress_algo='snappy')
        records = self._download_data(table, compress=True, compress_algo='snappy')
        self._assert_reads_data_equal(records, data)
        self._delete_table(table)

    def _gen_random_bigint(self):
        return random.randint(*types.bigint._bounds)

    def _gen_random_string(self, max_length=15):
        gen_letter = lambda: letters[random.randint(0, 51)]
        return to_str(''.join([gen_letter() for _ in range(random.randint(1, 15))]))

    def _gen_random_double(self):
        return random.uniform(-2**32, 2**32)

    def _gen_random_datetime(self):
        return datetime.fromtimestamp(random.randint(0, int(time.time())))

    def _gen_random_boolean(self):
        return random.uniform(-1, 1) > 0

    def _gen_random_decimal(self):
        return Decimal(str(self._gen_random_double()))

    def _gen_random_array_type(self):
        t = random.choice(['string', 'bigint', 'double', 'boolean'])
        return types.Array(t)

    def _gen_random_map_type(self):
        random_key_type = random.choice(['bigint', 'string'])
        random_value_type = random.choice(['bigint', 'string', 'double'])

        return types.Map(random_key_type, random_value_type)

    def _gen_random_array(self, random_type, size=None):
        size = size or random.randint(100, 500)

        random_type = types.validate_data_type(random_type)
        if isinstance(random_type, types.Array):
            random_type = random_type.value_type
        method = getattr(self, '_gen_random_%s' % random_type.name)
        array = [method() for _ in range(size)]

        return array

    def _gen_random_map(self, random_map_type):
        size = random.randint(100, 500)

        random_map_type = types.validate_data_type(random_map_type)

        key_arrays = self._gen_random_array(random_map_type.key_type, size)
        value_arrays = self._gen_random_array(random_map_type.value_type, size)

        m = OrderedDict(zip(key_arrays, value_arrays))
        return m

    def _gen_table(self, partition=None, partition_type=None, partition_val=None, size=100):
        def gen_name(name):
            if '<' in name:
                name = name.split('<', 1)[0]
            if len(name) > 4:
                name = name[:4]
            else:
                name = name[:2]
            return name

        test_table_name = tn('pyodps_test_tunnel')
        types = ['bigint', 'string', 'double', 'datetime', 'boolean', 'decimal']
        types.append(self._gen_random_array_type().name)
        types.append(self._gen_random_map_type().name)
        random.shuffle(types)
        names = [gen_name(t) for t in types]

        self.odps.delete_table(test_table_name, if_exists=True)
        partition_names = [partition, ] if partition else None
        partition_types = [partition_type, ] if partition_type else None
        table = self.odps.create_table(
            test_table_name,
            Schema.from_lists(names, types, partition_names=partition_names,
                              partition_types=partition_types))
        if partition_val:
            table.create_partition('%s=%s' % (partition, partition_val))

        data = []
        for i in range(size):
            record = []
            for t in types:
                n = t.split('<', 1)[0]
                method = getattr(self, '_gen_random_'+n)
                if n in ('map', 'array'):
                    record.append(method(t))
                elif n == 'double' and i == 0:
                    record.append(float('nan'))
                else:
                    record.append(method())
            if partition is not None and partition_val is not None:
                record.append(partition_val)
            data.append(record)

        return table, data

    def _assert_reads_data_equal(self, reads, data, ignore_tz=True):
        for val1, val2 in zip(data, reads):
            for it1, it2 in zip(val1, val2):
                if isinstance(it1, dict):
                    self.assertEqual(len(it1), len(it2))
                    self.assertTrue(any(it1[k] == it2[k] for k in it1))
                elif isinstance(it1, list):
                    self.assertSequenceEqual(it1, it2)
                elif isinstance(it1, float) and math.isnan(it1) and \
                        isinstance(it2, float) and math.isnan(it2):
                    continue
                else:
                    self.assertEqual(it1, it2)

    @bothPyAndC
    def testTableUploadAndDownloadTunnel(self):
        table, data = self._gen_table()

        records = [table.new_record(values=d) for d in data]
        self.odps.write_table(table, 0, records)

        reads = list(self.odps.read_table(table, len(data)))
        for val1, val2 in zip(data, [r.values for r in reads]):
            for it1, it2 in zip(val1, val2):
                if isinstance(it1, dict):
                    self.assertEqual(len(it1), len(it2))
                    self.assertTrue(any(it1[k] == it2[k] for k in it1))
                elif isinstance(it1, list):
                    self.assertSequenceEqual(it1, it2)
                else:
                    if isinstance(it1, float) and math.isnan(it1) \
                            and isinstance(it2, float) and math.isnan(it2):
                        continue
                    self.assertEqual(it1, it2)

        table.drop()

    @bothPyAndC
    def testMultiTableUploadAndDownloadTunnel(self):
        table, data = self._gen_table(size=10)

        records = [table.new_record(values=d) for d in data]

        self.odps.write_table(table, 0, records[:5])
        self.odps.write_table(table, 0, records[5:])

        reads = list(self.odps.read_table(table, len(data)))
        for val1, val2 in zip(sorted(data), sorted([r.values for r in reads])):
            for it1, it2 in zip(val1, val2):
                if isinstance(it1, dict):
                    self.assertEqual(len(it1), len(it2))
                    self.assertTrue(any(it1[k] == it2[k] for k in it1))
                elif isinstance(it1, list):
                    self.assertSequenceEqual(it1, it2)
                elif isinstance(it1, float) and math.isnan(it1) and \
                        isinstance(it2, float) and math.isnan(it2):
                    continue
                else:
                    self.assertEqual(it1, it2)

    @bothPyAndC
    def testParallelTableUploadAndDownloadTunnel(self):
        p = 'ds=test'

        table, data = self._gen_table(partition=p.split('=', 1)[0], partition_type='string',
                                      partition_val=p.split('=', 1)[1])
        self.assertTrue(table.exist_partition(p))
        records = [table.new_record(values=d) for d in data]

        n_blocks = 5
        blocks = list(range(n_blocks))
        n_threads = 2
        thread_pool = ThreadPool(n_threads)

        def gen_block_records(block_id):
            c = len(data)
            st = int(c / n_blocks * block_id)
            if block_id < n_blocks - 1:
                ed = int(c / n_blocks * (block_id + 1))
            else:
                ed = c
            return records[st: ed]

        def write(w):
            def inner(arg):
                idx, r = arg
                w.write(idx, r)
            return inner

        with table.open_writer(partition=p, blocks=blocks) as writer:
            thread_pool.map(write(writer), [(i, gen_block_records(i)) for i in blocks])

        for step in range(1, 4):
            reads = []
            expected = []

            with table.open_reader(partition=p) as reader:
                count = reader.count

                for i in range(n_blocks):
                    start = int(count / n_blocks * i)
                    if i < n_blocks - 1:
                        end = int(count / n_blocks * (i + 1))
                    else:
                        end = count
                    for record in reader[start:end:step]:
                        reads.append(record)
                    expected.extend(data[start:end:step])

            self.assertEqual(len(expected), len(reads))
            for val1, val2 in zip(expected, [r.values for r in reads]):
                for it1, it2 in zip(val1[:-1], val2[:-1]):
                    if isinstance(it1, dict):
                        self.assertEqual(len(it1), len(it2))
                        self.assertTrue(any(it1[k] == it2[k] for k in it1))
                    elif isinstance(it1, list):
                        self.assertSequenceEqual(it1, it2)
                    elif isinstance(it1, float) and math.isnan(it1) and \
                            isinstance(it2, float) and math.isnan(it2):
                        continue
                    else:
                        self.assertEqual(it1, it2)

        table.drop()

    @odps2_typed_case
    @bothPyAndC
    def testPrimitiveTypes2(self):
        table_name = tn('test_hivetunnel_singleton_types')
        self.odps.delete_table(table_name, if_exists=True)

        table = self.odps.create_table(table_name, 'col1 tinyint, col2 smallint, col3 int, col4 float, col5 binary',
                                       lifecycle=1)
        self.assertListEqual(table.schema.types,
                             [types.tinyint, types.smallint, types.int_, types.float_, types.binary])

        contents = [
            [127, 32767, 1234321, 10.5432, b'Hello, world!'],
            [-128, -32768, 4312324, 20.1234, b'Excited!'],
            [-1, 10, 9875479, 20.1234, b'Bravo!'],
        ]
        self.odps.write_table(table_name, contents)
        written = list(self.odps.read_table(table_name))
        values = [list(v.values) for v in written]
        self.assertListAlmostEqual(contents, values, only_float=False, places=4)

        table.drop(if_exists=True)

    @odps2_typed_case
    @bothPyAndC
    def testDate(self):
        table_name = tn('test_hivetunnel_date_io')
        self.odps.delete_table(table_name, if_exists=True)

        table = self.odps.create_table(table_name, 'col1 int, col2 date', lifecycle=1)
        self.assertListEqual(table.schema.types, [types.int_, types.date])

        contents = [
            [0, date(2020, 2, 12)],
            [1, date(1900, 1, 1)],
            [2, date(2000, 3, 20)]
        ]
        self.odps.write_table(table_name, contents)
        written = list(self.odps.read_table(table_name))
        values = [list(v.values) for v in written]
        self.assertListEqual(contents, values)

        table.drop(if_exists=True)

    @pandas_case
    @odps2_typed_case
    @bothPyAndC
    def testTimestamp(self):
        import pandas as pd
        table_name = tn('test_hivetunnel_timestamp_io')
        self.odps.delete_table(table_name, if_exists=True)

        table = self.odps.create_table(table_name, 'col1 int, col2 timestamp', lifecycle=1)
        self.assertListEqual(table.schema.types, [types.int_, types.timestamp])

        contents = [
            [0, pd.Timestamp('2013-09-21 11:23:35.196045321')],
            [1, pd.Timestamp('1998-02-15 23:59:21.943829154')],
            [2, pd.Timestamp('2017-10-31 00:12:39.396583106')],
        ]
        self.odps.write_table(table_name, contents)
        written = list(self.odps.read_table(table_name))
        values = [list(v.values) for v in written]
        self.assertListEqual(contents, values)

        table.drop(if_exists=True)

    @odps2_typed_case
    @bothPyAndC
    def testLengthLimitTypes(self):
        table_name = tn('test_hivetunnel_length_limit_io')
        self.odps.delete_table(table_name, if_exists=True)

        table = self.odps.create_table(table_name, 'col1 int, col2 varchar(20), col3 char(30)', lifecycle=1)
        self.assertEqual(table.schema.types[0], types.int_)
        self.assertIsInstance(table.schema.types[1], types.Varchar)
        self.assertEqual(table.schema.types[1].size_limit, 20)
        self.assertIsInstance(table.schema.types[2], types.Char)
        self.assertEqual(table.schema.types[2].size_limit, 30)

        contents = [
            [0, 'agdesfdr', 'sadfklaslkjdvvn'],
            [1, 'sda;fkd', 'asdlfjjls;admc'],
            [2, 'aetlkakls;dfj', 'sadffafafsafsaf'],
        ]
        self.odps.write_table(table_name, contents)
        written = list(self.odps.read_table(table_name))

        contents = [r[:2] + [r[2] + ' ' * (30 - len(r[2]))] for r in contents]
        values = [list(v.values) for v in written]
        self.assertListEqual(contents, values)

        table.drop(if_exists=True)

    @odps2_typed_case
    @bothPyAndC
    def testDecimal2(self):
        table_name = tn('test_hivetunnel_decimal_io')
        self.odps.delete_table(table_name, if_exists=True)

        table = self.odps.create_table(table_name, 'col1 int, col2 decimal(6,2), '
                                                   'col3 decimal(10), col4 decimal(10,3)', lifecycle=1)
        self.assertEqual(table.schema.types[0], types.int_)
        self.assertIsInstance(table.schema.types[1], types.Decimal)
        # comment out due to behavior change of ODPS SQL
        # self.assertIsNone(table.schema.types[1].precision)
        # self.assertIsNone(table.schema.types[1].scale)
        self.assertIsInstance(table.schema.types[2], types.Decimal)
        self.assertEqual(table.schema.types[2].precision, 10)
        self.assertIsInstance(table.schema.types[3], types.Decimal)
        self.assertEqual(table.schema.types[3].precision, 10)
        self.assertEqual(table.schema.types[3].scale, 3)

        contents = [
            [0, Decimal('2.34'), Decimal('34567'), Decimal('56.789')],
            [1, Decimal('11.76'), Decimal('9321'), Decimal('19.125')],
            [2, Decimal('134.21'), Decimal('1642'), Decimal('999.214')],
        ]
        self.odps.write_table(table_name, contents)
        written = list(self.odps.read_table(table_name))
        values = [list(v.values) for v in written]
        self.assertListEqual(contents, values)

        table.drop(if_exists=True)

    @pandas_case
    @odps2_typed_case
    @bothPyAndC
    def testIntervals(self):
        import pandas as pd
        empty_table_name = tn('test_hivetunnel_interval_empty')
        self.odps.delete_table(empty_table_name, if_exists=True)
        empty_table = self.odps.create_table(empty_table_name, 'col1 int', if_not_exists=True)

        table_name = tn('test_hivetunnel_interval_io')
        self.odps.delete_table(table_name, if_exists=True)
        self.odps.execute_sql("create table %s lifecycle 1 as\n"
                              "select interval_day_time('2 1:2:3') as col1,"
                              "  interval_year_month('10-11') as col2\n"
                              "from %s" %
                              (table_name, empty_table_name))
        table = self.odps.get_table(table_name)
        self.assertListEqual(table.schema.types, [types.interval_day_time, types.interval_year_month])

        contents = [
            [pd.Timedelta(seconds=1048576, nanoseconds=428571428), Monthdelta(13)],
            [pd.Timedelta(seconds=934567126, nanoseconds=142857142), Monthdelta(-20)],
            [pd.Timedelta(seconds=91230401, nanoseconds=285714285), Monthdelta(50)],
        ]
        self.odps.write_table(table_name, contents)
        written = list(self.odps.read_table(table_name))
        values = [list(v.values) for v in written]
        self.assertListEqual(contents, values)

        table.drop()
        empty_table.drop()

    @odps2_typed_case
    @bothPyAndC
    def testStruct(self):
        table_name = tn('test_hivetunnel_struct_io')
        self.odps.delete_table(table_name, if_exists=True)

        col_def = 'col1 int, col2 struct<name:string,age:int,'\
                  'parents:map<varchar(20),smallint>,hobbies:array<varchar(100)>>'
        table = self.odps.create_table(table_name, col_def, lifecycle=1)
        self.assertEqual(table.schema.types[0], types.int_)
        self.assertIsInstance(table.schema.types[1], types.Struct)

        contents = [
            [0, {'name': 'user1', 'age': 20, 'parents': {'fa': 5, 'mo': 6},
                 'hobbies': ['worship', 'yacht']}],
            [1, {'name': 'user2', 'age': 65, 'parents': {'fa': 2, 'mo': 7},
                 'hobbies': ['ukelele', 'chess']}],
            [2, {'name': 'user3', 'age': 32, 'parents': {'fa': 1, 'mo': 3},
                 'hobbies': ['poetry', 'calligraphy']}],
        ]
        self.odps.write_table(table_name, contents)
        written = list(self.odps.read_table(table_name))
        values = [list(v.values) for v in written]
        self.assertListEqual(contents, values)

        table.drop(if_exists=True)

    @bothPyAndC
    def testAsyncTableUploadAndDownload(self):
        table, data = self._gen_table()

        records = [table.new_record(values=d) for d in data]
        self.odps.write_table(table, 0, records)

        reads = list(self.odps.read_table(table, len(data), async_=True))
        for val1, val2 in zip(data, [r.values for r in reads]):
            for it1, it2 in zip(val1, val2):
                if isinstance(it1, dict):
                    self.assertEqual(len(it1), len(it2))
                    self.assertTrue(any(it1[k] == it2[k] for k in it1))
                elif isinstance(it1, list):
                    self.assertSequenceEqual(it1, it2)
                else:
                    if isinstance(it1, float) and math.isnan(it1) \
                            and isinstance(it2, float) and math.isnan(it2):
                        continue
                    self.assertEqual(it1, it2)

        table.drop()


if __name__ == '__main__':
    unittest.main()
