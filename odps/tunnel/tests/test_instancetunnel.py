#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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
from collections import OrderedDict
from datetime import datetime
from decimal import Decimal

import mock
import pytest

try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters  # noqa: F401

from ... import options
from ...models import TableSchema
from ...tests.core import get_code_mode, py_and_c, tn
from .. import InstanceTunnel, TableTunnel


def _reloader():
    from ...conftest import get_config

    cfg = get_config()
    cfg.tunnel = TableTunnel(cfg.odps, endpoint=cfg.odps._tunnel_endpoint)


py_and_c_deco = py_and_c(
    [
        "odps.models.record",
        "odps.models",
        "odps.tunnel.io.reader",
        "odps.tunnel.io.writer",
        "odps.tunnel.tabletunnel",
        "odps.tunnel.instancetunnel",
    ],
    _reloader,
)


@pytest.fixture
def instance_tunnel(odps):
    return InstanceTunnel(odps)


def _upload_data(tunnel, test_table, records, compress=False, **kw):
    upload_ss = tunnel.create_upload_session(test_table, **kw)
    # make sure session reprs work well
    repr(upload_ss)
    writer = upload_ss.open_record_writer(0, compress=compress)

    # test use right py or c writer
    assert get_code_mode() == writer._mode()

    for r in records:
        record = upload_ss.new_record()
        # test record
        assert get_code_mode() == record._mode()
        for i, it in enumerate(r):
            record[i] = it
        writer.write(record)
    writer.close()
    upload_ss.commit([0])


def _download_instance_data(
    instance_tunnel, test_instance, compress=False, columns=None, **kw
):
    count = kw.pop("count", 3)
    download_ss = instance_tunnel.create_download_session(test_instance, **kw)
    # make sure session reprs work well
    repr(download_ss)
    with download_ss.open_record_reader(
        0, count, compress=compress, columns=columns
    ) as reader:
        # test use right py or c writer
        assert get_code_mode() == reader._mode()

        records = []

        for record in reader:
            records.append(tuple(record.values))
            assert get_code_mode() == record._mode()

        return records


def _gen_data():
    return [
        (
            "hello \x00\x00 world",
            2**63 - 1,
            math.pi,
            datetime(2015, 9, 19, 2, 11, 25, 33000),
            True,
            Decimal("3.14"),
            ["simple", "easy"],
            OrderedDict({"s": 1}),
        ),
        (
            "goodbye",
            222222,
            math.e,
            datetime(2020, 3, 10),
            False,
            Decimal("2.555555"),
            ["true", None],
            OrderedDict({"true": 1}),
        ),
        (
            "c" * 300,
            -(2**63) + 1,
            -2.222,
            datetime(1990, 5, 25, 3, 10),
            True,
            Decimal(22222),
            ["false"],
            OrderedDict({"false": 0}),
        ),
    ]


def _create_table(odps, table_name):
    fields = ["id", "int_num", "float_num", "dt", "bool", "dec", "arr", "m"]
    types = [
        "string",
        "bigint",
        "double",
        "datetime",
        "boolean",
        "decimal",
        "array<string>",
        "map<string,bigint>",
    ]

    odps.delete_table(table_name, if_exists=True)
    return odps.create_table(
        table_name, TableSchema.from_lists(fields, types), lifecycle=1
    )


def _create_partitioned_table(odps, table_name):
    fields = ["id", "int_num", "float_num", "dt", "bool", "dec", "arr", "m"]
    types = [
        "string",
        "bigint",
        "double",
        "datetime",
        "boolean",
        "decimal",
        "array<string>",
        "map<string,bigint>",
    ]

    odps.delete_table(table_name, if_exists=True)
    return odps.create_table(
        table_name,
        TableSchema.from_lists(fields, types, ["ds"], ["string"]),
        lifecycle=1,
    )


def _delete_table(odps, table_name):
    odps.delete_table(table_name)


@py_and_c_deco
def test_download_by_raw_tunnel(config, instance_tunnel):
    test_table_name = tn("pyodps_test_raw_inst_tunnel")
    _create_table(config.odps, test_table_name)
    data = _gen_data()

    _upload_data(config.tunnel, test_table_name, data)
    inst = config.odps.execute_sql("select * from %s" % test_table_name)
    records = _download_instance_data(instance_tunnel, inst)
    assert list(data) == list(records)

    _delete_table(config.odps, test_table_name)


@py_and_c_deco
def test_tunnel_read_with_retry(config, instance_tunnel):
    from ..instancetunnel import InstanceDownloadSession

    test_table_name = tn("pyodps_test_inst_tunnel_with_retry")
    _create_table(config.odps, test_table_name)
    data = _gen_data()

    _upload_data(config.tunnel, test_table_name, data)
    inst = config.odps.execute_sql("select * from %s" % test_table_name)

    try:
        ranges = []
        original = InstanceDownloadSession._build_input_stream

        def new_build_input_stream(self, start, count, *args, **kw):
            ranges.append((start, count))
            assert start in (0, 2)
            assert start == 0 or count == session.count - 2
            return original(self, start, count, *args, **kw)

        with mock.patch(
            "odps.tunnel.instancetunnel.InstanceDownloadSession._build_input_stream",
            new=new_build_input_stream,
        ):
            session = instance_tunnel.create_download_session(inst)
            reader = session.open_record_reader(0, session.count)

            reader._inject_error(2, ValueError)
            result = [tuple(r.values) for r in reader]

        assert ranges == [(0, 3), (2, 1)]
        assert data == result
    finally:
        _delete_table(config.odps, test_table_name)


@py_and_c_deco
@pytest.mark.parametrize("algo, module", [(None, None), ("snappy", "snappy")])
def test_upload_and_download_with_compress(config, instance_tunnel, algo, module):
    raw_chunk_size = options.chunk_size
    options.chunk_size = 16
    if module:
        pytest.importorskip(module)

    try:
        test_table_name = tn("pyodps_test_zlib_inst_tunnel")
        _create_table(config.odps, test_table_name)
        data = _gen_data()

        _upload_data(
            config.tunnel, test_table_name, data, compress=True, compress_algo=algo
        )
        inst = config.odps.execute_sql("select * from %s" % test_table_name)
        records = _download_instance_data(
            instance_tunnel, inst, compress=True, compress_algo=algo
        )
        assert data == records

        _delete_table(config.odps, test_table_name)
    finally:
        options.chunk_size = raw_chunk_size


@py_and_c_deco
def test_partition_upload_and_download_by_raw_tunnel(config, instance_tunnel):
    test_table_name = tn("pyodps_test_raw_partition_tunnel")
    test_table_partition = "ds=test"
    config.odps.delete_table(test_table_name, if_exists=True)

    table = _create_partitioned_table(config.odps, test_table_name)
    table.create_partition(test_table_partition)
    data = _gen_data()

    _upload_data(
        config.tunnel, test_table_name, data, partition_spec=test_table_partition
    )
    inst = config.odps.execute_sql("select * from %s where ds='test'" % test_table_name)
    records = _download_instance_data(instance_tunnel, inst)
    assert data == [r[:-1] for r in records]

    _delete_table(config.odps, test_table_name)


def test_instance_tunnel_with_quota(odps_with_tunnel_quota, config):
    from odps.rest import RestClient

    orig_request = RestClient.request

    def patch_request(self, *args, **kw):
        if self._endpoint == tunnel_endpoint:
            assert kw["params"]["quotaName"] == quota_name
        return orig_request(self, *args, **kw)

    odps = odps_with_tunnel_quota
    table_name = tn("test_table_tunnel_with_quota")
    odps.delete_table(table_name, if_exists=True)

    quota_name = config.get("test", "default_tunnel_quota_name")
    tunnel = InstanceTunnel(odps, quota_name=quota_name)
    tunnel_endpoint = tunnel.tunnel_rest.endpoint
    tb = odps.create_table(table_name, "col1 string", lifecycle=1)
    with tb.open_writer() as writer:
        writer.write([["data"]])

    inst = odps.execute_sql("select * from " + table_name)

    with mock.patch("odps.rest.RestClient.request", new=patch_request):
        download_session = tunnel.create_download_session(inst)
        assert download_session.quota_name == quota_name
        download_session.reload()
        assert download_session.quota_name == quota_name

        reader = download_session.open_record_reader(0, 1)
        assert list(reader)[0][0] == "data"

    tb.drop()
