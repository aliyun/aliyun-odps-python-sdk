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

from __future__ import print_function

import cProfile
import json
import os
import time
from contextlib import contextmanager
from pstats import Stats

import pytest

if bool(json.loads(os.getenv("FORCE_PY", "0"))):
    from odps import options

    options.force_py = True

from datetime import datetime

from odps.compat import Decimal
from odps.conftest import odps, tunnel  # noqa: F401
from odps.models import TableSchema

# remember to reset False before committing
ENABLE_PROFILE = bool(json.loads(os.getenv("ENABLE_PROFILE", "0")))
DUMP_PROFILE = bool(json.loads(os.getenv("DUMP_PROFILE", "0")))

COMPRESS_DATA = True
BUFFER_SIZE = 1024 * 1024
DATA_AMOUNT = 100000
STRING_LITERAL = (
    "Soft kitty, warm kitty, little ball of fur; happy kitty, sleepy kitty, purr, purr"
)
NUMERIC_ONLY = bool(json.loads(os.getenv("NUMERIC_ONLY", "0")))


@pytest.fixture
def schema():
    fields = ["a", "b", "c", "d", "e", "f"]
    types = ["bigint", "double", "datetime", "boolean", "string", "decimal"]
    return TableSchema.from_lists(fields, types)


@contextmanager
def profiled():
    if ENABLE_PROFILE:
        pr = cProfile.Profile()
        pr.enable()
    try:
        yield
    finally:
        if ENABLE_PROFILE:
            if DUMP_PROFILE:
                pr.dump_stats("profile.out")
            p = Stats(pr)
            p.strip_dirs()
            p.sort_stats("time")
            p.print_stats(40)
            p.print_callees("types.py:846\(validate_value", 20)
            p.print_callees("types.py:828\(_validate_primitive_value", 20)
            p.print_callees("tabletunnel.py:185\(write", 20)


def test_write(odps, schema, tunnel):
    table_name = "pyodps_test_tunnel_write_performance"
    odps.create_table(table_name, schema, if_not_exists=True)
    ss = tunnel.create_upload_session(table_name)
    r = ss.new_record()

    start = time.time()
    with ss.open_record_writer(0) as writer, profiled():
        for i in range(DATA_AMOUNT):
            r[0] = 2**63 - 1
            r[1] = 0.0001
            r[2] = datetime(2015, 11, 11) if not NUMERIC_ONLY else None
            r[3] = True
            r[4] = STRING_LITERAL if not NUMERIC_ONLY else None
            r[5] = Decimal("3.15") if not NUMERIC_ONLY else None
            writer.write(r)
        n_bytes = writer.n_bytes
    print(
        n_bytes, "bytes", float(n_bytes) / 1024 / 1024 / (time.time() - start), "MiB/s"
    )
    ss.commit([0])
    odps.delete_table(table_name, if_exists=True)


def test_read(odps, schema, tunnel):
    table_name = "pyodps_test_tunnel_read_performance"
    odps.delete_table(table_name, if_exists=True)
    t = odps.create_table(table_name, schema)

    def gen_data():
        for i in range(DATA_AMOUNT):
            r = t.new_record()
            r[0] = 2**63 - 1
            r[1] = 0.0001
            r[2] = datetime(2015, 11, 11) if not NUMERIC_ONLY else None
            r[3] = True
            r[4] = STRING_LITERAL if not NUMERIC_ONLY else None
            r[5] = Decimal("3.15") if not NUMERIC_ONLY else None
            yield r

    odps.write_table(t, gen_data())

    ds = tunnel.create_download_session(table_name)

    start = time.time()
    cnt = 0
    with ds.open_record_reader(0, ds.count) as reader, profiled():
        for _ in reader:
            cnt += 1
        n_bytes = reader.n_bytes
    print(
        n_bytes, "bytes", float(n_bytes) / 1024 / 1024 / (time.time() - start), "MiB/s"
    )
    assert DATA_AMOUNT == cnt
    odps.delete_table(table_name, if_exists=True)


def test_buffered_write(odps, schema, tunnel):
    table_name = "test_tunnel_bufferred_write"
    odps.create_table(table_name, schema, if_not_exists=True)
    ss = tunnel.create_upload_session(table_name)
    r = ss.new_record()

    start = time.time()
    with ss.open_record_writer(
        buffer_size=BUFFER_SIZE, compress=COMPRESS_DATA
    ) as writer:
        for i in range(DATA_AMOUNT):
            r[0] = 2**63 - 1
            r[1] = 0.0001
            r[2] = datetime(2015, 11, 11) if not NUMERIC_ONLY else None
            r[3] = True
            r[4] = STRING_LITERAL if not NUMERIC_ONLY else None
            r[5] = Decimal("3.15") if not NUMERIC_ONLY else None
            writer.write(r)
        n_bytes = writer.n_bytes
    print(
        n_bytes, "bytes", float(n_bytes) / 1024 / 1024 / (time.time() - start), "MiB/s"
    )
    ss.commit(writer.get_blocks_written())
    odps.delete_table(table_name, if_exists=True)
