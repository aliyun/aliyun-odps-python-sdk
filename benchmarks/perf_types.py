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

import cProfile
from datetime import datetime
from decimal import Decimal
from pstats import Stats

import pytest

from odps.models import Record, TableSchema

COMPRESS_DATA = True
BUFFER_SIZE = 1024 * 1024
DATA_AMOUNT = 100000
STRING_LITERAL = (
    "Soft kitty, warm kitty, little ball of fur; happy kitty, sleepy kitty, purr, purr"
)


@pytest.fixture
def schema():
    pr = cProfile.Profile()
    pr.enable()
    fields = [
        "bigint",
        "double",
        "datetime",
        "boolean",
        "string",
        "decimal",
        "array",
        "map",
        "struct",
    ]
    types = [
        "bigint",
        "double",
        "datetime",
        "boolean",
        "string",
        "decimal",
        "array<bigint>",
        "map<string, bigint>",
        "struct<key:string, value:bigint>",
    ]
    try:
        schema = TableSchema.from_lists(fields, types)
        schema.build_snapshot()
        yield schema
    finally:
        p = Stats(pr)
        p.strip_dirs()
        p.sort_stats("cumtime")
        p.print_stats(40)


def test_set_record_field_bigint(schema):
    r = Record(schema=schema)
    for _ in range(10**6):
        r["bigint"] = 2**63 - 1


def test_set_record_field_double(schema):
    r = Record(schema=schema)
    for _ in range(10**6):
        r["double"] = 0.0001


def test_set_record_field_boolean(schema):
    r = Record(schema=schema)
    for _ in range(10**6):
        r["boolean"] = False


def test_set_record_field_string(schema):
    r = Record(schema=schema)
    for _ in range(10**6):
        r["string"] = STRING_LITERAL


def test_set_record_field_datetime(schema):
    r = Record(schema=schema)
    for _ in range(10**6):
        r["datetime"] = datetime(2016, 1, 1)


def test_set_record_field_decimal(schema):
    r = Record(schema=schema)
    for _ in range(10**6):
        r["decimal"] = Decimal("1.111111")


def test_set_record_field_array(schema):
    r = Record(schema=schema)
    for _ in range(10**6):
        r["array"] = [2**63 - 1]


def test_set_record_field_map(schema):
    r = Record(schema=schema)
    for _ in range(10**6):
        r["map"] = {"data_key": 2**63 - 1}


def test_set_record_field_struct(schema):
    r = Record(schema=schema)
    for _ in range(10**6):
        r["struct"] = ("data_key", 2**63 - 1)
