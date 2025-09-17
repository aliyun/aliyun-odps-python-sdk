# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import datetime
import itertools

import pytest

try:
    import pandas as pd
except ImportError:
    pd = None

from ...compat import Decimal
from ...models import Record
from ...types import Column, OdpsSchema
from .. import hasher as py_hasher

hasher_mods = [py_hasher]

try:
    from .. import hasher_c as c_hasher

    hasher_mods.append(c_hasher)
except ImportError:
    c_hasher = None


params = list(itertools.product(hasher_mods, {None, pd}))


def _build_schema_and_record(pd):
    columns = [
        Column("col1", "bigint"),
        Column("col2", "float"),
        Column("col3", "double"),
        Column("col4", "boolean"),
        Column("col5", "string"),
        Column("col6", "date"),
        Column("col7", "datetime"),
    ]
    values = [
        145680,
        134.562,
        15672.56271,
        True,
        "hello",
        datetime.date(2022, 12, 5),
        datetime.datetime(2023, 6, 11, 22, 33, 11),
    ]
    if pd is not None:
        columns.extend(
            [
                Column("col8", "timestamp"),
                Column("col9", "interval_day_time"),
            ]
        )
        values.extend(
            [
                pd.Timestamp(2022, 6, 11, 22, 33, 1, 134561, 241),
                pd.Timedelta(
                    days=128, hours=10, minutes=5, seconds=17, microseconds=11
                ),
            ]
        )
    schema = OdpsSchema(columns)
    record = Record(schema=schema, values=values)
    return schema, record


@pytest.mark.parametrize("hasher_mod, pd", params)
def test_default_hasher(hasher_mod, pd):
    assert hasher_mod.hash_value("default", "bigint", 145680) == 1063204611
    assert hasher_mod.hash_value("default", "float", 134.562) == -1512465477
    assert hasher_mod.hash_value("default", "double", 15672.56271) == 1254569207
    assert hasher_mod.hash_value("default", "boolean", True) == 388737479
    assert hasher_mod.hash_value("default", "string", "hello".encode()) == -1259046373
    assert (
        hasher_mod.hash_value("default", "date", datetime.date(2022, 12, 5))
        == 903574500
    )
    assert (
        hasher_mod.hash_value(
            "default", "datetime", datetime.datetime(2022, 6, 11, 22, 33, 11)
        )
        == -2026178719
    )
    if pd is not None:
        assert (
            hasher_mod.hash_value(
                "default", "timestamp", pd.Timestamp("2023-07-05 11:24:15.145673214")
            )
            == -31960127
        )
        assert (
            hasher_mod.hash_value(
                "default",
                "interval_day_time",
                pd.Timedelta(seconds=100002, microseconds=2000, nanoseconds=1),
            )
            == -1088782317
        )

    schema, rec = _build_schema_and_record(pd)
    col_names = [c.name for c in schema.columns]
    rec_hasher = hasher_mod.RecordHasher(schema, "default", col_names)
    if pd is not None:
        assert rec_hasher.hash_record(rec) == 91440730
    else:
        assert rec_hasher.hash_record(rec) == 99191800

    assert hasher_mod.hash_value("default", "decimal(4,2)", Decimal("0")) == 0
    assert hasher_mod.hash_value("default", "decimal(4,2)", Decimal("-1")) == 1405574141
    assert (
        hasher_mod.hash_value("default", "decimal(18,2)", Decimal("12.34"))
        == -904458774
    )
    assert (
        hasher_mod.hash_value("default", "decimal(18,2)", Decimal("-9.8e11"))
        == -1816428053
    )
    assert (
        hasher_mod.hash_value("default", "decimal(38,18)", Decimal("6.4"))
        == -1846789132
    )


@pytest.mark.parametrize("hasher_mod, pd", params)
def test_legacy_hasher(hasher_mod, pd):
    assert hasher_mod.hash_value("legacy", "bigint", 145680) == 145680
    assert hasher_mod.hash_value("legacy", "float", 134.562) == 1124503519
    assert hasher_mod.hash_value("legacy", "double", 15672.56271) == 1177487321
    assert hasher_mod.hash_value("legacy", "boolean", False) == -978963218
    assert hasher_mod.hash_value("legacy", "string", "hello".encode()) == 99162322
    assert (
        hasher_mod.hash_value("legacy", "date", datetime.date(2022, 12, 5))
        == 1670198400
    )
    assert (
        hasher_mod.hash_value(
            "legacy", "datetime", datetime.datetime(2022, 6, 11, 22, 33, 11)
        )
        == 1395582425
    )
    if pd is not None:
        assert (
            hasher_mod.hash_value(
                "legacy", "timestamp", pd.Timestamp("2023-07-05 11:24:15.145673214")
            )
            == -779619479
        )
        assert (
            hasher_mod.hash_value(
                "legacy",
                "interval_day_time",
                pd.Timedelta(seconds=100002, microseconds=2000, nanoseconds=1),
            )
            == -2145458903
        )

    schema, rec = _build_schema_and_record(pd)
    col_names = [c.name for c in schema.columns]
    rec_hasher = hasher_mod.RecordHasher(schema, "legacy", col_names)
    if pd is not None:
        assert rec_hasher.hash_record(rec) == 1171650329
        assert rec_hasher.hash_list(list(rec.values), need_index=False) == 1171650329
        assert rec_hasher.hash_list(list(rec.values), need_index=True) == 1171650329
    else:
        assert rec_hasher.hash_record(rec) == 1259167848
        assert rec_hasher.hash_list(list(rec.values), need_index=False) == 1259167848
        assert rec_hasher.hash_list(list(rec.values), need_index=True) == 1259167848

    assert hasher_mod.hash_value("legacy", "decimal(4,2)", Decimal("0")) == 0
    assert hasher_mod.hash_value("legacy", "decimal(4,2)", Decimal("-1")) == 99
    assert hasher_mod.hash_value("legacy", "decimal(18,2)", Decimal("12.34")) == 1234
    assert (
        hasher_mod.hash_value("default", "decimal(18,2)", Decimal("-9.8e11"))
        == -1816428053
    )
    assert (
        hasher_mod.hash_value("legacy", "decimal(38,18)", Decimal("6.4")) == 978411031
    )
