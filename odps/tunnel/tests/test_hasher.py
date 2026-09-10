# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from decimal import Decimal

from ...compat import Version
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
        # pandas < 2.0: nanosecond is positional-or-keyword and gets
        # clobbered by the positional branch, so pass it positionally.
        # pandas >= 2.0: nanosecond is keyword-only, pass it as a kwarg.
        if Version(pd.__version__) < Version("2.0.0"):
            ts_value = pd.Timestamp(2022, 6, 11, 22, 33, 1, 134561, 241)
        else:
            ts_value = pd.Timestamp(2022, 6, 11, 22, 33, 1, 134561, nanosecond=241)
        values.extend(
            [
                ts_value,
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
    # Large bigint values that overflow 64-bit intermediates
    assert hasher_mod.hash_value("default", "bigint", -4149976519821344517) == 808790275
    assert hasher_mod.hash_value("default", "bigint", 6812388553834026379) == 319711033
    assert hasher_mod.hash_value("default", "bigint", 5641369577242833675) == 1388740052
    assert hasher_mod.hash_value("default", "float", 134.562) == -1512465477
    assert hasher_mod.hash_value("default", "double", 15672.56271) == 1254569207
    assert hasher_mod.hash_value("default", "boolean", True) == 388737479
    assert hasher_mod.hash_value("default", "string", "eragf".encode()) == 1281892457
    assert (
        hasher_mod.hash_value("default", "string", "abcdefghijklmnop".encode())
        == -458446633
    )
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
    # Large bigint values that overflow 32-bit in (val >> 32) ^ val
    assert hasher_mod.hash_value("legacy", "bigint", -4149976519821344517) == 2076054188
    assert hasher_mod.hash_value("legacy", "bigint", 6812388553834026379) == -1849908544
    assert hasher_mod.hash_value("legacy", "bigint", 5641369577242833675) == -1945429834
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


@pytest.mark.parametrize("hasher_mod", hasher_mods)
def test_record_hasher_pk_not_at_beginning(hasher_mod):
    """Test RecordHasher when PK columns are not at the beginning of the schema."""
    columns = [
        Column("col1", "string"),
        Column("col2", "string"),
        Column("col3", "string"),
    ]
    schema = OdpsSchema(columns)
    # Hash keys are at positions 1 and 2, not at the beginning
    rec_hasher = hasher_mod.RecordHasher(schema, "default", ["col2", "col3"])

    record = Record(schema=schema, values=["non_pk_val", "pk_val_1", "pk_val_2"])
    hash_result = rec_hasher.hash_record(record)

    # Verify hash_list with need_index=True also works
    hash_list_result = rec_hasher.hash_list(list(record.values), need_index=True)
    assert hash_result == hash_list_result

    # Verify hash_list with need_index=False works (only PK values passed)
    hash_no_idx_result = rec_hasher.hash_list(
        ["pk_val_1", "pk_val_2"], need_index=False
    )
    assert hash_result == hash_no_idx_result


@pytest.mark.parametrize("hasher_mod", hasher_mods)
def test_hash_string_signed_bytes(hasher_mod):
    """Test that hash_string treats bytes as signed (-128..127).

    Python's bytes iteration yields unsigned values (0..255), but the
    reference implementation treats byte arrays as signed. This test
    verifies the pure-Python fallback matches for bytes >= 0x80.
    """
    default_hasher = hasher_mod.get_hasher("default")
    legacy_hasher = hasher_mod.get_hasher("legacy")

    # Single high-byte values where signed vs unsigned interpretation diverges
    assert default_hasher.hash_string(b"\x80") == 656789031
    assert default_hasher.hash_string(b"\xfe") == 613435680
    assert default_hasher.hash_string(b"\xff") == 306848916

    assert legacy_hasher.hash_string(b"\x80") == -128
    assert legacy_hasher.hash_string(b"\xfe") == -2
    assert legacy_hasher.hash_string(b"\xff") == -1

    # Multi-byte with high bytes
    assert default_hasher.hash_string(b"\x80\x81\x82") == 1267950483
    assert legacy_hasher.hash_string(b"\x80\x81\x82") == -127071


@pytest.mark.parametrize("hasher_mod", hasher_mods)
def test_upsert_bucket_assignment(hasher_mod):
    """Test bucket assignment for the exact values from the field bug.

    The string "eragf" was hashed to bucket 15 by the buggy pure-Python
    hasher (no 32-bit truncation of intermediates), but the server computed
    bucket 9, causing "bucket id mismatch: expected=15, computed=9".
    """
    schema = OdpsSchema([Column("a", "string"), Column("b", "bigint")])
    rec_hasher = hasher_mod.RecordHasher(schema, "default", ["a"])
    num_buckets = 16

    test_data = [("abcd", 12345), ("efgh", 94512), ("eragf", 434)]
    expected_buckets = [4, 13, 9]

    for (s, b), expected_bucket in zip(test_data, expected_buckets):
        rec = Record(schema=schema, values=[s, b])
        hash_val = rec_hasher.hash_record(rec)
        bucket = hash_val % num_buckets
        assert (
            bucket == expected_bucket
        ), f"Bucket mismatch for ({s!r}, {b}): got {bucket}, expected {expected_bucket}"
