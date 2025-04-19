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

import pytest

try:
    import pandas as pd
    import pyarrow as pa
except ImportError:
    pa = pd = None

from ...compat import datetime_utcnow
from ...models import Record, TableSchema
from ...tests.core import pandas_case, pyarrow_case
from ..core import FunctionCall, VisitedExpressions

_routine_expr_str = """[
  {
    "leafExprDesc": {
      "reference": {
        "name": "dt"
      },
      "type": "timestamp"
    }
  },
  {
    "leafExprDesc": {
      "constant": "%(interval)s",
      "type": "string"
    }
  },
  {
    "functionCall": {
      "name": "trunc_time",
      "type": "string"
    }
  }
]"""


def test_expression_parse():
    parsed = VisitedExpressions.parse(_routine_expr_str % dict(interval="day"))
    assert isinstance(parsed, FunctionCall)
    assert str(parsed) == "trunc_time(`dt`, 'day')"
    assert parsed.references == ["dt"]

    expr_str = """[
      {
        "functionCall": {
          "name": "current_timestamp_ntz",
          "type": "timestamp"
        }
      },
      {
        "leafExprDesc": {
          "constant": "hour",
          "type": "string"
        }
      },
      {
        "functionCall": {
          "name": "trunc_time",
          "type": "string"
        }
      },
      {
        "leafExprDesc": {
          "constant": "day",
          "type": "string"
        }
      },
      {
        "functionCall": {
          "name": "TRUNC_time",
          "type": "string"
        }
      }
    ]"""
    parsed = VisitedExpressions.parse(expr_str)
    assert isinstance(parsed, FunctionCall)
    assert (
        str(parsed) == "trunc_time(trunc_time(current_timestamp_ntz(), 'hour'), 'day')"
    )


interval_fmt_combine = [
    ("YEAR", "%Y"),
    ("MONTH", "%Y-%m"),
    ("DAY", "%Y-%m-%d"),
    ("HOUR", "%Y-%m-%d %H:00:00"),
]


@pytest.mark.parametrize("interval,fmt", interval_fmt_combine)
def test_expression_call_record(interval, fmt):
    schema = TableSchema.from_lists(["dt", "pt"], ["datetime", "string"])
    record = Record(schema=schema)

    parsed = VisitedExpressions.parse(_routine_expr_str % dict(interval=interval))
    record["dt"] = datetime_utcnow()
    assert parsed.eval(record) == record["dt"].strftime(fmt)


@pytest.mark.parametrize("interval,fmt", interval_fmt_combine)
@pandas_case
def test_expression_call_series(interval, fmt):
    dt_array = [
        pd.Timestamp("2025-02-25 11:23:41"),
        pd.Timestamp("2025-02-26 12:23:41"),
        pd.Timestamp("2025-02-26 13:23:41"),
        pd.Timestamp("2025-02-26 14:23:41"),
        pd.Timestamp("2025-02-27 15:23:41"),
    ]
    df = pd.DataFrame({"dt": dt_array})

    parsed = VisitedExpressions.parse(_routine_expr_str % dict(interval=interval))
    expected = df.dt.dt.strftime(fmt)
    pd.testing.assert_series_equal(parsed.eval(df), expected)


@pytest.mark.parametrize("interval,fmt", interval_fmt_combine)
@pyarrow_case
def test_expression_call_arrow_record_batch(interval, fmt):
    pt_format = "%Y-%m-%d %H:%M:%S"
    dt_array = [
        datetime.datetime.strptime("2025-02-25 11:23:41", pt_format),
        datetime.datetime.strptime("2025-02-26 12:23:41", pt_format),
        datetime.datetime.strptime("2025-02-26 13:23:41", pt_format),
        datetime.datetime.strptime("2025-02-26 14:23:41", pt_format),
        datetime.datetime.strptime("2025-02-27 15:23:41", pt_format),
    ]
    tb = pa.RecordBatch.from_arrays([pa.array(dt_array)], names=["dt"])

    parsed = VisitedExpressions.parse(_routine_expr_str % dict(interval=interval))
    expected = pa.array([c.strftime(fmt) for c in dt_array])
    pd.testing.assert_series_equal(parsed.eval(tb).to_pandas(), expected.to_pandas())
