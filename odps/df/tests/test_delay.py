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

import functools

import pytest

from ...tests.core import get_result, tn
from ...models import TableSchema
from .. import DataFrame, Delay


@pytest.fixture
def setup(odps):
    test_table_name = tn('pyodps_test_delay')
    schema = TableSchema.from_lists(['id', 'name', 'value'], ['bigint', 'string', 'bigint'])

    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema)
    data = [[1, 'name1', 1], [2, 'name1', 2], [3, 'name1', 3],
            [4, 'name2', 1], [5, 'name2', 2], [6, 'name2', 3]]

    with table.open_writer() as w:
        w.write(data)

    return DataFrame(table), data


def test_sync_execute(setup):
    df, data = setup
    delay = Delay()
    filtered = df[df.id > 0].cache()
    sub_futures = [filtered[filtered.value == i].execute(delay=delay) for i in range(1, 3)]
    delay.execute(timeout=10 * 60)

    assert all(f.done() for f in sub_futures) is True
    for i in range(1, 3):
        assert get_result(sub_futures[i - 1].result()) == [d for d in data if d[2] == i]

    # execute on executed delay
    sub_future = filtered[filtered.value == 3].execute(delay=delay)
    delay.execute(timeout=10 * 60)
    assert sub_future.done() is True
    assert get_result(sub_future.result()) == [d for d in data if d[2] == 3]


def test_async_execute(setup):
    def make_filter(df, cnt):
        def waiter(val, c):
            import time
            time.sleep(30 * c)
            return val

        f_df = df[df.value == cnt]
        return f_df[f_df.exclude('value'), f_df.value.map(functools.partial(waiter, cnt))]

    df, data = setup
    delay = Delay()
    filtered = df[df.id > 0].cache()
    sub_futures = [make_filter(filtered, i).execute(delay=delay) for i in range(1, 4)]
    future = delay.execute(async_=True, n_parallel=3)
    pytest.raises(RuntimeError, delay.execute)

    try:
        for i in range(1, 4):
            assert future.done() is False
            assert any(f.done() for f in sub_futures[i - 1:]) is False
            assert all(f.done() for f in sub_futures[:i - 1]) is True
            assert get_result(sub_futures[i - 1].result()) == [d for d in data if d[2] == i]
        assert all(f.done() for f in sub_futures) is True
        future.result(timeout=10 * 60)
        assert future.done() is True
    finally:
        future.result()


def test_persist_execute(odps, setup):
    df, data = setup
    delay = Delay()
    filtered = df[df.id > 0].cache()

    persist_table_name = tn('pyodps_test_delay_persist')
    schema = TableSchema.from_lists(
        ['id', 'name', 'value'], ['bigint', 'string', 'bigint'],
        ['pt', 'ds'], ['string', 'string']
    )
    odps.delete_table(persist_table_name, if_exists=True)
    odps.create_table(persist_table_name, schema)

    future1 = filtered[filtered.value > 2].persist(persist_table_name, partition='pt=a,ds=d1', delay=delay)
    future2 = filtered[filtered.value < 2].persist(persist_table_name, partition='pt=a,ds=d2', delay=delay)

    delay.execute()
    df1 = future1.result()
    df2 = future2.result()

    assert [c.lhs.name for c in df1.predicate.children()] == ['pt', 'ds']
    result1 = get_result(df1.execute())
    assert [r[:-2] for r in result1] == [d for d in data if d[2] > 2]
    assert [c.lhs.name for c in df2.predicate.children()] == ['pt', 'ds']
    result2 = get_result(df2.execute())
    assert [r[:-2] for r in result2] == [d for d in data if d[2] < 2]
