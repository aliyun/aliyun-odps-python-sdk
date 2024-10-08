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

from .. import ExecutionContext, get_execution_context


def test_get_counter():
    ctx = get_execution_context()
    counters = ctx.get_counters()
    c = counters.get_counter("test_group", "test")
    assert "test" == c.get_name()
    assert 0 == c.get_value()

    c.increment(1)
    assert 1 == c.get_value()


def test_single_counters():
    ctx1 = get_execution_context()
    counters = ctx1.get_counters()
    counters.get_counter("test_group", "test")
    assert 1 == counters.size()

    ctx2 = ExecutionContext()
    counters2 = ctx2.get_counters()
    counters2.get_counter("test_group2", "test2")
    assert 2 == counters2.size()
