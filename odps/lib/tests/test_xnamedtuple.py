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

import sys

import pytest

from ..xnamedtuple import xnamedtuple


def test_xnamedtuple():
    # test normal use
    tp_cls = xnamedtuple("TPCls", "field1 field2")
    tp_val = tp_cls("val1", "val2")
    assert tp_val.field1 == "val1"
    assert tp_val.field2 == "val2"
    assert tp_val["field1"] == "val1"
    assert tp_val["field2"] == "val2"
    assert tp_val.asdict()["field1"] == "val1"
    assert tp_val.asdict()["field2"] == "val2"
    assert tp_val.get("field3") is None
    assert sorted(tp_val.keys()) == ["field1", "field2"]
    assert sorted(tp_val.values()) == ["val1", "val2"]
    assert sorted(tp_val.items()) == [("field1", "val1"), ("field2", "val2")]

    new_tp = tp_val.replace(field2="new_val2")
    assert new_tp.field2 == "new_val2"

    with pytest.raises(KeyError):
        _ = tp_val["field3"]
    with pytest.raises(AttributeError):
        _ = tp_val.field3

    # test fields with reserved words
    tp_cls = xnamedtuple("TPCls", "abc def values")
    tp_val = tp_cls("val1", "val2", "val3")
    # make sure reserved fields can also be obtained
    assert getattr(tp_val, "def") == "val2"
    assert tp_val.values == "val3"
