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

import sys
import time

import pytest

from ... import types as odps_types
from ..tools.runners import _convert_value

pytestmark = pytest.mark.skipif(
    sys.version_info[0] != 2, reason="Only needed for Python 2"
)


def test_py2_convert():
    cur_mills = int(1000 * time.time())
    assert 123 == _convert_value("123", odps_types.bigint)
    assert cur_mills == _convert_value(cur_mills, odps_types.datetime)
    assert [cur_mills] == _convert_value(
        [cur_mills], odps_types.Array(odps_types.datetime)
    )
    assert {"key": cur_mills} == _convert_value(
        {"key": cur_mills}, odps_types.Map(odps_types.string, odps_types.datetime)
    )

    struct_type = odps_types.Struct({"key": odps_types.datetime})
    assert (cur_mills,) == _convert_value(
        struct_type.namedtuple_type(cur_mills), struct_type
    )
    assert (cur_mills,) == _convert_value({"key": cur_mills}, struct_type)
