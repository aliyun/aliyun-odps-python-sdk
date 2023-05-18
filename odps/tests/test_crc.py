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

from .. import crc
from ..tests.core import py_and_c


@py_and_c("odps.crc")
def test_crc32c():
    crc_obj = crc.Crc32c()
    assert 0 == crc_obj.getvalue()
    buf = bytearray(b'abc')
    crc_obj.update(buf)
    assert 910901175 == crc_obj.getvalue()
    buf = bytearray(b'1111111111111111111')
    crc_obj.update(buf)
    assert 2917307201 == crc_obj.getvalue()
