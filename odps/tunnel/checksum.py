#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import struct

import six

from ..crc import Crc32c
from .. import utils


class Checksum(object):
    TRUE = bytearray([1])
    FALSE = bytearray([0])
    
    def __init__(self):
        self.crc = Crc32c()

    def update_bool(self, val):
        assert isinstance(val, bool)

        val = self.TRUE if val else self.FALSE
        self.update(val)

    def update_int(self, val):
        assert isinstance(val, six.integer_types)

        val = struct.pack('<i', val)
        self.update(val)

    def update_long(self, val):
        assert isinstance(val, six.integer_types)

        val = struct.pack('<q', val)
        self.update(val)

    def update_float(self, val):
        assert isinstance(val, float)

        val = struct.pack('<d', val)
        self.update(val)
            
    def update(self, b, off=None, length=None):
        b = utils.to_binary(b)

        off = off or 0
        length = length or len(b)
        self.crc.update(b, off, length)
        
    def getvalue(self):
        return self.crc.getvalue()
    
    def reset(self):
        return self.crc.reset()
