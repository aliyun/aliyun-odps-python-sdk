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

from ..crc import Crc32c, Crc32
from .. import utils
from ..compat import six
from ..types import integer_builtins, float_builtins


class Checksum(object):
    TRUE = bytearray([1])
    FALSE = bytearray([0])
    
    def __init__(self, method='crc32c'):
        self.crc = Crc32c() if method.lower() == 'crc32c' else Crc32()

    def _mode(self):
        # use for UT to check if use c extension
        try:
            from ..crc32c_c import Crc32c

            return 'c' if isinstance(self.crc, Crc32c) else 'py'
        except ImportError:
            return 'py'

    def update_bool(self, val):
        assert isinstance(val, bool)

        val = self.TRUE if val else self.FALSE
        self.update(val)

    def update_int(self, val):
        assert isinstance(val, integer_builtins)

        val = struct.pack('<i', val)
        self.update(val)

    def update_long(self, val):
        assert isinstance(val, integer_builtins)

        val = struct.pack('<q', val)
        self.update(val)

    def update_float(self, val):
        assert isinstance(val, float_builtins)

        val = struct.pack('<d', val)
        self.update(val)
            
    def update(self, b):
        b = bytearray(utils.to_binary(b))
        self.crc.update(b)
        
    def getvalue(self):
        return self.crc.getvalue()
    
    def reset(self):
        return self.crc.reset()
