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


import struct

from ..crc import Crc32c, Crc32
from .. import utils


class Checksum(object):
    TRUE = bytearray([1])
    FALSE = bytearray([0])
    
    def __init__(self, method='crc32c'):
        self.crc = Crc32c() if method.lower() == 'crc32c' else Crc32()

    def _mode(self):
        # use for UT to check if use c extension
        try:
            from ..src.crc32c_c import Crc32c

            return 'c' if isinstance(self.crc, Crc32c) else 'py'
        except ImportError:
            return 'py'

    def update_bool(self, val):
        assert isinstance(val, bool)

        val = self.TRUE if val else self.FALSE
        self._update(val)

    def update_int(self, val):
        val = struct.pack('<i', val)
        self._update(val)

    def update_long(self, val):
        val = struct.pack('<q', val)
        self._update(val)

    def update_float(self, val):
        val = struct.pack('<f', val)
        self._update(val)

    def update_double(self, val):
        val = struct.pack('<d', val)
        self._update(val)

    def _update(self, b):
        # update crc without type checking
        self.crc.update(bytearray(b))
            
    def update(self, b):
        b = utils.to_binary(b)
        self._update(b)
        
    def getvalue(self):
        return self.crc.getvalue()
    
    def reset(self):
        return self.crc.reset()
