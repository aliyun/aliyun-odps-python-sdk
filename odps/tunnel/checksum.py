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


from odps.utils import PyCrc32c, DataStream


class Checksum(object):
    TRUE = DataStream(chr(1))
    FALSE = DataStream(chr(0))
    
    def __init__(self):
        self.crc = PyCrc32c()
        self.byte_buffer = DataStream(8)
        
    def update_(self, v, force=None):
        def _update_boolean(v):
            b = self.TRUE if v else self.FALSE
            self.crc.update(b, 0, 1)
        def _update_int(v):
            self.byte_buffer.clear()
            self.byte_buffer.append(v, '<i')
            self.crc.update(self.byte_buffer, 0, 4)
        def _update_long(v):
            self.byte_buffer.clear()
            self.byte_buffer.append(v, '<q')
            self.crc.update(self.byte_buffer, 0, 8)
        def _update_float(v):
            self.byte_buffer.clear()
            self.byte_buffer.append(v, '<d')
            self.crc.update(self.byte_buffer, 0, 8)
        
        if force is not None:
            if force == 'boolean':
                _update_boolean(v)
            elif force == 'int':
                _update_int(v)
            elif force == 'long':
                _update_long(v)
            elif force == 'float':
                _update_float(v)
            else:
                raise ValueError(
                    'Force can only be boolean, int, long or float')
        else:
            if v is True or v is False:
                _update_boolean(v)
            elif isinstance(v, int):
                _update_int(v)
            elif isinstance(v, long):
                _update_long(v)
            elif isinstance(v, float):
                _update_float(v)
            
    def update(self, b, off, length):
        self.crc.update(b, off, length)
        
    def get_value(self):
        return self.crc.get_value()
    
    def reset(self):
        return self.crc.reset()
