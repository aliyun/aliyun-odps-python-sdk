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

from libc.stdint cimport *

cdef int32_t get_varint32(object input)

cdef int64_t get_varint64(object input)

cdef int32_t get_signed_varint32(object input)

cdef int64_t get_signed_varint64(object input)

cdef int set_varint32(int32_t varint, bytearray buf)

cdef int set_varint64(int64_t varint, bytearray buf)

cdef int set_signed_varint32(int32_t varint, bytearray buf)

cdef int set_signed_varint64(int64_t varint, bytearray buf)