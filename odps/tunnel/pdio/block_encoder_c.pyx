# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from libc.stdint cimport *
from libc.string cimport *

from . import errno

cdef:
    int BD_SUCCESS = errno.BD_SUCCESS
    int BD_BUFFER_EXHAUSTED = errno.BD_BUFFER_EXHAUSTED
    int BD_CHECKSUM_INVALID = errno.BD_CHECKSUM_INVALID
    int BD_COUNT_NOT_MATCH = errno.BD_COUNT_NOT_MATCH
    int BD_INVALID_STREAM_DATA = errno.BD_INVALID_STREAM_DATA
    int BD_INVALID_PB_TAG = errno.BD_INVALID_PB_TAG


cdef class Encoder:
    cdef init(self, char *buf_ptr, int buf_len):
        self._buf_ptr = buf_ptr
        self._buf_len = buf_len
        self._last_error = BD_SUCCESS
        self._pos = 0

    cdef int position(self) nogil:
        return self._pos

    cdef int get_last_error(self) nogil:
        return self._last_error

    cdef void set_last_error(self, int errno) nogil:
        self._last_error = errno

    cdef int append_tag(self, int field_num, int wire_type) nogil:
        cdef int key
        key = (field_num << 3) | wire_type
        cdef int size = self._set_varint64(key)
        return size

    cdef int append_sint32(self, int32_t value) nogil:
        return self._set_signed_varint32(value)

    cdef int append_uint32(self, uint32_t value) nogil:
        return self._set_varint32(value)

    cdef int append_sint64(self, int64_t value) nogil:
        return self._set_signed_varint64(value)

    cdef int append_uint64(self, uint64_t value) nogil:
        return self._set_varint64(value)

    cdef int append_bool(self, bint value) nogil:
        return self._set_varint32(value)

    cdef int append_float(self, float value) nogil:
        if self._pos + sizeof(float) >= self._buf_len:
            self._last_error = BD_BUFFER_EXHAUSTED
            return -1
        memcpy(self._buf_ptr + self._pos, &value, sizeof(float))
        self._pos += sizeof(float)
        return sizeof(float)

    cdef int append_double(self, double value) nogil:
        if self._pos + sizeof(double) >= self._buf_len:
            self._last_error = BD_BUFFER_EXHAUSTED
            return -1
        memcpy(self._buf_ptr + self._pos, &value, sizeof(double))
        self._pos += sizeof(double)
        return sizeof(double)

    cdef int append_string(self, const char *ptr, int value_len) nogil:
        cdef int size = self._set_varint32(value_len)
        if self._last_error:
            return -1
        if self._pos + value_len >= self._buf_len:
            self._last_error = BD_BUFFER_EXHAUSTED
            return -1
        memcpy(self._buf_ptr + self._pos, ptr, value_len)
        return size + value_len

    cdef int _set_varint32(self, int32_t varint) nogil:
        """
        Serialize an integer into a protobuf varint; return the number of bytes
        serialized.
        """

        # Negative numbers are always 10 bytes, so we need a uint64_t to
        # facilitate encoding
        cdef uint64_t enc = varint
        bits = enc & 0x7f
        enc >>= 7
        cdef int idx = 0
        while enc:
            if self._pos + idx >= self._buf_len:
                self._last_error = BD_BUFFER_EXHAUSTED
                return -1
            else:
                self._buf_ptr[self._pos + idx] = bits | 0x80
            bits = enc & 0x7f
            enc >>= 7
            idx += 1

        if self._pos + idx >= self._buf_len:
            self._last_error = BD_BUFFER_EXHAUSTED
            return -1
        self._buf_ptr[self._pos + idx] = bits
        self._pos += idx + 1
        return idx + 1

    cdef int _set_varint64(self, int64_t varint) nogil:
        """
        Serialize an integer into a protobuf varint; return the number of bytes
        serialized.
        """

        # Negative numbers are always 10 bytes, so we need a uint64_t to
        # facilitate encoding
        cdef uint64_t enc = varint
        bits = enc & 0x7f
        enc >>= 7
        cdef int idx = 0
        while enc:
            if self._pos + idx >= self._buf_len:
                self._last_error = BD_BUFFER_EXHAUSTED
                return -1
            else:
                self._buf_ptr[self._pos + idx] = bits | 0x80
            bits = enc & 0x7f
            enc >>= 7
            idx += 1

        if self._pos + idx >= self._buf_len:
            self._last_error = BD_BUFFER_EXHAUSTED
            return -1
        self._buf_ptr[self._pos + idx] = bits
        self._pos += idx + 1
        return idx + 1

    cdef int _set_signed_varint32(self, int32_t varint) nogil:
        """
        Serialize an integer into a signed protobuf varint; return the number of
        bytes serialized.
        """
        cdef uint32_t enc
        cdef int idx = 0

        enc = (varint << 1) ^ (varint >> 31) # zigzag encoding
        bits = enc & 0x7f
        enc >>= 7
        while enc:
            if self._pos + idx >= self._buf_len:
                self._last_error = BD_BUFFER_EXHAUSTED
                return -1
            else:
                self._buf_ptr[self._pos + idx] = bits | 0x80
            bits = enc & 0x7f
            enc >>= 7
            idx += 1

        if self._pos + idx >= self._buf_len:
            self._last_error = BD_BUFFER_EXHAUSTED
            return -1
        self._buf_ptr[self._pos + idx] = bits
        self._pos += idx + 1
        return idx + 1


    cdef int _set_signed_varint64(self, int64_t varint) nogil:
        """
        Serialize an integer into a signed protobuf varint; return the number of
        bytes serialized.
        """
        cdef uint64_t enc
        cdef int idx = 0

        enc = (varint << 1) ^ (varint >> 63) # zigzag encoding
        bits = enc & 0x7f
        enc >>= 7
        while enc:
            if self._pos + idx >= self._buf_len:
                self._last_error = BD_BUFFER_EXHAUSTED
                return -1
            else:
                self._buf_ptr[self._pos + idx] = bits | 0x80
            bits = enc & 0x7f
            enc >>= 7
            idx += 1

        if self._pos + idx >= self._buf_len:
            self._last_error = BD_BUFFER_EXHAUSTED
            return -1
        self._buf_ptr[self._pos + idx] = bits
        self._pos += idx + 1
        return idx + 1
