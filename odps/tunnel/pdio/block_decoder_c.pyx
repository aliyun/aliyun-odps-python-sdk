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
from libcpp.string cimport string

from . import errno
from ..pb.wire_format import TAG_TYPE_BITS as PY_TAG_TYPE_BITS, _TAG_TYPE_MASK as _PY_TAG_TYPE_MASK

cdef:
    int TAG_TYPE_BITS = PY_TAG_TYPE_BITS
    int _TAG_TYPE_MASK = _PY_TAG_TYPE_MASK

    int BD_SUCCESS = errno.BD_SUCCESS
    int BD_BUFFER_EXHAUSTED = errno.BD_BUFFER_EXHAUSTED
    int BD_CHECKSUM_INVALID = errno.BD_CHECKSUM_INVALID
    int BD_COUNT_NOT_MATCH = errno.BD_COUNT_NOT_MATCH
    int BD_INVALID_STREAM_DATA = errno.BD_INVALID_STREAM_DATA
    int BD_INVALID_PB_TAG = errno.BD_INVALID_PB_TAG


cdef class Decoder:
    cdef init(self, char *buf_ptr, int buf_len):
        self._buf_ptr = buf_ptr
        self._buf_len = buf_len
        self._last_error = BD_SUCCESS
        self._pos = 0

    cdef int get_last_error(self) nogil:
        return self._last_error

    cdef void set_last_error(self, int errno) nogil:
        self._last_error = errno

    cdef int position(self) nogil:
        return self._pos

    cdef void add_offset(self, int n) nogil:
        self._pos += n

    cdef int32_t read_field_number(self) nogil:
        cdef int32_t tag_and_type
        tag_and_type = self.read_uint32()
        return tag_and_type >> TAG_TYPE_BITS

    cdef FieldParam read_field_number_and_wire_type(self) nogil:
        cdef int32_t tag_and_type
        cdef FieldParam fp
        tag_and_type = self.read_uint32()
        fp.field_number = tag_and_type >> TAG_TYPE_BITS
        fp.wire_type = tag_and_type & _TAG_TYPE_MASK
        return fp

    cdef int32_t read_sint32(self) nogil:
        return self._get_signed_varint32()

    cdef uint32_t read_uint32(self) nogil:
        return self._get_varint32()

    cdef int64_t read_sint64(self) nogil:
        return self._get_signed_varint64()

    cdef uint64_t read_uint64(self) nogil:
        return self._get_varint64()

    cdef bint read_bool(self) nogil:
        return self._get_varint32()

    cdef double read_double(self) nogil:
        cdef double retval

        if sizeof(double) + self._pos > self._buf_len:
            self._last_error = BD_BUFFER_EXHAUSTED
            return 0.0

        memcpy(&retval, self._buf_ptr + self._pos, sizeof(double))
        self._pos += sizeof(double)
        return retval

    cdef float read_float(self) nogil:
        cdef float retval

        if sizeof(float) + self._pos > self._buf_len:
            self._last_error = BD_BUFFER_EXHAUSTED
            return 0.0

        memcpy(&retval, self._buf_ptr + self._pos, sizeof(float))
        self._pos += sizeof(float)
        return retval

    cdef string read_string(self) nogil:
        cdef int size
        cdef int old_pos = self._pos

        size = self.read_uint32()
        if self._last_error:
            return string()
        if size + self._pos > self._buf_len:
            self._pos = old_pos
            self._last_error = BD_BUFFER_EXHAUSTED
            return string()

        return string(self._buf_ptr + self._pos, size)

    cdef int _read_input_byte(self) nogil:
        cdef int ret
        if self._pos < self._buf_len:
            self._last_error = BD_SUCCESS
            ret = self._buf_ptr[self._pos]
            self._pos += 1
            return ret
        else:
            self._last_error = BD_BUFFER_EXHAUSTED
            return 0

    cdef int32_t _get_varint32(self) nogil:
        """
        Deserialize a protobuf varint read from input stream; update
        offset based on number of bytes consumed.
        """
        cdef int32_t value = 0
        cdef int32_t base = 1
        cdef int index = 0
        cdef int val_byte

        while True:
            val_byte = self._read_input_byte()
            if self._last_error != BD_SUCCESS:
                return 0
            value += (val_byte & 0x7F) * base
            if val_byte & 0x80:
                base *= 128
            else:
                return value

    cdef int64_t _get_varint64(self) nogil:
        """
        Deserialize a protobuf varint read from input stream; update
        offset based on number of bytes consumed.
        """
        cdef int64_t value = 0
        cdef int64_t base = 1
        cdef int index = 0
        cdef int val_byte

        while True:
            val_byte = self._read_input_byte()
            if self._last_error != BD_SUCCESS:
                return 0
            value += (val_byte & 0x7F) * base
            if val_byte & 0x80:
                base *= 128
            else:
                return value

    cdef int32_t _get_signed_varint32(self) nogil:
        """
        Deserialize a signed protobuf varint read from input stream;
        update offset based on number of bytes consumed.
        """
        cdef uint32_t value = 0
        cdef int32_t base = 1
        cdef int index = 0
        cdef int val_byte

        while True:
            val_byte = self._read_input_byte()
            if self._last_error != BD_SUCCESS:
                return 0
            value += (val_byte & 0x7F) * base
            if val_byte & 0x80:
                base *= 128
            else:
                return <int32_t>((value >> 1) ^ (-(value & 1))) # zigzag decoding

    cdef int64_t _get_signed_varint64(self) nogil:
        """
        Deserialize a signed protobuf varint read from input stream;
        update offset based on number of bytes consumed.
        """
        cdef uint64_t value = 0
        cdef int64_t base = 1
        cdef int index = 0
        cdef int val_byte

        while True:
            val_byte = self._read_input_byte()
            if self._last_error != BD_SUCCESS:
                return 0
            value += (val_byte & 0x7F) * base
            if val_byte & 0x80:
                base *= 128
            else:
                return <int64_t>((value >> 1) ^ (-(value & 1))) # zigzag decoding
