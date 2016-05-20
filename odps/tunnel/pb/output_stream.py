# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.
# http://code.google.com/p/protobuf/
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

"""This implementation is modified from google's original Protobuf implementation.
The original author is: robinson@google.com (Will Robinson).
Modified by onesuperclark@gmail.com(onesuper).
"""

import array
import struct

from . import errors
from . import wire_format


class OutputStream(object):
    """Contains all logic for writing bits, and ToString() to get the result."""

    def __init__(self):
        self._buffer = array.array('B')

    def append_raw_bytes(self, raw_bytes):
        """Appends raw_bytes to our internal buffer."""
        self._buffer.fromstring(raw_bytes)

    def append_little_endian32(self, unsigned_value):
        """Appends an unsigned 32-bit integer to the internal buffer,
        in little-endian byte order.
        """
        if not 0 <= unsigned_value <= wire_format.UINT32_MAX:
            raise errors.EncodeError(
                'Unsigned 32-bit out of range: %d' % unsigned_value)
        self._buffer.fromstring(struct.pack(
            wire_format.FORMAT_UINT32_LITTLE_ENDIAN, unsigned_value))

    def append_little_endian64(self, unsigned_value):
        """Appends an unsigned 64-bit integer to the internal buffer,
        in little-endian byte order.
        """
        if not 0 <= unsigned_value <= wire_format.UINT64_MAX:
            raise errors.EncodeError(
                'Unsigned 64-bit out of range: %d' % unsigned_value)
        self._buffer.fromstring(struct.pack(
            wire_format.FORMAT_UINT64_LITTLE_ENDIAN, unsigned_value))

    def append_varint32(self, value):
        """Appends a signed 32-bit integer to the internal buffer,
        encoded as a varint.  (Note that a negative varint32 will
        always require 10 bytes of space.)
        """
        if not wire_format.INT32_MIN <= value <= wire_format.INT32_MAX:
            raise errors.EncodeError('Value out of range: %d' % value)
        self.append_varint64(value)

    def append_var_uint32(self, value):
        """Appends an unsigned 32-bit integer to the internal buffer,
        encoded as a varint.
        """
        if not 0 <= value <= wire_format.UINT32_MAX:
            raise errors.EncodeError('Value out of range: %d' % value)
        self.append_var_uint64(value)

    def append_varint64(self, value):
        """Appends a signed 64-bit integer to the internal buffer,
        encoded as a varint.
        """
        if not wire_format.INT64_MIN <= value <= wire_format.INT64_MAX:
            raise errors.EncodeError('Value out of range: %d' % value)
        if value < 0:
            value += (1 << 64)
        self.append_var_uint64(value)

    def append_var_uint64(self, unsigned_value):
        """Appends an unsigned 64-bit integer to the internal buffer,
        encoded as a varint.
        """
        if not 0 <= unsigned_value <= wire_format.UINT64_MAX:
            raise errors.EncodeError('Value out of range: %d' % unsigned_value)
        while True:
            bits = unsigned_value & 0x7f
            unsigned_value >>= 7
            if unsigned_value:
                bits |= 0x80
            self._buffer.append(bits)
            if not unsigned_value:
                break

    def tostring(self):
        """Returns a string containing the bytes in our internal buffer."""
        return self._buffer.tostring()

    def __len__(self):
        return len(self._buffer)
