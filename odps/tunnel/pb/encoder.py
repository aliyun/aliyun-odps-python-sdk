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

import struct

from . import errors
from . import wire_format
from . import output_stream


class Encoder(object):
    """Encodes logical protocol buffer fields to the wire format."""

    def __init__(self):
        self._stream = output_stream.OutputStream()

    def tostring(self):
        """Returns all values encoded in this object as a string."""
        return self._stream.tostring()

    def __len__(self):
        return len(self._stream)

    def append_tag(self, field_number, wire_type):
        """Appends a tag containing field number and wire type information."""
        self._stream.append_var_uint32(wire_format.pack_tag(field_number, wire_type))

    def append_int32(self, value):
        """Appends a 32-bit integer to our buffer, varint-encoded."""
        self._stream.append_varint32(value)

    def append_int64(self, value):
        """Appends a 64-bit integer to our buffer, varint-encoded."""
        self._stream.append_varint64(value)

    def append_uint32(self, unsigned_value):
        """Appends an unsigned 32-bit integer to our buffer, varint-encoded."""
        self._stream.append_var_uint32(unsigned_value)

    def append_uint64(self, unsigned_value):
        """Appends an unsigned 64-bit integer to our buffer, varint-encoded."""
        self._stream.append_var_uint64(unsigned_value)

    def append_sint32(self, value):
        """Appends a 32-bit integer to our buffer, zigzag-encoded and then
        varint-encoded.
        """
        zigzag_value = wire_format.zig_zag_encode(value)
        self._stream.append_var_uint32(zigzag_value)

    def append_sint64(self, value):
        """Appends a 64-bit integer to our buffer, zigzag-encoded and then
        varint-encoded.
        """
        zigzag_value = wire_format.zig_zag_encode(value)
        self._stream.append_var_uint64(zigzag_value)

    def append_fixed32(self, unsigned_value):
        """Appends an unsigned 32-bit integer to our buffer, in little-endian
        byte-order.
        """
        self._stream.append_little_endian32(unsigned_value)

    def append_fixed64(self, unsigned_value):
        """Appends an unsigned 64-bit integer to our buffer, in little-endian
        byte-order.
        """
        self._stream.append_little_endian64(unsigned_value)

    def append_sfixed32(self, value):
        """Appends a signed 32-bit integer to our buffer, in little-endian
        byte-order.
        """
        sign = (value & 0x80000000) and -1 or 0
        if value >> 32 != sign:
            raise errors.EncodeError('SFixed32 out of range: %d' % value)
        self._stream.append_little_endian32(value & 0xffffffff)

    def append_sfixed64(self, value):
        """Appends a signed 64-bit integer to our buffer, in little-endian
        byte-order.
        """
        sign = (value & 0x8000000000000000) and -1 or 0
        if value >> 64 != sign:
            raise errors.EncodeError('SFixed64 out of range: %d' % value)
        self._stream.append_little_endian64(value & 0xffffffffffffffff)

    def append_float(self, value):
        """Appends a floating-point number to our buffer."""
        self._stream.append_raw_bytes(struct.pack('f', value))

    def append_double(self, value):
        """Appends a double-precision floating-point number to our buffer."""
        self._stream.append_raw_bytes(struct.pack('d', value))

    def append_bool(self, value):
        """Appends a boolean to our buffer."""
        self.append_int32(value)

    def append_enum(self, value):
        """Appends an enum value to our buffer."""
        self.append_int32(value)

    def append_string(self, value):
        """Appends a length-prefixed string to our buffer, with the
        length varint-encoded.
        """
        self._stream.append_var_uint32(len(value))
        self._stream.append_raw_bytes(value)

    def append_bytes(self, value):
        """Appends a length-prefixed sequence of bytes to our buffer, with the
        length varint-encoded.
        """
        self.append_string(value)
