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

from . import input_stream
from . import wire_format


class Decoder(object):
    """Decodes logical protocol buffer fields from the wire."""

    def __init__(self, input):
        """Initializes the decoder to read from input stream.
        """
        self._stream = input_stream.InputStream(input)

    def position(self):
        """Returns the 0-indexed position in |s|."""
        return self._stream.position()

    def read_field_number_and_wire_type(self):
        """Reads a tag from the wire. Returns a (field_number, wire_type) pair."""
        tag_and_type = self.read_uint32()
        return wire_format.unpack_tag(tag_and_type)

    def read_int32(self):
        """Reads and returns a signed, varint-encoded, 32-bit integer."""
        return self._stream.read_varint32()

    def read_int64(self):
        """Reads and returns a signed, varint-encoded, 64-bit integer."""
        return self._stream.read_varint64()

    def read_uint32(self):
        """Reads and returns an signed, varint-encoded, 32-bit integer."""
        return self._stream.read_var_uint32()

    def read_uint64(self):
        """Reads and returns an signed, varint-encoded,64-bit integer."""
        return self._stream.read_var_uint64()

    def read_sint32(self):
        """Reads and returns a signed, zigzag-encoded, varint-encoded,
        32-bit integer."""
        return wire_format.zig_zag_decode(self._stream.read_var_uint32())

    def read_sint64(self):
        """Reads and returns a signed, zigzag-encoded, varint-encoded,
        64-bit integer."""
        return wire_format.zig_zag_decode(self._stream.read_var_uint64())

    def read_fixed32(self):
        """Reads and returns an unsigned, fixed-width, 32-bit integer."""
        return self._stream.read_little_endian32()

    def read_fixed64(self):
        """Reads and returns an unsigned, fixed-width, 64-bit integer."""
        return self._stream.read_little_endian64()

    def read_sfixed32(self):
        """Reads and returns a signed, fixed-width, 32-bit integer."""
        value = self._stream.read_little_endian32()
        if value >= (1 << 31):
            value -= (1 << 32)
        return value

    def read_sfixed64(self):
        """Reads and returns a signed, fixed-width, 64-bit integer."""
        value = self._stream.read_little_endian64()
        if value >= (1 << 63):
            value -= (1 << 64)
        return value

    def read_float(self):
        """Reads and returns a 4-byte floating-point number."""
        serialized = self._stream.read_string(4)
        return struct.unpack('f', serialized)[0]

    def read_double(self):
        """Reads and returns an 8-byte floating-point number."""
        serialized = self._stream.read_string(8)
        return struct.unpack('d', serialized)[0]

    def read_bool(self):
        """Reads and returns a bool."""
        i = self._stream.read_var_uint32()
        return bool(i)

    def read_enum(self):
        """Reads and returns an enum value."""
        return self._stream.read_var_uint32()

    def read_string(self):
        """Reads and returns a length-delimited string."""
        length = self._stream.read_var_uint32()
        return self._stream.read_string(length)

    def ReadBytes(self):
        """Reads and returns a length-delimited byte sequence."""
        return self.read_string()
