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


class InputStream(object):
    """Contains all logic for reading bits, and dealing with stream position.

    If an InputStream method ever raises an exception, the stream is left
    in an indeterminate state and is not safe for further use.
    """

    def __init__(self, input):
        self._input = input
        self._pos = 0

    def position(self):
        """Returns the current position in the stream, or equivalently, the
        number of bytes read so far.
        """
        return self._pos

    def read_string(self, size):
        """Reads up to 'size' bytes from the stream, stopping early
        only if we reach the end of the stream.  Returns the bytes read
        as a string.
        """
        if size < 0:
            raise errors.DecodeError('Negative size %d' % size)
        s = self._input.read(size)
        if len(s) != size:
            raise errors.DecodeError('String claims to have %d bytes, but read %d' % (size, len(s)))
        self._pos += len(s)  # Only advance by the number of bytes actually read.
        return s

    def read_little_endian32(self):
        """Interprets the next 4 bytes of the stream as a little-endian
        encoded, unsiged 32-bit integer, and returns that integer.
        """
        try:
            i = struct.unpack(wire_format.FORMAT_UINT32_LITTLE_ENDIAN,
                              self._input.read(4))
            self._pos += 4
            return i[0]  # unpack() result is a 1-element tuple.
        except struct.error as e:
            raise errors.DecodeError(e)

    def read_little_endian64(self):
        """Interprets the next 8 bytes of the stream as a little-endian
        encoded, unsiged 64-bit integer, and returns that integer.
        """
        try:
            i = struct.unpack(wire_format.FORMAT_UINT64_LITTLE_ENDIAN,
                              self._input.read(8))
            self._pos += 8
            return i[0]  # unpack() result is a 1-element tuple.
        except struct.error as e:
            raise errors.DecodeError(e)

    def read_varint32(self):
        """Reads a varint from the stream, interprets this varint
        as a signed, 32-bit integer, and returns the integer.
        """
        i = self.read_varint64()
        if not wire_format.INT32_MIN <= i <= wire_format.INT32_MAX:
            raise errors.DecodeError('Value out of range for int32: %d' % i)
        return int(i)

    def read_var_uint32(self):
        """Reads a varint from the stream, interprets this varint
        as an unsigned, 32-bit integer, and returns the integer.
        """
        i = self.read_var_uint64()
        if i > wire_format.UINT32_MAX:
            raise errors.DecodeError('Value out of range for uint32: %d' % i)
        return i

    def read_varint64(self):
        """Reads a varint from the stream, interprets this varint
        as a signed, 64-bit integer, and returns the integer.
        """
        i = self.read_var_uint64()
        if i > wire_format.INT64_MAX:
            i -= (1 << 64)
        return i

    def read_var_uint64(self):
        """Reads a varint from the stream, interprets this varint
        as an unsigned, 64-bit integer, and returns the integer.
        """
        i = self._read_varint_helper()
        if not 0 <= i <= wire_format.UINT64_MAX:
            raise errors.DecodeError('Value out of range for uint64: %d' % i)
        return i

    def _read_varint_helper(self):
        """Helper for the various varint-reading methods above.
        Reads an unsigned, varint-encoded integer from the stream and
        returns this integer.

        Does no bounds checking except to ensure that we read at most as many bytes
        as could possibly be present in a varint-encoded 64-bit number.
        """
        result = 0
        shift = 0
        while 1:
            if shift >= 64:
                raise errors.DecodeError('Too many bytes when decoding varint.')
            try:
                b = ord(self._input.read(1))
            except IndexError:
                raise errors.DecodeError('Truncated varint.')
            self._pos += 1
            result |= ((b & 0x7f) << shift)
            shift += 7
            if not (b & 0x80):
                return result
