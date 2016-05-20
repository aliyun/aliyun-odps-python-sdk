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

TAG_TYPE_BITS = 3  # Number of bits used to hold type info in a proto tag.
_TAG_TYPE_MASK = (1 << TAG_TYPE_BITS) - 1  # 0x7

# These numbers identify the wire type of a protocol buffer value.
# We use the least-significant TAG_TYPE_BITS bits of the varint-encoded
# tag-and-type to store one of these WIRETYPE_* constants.
# These values must match WireType enum in //net/proto2/public/wire_format.h.
WIRETYPE_VARINT = 0
WIRETYPE_FIXED64 = 1
WIRETYPE_LENGTH_DELIMITED = 2
WIRETYPE_START_GROUP = 3
WIRETYPE_END_GROUP = 4
WIRETYPE_FIXED32 = 5
_WIRETYPE_MAX = 5

# Bounds for various integer types.
INT32_MAX = int((1 << 31) - 1)
INT32_MIN = int(-(1 << 31))
UINT32_MAX = (1 << 32) - 1

INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
UINT64_MAX = (1 << 64) - 1

# "struct" format strings that will encode/decode the specified formats.
FORMAT_UINT32_LITTLE_ENDIAN = '<I'
FORMAT_UINT64_LITTLE_ENDIAN = '<Q'

# We'll have to provide alternate implementations of AppendLittleEndian*() on
# any architectures where these checks fail.
if struct.calcsize(FORMAT_UINT32_LITTLE_ENDIAN) != 4:
    raise AssertionError('Format "I" is not a 32-bit number.')
if struct.calcsize(FORMAT_UINT64_LITTLE_ENDIAN) != 8:
    raise AssertionError('Format "Q" is not a 64-bit number.')


def pack_tag(field_number, wire_type):
    """Returns an unsigned 32-bit integer that encodes the field number and
    wire type information in standard protocol message wire format.

    Args:
      field_number: Expected to be an integer in the range [1, 1 << 29)
      wire_type: One of the WIRETYPE_* constants.
    """
    if not 0 <= wire_type <= _WIRETYPE_MAX:
        raise errors.EncodeError('Unknown wire type: %d' % wire_type)
    return (field_number << TAG_TYPE_BITS) | wire_type


def unpack_tag(tag):
    """The inverse of PackTag().  Given an unsigned 32-bit number,
    returns a (field_number, wire_type) tuple.
    """
    return (tag >> TAG_TYPE_BITS), (tag & _TAG_TYPE_MASK)


def zig_zag_encode(value):
    """ZigZag Transform:  Encodes signed integers so that they can be
    effectively used with varint encoding.  See wire_format.h for
    more details.
    """
    if value >= 0:
        return value << 1
    return ((value << 1) ^ (~0)) | 0x1


def zig_zag_decode(value):
    """Inverse of ZigZagEncode()."""
    if not value & 0x1:
        return value >> 1
    return (value >> 1) ^ (~0)


# The *ByteSize() functions below return the number of bytes required to
# serialize "field number + type" information and then serialize the value.


def int32_byte_size(field_number, int32):
    return int64_byte_size(field_number, int32)


def int64_byte_size(field_number, int64):
    # Have to convert to uint before calling UInt64ByteSize().
    return UInt64ByteSize(field_number, 0xffffffffffffffff & int64)


def uint32_byte_size(field_number, uint32):
    return UInt64ByteSize(field_number, uint32)


def UInt64ByteSize(field_number, uint64):
    return _tag_byte_size(field_number) + _var_uint64_byte_size_no_tag(uint64)


def sint32_byte_size(field_number, int32):
    return uint32_byte_size(field_number, zig_zag_encode(int32))


def sint64_byte_size(field_number, int64):
    return UInt64ByteSize(field_number, zig_zag_encode(int64))


def fixed32_byte_size(field_number, fixed32):
    return _tag_byte_size(field_number) + 4


def fixed64_byte_size(field_number, fixed64):
    return _tag_byte_size(field_number) + 8


def sfixed32_byte_size(field_number, sfixed32):
    return _tag_byte_size(field_number) + 4


def sfixed64_byte_size(field_number, sfixed64):
    return _tag_byte_size(field_number) + 8


def float_byte_size(field_number, flt):
    return _tag_byte_size(field_number) + 4


def double_byte_size(field_number, double):
    return _tag_byte_size(field_number) + 8


def bool_byte_size(field_number, b):
    return _tag_byte_size(field_number) + 1


def enum_byte_size(field_number, enum):
    return uint32_byte_size(field_number, enum)


def string_byte_size(field_number, string):
    return (_tag_byte_size(field_number)
            + _var_uint64_byte_size_no_tag(len(string))
            + len(string))


def bytes_byte_size(field_number, b):
    return string_byte_size(field_number, b)


def group_byte_size(field_number, message):
    return (2 * _tag_byte_size(field_number)  # START and END group.
            + message.ByteSize())


def message_byte_size(field_number, message):
    return (_tag_byte_size(field_number)
            + _var_uint64_byte_size_no_tag(message.ByteSize())
            + message.ByteSize())


def message_set_item_byte_size(field_number, msg):
    # First compute the sizes of the tags.
    # There are 2 tags for the beginning and ending of the repeated group, that
    # is field number 1, one with field number 2 (type_id) and one with field
    # number 3 (message).
    total_size = (2 * _tag_byte_size(1) + _tag_byte_size(2) + _tag_byte_size(3))

    # Add the number of bytes for type_id.
    total_size += _var_uint64_byte_size_no_tag(field_number)

    message_size = msg.ByteSize()

    # The number of bytes for encoding the length of the message.
    total_size += _var_uint64_byte_size_no_tag(message_size)

    # The size of the message.
    total_size += message_size
    return total_size


# Private helper functions for the *ByteSize() functions above.


def _tag_byte_size(field_number):
    """Returns the bytes required to serialize a tag with this field number."""
    # Just pass in type 0, since the type won't affect the tag+type size.
    return _var_uint64_byte_size_no_tag(pack_tag(field_number, 0))


def _var_uint64_byte_size_no_tag(uint64):
    """Returns the bytes required to serialize a single varint.
    uint64 must be unsigned.
    """
    if uint64 > UINT64_MAX:
        raise errors.EncodeError('Value out of range: %d' % uint64)
    bytes = 1
    while uint64 > 0x7f:
        bytes += 1
        uint64 >>= 7
    return bytes
