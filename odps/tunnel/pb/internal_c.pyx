from libc.stdint cimport *
from libc.string cimport *

from util_c cimport *

cpdef int append_tag(bytearray buf, int field_idx, int wire_type):
    cdef int key
    key = (field_idx << 3) | wire_type
    size = set_varint64(key, buf)
    return size

cpdef int append_sint32(bytearray buf, int32_t value):
    return set_signed_varint32(value, buf)

cpdef int append_uint32(bytearray buf, uint32_t value):
    return set_varint32(value, buf)

cpdef int append_sint64(bytearray buf, int64_t value):
    return set_signed_varint64(value, buf)

cpdef int append_uint64(bytearray buf, uint64_t value):
    return set_varint64(value, buf)

cpdef int append_bool(bytearray buf, bint value):
    return set_varint32(value, buf)

cpdef int append_double(bytearray buf, double value):
    buf += (<unsigned char *>&value)[:sizeof(double)]
    return sizeof(double)

cpdef int append_float(bytearray buf, float value):
    buf += (<unsigned char *>&value)[:sizeof(float)]
    return sizeof(float)

cpdef int append_string(bytearray buf, const unsigned char * value):
    size = set_varint32(len(value), buf)
    buf += value
    return size + len(value)

