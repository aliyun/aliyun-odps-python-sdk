from libc.stdint cimport *
from libc.string cimport *

from util_c cimport *

cpdef int append_tag(bytearray buf, int field_num, int wire_type)

cpdef int append_sint32(bytearray buf, int32_t value)

cpdef int append_uint32(bytearray buf, uint32_t value)

cpdef int append_sint64(bytearray buf, int64_t value)

cpdef int append_uint64(bytearray buf, uint64_t value)

cpdef int append_bool(bytearray buf, bint value)

cpdef int append_double(bytearray buf, double value)

cpdef int append_float(bytearray buf, float value)

cpdef int append_string(bytearray buf, const unsigned char* value)
