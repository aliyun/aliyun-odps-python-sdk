from libc.stdint cimport *
from libc.string cimport *

from internal_c cimport *

cdef class Encoder:
    cdef bytearray _buffer

    cpdef bytes tostring(self)

    cpdef int append_tag(self, int field_num, int wire_type)

    cpdef int append_sint32(self, int32_t value)

    cpdef int append_uint32(self, uint32_t value)

    cpdef int append_sint64(self, int64_t value)

    cpdef int append_uint64(self, uint64_t value)

    cpdef int append_bool(self, bint value)

    cpdef int append_double(self, double value)

    cpdef int append_float(self, float value)

    cpdef int append_string(self, const unsigned char* value)
