from libc.stdint cimport *
from libc.string cimport *

from util_c cimport *

cdef class Decoder:
    cdef int _pos
    cdef object _stream

    cpdef int position(self)

    cpdef add_offset(self, int n)

    cpdef read_field_number_and_wire_type(self)

    cpdef int32_t read_sint32(self)

    cpdef uint32_t read_uint32(self)

    cpdef int64_t read_sint64(self)

    cpdef uint64_t read_uint64(self)

    cpdef bint read_bool(self)

    cpdef double read_double(self)

    cpdef float read_float(self)

    cpdef bytes read_string(self)