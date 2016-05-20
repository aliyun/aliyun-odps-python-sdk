from libc.stdint cimport *
from libc.string cimport *

from internal_c cimport *

cdef class Encoder:
    def __cinit__(self):
        self._buffer = bytearray()

    def __dealloc__(self):
        pass

    def __len__(self):
        return len(self._buffer)

    cpdef bytes tostring(self):
        return bytes(self._buffer)

    cpdef int append_tag(self, int field_num, int wire_type):
        return append_tag(self._buffer, field_num, wire_type)

    cpdef int append_sint32(self, int32_t value):
        return append_sint32(self._buffer, value)

    cpdef int append_uint32(self, uint32_t value):
        return append_uint32(self._buffer, value)

    cpdef int append_sint64(self, int64_t value):
        return append_sint64(self._buffer, value)

    cpdef int append_uint64(self, uint64_t value):
        return append_uint64(self._buffer, value)

    cpdef int append_bool(self, bint value):
        return append_bool(self._buffer, value)

    cpdef int append_float(self, float value):
        return append_float(self._buffer, value)

    cpdef int append_double(self, double value):
        return append_double(self._buffer, value)

    cpdef int append_string(self, const unsigned char *value):
        return append_string(self._buffer, value)

