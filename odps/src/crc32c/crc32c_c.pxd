from libc.stdint cimport *

cdef class Crc32c:

    cdef uint32_t _crc

    cpdef update(self, bytearray buf)

    cpdef uint32_t getvalue(self)

    cpdef reset(self)