from libc.stdint cimport *


cdef extern from "crc32c.c":
    uint32_t crc32c(uint32_t crc, const void *buf, size_t length)

cdef class Crc32c:
    def __cinit__(self):
        self._crc = 0

    cpdef uint32_t getvalue(self):
        return self._crc

    cpdef update(self, bytearray buf):
        cdef char* cstring = <char *>buf
        self._crc = crc32c(self._crc, <const void*>cstring, len(buf))

    cpdef reset(self):
        self._crc = 0