from libc.stdint cimport *

cdef char * read_stream(object input, int size)

cdef int32_t get_varint32(object input)

cdef int64_t get_varint64(object input)

cdef int32_t get_signed_varint32(object input)

cdef int64_t get_signed_varint64(object input)

cdef int set_varint32(int32_t varint, bytearray buf)

cdef int set_varint64(int64_t varint, bytearray buf)

cdef int set_signed_varint32(int32_t varint, bytearray buf)

cdef int set_signed_varint64(int64_t varint, bytearray buf)