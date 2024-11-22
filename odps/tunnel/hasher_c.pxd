from libc.stdint cimport *
from libcpp.vector cimport vector

from ..src.types_c cimport BaseRecord, SchemaSnapshot
from ..src.utils_c cimport CMillisecondsConverter


cdef class AbstractHasher:
    cdef int32_t c_hash_bigint(self, int64_t val) noexcept nogil
    cdef int32_t c_hash_float(self, float val) nogil
    cdef int32_t c_hash_double(self, double val) nogil
    cdef int32_t c_hash_bool(self, bint val) nogil
    cdef int32_t c_hash_string(self, char *ptr, size_t size) nogil


cdef class DefaultHasher(AbstractHasher):
    cdef int32_t c_hash_bigint(self, int64_t val) noexcept nogil
    cdef int32_t c_hash_float(self, float val) nogil
    cdef int32_t c_hash_double(self, double val) nogil
    cdef int32_t c_hash_bool(self, bint val) nogil
    cdef int32_t c_hash_string(self, char *ptr, size_t size) nogil


cdef class LegacyHasher(AbstractHasher):
    cdef int32_t c_hash_bigint(self, int64_t val) noexcept nogil
    cdef int32_t c_hash_float(self, float val) nogil
    cdef int32_t c_hash_double(self, double val) nogil
    cdef int32_t c_hash_bool(self, bint val) nogil
    cdef int32_t c_hash_string(self, char *ptr, size_t size) nogil


cpdef AbstractHasher get_hasher(hasher_type)


cdef class FieldHasher:
    cdef AbstractHasher _hasher
    cdef int32_t hash_object(self, object value) except? -1


cdef class RecordHasher:
    cdef:
        AbstractHasher _hasher
        SchemaSnapshot _schema_snapshot
        vector[int32_t] _col_ids
        list _idx_to_hash_fun
        CMillisecondsConverter _mills_converter

    cpdef int32_t hash(self, BaseRecord record)


cpdef int32_t hash_value(hasher_type, data_type, value)
