from libcpp.string cimport string

cdef extern from "<sstream>" namespace "std" nogil:
    cdef cppclass stringstream:
        stringstream() except +
        stringstream(const string &str) except +
        stringstream& push "operator<<" (bint val)
        stringstream& push "operator<<" (short val)
        stringstream& push "operator<<" (unsigned short val)
        stringstream& push "operator<<" (int val)
        stringstream& push "operator<<" (unsigned int val)
        stringstream& push "operator<<" (long val)
        stringstream& push "operator<<" (unsigned long val)
        stringstream& push "operator<<" (float val)
        stringstream& push "operator<<" (double val)
        stringstream& push "operator<<" (long double val)
        stringstream& push "operator<<" (void* val)
        stringstream()
        unsigned int tellp()
        stringstream& put(char c)
        stringstream& write(const char *c, size_t n)
        string to_string "str" () const
