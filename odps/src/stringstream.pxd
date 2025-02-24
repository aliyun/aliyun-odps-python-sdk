# Copyright 1999-2025 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
