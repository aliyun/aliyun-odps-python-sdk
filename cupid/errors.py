# Copyright 1999-2022 Alibaba Group Holding Ltd.
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


class SubprocessStreamEOFError(IOError):
    pass


class CupidError(RuntimeError):
    pass


class InstanceRecycledError(CupidError):
    pass


class CupidUserError(CupidError):
    pass


class CupidMasterTimeoutError(CupidError):
    pass


class CupidCppError(CupidError):
    cpp_error_type = None

    def __new__(cls, err_type=None, err_msg=None):
        if cls is not CupidCppError:
            return CupidError.__new__(cls)
        if err_type in _cpp_errors:
            return CupidError.__new__(_cpp_errors[err_type])
        else:
            return CupidError.__new__(cls)

    def __init__(self, err_type=None, err_msg=None):
        self._err_type = err_type
        self._err_msg = err_msg

    def __str__(self):
        return '%s: %s' % (self._err_type, self._err_msg)


class CupidReplyTimeoutError(CupidCppError):
    cpp_error_type = 'ChannelTimeOutException'


class CupidChannelReplyError(CupidCppError):
    cpp_error_type = 'ChannelReplyException'


class CupidClientClosedError(CupidCppError):
    cpp_error_type = 'ChannelClientClosedException'


_cpp_errors = dict()
for _ex in globals().copy().values():
    if not isinstance(_ex, type) or not issubclass(_ex, CupidCppError):
        continue
    if getattr(_ex, 'cpp_error_type', None) is None:
        continue
    _cpp_errors[_ex.cpp_error_type] = _ex
