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

import os
import json
from ctypes import CDLL, RTLD_GLOBAL, create_string_buffer, \
    c_char, c_void_p, c_char_p, c_int32, c_uint32, byref, POINTER
from ctypes.util import find_library
from io import IOBase

from odps.compat import BytesIO, six
from odps.utils import to_binary
from ..errors import SubprocessStreamEOFError, CupidCppError
from ..utils import get_environ

_lib_path = find_library('odps_subprocess')
if _lib_path is None:
    ld_paths = (get_environ('LD_LIBRARY_PATH') or "").split(':')
    for p in ld_paths:
        so_path = os.path.join(p, 'libodps_subprocess.so')
        if os.path.exists(so_path):
            _lib_path = so_path
            break
if _lib_path is None:
    _lib_path = os.path.join(os.getcwd(), 'libodps_subprocess.so')

try:
    odps_subproc = CDLL(_lib_path, mode=RTLD_GLOBAL)
except OSError:
    raise ImportError('Failed to load libodps_subprocess.so.')

Subprocess_Container_Init = odps_subproc.Subprocess_Container_Init

Subprocess_StdString_Size = odps_subproc.Subprocess_StdString_Size
Subprocess_StdString_Size.argtypes = [c_void_p]
Subprocess_StdString_Size.restype = c_uint32

Subprocess_StdString_CopyTo = odps_subproc.Subprocess_StdString_CopyTo
Subprocess_StdString_CopyTo.argtypes = [c_char_p, c_void_p, c_uint32]

Subprocess_StdString_delete = odps_subproc.Subprocess_StdString_delete
Subprocess_StdString_delete.argtypes = [c_void_p]

Subprocess_StartFDReceiver = odps_subproc.Subprocess_StartFDReceiver

Subprocess_StdIStream_eof = odps_subproc.Subprocess_StdIStream_eof
Subprocess_StdIStream_eof.argtypes = [c_void_p]
Subprocess_StdIStream_eof.restype = c_int32

Subprocess_StdIStream_read = odps_subproc.Subprocess_StdIStream_read
Subprocess_StdIStream_read.argtypes = [c_void_p, c_char_p, c_uint32]
Subprocess_StdIStream_read.restype = c_uint32

Subprocess_StdIStream_seekg = odps_subproc.Subprocess_StdIStream_seekg
Subprocess_StdIStream_seekg.argtypes = [c_void_p, c_int32, c_int32]

Subprocess_StdIStream_tellg = odps_subproc.Subprocess_StdIStream_tellg
Subprocess_StdIStream_tellg.argtypes = [c_void_p]
Subprocess_StdIStream_tellg.restype = c_uint32

Subprocess_StdIStream_getline = odps_subproc.Subprocess_StdIStream_getline
Subprocess_StdIStream_getline.argtypes = [c_void_p]
Subprocess_StdIStream_getline.restype = c_void_p

Subprocess_StdIStream_delete = odps_subproc.Subprocess_StdIStream_delete
Subprocess_StdIStream_delete.argtypes = [c_void_p]

Subprocess_StdOStream_write = odps_subproc.Subprocess_StdOStream_write
Subprocess_StdOStream_write.argtypes = [c_void_p, c_char_p, c_uint32]

Subprocess_StdOStream_flush = odps_subproc.Subprocess_StdOStream_flush
Subprocess_StdOStream_flush.argtypes = [c_void_p]

Subprocess_StdOStream_delete = odps_subproc.Subprocess_StdOStream_delete
Subprocess_StdOStream_delete.argtypes = [c_void_p]

ChannelConf_new = odps_subproc.ChannelConf_new
ChannelConf_new.restype = c_void_p

ChannelConf_delete = odps_subproc.ChannelConf_delete
ChannelConf_delete.argtypes = [c_void_p]

ChannelConf_SetInt = odps_subproc.ChannelConf_SetInt
ChannelConf_SetInt.argtypes = [c_void_p, c_char_p, c_int32]

ChannelResultInputStreamPtr_GetResult = odps_subproc.ChannelResultInputStreamPtr_GetResult
ChannelResultInputStreamPtr_GetResult.argtypes = [c_void_p, c_int32, POINTER(c_void_p)]
ChannelResultInputStreamPtr_GetResult.restype = c_void_p

ChannelResultInputStreamPtr_GetStream = odps_subproc.ChannelResultInputStreamPtr_GetStream
ChannelResultInputStreamPtr_GetStream.argtypes = [c_void_p]
ChannelResultInputStreamPtr_GetStream.restype = c_void_p

ChannelResultInputStreamPtr_GetFD = odps_subproc.ChannelResultInputStreamPtr_GetFD
ChannelResultInputStreamPtr_GetFD.argtypes = [c_void_p]
ChannelResultInputStreamPtr_GetFD.restype = c_int32

ChannelResultInputStreamPtr_delete = odps_subproc.ChannelResultInputStreamPtr_delete
ChannelResultInputStreamPtr_delete.argtypes = [c_void_p]

ChannelResultOutputStreamPtr_GetResult = odps_subproc.ChannelResultOutputStreamPtr_GetResult
ChannelResultOutputStreamPtr_GetResult.argtypes = [c_void_p, c_int32, POINTER(c_void_p)]
ChannelResultOutputStreamPtr_GetResult.restype = c_void_p

ChannelResultOutputStreamPtr_GetStream = odps_subproc.ChannelResultOutputStreamPtr_GetStream
ChannelResultOutputStreamPtr_GetStream.argtypes = [c_void_p]
ChannelResultOutputStreamPtr_GetStream.restype = c_void_p

ChannelResultOutputStreamPtr_GetFD = odps_subproc.ChannelResultOutputStreamPtr_GetFD
ChannelResultOutputStreamPtr_GetFD.argtypes = [c_void_p]
ChannelResultOutputStreamPtr_GetFD.restype = c_int32

ChannelResultOutputStreamPtr_delete = odps_subproc.ChannelResultOutputStreamPtr_delete
ChannelResultOutputStreamPtr_delete.argtypes = [c_void_p]

ChannelSlaveClient_new = odps_subproc.ChannelSlaveClient_new
ChannelSlaveClient_new.argtypes = [c_int32, c_int32, c_char_p]
ChannelSlaveClient_new.restype = c_void_p

ChannelSlaveClient_delete = odps_subproc.ChannelSlaveClient_delete
ChannelSlaveClient_delete.argtypes = [c_void_p]

ChannelSlaveClient_Start = odps_subproc.ChannelSlaveClient_Start
ChannelSlaveClient_Start.argtypes = [c_void_p]

ChannelSlaveClient_Stop = odps_subproc.ChannelSlaveClient_Stop
ChannelSlaveClient_Stop.argtypes = [c_void_p]

ChannelSlaveClient_SyncCall = odps_subproc.ChannelSlaveClient_SyncCall
ChannelSlaveClient_SyncCall.argtypes = [c_void_p, c_char_p, c_char_p, c_uint32, c_int32, POINTER(c_void_p)]
ChannelSlaveClient_SyncCall.restype = c_void_p

ChannelSlaveClient_CreateInputStream = odps_subproc.ChannelSlaveClient_CreateInputStream
ChannelSlaveClient_CreateInputStream.argtypes = [c_void_p, c_char_p, c_char_p, c_uint32, POINTER(c_void_p)]
ChannelSlaveClient_CreateInputStream.restype = c_void_p

ChannelSlaveClient_CreateOutputStream = odps_subproc.ChannelSlaveClient_CreateOutputStream
ChannelSlaveClient_CreateOutputStream.argtypes = [c_void_p, c_char_p, c_char_p, c_uint32, POINTER(c_void_p)]
ChannelSlaveClient_CreateOutputStream.restype = c_void_p

ChannelApiError_GetErrorType = odps_subproc.ChannelApiError_GetErrorType
ChannelApiError_GetErrorType.argtypes = [c_void_p]
ChannelApiError_GetErrorType.restype = c_void_p

ChannelApiError_GetErrorMessage = odps_subproc.ChannelApiError_GetErrorMessage
ChannelApiError_GetErrorMessage.argtypes = [c_void_p]
ChannelApiError_GetErrorMessage.restype = c_void_p

ChannelApiError_delete = odps_subproc.ChannelApiError_delete
ChannelApiError_delete.argtypes = [c_void_p]

READ_CHUNK = 2048


def _read_std_string_ptr(std_string):
    std_string_len = Subprocess_StdString_Size(std_string)
    p = create_string_buffer(std_string_len)
    Subprocess_StdString_CopyTo(p, std_string, std_string_len)
    Subprocess_StdString_delete(std_string)
    return p


def _call_with_raise(func, *args):
    err_pp = c_void_p()
    err_ptr = 0
    try:
        args = args + (byref(err_pp), )
        res = func(*args)
        if res:
            return res
        else:
            err_ptr = err_pp.value
            err_type_ptr = ChannelApiError_GetErrorType(err_ptr)
            err_type = _read_std_string_ptr(err_type_ptr).raw
            err_message_ptr = ChannelApiError_GetErrorMessage(err_ptr)
            err_message = _read_std_string_ptr(err_message_ptr).raw
            raise CupidCppError(err_type, err_message)
    finally:
        if err_ptr:
            ChannelApiError_delete(err_ptr)


class IStreamWrapper(IOBase):
    def __init__(self, ptr, free=True):
        self._ptr = ptr
        self._free = free

    def __del__(self):
        self.close()

    def close(self):
        if self._free and self._ptr is not None:
            Subprocess_StdIStream_delete(self._ptr)
            self._ptr = None

    @property
    def closed(self):
        if self._ptr is None:
            return True
        else:
            return False

    def _reset(self):
        self._ptr = None

    def readable(self):
        return self._ptr is not None

    def read(self, size=-1):
        if Subprocess_StdIStream_eof(self._ptr):
            raise SubprocessStreamEOFError('Subprocess stream exhausted')
        if size < 0:
            buf_io = BytesIO()
            while True:
                try:
                    chunk = self.read(READ_CHUNK)
                except SubprocessStreamEOFError:
                    chunk = b''
                buf_io.write(chunk)
                if len(chunk) < READ_CHUNK:
                    break
            return buf_io.getvalue()
        else:
            p = create_string_buffer(size)
            Subprocess_StdIStream_read(self._ptr, p, size)
            return p.raw

    def readinto(self, b, offset=0):
        if Subprocess_StdIStream_eof(self._ptr):
            raise SubprocessStreamEOFError('Subprocess stream exhausted')
        size = len(b) - offset
        array_cls = c_char * size
        p = array_cls.from_buffer(b, offset)
        ret_size = Subprocess_StdIStream_read(self._ptr, p, size)
        if ret_size == 0 and Subprocess_StdIStream_eof(self._ptr):
            raise SubprocessStreamEOFError('Subprocess stream exhausted')
        return ret_size

    def readline(self, size=-1):
        str_ptr = Subprocess_StdIStream_getline(self._ptr, size)
        p = _read_std_string_ptr(str_ptr)
        return p.raw

    def readlines(self, hint=-1):
        lines = []
        while True:
            line = self.readline(hint)
            lines.append(line)
            if hint == 0 or Subprocess_StdIStream_eof(self._ptr):
                break
            if hint > 0:
                hint -= len(line)
        return lines

    def __iter__(self):
        while True:
            line = self.readline()
            yield line
            if Subprocess_StdIStream_eof(self._ptr):
                break

    def seek(self, offset, whence=0):
        Subprocess_StdIStream_seekg(self._ptr, offset, whence)

    def tell(self):
        return Subprocess_StdIStream_tellg(self._ptr)


class OStreamWrapper(IOBase):
    def __init__(self, ptr, free=True):
        self._ptr = ptr
        self._free = free

    def __del__(self):
        if self._ptr is not None:
            self.flush()
        self.close()

    def close(self):
        if self._free and self._ptr is not None:
            Subprocess_StdOStream_delete(self._ptr)
            self._ptr = None

    @property
    def closed(self):
        if self._ptr is None:
            return True
        else:
            return False

    def _reset(self):
        self._ptr = None

    def writable(self):
        return self._ptr is not None

    def write(self, data, length=None):
        length = length or len(data)
        atype = c_char * length
        Subprocess_StdOStream_write(self._ptr, atype.from_buffer_copy(data), length)

    def flush(self):
        if self._ptr is not None:
            Subprocess_StdOStream_flush(self._ptr)


class ChannelResultInputStreamWrapper(IStreamWrapper):
    def __init__(self, ptr):
        self._result_ptr = ptr
        is_ptr = ChannelResultInputStreamPtr_GetStream(ptr)
        super(ChannelResultInputStreamWrapper, self).__init__(is_ptr, False)

    def __del__(self):
        self.close()

    def fileno(self):
        return ChannelResultInputStreamPtr_GetFD(self._result_ptr)

    def result(self, timeout=-1):
        # Stream object already freed in C end
        self._reset()

        pstr = _call_with_raise(ChannelResultInputStreamPtr_GetResult, self._result_ptr, timeout)
        p = _read_std_string_ptr(pstr)
        return p.raw

    def close(self):
        if self._result_ptr is not None:
            ChannelResultInputStreamPtr_delete(self._result_ptr)
        self._result_ptr = None
        super(ChannelResultInputStreamWrapper, self).close()


class ChannelResultOutputStreamWrapper(OStreamWrapper):
    def __init__(self, ptr):
        self._result_ptr = ptr
        os_ptr = ChannelResultOutputStreamPtr_GetStream(ptr)
        super(ChannelResultOutputStreamWrapper, self).__init__(os_ptr, False)

    def __del__(self):
        self.close()

    def fileno(self):
        return ChannelResultOutputStreamPtr_GetFD(self._result_ptr)

    def result(self, timeout=-1):
        # Stream object already freed in C end
        self._reset()

        pstr = _call_with_raise(ChannelResultOutputStreamPtr_GetResult, self._result_ptr, timeout)
        p = _read_std_string_ptr(pstr)
        return p.raw

    def close(self):
        if self._result_ptr is not None:
            self.flush()
            ChannelResultOutputStreamPtr_delete(self._result_ptr)
        self._result_ptr = None
        super(ChannelResultOutputStreamWrapper, self).close()


class ChannelOutputWriter(object):
    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        self._stream.write(data)

    def close(self):
        # sync
        self._stream.result()
        self._stream.close()

    @property
    def closed(self):
        return self._stream.closed

    def flush(self):
        self._stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ChannelConf(object):
    def __init__(self):
        self._ptr = ChannelConf_new()

    def __del__(self):
        if self._ptr is not None:
            ChannelConf_delete(self._ptr)
        self._ptr = None

    def set_integer(self, key, value):
        if isinstance(key, six.string_types):
            key = key.encode()
        ChannelConf_SetInt(self._ptr, key, value)


class ChannelSlaveClient(object):
    def __init__(self, request_channel_nums, response_channel_nums, client_name):
        if isinstance(client_name, six.string_types):
            client_name = client_name.encode()
        self._ptr = ChannelSlaveClient_new(request_channel_nums, response_channel_nums, client_name)

    def __del__(self):
        if self._ptr is not None:
            ChannelSlaveClient_delete(self._ptr)
        self._ptr = None

    def start(self):
        ChannelSlaveClient_Start(self._ptr)

    def stop(self):
        ChannelSlaveClient_Stop(self._ptr)

    def sync_call(self, method, parameter, timeout=-1):
        method = to_binary(method)
        parameter = to_binary(parameter)
        result_ptr = _call_with_raise(ChannelSlaveClient_SyncCall, self._ptr, method,
                                      parameter, len(parameter), timeout)
        p = _read_std_string_ptr(result_ptr)
        return p.raw

    def create_file_reader(self, method, params):
        if isinstance(method, six.string_types):
            method = method.encode()
        if isinstance(params, six.string_types):
            params = params.encode()
        ptr = _call_with_raise(ChannelSlaveClient_CreateInputStream, self._ptr, method, params, len(params))
        return ChannelResultInputStreamWrapper(ptr)

    def create_file_writer(self, method, params):
        if isinstance(method, six.string_types):
            method = method.encode()
        if isinstance(params, six.string_types):
            params = params.encode()
        ptr = _call_with_raise(ChannelSlaveClient_CreateOutputStream, self._ptr, method, params, len(params))
        return ChannelResultOutputStreamWrapper(ptr)

    def create_record_reader(self, label, schema, columns=None):
        from ..io.table import CupidRecordReader

        params = dict(type='ReadByLabel', label=label, arrow=False, batch=False)
        stream = self.create_file_reader('createTableInputStream', json.dumps(params).encode())
        reader = CupidRecordReader(schema, stream, columns=columns)
        return reader

    def create_record_writer(self, label, schema):
        from ..io.table import CupidRecordWriter
        params = json.dumps(dict(type='WriteByLabel', label=label, arrow=False, Batch=False))
        stream = self.create_file_writer('createTableOutputStream', params.encode())
        writer = CupidRecordWriter(schema, stream)
        return writer

    def create_arrow_writer(self, label):
        params = json.dumps(dict(type='WriteByLabel', label=label, arrow=True, batch=True))
        stream = self.create_file_writer('createTableOutputStream', params.encode())
        return ChannelOutputWriter(stream)

    create_table_reader = create_record_reader
    create_table_writer = create_record_writer

    def create_pandas_reader(self, label, schema, columns=None):
        try:
            from ..io.table import CupidPandasReader
        except ImportError:
            return None

        params = json.dumps(dict(type='ReadByLabel', label=label))
        stream = self.create_file_reader('createTableInputStream', params.encode())
        reader = CupidPandasReader(schema, stream, columns=columns)
        return reader

    def create_pandas_writer(self, label, schema):
        try:
            from ..io.table import CupidPandasWriter
        except ImportError:
            return None

        params = json.dumps(dict(type='WriteByLabel', label=label))
        stream = self.create_file_writer('createTableOutputStream', params.encode())
        writer = CupidPandasWriter(schema, stream)
        return writer
