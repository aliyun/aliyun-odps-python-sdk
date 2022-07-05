#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import sys
import struct

from requests.exceptions import StreamConsumedError

from . import io
from .base import BaseTunnel
from .checksum import Checksum
from .errors import TunnelError
from .. import serializers, options
from ..models import errors
from ..compat import irange, Enum, six
from ..utils import to_binary, to_text

MAX_CHUNK_SIZE = 256 * 1024 * 1024
MIN_CHUNK_SIZE = 1
CHECKSUM_SIZE = 4

CHECKSUM_PACKER = '>i' if six.PY2 else '>I'


class VolumeTunnel(BaseTunnel):
    def create_download_session(self, volume, partition_spec, file_name, download_id=None, compress_option=None,
                                compress_algo=None, compress_level=None, compress_strategy=None):
        if not isinstance(volume, six.string_types):
            volume = volume.name
        volume = self._project.volumes[volume]
        if compress_option is None and compress_algo is not None:
            compress_option = io.CompressOption(
                compress_algo=compress_algo, level=compress_level, strategy=compress_strategy)

        return VolumeDownloadSession(self.tunnel_rest, volume, partition_spec, file_name, download_id=download_id,
                                     compress_option=compress_option)

    def create_upload_session(self, volume, partition_spec, upload_id=None, compress_option=None,
                              compress_algo=None, compress_level=None, compress_strategy=None):
        if not isinstance(volume, six.string_types):
            volume = volume.name
        volume = self._project.volumes[volume]
        if compress_option is None and compress_algo is not None:
            compress_option = io.CompressOption(
                compress_algo=compress_algo, level=compress_level, strategy=compress_strategy)

        return VolumeUploadSession(self.tunnel_rest, volume, partition_spec, upload_id=upload_id,
                                   compress_option=compress_option)


class VolumeFSTunnel(BaseTunnel):
    def open_reader(self, volume, path, start=None, length=None, compress_option=None, compress_algo=None,
                    compress_level=None, compress_strategy=None):
        if not isinstance(volume, six.string_types):
            volume = volume.name
        volume = self._project.volumes[volume]

        if start is None:
            start = 0
        if length is None:
            file_obj = volume[path]
            length = file_obj.length

        headers = {
            'Range': 'bytes={0}-{1}'.format(start, start + length - 1),
            'x-odps-volume-fs-path': '/' + volume.name + '/' + path.lstrip('/'),
        }

        if compress_option is not None:
            if compress_option.algorithm == io.CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Accept-Encoding'] = 'deflate'
            elif compress_option.algorithm != io.CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')

        url = volume.resource(client=self.tunnel_rest)
        resp = self.tunnel_rest.get(url, headers=headers, stream=True)
        if not self.tunnel_rest.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e

        if compress_option is None and compress_algo is not None:
            compress_option = io.CompressOption(
                compress_algo=compress_algo, level=compress_level, strategy=compress_strategy)

        content_encoding = resp.headers.get('Content-Encoding')
        if content_encoding is not None:
            compress = True
        else:
            compress = False

        option = compress_option if compress else None
        return VolumeReader(self.tunnel_rest, resp, option)

    def open_writer(self, volume, path, replication=None, compress_option=None, compress_algo=None,
                    compress_level=None, compress_strategy=None):
        if not isinstance(volume, six.string_types):
            volume = volume.name
        volume = self._project.volumes[volume]

        headers = {
            'Content-Type': 'application/octet-stream',
            'Transfer-Encoding': 'chunked',
            'x-odps-volume-fs-path': '/' + volume.name + '/' + path.lstrip('/'),
        }
        params = {}

        if compress_option is None and compress_algo is not None:
            compress_option = io.CompressOption(
                compress_algo=compress_algo, level=compress_level, strategy=compress_strategy)
        if compress_option is not None:
            if compress_option.algorithm == io.CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Content-Encoding'] = 'deflate'
            elif compress_option.algorithm != io.CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')

        if replication:
            params['replication'] = replication

        url = volume.resource(client=self.tunnel_rest)

        chunk_upload = lambda data: self.tunnel_rest.post(url, data=data, params=params, headers=headers)
        if compress_option is None and compress_algo is not None:
            compress_option = io.CompressOption(
                compress_algo=compress_algo, level=compress_level, strategy=compress_strategy)
        return VolumeFSWriter(self.tunnel_rest, chunk_upload, volume, path, compress_option)


class VolumeDownloadSession(serializers.JSONSerializableModel):
    __slots__ = 'id', '_client', 'project_name', 'volume_name', 'partition_spec', 'file_name', '_compress_option'

    class Status(Enum):
        UNKNOWN = 'UNKNOWN'
        NORMAL = 'NORMAL'
        CLOSED = 'CLOSED'
        EXPIRED = 'EXPIRED'

    id = serializers.JSONNodeField('DownloadID')
    status = serializers.JSONNodeField('Status',
                                       parse_callback=lambda v: VolumeDownloadSession.Status(v.upper()))
    file_name = serializers.JSONNodeField('File', 'FileName')
    file_length = serializers.JSONNodeField('File', 'FileLength')
    volume_name = serializers.JSONNodeField('Partition', 'Volume')
    partition_spec = serializers.JSONNodeField('Partition', 'Partition')

    def __init__(self, client, volume, partition_spec, file_name=None, download_id=None, compress_option=None):
        super(VolumeDownloadSession, self).__init__()

        self._client = client
        self._compress_option = compress_option
        self.project_name = volume.project.name
        self.volume_name = volume.name
        self.partition_spec = partition_spec
        self.file_name = file_name

        if download_id is None:
            self._init()
        else:
            self.id = download_id
            self.reload()
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return (
            "<VolumeDownloadSession id=%s project_name=%s volume_name=%s partition_spec=%s>"
            % (self.id, self.project_name, self.volume_name, self.partition_spec)
        )

    def resource(self):
        return self._client.endpoint + '/projects/%s/tunnel/downloads' % self.project_name

    def _init(self):
        headers = {'Content-Length': '0'}
        params = dict(type='volumefile', target='/'.join([self.project_name, self.volume_name,
                                                          self.partition_spec, self.file_name]))

        url = self.resource()
        resp = self._client.post(url, {}, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
        else:
            e = TunnelError.parse(resp)
            raise e

    def reload(self):
        headers = {'Content-Length': '0'}
        params = {}

        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

        url = self.resource() + '/' + str(self.id)
        resp = self._client.get(url, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
        else:
            e = TunnelError.parse(resp)
            raise e

    def open(self, start=0, length=sys.maxsize):
        compress_option = self._compress_option or io.CompressOption()

        params = {}

        headers = {'Content-Length': 0, 'x-odps-tunnel-version': 4}
        if compress_option.algorithm == io.CompressOption.CompressAlgorithm.ODPS_ZLIB:
            headers['Accept-Encoding'] = 'deflate'
        elif compress_option.algorithm != io.CompressOption.CompressAlgorithm.ODPS_RAW:
            raise TunnelError('invalid compression option')

        params['data'] = ''
        params['range'] = '(%s,%s)' % (start, length)

        url = self.resource()
        resp = self._client.get(url + '/' + self.id, params=params, headers=headers, stream=True)
        if not self._client.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e

        content_encoding = resp.headers.get('Content-Encoding')
        if content_encoding is not None:
            if content_encoding == 'deflate':
                self._compress_option = io.CompressOption(
                    io.CompressOption.CompressAlgorithm.ODPS_ZLIB, -1, 0)
            else:
                raise TunnelError('Invalid content encoding')
            compress = True
        else:
            compress = False

        option = compress_option if compress else None
        return VolumeReader(self._client, resp, option)


class VolumeReader(object):
    def __init__(self, client, response, compress_option):
        self._client = client
        self._response = io.RequestsInputStream(response)
        self._compress_option = compress_option
        self._crc = Checksum(method='crc32')
        self._buffer_size = 0
        self._initialized = False
        self._last_line_ending = None
        self._eof = False

        # buffer part left by sized read or read-line operation, see read()
        self._left_part = None
        self._left_part_pos = 0

        # left part of checksum block when chunked, see _read_buf()
        self._chunk_left = None

    def _raw_read(self, l):
        return self._response.read(l)

    def _init_buf(self):
        size_buf = self._raw_read(4)
        if not size_buf:
            raise IOError('Tunnel reader breaks unexpectedly.')
        self._crc.update(size_buf)
        chunk_size = struct.unpack('>I', size_buf)[0]
        if chunk_size > MAX_CHUNK_SIZE or chunk_size < MIN_CHUNK_SIZE:
            raise IOError("ChunkSize should be in [%d, %d], now is %d." % (MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, chunk_size))
        self._buffer_size = CHECKSUM_SIZE + chunk_size

    def _read_buf(self):
        has_stuff = False

        data_buffer = six.BytesIO()
        if self._chunk_left:
            # we have cached chunk left, add to buffer
            data_buffer.write(self._chunk_left)
            self._chunk_left = None
        while data_buffer.tell() < self._buffer_size:
            try:
                # len(buf) might be less than _buffer_size
                buf = self._raw_read(self._buffer_size)
                if not buf:
                    break
                data_buffer.write(buf)
                has_stuff = True
            except StopIteration:
                break
            except StreamConsumedError:
                break
        if not has_stuff:
            return None

        # check if we need to store the rest part.
        if data_buffer.tell() <= self._buffer_size:
            buf = data_buffer.getvalue()
        else:
            buf_all = data_buffer.getvalue()
            buf, self._chunk_left = buf_all[:self._buffer_size], buf_all[self._buffer_size:]

        if len(buf) >= CHECKSUM_SIZE:
            self._data_size = len(buf) - CHECKSUM_SIZE
            self._crc.update(buf[:self._data_size])
            checksum = struct.unpack_from(CHECKSUM_PACKER, buf, self._data_size)[0]
            if checksum != self._crc.getvalue():
                raise IOError('CRC check error in VolumeReader.')
        else:
            raise IOError('Invalid VolumeReader.')
        return bytearray(buf[:self._data_size])

    def read(self, size=None, break_line=False):
        if size is None:
            size = sys.maxsize
        if self._eof:
            return None
        if size == 0:
            return six.binary_type()

        if not self._initialized:
            self._initialized = True
            self._init_buf()

        has_stuff = False

        out_buf = six.BytesIO()
        if self._left_part:
            if break_line:
                # deal with Windows line endings
                if self._left_part[self._left_part_pos] == ord('\n') and self._last_line_ending == ord('\r'):
                    self._last_line_ending = None
                    self._left_part_pos += 1

                for idx in irange(self._left_part_pos, len(self._left_part)):
                    if self._left_part[idx] not in (ord('\r'), ord('\n')):
                        continue
                    self._last_line_ending = self._left_part[idx]
                    self._left_part[idx] = ord('\n')
                    ret = self._left_part[self._left_part_pos:idx + 1]
                    self._left_part_pos = idx + 1
                    if self._left_part_pos == len(self._left_part):
                        self._left_part = None
                        self._left_part_pos = 0
                    return bytes(ret)
            if len(self._left_part) - self._left_part_pos >= size:
                ret = self._left_part[self._left_part_pos:self._left_part_pos + size]
                self._left_part_pos += size
                return bytes(ret)
            else:
                out_buf.write(bytes(self._left_part[self._left_part_pos:]))
                self._left_part = None
                self._left_part_pos = 0
                has_stuff = True
        length_left = size - out_buf.tell()
        while length_left > 0:
            buf = self._read_buf()
            if buf is None:
                self._eof = True
                break
            has_stuff = True
            start_pos = 0
            if break_line:
                if buf[0] == ord('\n') and self._last_line_ending == ord('\r'):
                    start_pos = 1
                for idx in irange(start_pos, len(buf)):
                    if buf[idx] not in (ord('\r'), ord('\n')):
                        continue
                    self._last_line_ending = buf[idx]
                    buf[idx] = ord('\n')
                    out_buf.write(bytes(buf[start_pos:idx + 1]))
                    if idx + 1 < len(buf):
                        self._left_part = buf[idx + 1:]
                        self._left_part_pos = 0
                    return out_buf.getvalue()

            if len(buf) >= length_left:
                out_buf.write(bytes(buf[start_pos:start_pos + length_left]))
                if len(buf) > length_left:
                    self._left_part = buf[start_pos + length_left:]
                    self._left_part_pos = 0
                length_left = 0
            else:
                out_buf.write(bytes(buf[start_pos:start_pos + self._data_size]))
                length_left -= self._data_size
        return out_buf.getvalue() if has_stuff else None

    def _it(self, size=sys.maxsize, encoding='utf-8'):
        while True:
            line = self.readline(size, encoding=encoding)
            if line is None:
                break
            yield line

    def readline(self, size=sys.maxsize, encoding='utf-8'):
        line = self.read(size, break_line=True)
        return to_text(line, encoding=encoding)

    def readlines(self, size=sys.maxsize, encoding='utf-8'):
        return [line for line in self._it(size, encoding=encoding)]

    def __iter__(self):
        return self._it()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class VolumeUploadSession(serializers.JSONSerializableModel):
    __slots__ = 'id', '_client', '_compress_option', 'project_name', 'volume_name', 'partition_spec'

    class Status(Enum):
        UNKNOWN = 'UNKNOWN'
        NORMAL = 'NORMAL'
        CLOSING = 'CLOSING'
        CLOSED = 'CLOSED'
        CANCELED = 'CANCELED'
        EXPIRED = 'EXPIRED'
        CRITICAL = 'CRITICAL'

    class UploadFile(serializers.JSONSerializableModel):
        file_name = serializers.JSONNodeField('FileName')
        file_length = serializers.JSONNodeField('FileLength')

    id = serializers.JSONNodeField('UploadID')
    status = serializers.JSONNodeField('Status',
                                       parse_callback=lambda v: VolumeUploadSession.Status(v.upper()))
    file_list = serializers.JSONNodesReferencesField(UploadFile, 'FileList')

    def __init__(self, client, volume, partition_spec, upload_id=None, compress_option=None):
        super(VolumeUploadSession, self).__init__()

        self._client = client
        self._compress_option = compress_option
        self.project_name = volume.project.name
        self.volume_name = volume.name
        self.partition_spec = partition_spec

        if upload_id is None:
            self._init()
        else:
            self.id = upload_id
            self.reload()
        self._compress_option = compress_option
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return (
            "<VolumeUploadSession id=%s project_name=%s volume_name=%s partition_spec=%s>"
            % (self.id, self.project_name, self.volume_name, self.partition_spec)
        )

    def resource(self):
        return self._client.endpoint + '/projects/%s/tunnel/uploads' % self.project_name

    def _init(self):
        headers = {'Content-Length': '0'}
        params = dict(type='volumefile', target='/'.join([self.project_name, self.volume_name,
                                                          self.partition_spec]) + '/')

        url = self.resource()
        resp = self._client.post(url, {}, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
        else:
            e = TunnelError.parse(resp)
            raise e

    def reload(self):
        headers = {'Content-Length': '0'}
        params = {}

        url = self.resource() + '/' + str(self.id)
        resp = self._client.get(url, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
        else:
            e = TunnelError.parse(resp)
            raise e

    @staticmethod
    def _format_file_name(file_name):
        buf = six.StringIO()
        if file_name and file_name[0] == '/':
            raise TunnelError("FileName cannot start with '/', file name is " + file_name)
        pre_slash = False
        for ch in file_name:
            if ch == '/':
                if not pre_slash:
                    buf.write(ch)
                pre_slash = True
            else:
                buf.write(ch)
                pre_slash = False
        return buf.getvalue()

    def open(self, file_name, compress=False, append=False):
        compress_option = self._compress_option or io.CompressOption()
        headers = {'Content-Type': 'test/plain', 'Transfer-Encoding': 'chunked', 'x-odps-tunnel-version': 4}
        params = {}

        if compress:
            if compress_option.algorithm == io.CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Content-Encoding'] = 'deflate'
            elif compress_option.algorithm != io.CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')

        file_name = self._format_file_name(file_name)
        params['blockid'] = file_name
        if append:
            params['resume'] = ''

        url = self.resource() + '/' + self.id

        chunk_uploader = lambda data: self._client.post(url, data=data, params=params, headers=headers)
        option = compress_option if compress else None
        return VolumeWriter(self._client, chunk_uploader, option)

    def commit(self, files):
        if not files:
            raise ValueError('`files` not supplied')
        if isinstance(files, six.string_types):
            files = [files, ]
        formatted = [self._format_file_name(fn) for fn in files]

        self.reload()
        files_uploading = set(f.file_name for f in self.file_list)

        if len(files_uploading) != len(formatted):
            raise TunnelError("File number not match, server: %d, client: %d" % (len(files_uploading), len(formatted)))
        for fn in (fn for fn in formatted if fn not in files_uploading):
            raise TunnelError("File not exits on server, file name is " + fn)

        self._complete_upload()

    def _complete_upload(self):
        headers = {'Content-Length': '0'}
        params = {}

        url = self.resource() + '/' + self.id
        resp = self._client.put(url, {}, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
        else:
            e = TunnelError.parse(resp)
            raise e


class VolumeWriter(object):
    CHUNK_SIZE = 512 * 1024

    def __init__(self, client, uploader, compress_option):
        self._client = client
        self._compress_option = compress_option
        self._req_io = io.RequestsIO(uploader, chunk_size=options.chunk_size)

        if compress_option is None:
            self._writer = self._req_io
        elif compress_option.algorithm == \
                io.CompressOption.CompressAlgorithm.ODPS_RAW:
            self._writer = self._req_io
        elif compress_option.algorithm == \
                io.CompressOption.CompressAlgorithm.ODPS_ZLIB:
            self._writer = io.DeflateOutputStream(self._req_io)
        else:
            raise errors.InvalidArgument('Invalid compression algorithm.')

        self._crc = Checksum(method='crc32')
        self._initialized = False
        self._chunk_offset = 0

    def _init_writer(self):
        chunk_bytes = struct.pack('>I', self.CHUNK_SIZE)
        self._writer.write(chunk_bytes)
        self._crc.update(chunk_bytes)
        self._chunk_offset = 0

    def write(self, buf, encoding='utf-8'):
        buf = to_binary(buf, encoding=encoding)
        if isinstance(buf, six.integer_types):
            buf = bytes(bytearray([buf, ]))
        elif isinstance(buf, six.BytesIO):
            buf = buf.getvalue()
        if not self._initialized:
            self._initialized = True
            self._init_writer()
            self._req_io.start()

        if not buf:
            raise IOError('Invalid data buffer!')
        processed = 0
        while processed < len(buf):
            if self._chunk_offset == self.CHUNK_SIZE:
                checksum = self._crc.getvalue()
                self._writer.write(struct.pack(CHECKSUM_PACKER, checksum))
                self._chunk_offset = 0
            else:
                size = self.CHUNK_SIZE - self._chunk_offset if len(buf) - processed > self.CHUNK_SIZE - self._chunk_offset\
                    else len(buf) - processed
                write_chunk = buf[processed:processed + size]
                self._writer.write(write_chunk)
                self._crc.update(write_chunk)
                processed += size
                self._chunk_offset += size

    def close(self):
        if not self._initialized:
            self._initialized = True
            self._init_writer()

        if self._chunk_offset != 0:
            checksum = self._crc.getvalue()
            self._writer.write(struct.pack(CHECKSUM_PACKER, checksum))
        self._writer.flush()
        result = self._req_io.finish()
        if result is None:
            raise TunnelError('No results returned in VolumeWriter.')
        if not self._client.is_ok(result):
            e = TunnelError.parse(result)
            raise e
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if an error occurs inside the with block, we do not commit
        if exc_val is not None:
            return
        self.close()


class VolumeFSWriter(VolumeWriter):
    def __init__(self, client, uploader, volume, path, compress_option):
        self._volume = volume
        self._path = path
        super(VolumeFSWriter, self).__init__(client, uploader, compress_option)

    def close(self):
        result = super(VolumeFSWriter, self).close()
        if 'x-odps-volume-sessionid' not in result.headers:
            raise TunnelError('No session id returned in response.')
        headers = {
            'x-odps-volume-fs-path': '/' + self._volume.name + '/' + self._path.lstrip('/'),
            'x-odps-volume-sessionid': result.headers.get('x-odps-volume-sessionid'),
        }
        commit_result = self._client.put(self._volume.resource(client=self._client), None, headers=headers)
        if not self._client.is_ok(commit_result):
            e = TunnelError.parse(commit_result)
            raise e
        return result
