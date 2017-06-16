#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from .base import BaseTunnel
from .io.reader import TunnelRecordReader
from .io.stream import CompressOption, SnappyRequestsInputStream, RequestsInputStream
from .errors import TunnelError
from .. import errors, serializers, types
from ..compat import Enum, six
from ..models import Projects, Schema


class InstanceDownloadSession(serializers.JSONSerializableModel):
    __slots__ = '_client', '_instance', '_limit_enabled', '_compress_option'

    class Status(Enum):
        Unknown = 'UNKNOWN'
        Normal = 'NORMAL'
        Closes = 'CLOSES'
        Expired = 'EXPIRED'

    id = serializers.JSONNodeField('DownloadID')
    status = serializers.JSONNodeField(
        'Status', parse_callback=lambda s: InstanceDownloadSession.Status(s.upper()))
    count = serializers.JSONNodeField('RecordCount')
    schema = serializers.JSONNodeReferenceField(Schema, 'Schema')

    def __init__(self, client, instance, download_id=None, limit_enabled=False,
                 compress_option=None):
        super(InstanceDownloadSession, self).__init__()

        self._client = client
        self._instance = instance
        self._limit_enabled = limit_enabled

        if download_id is None:
            self._init()
        else:
            self.id = download_id
            self.reload()
        self._compress_option = compress_option

    def _init(self):
        params = {'downloads': ''}
        headers = {'Content-Length': 0}

        if self._limit_enabled:
            params['instance_tunnel_limit_enabled'] = ''

        url = self._instance.resource()
        resp = self._client.post(url, {}, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
            if self.schema is not None:
                self.schema.build_snapshot()
        else:
            e = TunnelError.parse(resp)
            raise e

    def reload(self):
        params = {'downloadid': self.id}
        headers = {'Content-Length': 0}

        url = self._instance.resource()
        resp = self._client.get(url, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
            if self.schema is not None:
                self.schema.build_snapshot()
        else:
            e = TunnelError.parse(resp)
            raise e

    def open_record_reader(self, start, count, compress=False, columns=None):
        compress_option = self._compress_option or CompressOption()

        params = {}
        headers = {'Content-Length': 0, 'x-odps-tunnel-version': 4}
        if compress:
            if compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Accept-Encoding'] = 'deflate'
            elif compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_SNAPPY:
                headers['Accept-Encoding'] = 'x-snappy-framed'
            elif compress_option.algorithm != \
                    CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')
        params['downloadid'] = self.id
        params['data'] = ''
        params['rowrange'] = '(%s,%s)' % (start, count)
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, types.Column) else col
            params['columns'] = ','.join(col_name(col) for col in columns)

        url = self._instance.resource()
        resp = self._client.get(url, stream=True, params=params, headers=headers)
        if not self._client.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e

        content_encoding = resp.headers.get('Content-Encoding')
        if content_encoding is not None:
            if content_encoding == 'deflate':
                self._compress_option = CompressOption(
                    CompressOption.CompressAlgorithm.ODPS_ZLIB, -1, 0)
            elif content_encoding == 'x-snappy-framed':
                self._compress_option = CompressOption(
                    CompressOption.CompressAlgorithm.ODPS_SNAPPY, -1, 0)
            else:
                raise TunnelError('invalid content encoding')
            compress = True
        else:
            compress = False

        option = compress_option if compress else None
        if option is None:
            input_stream = RequestsInputStream(resp)  # create a file-like object from body
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_RAW:
            input_stream = RequestsInputStream(resp)  # create a file-like object from body
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_ZLIB:
            input_stream = RequestsInputStream(resp)  # Requests automatically decompress gzip data!
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_SNAPPY:
            input_stream = SnappyRequestsInputStream(resp)
        else:
            raise errors.InvalidArgument('Invalid compression algorithm.')

        return TunnelRecordReader(self.schema, input_stream, columns=columns)


class InstanceTunnel(BaseTunnel):
    def create_download_session(self, instance, download_id=None, limit_enabled=False, compress_option=None,
                                compress_algo=None, compres_level=None, compress_strategy=None, **_):
        if not isinstance(instance, six.string_types):
            instance = instance.id
        instance = Projects(client=self.tunnel_rest)[self._project.name].instances[instance]
        compress_option = compress_option
        if compress_option is None and compress_algo is not None:
            compress_option = CompressOption(
                compress_algo=compress_algo, level=compres_level, strategy=compress_strategy)

        return InstanceDownloadSession(self.tunnel_rest, instance, download_id=download_id,
                                       limit_enabled=limit_enabled, compress_option=compress_option)
