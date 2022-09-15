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

from .base import BaseTunnel, TUNNEL_VERSION
from .io.reader import TunnelRecordReader, TunnelArrowReader
from .io.stream import CompressOption, SnappyRequestsInputStream, RequestsInputStream
from .errors import TunnelError
from .. import errors, serializers, types
from ..compat import Enum, six
from ..config import options
from ..models import Projects, Schema
from ..models.table import TableSchema

try:
    import numpy as np
except ImportError:
    np = None


class InstanceDownloadSession(serializers.JSONSerializableModel):
    __slots__ = '_client', '_instance', '_limit_enabled', '_compress_option', '_sessional', \
                '_session_task_name', '_session_subquery_id'

    class Status(Enum):
        Unknown = 'UNKNOWN'
        Normal = 'NORMAL'
        Closes = 'CLOSES'
        Expired = 'EXPIRED'
        Failed = 'FAILED'
        Initiating = 'INITIATING'

    id = serializers.JSONNodeField('DownloadID')
    status = serializers.JSONNodeField(
        'Status', parse_callback=lambda s: InstanceDownloadSession.Status(s.upper()))
    count = serializers.JSONNodeField('RecordCount')
    schema = serializers.JSONNodeReferenceField(Schema, 'Schema')

    def __init__(self, client, instance, download_id=None, limit=None,
                 compress_option=None, **kw):
        super(InstanceDownloadSession, self).__init__()

        self._client = client
        self._instance = instance
        self._limit_enabled = limit if limit is not None else kw.get('limit_enabled', False)
        self._sessional = kw.pop("sessional", False)
        self._session_task_name = kw.pop("session_task_name", "")
        self._session_subquery_id = int(kw.pop("session_subquery_id", -1))
        if self._sessional and ((not self._session_task_name) or (self._session_subquery_id == -1)):
            raise TunnelError(
                "Taskname('session_task_name') and Subquery ID ('session_subquery_id') "
                "keyword argument must be provided for session instance tunnels."
            )

        if download_id is None:
            self._init()
        else:
            self.id = download_id
            self.reload()
        self._compress_option = compress_option
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return "<InstanceDownloadSession id=%s project_name=%s instance_id=%s>" % (
            self.id,
            self._instance.project.name,
            self._instance.id,
        )

    def _init(self):
        params = {'downloads': ''}
        headers = {
            'Content-Length': 0,
            'x-odps-tunnel-version': TUNNEL_VERSION,
        }
        # Now we use DirectDownloadMode to fetch session results(any other method is removed)
        # This mode, only one request needed. So we don't have to send request here ..
        if not self._sessional:
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
        if not self._sessional:
            params = {'downloadid': self.id}
            headers = {
                'Content-Length': 0,
                'x-odps-tunnel-version': TUNNEL_VERSION,
            }
            if self._sessional:
                params['cached'] = ''
                params['taskname'] = self._session_task_name

            url = self._instance.resource()
            resp = self._client.get(url, params=params, headers=headers)
            if self._client.is_ok(resp):
                self.parse(resp, obj=self)
                if self.schema is not None:
                    self.schema.build_snapshot()
            else:
                e = TunnelError.parse(resp)
                raise e
        else:
            self.status = InstanceDownloadSession.Status.Normal

    def _open_reader(self, start, count, compress=False, columns=None, arrow=False, reader_cls=None):
        compress_option = self._compress_option or CompressOption()

        params = {}
        headers = {'x-odps-tunnel-version': TUNNEL_VERSION}
        if self._sessional:
            params['cached'] = ''
            params['taskname'] = self._session_task_name
            params['queryid'] = str(self._session_subquery_id)
        else:
            params['downloadid'] = self.id
            params['rowrange'] = '(%s,%s)' % (start, count)
            headers['Content-Length'] = 0
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
        params['data'] = ''
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, types.Column) else col
            params['columns'] = ','.join(col_name(col) for col in columns)

        if arrow:
            params['arrow'] = ''

        url = self._instance.resource()
        resp = self._client.get(url, stream=True, params=params, headers=headers)
        if not self._client.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e

        if self._sessional:
            # in DirectDownloadMode, the schema is brought back in HEADER.
            # handle this.
            schema_json = resp.headers.get('odps-tunnel-schema')
            self.schema = TableSchema()
            self.schema = self.schema.deserial(schema_json)

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

        return reader_cls(self.schema, input_stream, columns=columns)

    def open_record_reader(self, start, count, compress=False, columns=None):
        return self._open_reader(
            start, count, compress=compress, columns=columns, reader_cls=TunnelRecordReader
        )

    def open_arrow_reader(self, start, count, compress=False, columns=None):
        return self._open_reader(
            start, count, compress=compress, columns=columns, arrow=True, reader_cls=TunnelArrowReader
        )

    if np is not None:
        def open_pandas_reader(self, start, count, compress=False, columns=None):
            from .pdio.pdreader_c import TunnelPandasReader
            return self._open_reader(
                start, count, compress=compress, columns=columns, reader_cls=TunnelPandasReader
            )


class InstanceTunnel(BaseTunnel):
    def create_download_session(self, instance, download_id=None, limit=None, compress_option=None,
                                compress_algo=None, compress_level=None, compress_strategy=None, **kw):
        if not isinstance(instance, six.string_types):
            instance = instance.id
        instance = Projects(client=self.tunnel_rest)[self._project.name].instances[instance]
        compress_option = compress_option
        if compress_option is None and compress_algo is not None:
            compress_option = CompressOption(
                compress_algo=compress_algo, level=compress_level, strategy=compress_strategy)

        if limit is None:
            limit = kw.get('limit_enabled', False)
        return InstanceDownloadSession(self.tunnel_rest, instance, download_id=download_id,
                                       limit=limit, compress_option=compress_option, **kw)
