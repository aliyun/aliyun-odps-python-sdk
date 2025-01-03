#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import logging
import sys
from collections import OrderedDict

import requests

from .. import serializers, types
from ..compat import Enum, six
from ..config import options
from ..models import Projects, TableSchema
from .base import TUNNEL_VERSION, BaseTunnel
from .errors import TunnelError
from .io.reader import TunnelArrowReader, TunnelRecordReader
from .io.stream import CompressOption, get_decompress_stream

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class InstanceDownloadSession(serializers.JSONSerializableModel):
    __slots__ = (
        "_client",
        "_instance",
        "_limit_enabled",
        "_compress_option",
        "_sessional",
        "_session_task_name",
        "_session_subquery_id",
        "_quota_name",
        "_timeout",
        "_tags",
    )

    class Status(Enum):
        Unknown = "UNKNOWN"
        Normal = "NORMAL"
        Closes = "CLOSES"
        Expired = "EXPIRED"
        Failed = "FAILED"
        Initiating = "INITIATING"

    id = serializers.JSONNodeField("DownloadID")
    status = serializers.JSONNodeField(
        "Status", parse_callback=lambda s: InstanceDownloadSession.Status(s.upper())
    )
    count = serializers.JSONNodeField("RecordCount")
    schema = serializers.JSONNodeReferenceField(TableSchema, "Schema")
    quota_name = serializers.JSONNodeField("QuotaName")

    def __init__(
        self,
        client,
        instance,
        download_id=None,
        limit=None,
        compress_option=None,
        quota_name=None,
        timeout=None,
        tags=None,
        **kw
    ):
        super(InstanceDownloadSession, self).__init__()

        self._client = client
        self._instance = instance
        self._limit_enabled = (
            limit if limit is not None else kw.get("limit_enabled", False)
        )
        self._quota_name = quota_name
        self._sessional = kw.pop("sessional", False)
        self._session_task_name = kw.pop("session_task_name", "")
        self._session_subquery_id = int(kw.pop("session_subquery_id", -1))
        self._timeout = timeout
        if self._sessional and (
            (not self._session_task_name) or (self._session_subquery_id == -1)
        ):
            raise TunnelError(
                "Taskname('session_task_name') and Subquery ID ('session_subquery_id') "
                "keyword argument must be provided for session instance tunnels."
            )

        self._tags = tags or options.tunnel.tags
        if isinstance(self._tags, six.string_types):
            self._tags = self._tags.split(",")

        if download_id is None:
            self._init()
        else:
            self.id = download_id
            self.reload()
        self._compress_option = compress_option

        logger.info("Tunnel session created: %r", self)
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        repr_kw = OrderedDict(
            [
                ("id", self.id),
                ("project_name", self._instance.project.name),
                ("instance_id", self._instance.id),
                ("subquery_id", self._session_subquery_id if self._sessional else None),
                ("limited", self._limit_enabled if self._limit_enabled else None),
            ]
        )
        repr_kw = OrderedDict([(k, v) for k, v in repr_kw.items() if v is not None])
        return "<InstanceDownloadSession %s>" % " ".join(
            "%s=%s" % (k, v) for k, v in repr_kw.items()
        )

    def _init(self):
        params = {}
        headers = {
            "Content-Length": 0,
            "x-odps-tunnel-version": TUNNEL_VERSION,
        }
        if self._tags:
            headers["odps-tunnel-tags"] = ",".join(self._tags)
        if self._quota_name is not None:
            params["quotaName"] = self._quota_name

        # Now we use DirectDownloadMode to fetch session results(any other method is removed)
        # This mode, only one request needed. So we don't have to send request here ..
        if not self._sessional:
            if self._limit_enabled:
                params["instance_tunnel_limit_enabled"] = ""
            url = self._instance.resource()
            try:
                resp = self._client.post(
                    url,
                    {},
                    action="downloads",
                    params=params,
                    headers=headers,
                    timeout=self._timeout,
                )
            except requests.exceptions.ReadTimeout:
                if callable(options.tunnel_session_create_timeout_callback):
                    options.tunnel_session_create_timeout_callback(*sys.exc_info())
                raise
            if self._client.is_ok(resp):
                self.parse(resp, obj=self)
                if self.schema is not None:
                    self.schema.build_snapshot()
            else:
                e = TunnelError.parse(resp)
                raise e

    def reload(self):
        if not self._sessional:
            params = {"downloadid": self.id}
            if self._quota_name is not None:
                params["quotaName"] = self._quota_name
            headers = {
                "Content-Length": 0,
                "x-odps-tunnel-version": TUNNEL_VERSION,
            }
            if self._tags:
                headers["odps-tunnel-tags"] = ",".join(self._tags)
            if self._sessional:
                params["cached"] = ""
                params["taskname"] = self._session_task_name

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

    def _build_input_stream(
        self, start, count, compress=False, columns=None, arrow=False
    ):
        compress_option = self._compress_option or CompressOption()

        params = {}
        headers = {"x-odps-tunnel-version": TUNNEL_VERSION}
        if self._quota_name is not None:
            params["quotaName"] = self._quota_name
        if self._tags:
            headers["odps-tunnel-tags"] = ",".join(self._tags)
        if self._sessional:
            params["cached"] = ""
            params["taskname"] = self._session_task_name
            params["queryid"] = str(self._session_subquery_id)
        else:
            params["downloadid"] = self.id
            params["rowrange"] = "(%s,%s)" % (start, count)
            headers["Content-Length"] = 0
        if compress:
            encoding = compress_option.algorithm.get_encoding()
            if encoding is not None:
                headers["Accept-Encoding"] = encoding
        params["data"] = ""
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, types.Column) else col
            params["columns"] = ",".join(col_name(col) for col in columns)

        if arrow:
            params["arrow"] = ""

        url = self._instance.resource()
        resp = self._client.get(url, stream=True, params=params, headers=headers)
        if not self._client.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e

        if self._sessional:
            # in DirectDownloadMode, the schema is brought back in HEADER.
            # handle this.
            schema_json = resp.headers.get("odps-tunnel-schema")
            self.schema = TableSchema()
            self.schema = self.schema.deserial(schema_json)

        content_encoding = resp.headers.get("Content-Encoding")
        if content_encoding is not None:
            compress_algo = CompressOption.CompressAlgorithm.from_encoding(
                content_encoding
            )
            if compress_algo != compress_option.algorithm:
                compress_option = self._compress_option = CompressOption(
                    compress_algo, -1, 0
                )
            compress = True
        else:
            compress = False

        option = compress_option if compress else None
        return get_decompress_stream(resp, option)

    def _open_reader(
        self, start, count, compress=False, columns=None, arrow=False, reader_cls=None
    ):
        stream_kw = dict(compress=compress, columns=columns, arrow=arrow)
        initial_stream_cache = [None]

        def stream_creator(cursor, cache=False):
            if cursor == 0 and initial_stream_cache[0] is not None:
                initial_stream_cache[0], stream = None, initial_stream_cache[0]
                return stream
            attempt_count = count - cursor if count is not None else None
            stream = self._build_input_stream(
                start + cursor, attempt_count, **stream_kw
            )
            if cache:
                initial_stream_cache[0] = stream
            return stream

        # for MCQA we must obtain schema from the first stream, hence the first reader
        # must be created beforehand and then cached for the reader class
        stream_creator(0, True)
        return reader_cls(self.schema, stream_creator, columns=columns)

    def open_record_reader(self, start, count, compress=False, columns=None, **_):
        return self._open_reader(
            start,
            count,
            compress=compress,
            columns=columns,
            reader_cls=TunnelRecordReader,
        )

    def open_arrow_reader(self, start, count, compress=False, columns=None, **_):
        return self._open_reader(
            start,
            count,
            compress=compress,
            columns=columns,
            arrow=True,
            reader_cls=TunnelArrowReader,
        )


class InstanceTunnel(BaseTunnel):
    def create_download_session(
        self,
        instance,
        download_id=None,
        limit=None,
        compress_option=None,
        compress_algo=None,
        compress_level=None,
        compress_strategy=None,
        timeout=None,
        tags=None,
        **kw
    ):
        if not isinstance(instance, six.string_types):
            instance = instance.id
        instance = Projects(client=self.tunnel_rest)[self._project.name].instances[
            instance
        ]
        compress_option = compress_option
        if compress_option is None and compress_algo is not None:
            compress_option = CompressOption(
                compress_algo=compress_algo,
                level=compress_level,
                strategy=compress_strategy,
            )

        if limit is None:
            limit = kw.get("limit_enabled", False)
        return InstanceDownloadSession(
            self.tunnel_rest,
            instance,
            download_id=download_id,
            limit=limit,
            compress_option=compress_option,
            quota_name=self._quota_name,
            timeout=timeout,
            tags=tags,
            **kw
        )
