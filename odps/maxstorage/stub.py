# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

"""RPC layer for :mod:`odps.maxstorage`.

One method per ``StorageStub`` operation.  Each builds the URL
(``api/storage/v{api_version}``), query params (``Target``, ``Action``,
operation-specific), headers (route_token, accept-encoding), JSON body, and
delegates to the ``RestClient``.

Version gating (v3-era features only): only features introduced *after* the
v3 path rename are gated by ``_supports_v3()``:
- ``WriteMode`` query-param: only sent on write operations for v3+.
- ``get_min_uncommitted_staging_id``: v3-only.
- ``custom_file_name`` in blob headers: only serialized for v3+.

Not gated (predate v3, work on both v2 and v3 servers):
- Route token: always passed.
- ``WriteSchema`` with ``ColumnId``: always parsed.
"""

import json
import logging

from .errors import StorageServiceError
from .models.responses import (
    CloseWriteStreamResponse,
    CreateInstanceReadSessionResponse,
    CreateTableReadSessionResponse,
    CreateTableWriteSessionResponse,
    CreateWriteStreamResponse,
    GetTableReadSessionResponse,
    GetTableWriteSessionResponse,
    GetWriteStreamResponse,
    WriteStreamResponse,
)
from .models.schema import WriteSchema
from .options import _supports_v3

logger = logging.getLogger(__name__)

ROUTE_TOKEN_HEADER = "x-odps-max-storage-route-token"
WRITE_ACCESS_TOKEN_HEADER = "x-odps-max-storage-write-access-token"
STORAGE_API_PREFIX = "api/storage/v"


def _parse_json_response(resp):
    """Parse JSON body from a tunnel REST response."""
    try:
        if hasattr(resp, "json"):
            return resp.json()
        return json.loads(resp.text)
    except (ValueError, TypeError) as exc:
        snippet = resp.text[:200] if hasattr(resp, "text") else repr(resp)
        raise StorageServiceError(
            resp.status_code if hasattr(resp, "status_code") else 0,
            error_code="ParseError",
            message=f"Failed to parse JSON response: {exc}. Response snippet: {snippet}",
        ) from exc


def _update_request_id(response, resp):
    """Copy the ODPS request ID from HTTP response headers onto the model."""
    if hasattr(resp, "headers") and "x-odps-request-id" in resp.headers:
        response.request_id = resp.headers["x-odps-request-id"]


class StorageStub:
    """The RPC layer.  One method per ``StorageStub`` operation.

    Each method builds the URL, query params, headers, JSON body, and
    delegates to ``self._rest`` (a :class:`odps.rest.RestClient`).
    """

    def __init__(self, rest, api_version="2"):
        self._rest = rest
        self._api_version = str(api_version)

    @property
    def api_version(self):
        return self._api_version

    def _supports_v3(self):
        """True when ``api_version >= 3``.  Gates v3-era features only."""
        return _supports_v3(self._api_version)

    def _url(self):
        return self._rest.endpoint + "/" + STORAGE_API_PREFIX + self._api_version

    def _build_common_headers(self, route_token=None):
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if route_token:
            headers[ROUTE_TOKEN_HEADER] = route_token
        return headers

    def _build_params(self, target, action, extra=None):
        params = {"Action": action, "Target": target}
        if extra:
            params.update(extra)
        return params

    def _write_mode_params(self, write_mode):
        """Return ``{'WriteMode': value}`` for v3+; empty dict for v2."""
        if self._supports_v3() and write_mode is not None:
            return {"WriteMode": write_mode.value}
        return {}

    def _parse_response(self, resp, response_cls, set_route_token=True):
        """Parse a JSON response into ``response_cls``, set request_id/route_token."""
        resp_json = _parse_json_response(resp)
        response = response_cls.from_dict(resp_json)
        _update_request_id(response, resp)
        if set_route_token:
            route_token = resp.headers.get(ROUTE_TOKEN_HEADER)
            if route_token:
                response.route_token = route_token
        warning = getattr(response, "warning_message", None)
        if warning:
            logger.warning(warning)
        return response

    # ---- Table read ----

    def create_table_read_session(self, table_id, request):
        """Action=TableCreateReadSession.  No route token sent; read from response."""
        url = self._url()
        params = self._build_params(table_id.to_target(), "TableCreateReadSession")
        headers = self._build_common_headers()
        body = json.dumps(request.to_dict())
        resp = self._rest.post(url, data=body, params=params, headers=headers)
        return self._parse_response(resp, CreateTableReadSessionResponse)

    def get_table_read_session(self, table_id, session_id, refresh=False):
        """Action=TableGetReadSession.  Lowercase ``session_refresh`` param."""
        url = self._url()
        params = self._build_params(
            table_id.to_target(),
            "TableGetReadSession",
            {"SessionId": session_id, "session_refresh": str(refresh).lower()},
        )
        headers = self._build_common_headers()
        resp = self._rest.post(url, data="{}", params=params, headers=headers)
        return self._parse_response(resp, GetTableReadSessionResponse)

    def create_table_read_stream(
        self, table_id, split, request, route_token, accept_encoding=None
    ):
        """Action=TableRead.  Streaming download; returns raw response.

        ``accept_encoding`` (str, optional) sets the ``ACCEPT-ENCODING`` header
        so the server compresses the Arrow stream (e.g. ``zstd`` / ``x-lz4-frame``).
        """
        url = self._url()
        params = self._build_params(
            table_id.to_target(),
            "TableRead",
            {"SessionId": split.session_id},
        )
        # Split-subtype-dependent params
        if hasattr(split, "split_index"):
            params["Index"] = str(split.split_index)
        elif hasattr(split, "offset"):
            params["Offset"] = str(split.offset)
            params["Count"] = str(split.length)
        headers = self._build_common_headers(route_token=route_token)
        if accept_encoding:
            headers["ACCEPT-ENCODING"] = accept_encoding
        body = json.dumps(request.to_dict())
        return self._rest.post(
            url, data=body, params=params, headers=headers, stream=True
        )

    def preview(self, table_id, request):
        """Action=TablePreview.  Streaming download; no ACCEPT-ENCODING."""
        url = self._url()
        params = self._build_params(table_id.to_target(), "TablePreview")
        if request.limit is not None:
            params["Limit"] = str(request.limit)
        if request.partition is not None:
            params["Partition"] = request.partition
        headers = self._build_common_headers()
        body = json.dumps(request.to_dict())
        return self._rest.post(
            url, data=body, params=params, headers=headers, stream=True
        )

    # ---- Table write ----

    def create_table_write_session(self, table_id, request, write_mode):
        """Action=TableCreateWriteSession."""
        url = self._url()
        params = self._build_params(table_id.to_target(), "TableCreateWriteSession")
        params.update(self._write_mode_params(write_mode))
        headers = self._build_common_headers()
        body = json.dumps(request.to_dict())
        resp = self._rest.post(url, data=body, params=params, headers=headers)
        return self._parse_response(resp, CreateTableWriteSessionResponse)

    def get_table_write_session(self, table_id, session_id, write_mode):
        """Action=TableGetWriteSession.  Route token sent + read from response."""
        url = self._url()
        params = self._build_params(
            table_id.to_target(),
            "TableGetWriteSession",
            {"SessionId": session_id},
        )
        params.update(self._write_mode_params(write_mode))
        headers = self._build_common_headers()
        resp = self._rest.post(url, data="{}", params=params, headers=headers)
        return self._parse_response(resp, GetTableWriteSessionResponse)

    def commit_table_write_session(
        self,
        table_id,
        session_id,
        stream_ids,
        stream_versions,
        write_mode,
        route_token=None,
    ):
        """Action=TableCommitWriteSession."""
        url = self._url()
        params = self._build_params(
            table_id.to_target(),
            "TableCommitWriteSession",
            {"SessionId": session_id},
        )
        params.update(self._write_mode_params(write_mode))
        headers = self._build_common_headers(route_token=route_token)
        body = "{}"
        if stream_ids and stream_versions and len(stream_ids) == len(stream_versions):
            body = json.dumps(
                {
                    "StreamIds": stream_ids,
                    "StreamVersions": stream_versions,
                }
            )
        self._rest.post(url, data=body, params=params, headers=headers)

    def abort_table_write_session(
        self,
        table_id,
        session_id,
        write_mode,
        route_token=None,
    ):
        """Action=TableAbortWriteSession."""
        url = self._url()
        params = self._build_params(
            table_id.to_target(),
            "TableAbortWriteSession",
            {"SessionId": session_id},
        )
        params.update(self._write_mode_params(write_mode))
        headers = self._build_common_headers(route_token=route_token)
        self._rest.post(url, data="{}", params=params, headers=headers)

    def create_table_write_stream(
        self, table_id, session_id, request, route_token, write_mode
    ):
        """Action=TableCreateWriteStream.  Route token sent + read (if non-null)."""
        url = self._url()
        params = self._build_params(
            table_id.to_target(),
            "TableCreateWriteStream",
            {"SessionId": session_id},
        )
        params.update(self._write_mode_params(write_mode))
        headers = self._build_common_headers(route_token=route_token)
        body = json.dumps(request.to_dict())
        resp = self._rest.post(url, data=body, params=params, headers=headers)
        return self._parse_response(resp, CreateWriteStreamResponse)

    def get_write_stream(self, table_id, request, route_token, write_mode):
        """Action=TableGetWriteStream.  EO mode sends ExactlyOnceMode param."""
        url = self._url()
        params = self._build_params(
            table_id.to_target(),
            "TableGetWriteStream",
            {
                "SessionId": request.session_id,
                "StreamId": request.stream_id,
                "StreamVersion": str(request.stream_version),
            },
        )
        params.update(self._write_mode_params(write_mode))
        if request.exactly_once_mode is True:
            params["ExactlyOnceMode"] = "true"
        headers = self._build_common_headers(route_token=route_token)
        body = json.dumps(request.to_dict())
        resp = self._rest.post(url, data=body, params=params, headers=headers)
        return self._parse_response(resp, GetWriteStreamResponse, set_route_token=False)

    def write_table(
        self,
        table_id,
        session_id,
        stream_id,
        stream_version,
        record_count,
        arrow_body,
        route_token,
        *,
        streaming_table_id=None,
        streaming_schema_version=None,
        row_offset=-1,
        access_token=None,
        write_mode=None,
        compress_option=None
    ):
        """Action=TableWrite.  Streaming upload; returns raw response."""
        url = self._url()
        params = self._build_params(
            table_id.to_target(),
            "TableWrite",
            {
                "SessionId": session_id,
                "StreamId": str(stream_id),
                "StreamVersion": str(stream_version),
                "Count": str(record_count),
            },
        )
        params.update(self._write_mode_params(write_mode))
        if streaming_table_id is not None:
            params["TableId"] = streaming_table_id
        if streaming_schema_version is not None:
            params["SchemaVersion"] = str(streaming_schema_version)
        if row_offset >= 0:
            params["RowOffset"] = str(row_offset)
        headers = self._build_common_headers(route_token=route_token)
        headers["Content-Type"] = "application/octet-stream"
        if access_token:
            headers[WRITE_ACCESS_TOKEN_HEADER] = access_token
        return self._rest.post(url, data=arrow_body, params=params, headers=headers)

    def parse_write_stream_response(self, http_response):
        """Parse the writeTable response body into :class:`WriteStreamResponse`."""
        resp_json = _parse_json_response(http_response)
        return WriteStreamResponse.from_dict(resp_json)

    def close_write_stream(self, table_id, request, route_token, write_mode):
        """Action=TableCloseWriteStream."""
        url = self._url()
        params = self._build_params(
            table_id.to_target(),
            "TableCloseWriteStream",
            {"SessionId": request.session_id},
        )
        params.update(self._write_mode_params(write_mode))
        headers = self._build_common_headers(route_token=route_token)
        body = json.dumps(request.to_dict())
        resp = self._rest.post(url, data=body, params=params, headers=headers)
        return self._parse_response(
            resp, CloseWriteStreamResponse, set_route_token=False
        )

    def get_write_schema(self, table_id, session_id, route_token):
        """Fetch the write schema (column IDs, nested type info).

        Called by ``TableArrowBlobUploadWriter`` at construction to resolve
        blob column IDs.  Result may be cached on the session.
        """
        url = self._url()
        params = self._build_params(
            table_id.to_target(),
            "TableGetWriteSchema",
            {"SessionId": session_id},
        )
        headers = self._build_common_headers(route_token=route_token)
        resp = self._rest.post(url, data="{}", params=params, headers=headers)
        resp_json = _parse_json_response(resp)
        return WriteSchema.from_dict(resp_json.get("TableSchema"))

    # ---- Instance read ----

    def create_instance_read_session(self, instance_id, request):
        """Action=InstanceCreateReadSession."""
        url = self._url()
        params = self._build_params(
            instance_id.to_target(), "InstanceCreateReadSession"
        )
        headers = self._build_common_headers()
        body = json.dumps(request.to_dict())
        resp = self._rest.post(url, data=body, params=params, headers=headers)
        response = CreateInstanceReadSessionResponse.from_dict(
            _parse_json_response(resp)
        )
        _update_request_id(response, resp)
        return response

    def get_instance_read_session(self, instance_id, session_id):
        """Action=InstanceGetReadSession (GET)."""
        url = self._url()
        params = self._build_params(
            instance_id.to_target(),
            "InstanceGetReadSession",
            {"SessionId": session_id},
        )
        headers = self._build_common_headers()
        resp = self._rest.get(url, params=params, headers=headers)
        response = CreateInstanceReadSessionResponse.from_dict(
            _parse_json_response(resp)
        )
        _update_request_id(response, resp)
        return response

    def create_instance_read_stream(
        self, instance_id, session_id, count, offset, request, accept_encoding=None
    ):
        """Action=InstanceRead.  Streaming download; returns raw response.

        ``accept_encoding`` (str, optional) sets the ``ACCEPT-ENCODING`` header
        so the server compresses the Arrow stream (e.g. ``zstd`` / ``x-lz4-frame``).
        """
        url = self._url()
        params = self._build_params(
            instance_id.to_target(),
            "InstanceRead",
            {"SessionId": session_id},
        )
        if count is not None:
            params["Count"] = str(count)
        if offset is not None:
            params["Offset"] = str(offset)
        headers = self._build_common_headers()
        if accept_encoding:
            headers["ACCEPT-ENCODING"] = accept_encoding
        body = json.dumps(request.to_dict())
        return self._rest.post(
            url, data=body, params=params, headers=headers, stream=True
        )

    # ---- Blob ----

    def read_blobs(self, references, accept_encoding=None):
        """Action=BlobRead, Target=generic.blob.  Streaming download."""
        url = self._url()
        params = self._build_params("generic.blob", "BlobRead")
        headers = self._build_common_headers()
        if accept_encoding:
            headers["ACCEPT-ENCODING"] = accept_encoding
        body = json.dumps({"BlobReferences": references})
        return self._rest.post(
            url, data=body, params=params, headers=headers, stream=True
        )

    def table_write_blob(
        self, table_id, params_extra, data, route_token=None, content_encoding=None
    ):
        """Action=TableWriteBlob (single stream).  Streaming upload."""
        url = self._url()
        params = self._build_params(table_id.to_target(), "TableWriteBlob")
        params.update(params_extra)
        headers = self._build_common_headers(route_token=route_token)
        headers["Content-Type"] = "application/octet-stream"
        if content_encoding:
            headers["Content-Encoding"] = content_encoding
        return self._rest.post(url, data=data, params=params, headers=headers)

    def table_batch_write_blob(
        self, table_id, params_extra, data, route_token=None, content_encoding=None
    ):
        """Action=TableWriteBlob with Mode=Batch.  Streaming upload."""
        url = self._url()
        params = self._build_params(table_id.to_target(), "TableWriteBlob")
        params.update(params_extra)
        headers = self._build_common_headers(route_token=route_token)
        headers["Content-Type"] = "application/octet-stream"
        if content_encoding:
            headers["Content-Encoding"] = content_encoding
        return self._rest.post(url, data=data, params=params, headers=headers)
