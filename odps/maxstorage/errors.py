# -*- coding: utf-8 -*-
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

"""Error hierarchy for :mod:`odps.maxstorage`.

``MaxStorageError`` subclasses :class:`odps.errors.ODPSError` so it integrates
with the existing PyODPS error-handling machinery (``throw_if_parsable`` etc.).
The stub parses non-2xx JSON bodies (``{Code, Message}``) into
``StorageServiceError``; client-side problems become ``StorageClientError``.
"""

from ..errors import ODPSError


class MaxStorageError(ODPSError):
    """Base class for all maxstorage errors."""


class StorageClientError(MaxStorageError):
    """Client-side error (network, invalid parameters, etc.)."""


class BlobDownloadError(StorageClientError):
    """Blob download failure — carries the failing blob reference."""

    def __init__(self, message, failed_blob_ref=None):
        super().__init__(message)
        self.failed_blob_ref = failed_blob_ref


class StorageServiceError(MaxStorageError):
    """Server-side error.

    Fields
    ------
    http_status : int
        HTTP status code of the failing response.
    error_code : str
        Server error code parsed from the JSON body.
    message : str
        Human-readable error message.
    request_id : str
        ``x-odps-request-id`` response header (may be ``None``).
    """

    def __init__(self, http_status, error_code=None, message=None, request_id=None):
        self.http_status = http_status
        self.error_code = error_code
        self.request_id = request_id
        msg = message or ""
        if error_code:
            msg = f"{error_code}: {msg}" if msg else error_code
        super().__init__(msg)
