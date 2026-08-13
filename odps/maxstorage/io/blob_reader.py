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

"""Blob download framing parser for :mod:`odps.maxstorage`.

Parses the framed download protocol from a CRC-stripped (and optionally
decompressed) stream: per-blob
``[HeaderLen LE64][Header JSON][DataLen LE64][Data][FooterLen LE64][Footer JSON]``.
"""

import json
from dataclasses import dataclass
from typing import Optional

from ..errors import MaxStorageError
from ..options import _str_version_ge
from .crc import CrcStrippedInputStream, read_exact, read_le_long

_RAW_READ_BLOCK = 1024 * 64


@dataclass(eq=False)
class BlobRecord:
    """A single downloaded blob yielded by :class:`BlobDataIterator`.

    ``custom_file_name`` is only populated when the client targets API v3+;
    it is ``None`` on v2 (the ``CustomFileName`` header is not parsed).
    """

    data: bytes
    mime_type: Optional[str]
    custom_file_name: Optional[str] = None

    def __repr__(self):
        return (
            f"BlobRecord(data={self.data!r}, mime_type={self.mime_type!r}, "
            f"custom_file_name={self.custom_file_name!r})"
        )


class BlobDataIterator:
    """Iterator yielding :class:`BlobRecord` per blob."""

    # Parses the framed download protocol from a CRC-stripped (and optionally
    # decompressed) stream supplied by the caller.  Per-blob wire layout:
    #   [HeaderLen LE64][Header JSON][DataLen LE64][Data][FooterLen LE64][Footer JSON]
    # CRC-stripping and decompression are the caller's responsibility (see
    # BlobManager._wrap_download_stream); the iterator only handles frame
    # parsing / raw passthrough.
    #
    # Framing is determined by the number of blob references in the request,
    # not by inspecting blob bytes (which is ambiguous — raw blob data whose
    # first 8 bytes decode to a small LE int is indistinguishable from a frame
    # header):
    #   - Single reference (expected_count == 1): the server returns the raw
    #     unframed stream — just the blob bytes, no frame headers.  The
    #     iterator reads the entire decompressed payload as one blob.
    #   - Multiple references (expected_count > 1): the server returns a
    #     framed stream with one frame per blob.  Each frame's header
    #     carries the blob's MIME type and (v3+) custom file name; the
    #     iterator yields one :class:`BlobRecord` per frame.
    #
    # The server decides framing based on the request ref count: single-ref
    # responses are raw (no frame headers), multi-ref responses are framed.

    def __init__(
        self,
        raw_stream,
        api_version: str = "2",
        expected_count: Optional[int] = None,
        crc_strip: bool = True,
    ):
        if expected_count is None:
            raise ValueError(
                "BlobDataIterator requires expected_count (the number of blob "
                "references in the request): 1 for single-blob raw download, "
                ">1 for multi-blob framed download."
            )
        self._raw_stream = raw_stream
        self._current_stream = None
        self._finished = False
        self._first = True
        self._api_version = api_version
        self._supports_custom_file_name = _str_version_ge(api_version, 3)
        # Framing is selected by the request ref count:
        #   expected_count == 1  → raw (server omits framing for single ref)
        #   expected_count  > 1  → framed (server frames multi-blob responses)
        self._expected_count = expected_count
        self._framed = expected_count > 1
        self._frames_yielded = 0
        # When ``crc_strip`` is True (default) the iterator wraps the raw
        # stream in CrcStrippedInputStream, preserving backward compat for
        # direct construction (legacy storage_api_v2 client, unit tests).
        # ``BlobManager._wrap_download_stream`` pre-strips CRC, so it passes
        # ``crc_strip=False`` to avoid double-stripping.
        self._crc_strip = crc_strip
        # Track whether the final footer of the last yielded frame has been
        # consumed so __next__ can drain it before stopping.
        self._final_footer_consumed = False

    def _ensure_stream(self):
        if self._current_stream is not None:
            return
        if self._crc_strip:
            # Default: strip per-block CRC32C trailers from the wire stream.
            # Callers that pre-strip (BlobManager._wrap_download_stream) pass
            # ``crc_strip=False`` to avoid double-stripping.
            self._current_stream = CrcStrippedInputStream(self._raw_stream)
        else:
            self._current_stream = self._raw_stream

        # Framing was already determined in __init__ from expected_count.

    def _consume_previous(self):
        """Consume the footer of the previous blob if any data remains unread."""
        if self._current_stream is None:
            return
        footer_len = read_le_long(self._current_stream)
        if footer_len is None:
            return
        if footer_len < 0:
            raise MaxStorageError(
                f"Corrupt blob stream: negative footer length ({footer_len})."
            )
        if footer_len > 0:
            read_exact(self._current_stream, footer_len)

    def _consume_final_footer(self):
        """Consume the final footer, raising on missing/truncated footer.

        In contract mode (``expected_count`` set) the server promises a
        footer after every frame, so a clean EOF here means the final
        footer is missing — a truncation error.  A partial footer-length
        read or short footer body is also surfaced via ``read_le_long`` /
        ``read_exact``.
        """
        if self._current_stream is None:
            return
        footer_len = read_le_long(self._current_stream)
        if footer_len is None:
            if self._expected_count is not None:
                raise MaxStorageError(
                    "Truncated blob stream: final footer missing after "
                    f"{self._frames_yielded} frame(s)."
                )
            return
        if footer_len < 0:
            raise MaxStorageError(
                f"Corrupt blob stream: negative footer length ({footer_len})."
            )
        if footer_len > 0:
            read_exact(self._current_stream, footer_len)

    def __iter__(self) -> "BlobDataIterator":
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration
        self._ensure_stream()
        # Stop after yielding the expected number of frames in contract-driven
        # mode so a truncated/extra frame never over-reads the stream.
        if (
            self._expected_count is not None
            and self._frames_yielded >= self._expected_count
        ):
            # Consume the final footer of the last yielded frame before
            # stopping, draining the previous blob's footer
            # before concluding iteration.  A missing/truncated
            # final footer is thus surfaced as an error rather than
            # silently accepted.
            if not self._final_footer_consumed:
                self._final_footer_consumed = True
                self._consume_final_footer()
            self._finished = True
            raise StopIteration
        if self._framed is True:
            record = self._next_framed()
        else:
            record = self._next_raw()
        if record is not None:
            self._frames_yielded += 1
        return record

    def _next_raw(self):
        if self._first:
            self._first = False
            # Read the entire decompressed payload as a single blob.
            # Use block reads so any stream type (CrcStrippedInputStream,
            # SimpleInputStream, etc.) works regardless of whether its
            # read() accepts a no-arg / -1 "read all" sentinel.
            chunks = []
            while True:
                chunk = self._current_stream.read(_RAW_READ_BLOCK)
                if not chunk:
                    break
                chunks.append(chunk)
            data = b"".join(chunks)
            self._finished = True
            if not data:
                raise StopIteration
            return BlobRecord(data, None, None)
        self._finished = True
        raise StopIteration

    def _unexpected_eof(self):
        """Raise on premature EOF in contract-driven (expected_count) mode.

        The server promised exactly ``expected_count`` frames, so an early
        EOF is a truncation error.
        """
        self._finished = True
        if self._expected_count is not None:
            raise MaxStorageError(
                "Truncated blob stream: expected "
                f"{self._expected_count} frame(s), got "
                f"{self._frames_yielded} before EOF."
            )
        raise StopIteration

    def _parse_frame_header(self):
        """Parse the next frame header. Returns (mime_type, data_len, custom_file_name)."""
        header_len = read_le_long(self._current_stream)
        if header_len is None:
            self._unexpected_eof()
        if header_len < 0:
            raise MaxStorageError(
                f"Corrupt blob stream: negative header length ({header_len})."
            )

        mime_type = custom_file_name = None
        header_bytes = read_exact(self._current_stream, header_len)
        if header_bytes:
            try:
                header = json.loads(header_bytes.decode("utf-8"))
                mime_type = header.get("ContentType") or None
                if self._supports_custom_file_name:
                    custom_file_name = header.get("CustomFileName") or None
            except (ValueError, UnicodeDecodeError):
                pass

        data_len = read_le_long(self._current_stream)
        if data_len is None:
            self._unexpected_eof()
        if data_len < 0:
            raise MaxStorageError(
                f"Corrupt blob stream: negative data length ({data_len})."
            )

        return mime_type, data_len, custom_file_name

    def _next_framed(self):
        if not self._first:
            self._consume_previous()
        self._first = False
        mime_type, data_len, custom_file_name = self._parse_frame_header()
        data = read_exact(self._current_stream, data_len)
        return BlobRecord(data, mime_type, custom_file_name)

    def _parse_next_frame_header(self):
        self._first = False
        return self._parse_frame_header()

    def read_data(self, size: int = -1) -> bytes:
        """Read up to *size* bytes of the current blob's data (for BlobStreamReader)."""
        return self._current_stream.read(size)

    def skip_remaining_data_and_footer(self, remaining_bytes: int) -> None:
        """Skip *remaining_bytes* of unread data and the trailing footer."""
        if remaining_bytes > 0:
            self._current_stream.read(remaining_bytes)
        self._consume_previous()


class BlobStreamReader:
    """File-like reader for streaming blob data from a :class:`BlobDataIterator`.

    Provides ``read(size)`` for incremental reads of the current blob,
    ``mime_type`` / ``custom_file_name`` properties, and a ``next()``
    method to advance to the next blob.  Avoids materializing entire blobs.

    If ``next()`` is called before the current blob is fully read, the reader
    **auto-drains** the remaining bytes.  Returns ``None`` from ``next()`` when
    no more blobs remain.
    """

    def __init__(self, iterator: "BlobDataIterator"):
        self._iterator = iterator
        self._mime_type = None
        self._custom_file_name = None
        self._data_remaining = 0
        self._exhausted = False
        self._finished = False
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        self._iterator._ensure_stream()

        if self._iterator._finished:
            self._finished = True
            return

        if self._iterator._framed is True:
            # Enforce expected_count in streaming mode: stop before parsing a
            # frame beyond the promised count (matching the materialized
            # __next__ path) so extra frames are never exposed.
            if (
                self._iterator._expected_count is not None
                and self._iterator._frames_yielded >= self._iterator._expected_count
            ):
                self._finished = True
                return
            try:
                (
                    mime_type,
                    data_len,
                    custom_file_name,
                ) = self._iterator._parse_next_frame_header()
                self._mime_type = mime_type
                self._custom_file_name = custom_file_name
                self._data_remaining = data_len
                self._iterator._frames_yielded += 1
            except StopIteration:
                self._finished = True
        else:
            self._mime_type = None
            self._custom_file_name = None
            self._data_remaining = -1  # unknown; read until EOF
            self._iterator._finished = True

        if self._data_remaining == 0 and not self._finished:
            self._exhausted = True

    @property
    def mime_type(self) -> Optional[str]:
        self._ensure_loaded()
        return self._mime_type

    @property
    def custom_file_name(self) -> Optional[str]:
        self._ensure_loaded()
        return self._custom_file_name

    def read(self, size: int = -1) -> bytes:
        self._ensure_loaded()
        if self._finished or self._exhausted:
            return b""
        if self._data_remaining == 0:
            self._exhausted = True
            return b""
        if self._data_remaining < 0:
            data = self._iterator.read_data(size)
            if not data:
                self._exhausted = True
            return data
        if size < 0 or size > self._data_remaining:
            size = self._data_remaining
        data = self._iterator.read_data(size)
        if not data:
            # Stream ended before the declared data length was satisfied —
            # a truncation error, not a clean exhaustion.
            self._exhausted = True
            raise MaxStorageError(
                f"Truncated blob stream: expected {self._data_remaining}"
                " more byte(s) of data, got EOF (truncated stream)."
            )
        self._data_remaining -= len(data)
        if self._data_remaining <= 0:
            self._exhausted = True
        return data

    def next(self) -> Optional["BlobStreamReader"]:
        """Advance to the next blob in-place. Returns ``self`` or ``None``."""
        self._ensure_loaded()
        if not self._finished:
            # Always consume remaining data + footer, whether or not the
            # current blob was fully read.  When _exhausted is True the
            # data is gone but the footer still needs to be consumed;
            # skip_remaining_data_and_footer(0) handles that via
            # _consume_previous().
            self._iterator.skip_remaining_data_and_footer(self._data_remaining)
        if self._finished:
            return None

        self._mime_type = None
        self._custom_file_name = None
        self._data_remaining = 0
        self._exhausted = False
        self._loaded = False

        self._ensure_loaded()
        if self._finished:
            return None
        return self
