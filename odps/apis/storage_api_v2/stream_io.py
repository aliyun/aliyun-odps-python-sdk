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

"""Stream I/O, blob I/O, and Arrow I/O classes for the Storage API V2."""

import collections
import hashlib
import json
import logging
import struct
from io import BytesIO, IOBase

from requests import codes

try:
    import pyarrow as pa
except ImportError:
    pa = None

from ... import errors, options, utils
from ...tunnel.io import RequestsIO
from ...tunnel.io.stream import CompressOption, get_compress_stream
from .models import (
    Compression,
    Status,
    WriteBlobResponse,
    WriteStreamResponse,
    _update_request_id,
)

ROUTE_TOKEN_HEADER = "x-odps-max-storage-route-token"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stream I/O
# ---------------------------------------------------------------------------


class StreamReader(IOBase):
    """Stream reader for reading Arrow IPC formatted data from the server.

    Methods
    -------
    read(nbytes=None)
        Read data from the stream. Returns bytes.
    get_status()
        Check reading status (RUNNING or OK).
    get_request_id()
        Get request ID after completion.
    close()
        Close the stream.
    readable()
        Check if the stream can still be read.
    """

    def __init__(self, download):
        self._stopped = False
        raw_reader = download()

        self._raw_reader = raw_reader
        self._chunk_size = 65536
        self._buffers = collections.deque()
        self._eof = False

    def readable(self):
        return not self._stopped

    def _read_chunk(self):
        buf = self._raw_reader.raw.read(self._chunk_size)
        return buf

    def _fill_next_buffer(self):
        if self._eof:
            return
        data = self._read_chunk()
        if not data:
            self._eof = True
            return

        self._buffers.append(BytesIO(data))

    def read(self, nbytes=None):
        if self._stopped:
            return b""

        total_size = 0
        bufs = []
        while nbytes is None or total_size < nbytes:
            if not self._buffers:
                self._fill_next_buffer()
                if not self._buffers:
                    break

            to_read = nbytes - total_size if nbytes is not None else None
            buf = self._buffers[0].read(to_read)

            if not buf:
                self._buffers.popleft()
            else:
                bufs.append(buf)
                total_size += len(buf)

        return b"".join(bufs)

    def get_status(self):
        if not self._stopped:
            return Status.RUNNING
        else:
            return Status.OK

    def get_request_id(self):
        if not self._stopped:
            logger.error("The reader is not closed yet, please wait")
            return None

        if (
            self._raw_reader is not None
            and "x-odps-request-id" in self._raw_reader.headers
        ):
            return self._raw_reader.headers["x-odps-request-id"]
        else:
            return None

    def close(self):
        self._stopped = True


class StreamWriter(IOBase):
    """Stream writer for uploading binary data to the server.

    Expects Arrow IPC formatted binary data. Use :class:`ArrowWriter` to
    convert Arrow record batches to this format.

    Methods
    -------
    write(data)
        Write binary data to the stream. Returns True on success,
        False if writer is closed or error occurred.
    finish()
        Finish writing and close the upload. Returns
        (commit_message, success_bool). In Exactly-Once mode,
        the response body is also parsed into a WriteStreamResponse
        accessible via get_write_stream_response().
    get_status()
        Check writer status (RUNNING during write, OK after finish).
    get_request_id()
        Get request ID after finish() for debugging.
    get_route_token()
        Get route token from write response headers for session affinity.
        Call after finish(). Returns None if not available.
    get_write_stream_response()
        Get the parsed WriteStreamResponse after finish(). Returns None
        if finish() has not been called or if the response did not
        contain exactly-once data.
    writable()
        Check if the writer can still accept data.
    """

    def __init__(self, upload, on_route_token=None):
        self._req_io = RequestsIO(upload, chunk_size=options.chunk_size)
        self._req_io.start()
        self._res = None
        self._stopped = False
        self._on_route_token = on_route_token
        self._write_stream_response = None

    def writable(self):
        return not self._stopped

    def write(self, data):
        if self._stopped:
            return False

        self._req_io.write(data)
        return True

    def finish(self):
        self._stopped = True
        self._res = self._req_io.finish()

        if self._res is not None and self._res.status_code == codes["ok"]:
            route_token = self._res.headers.get(ROUTE_TOKEN_HEADER)
            if route_token and self._on_route_token:
                self._on_route_token(route_token)
            resp_json = self._res.json()
            commit_message = resp_json.get("CommitMessage")
            # Parse WriteStreamResponse for Exactly-Once mode
            if resp_json.get("ExactlyOnceRowOffset") is not None:
                self._write_stream_response = WriteStreamResponse()
                self._write_stream_response.parse(
                    self._res, obj=self._write_stream_response
                )
            return commit_message, True
        else:
            return None, False

    def get_write_stream_response(self):
        """Get the parsed WriteStreamResponse after finish().

        In Exactly-Once mode, the server response contains
        ExactlyOnceRowOffset which tracks the committed row position.
        This method returns that parsed response, or None if not
        available.
        """
        return self._write_stream_response

    def get_status(self):
        if not self._stopped:
            return Status.RUNNING
        else:
            return Status.OK

    def get_request_id(self):
        if not self._stopped:
            logger.error("The writer is not closed yet, please close first")
            return None

        if self._res is not None and "x-odps-request-id" in self._res.headers:
            return self._res.headers["x-odps-request-id"]
        else:
            return None

    def get_route_token(self):
        if not self._stopped:
            logger.error("The writer is not closed yet, please close first")
            return None

        if self._res is not None:
            return self._res.headers.get(ROUTE_TOKEN_HEADER)
        return None


class BlobStreamWriter(IOBase):
    """Stream writer for single blob upload with MD5 checksum verification.

    Automatically computes MD5 checksum and verifies it against the server
    response. Data can be compressed during upload using the Compression enum.

    Methods
    -------
    write(data)
        Write binary data chunks to the blob stream. Accepts bytes or
        string (converted to bytes). Returns True on success.
    finish()
        Finish writing and get server response with MD5 verification.
        Returns :class:`WriteBlobResponse`. Raises ChecksumError if
        MD5 mismatch detected.
    get_status()
        Check writer status (RUNNING or OK).
    get_request_id()
        Get request ID after finish.
    writable()
        Check if writer can accept more data.
    """

    def __init__(self, upload, compression=Compression.UNCOMPRESSED):
        self._req_io = RequestsIO(upload, chunk_size=options.chunk_size)
        self._req_io.start()
        self._res = None
        self._stopped = False
        self._md5_digest = hashlib.md5()

        # Map Compression enum to CompressOption.CompressAlgorithm
        compress_algo = None
        if compression == Compression.ZSTD:
            compress_algo = CompressOption.CompressAlgorithm.ODPS_ZSTD
        elif compression == Compression.LZ4_FRAME:
            compress_algo = CompressOption.CompressAlgorithm.ODPS_LZ4
        elif compression == Compression.UNCOMPRESSED:
            compress_algo = CompressOption.CompressAlgorithm.ODPS_RAW
        else:
            raise ValueError(
                f"Unsupported compression type: {compression}. "
                f"Supported values are Compression.ZSTD, Compression.LZ4_FRAME, "
                f"or Compression.UNCOMPRESSED."
            )

        compress_option = CompressOption(compress_algo=compress_algo)
        self._compressor = get_compress_stream(self._req_io, compress_option)

    def writable(self):
        return not self._stopped

    def write(self, data):
        if self._stopped:
            return False

        data = utils.to_binary(data)
        self._md5_digest.update(data)
        self._compressor.write(data)
        return True

    def finish(self):
        """Finish writing and verify MD5 checksum against server response.

        Returns:
            WriteBlobResponse on success, None on failure.
        """
        self._stopped = True
        self._compressor.flush()
        self._res = self._req_io.finish()

        if self._res is not None and self._res.status_code == codes["ok"]:
            response = WriteBlobResponse()
            response.parse(self._res, obj=response)
            _update_request_id(response, self._res)

            resp_json = self._res.json()
            md5_value = resp_json.get("MD5Value")
            if md5_value and md5_value != self._md5_digest.hexdigest():
                raise errors.ChecksumError(
                    f"MD5 value mismatch, expected: {md5_value}, "
                    f"actual: {self._md5_digest.hexdigest()}"
                )
            return response
        else:
            return None

    def get_status(self):
        if not self._stopped:
            return Status.RUNNING
        else:
            return Status.OK

    def get_request_id(self):
        if not self._stopped:
            logger.error("The writer is not closed yet, please close first")
            return None

        if self._res is not None and "x-odps-request-id" in self._res.headers:
            return self._res.headers["x-odps-request-id"]
        else:
            return None


# ---------------------------------------------------------------------------
# Blob I/O
# ---------------------------------------------------------------------------


def _read_le_long(stream):
    """Read an 8-byte little-endian signed integer from a stream."""
    data = stream.read(8)
    if len(data) < 8:
        return None
    return struct.unpack("<q", data)[0]


def _read_exact(stream, n):
    """Read exactly n bytes from a stream."""
    bio = BytesIO()
    while bio.tell() < n:
        chunk = stream.read(n - bio.tell())
        if not chunk:
            break
        bio.write(chunk)
    return bio.getvalue()


class _CRCStrippingStream:
    """File-like wrapper that strips CRC32C checksums from the underlying stream.

    Wire format: [4096 bytes data][4 bytes CRC32C] repeated, with a
    potentially shorter final block [N bytes data (1<=N<=4096)][4 bytes CRC32C].

    Each full block is 4100 bytes total (4096 data + 4 CRC).
    The last block is between 5 and 4099 bytes (1~4095 data + 4 CRC).

    This wrapper reads blocks from the raw stream on demand, strips the
    trailing CRC bytes, and serves the clean data through the standard
    ``read()`` interface.  It avoids buffering the entire stream in memory.
    """

    _CRC_BLOCK_SIZE = 4096
    _CRC_SIZE = 4
    _FULL_BLOCK_TOTAL = _CRC_BLOCK_SIZE + _CRC_SIZE  # 4100

    def __init__(self, raw_stream):
        self._raw_stream = raw_stream
        self._buffer = b""  # clean data ready to serve
        self._finished = False  # all raw blocks consumed

    def _fill(self):
        """Read one block from the raw stream and strip its CRC."""
        if self._finished:
            return
        block = _read_exact(self._raw_stream, self._FULL_BLOCK_TOTAL)
        if not block:
            self._finished = True
            return
        if len(block) == self._FULL_BLOCK_TOTAL:
            # Full block: first 4096 bytes are data, last 4 are CRC
            self._buffer += block[: self._CRC_BLOCK_SIZE]
        else:
            # Tail block: first (len-4) bytes are data, last 4 are CRC
            if len(block) > self._CRC_SIZE:
                self._buffer += block[: -self._CRC_SIZE]
            self._finished = True

    def peek(self, size):
        """Peek at up to *size* bytes of CRC-stripped data without consuming.

        Fills from the raw stream as needed, but does not advance the
        read position.  Returns fewer than *size* bytes if the stream
        ends before that many clean bytes are available.

        Parameters
        ----------
        size : int
            Maximum number of bytes to peek at.  Must be >= 0.

        Returns
        -------
        bytes
            Up to *size* bytes of data from the current position.
        """
        if size <= 0:
            return b""
        while len(self._buffer) < size and not self._finished:
            self._fill()
        return self._buffer[:size]

    def read(self, size=-1):
        """Read up to *size* bytes of CRC-stripped data.

        Parameters
        ----------
        size : int
            Maximum bytes to read. ``-1`` reads all remaining data.
        """
        if size == 0:
            return b""
        # Keep filling until we have enough data or the stream is exhausted
        while True:
            if size < 0:
                # Read-all mode: consume the entire raw stream
                if not self._finished:
                    self._fill()
                    continue
                break
            else:
                if len(self._buffer) >= size or self._finished:
                    break
                self._fill()
        if size < 0:
            result = self._buffer
            self._buffer = b""
            return result
        result = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return result


class BlobDataIterator:
    """Iterator that parses the framed blob download protocol.

    The download stream is processed through the following layers:
      1. Decompression (if the response is compressed, handled upstream
         via ``get_decompress_stream`` before passing to this iterator)
      2. CRC32C stripping -- strips per-block checksums
      3. This iterator parses [HeaderLen][Header][DataLen][Data][FooterLen][Footer] frames

    Yields (data_bytes, mime_type) tuples for each blob, where:

    - data_bytes : bytes
        The raw binary data of the blob.
    - mime_type : str or None
        MIME type metadata if it was provided during upload,
        otherwise None.

    For single blob downloads, the server may omit framing and send raw
    decompressed data directly. The iterator automatically detects this
    case and returns the entire payload as one blob.
    """

    def __init__(self, raw_stream):
        self._raw_stream = raw_stream
        self._current_stream = None
        self._finished = False
        self._first = True
        self._framed = None  # detected on first read

    @staticmethod
    def _is_framed(data):
        """Heuristic: check if the decompressed data starts with a protocol-frame
        header-length prefix (a small LE int64).  Blob data itself rarely starts
        with 8 bytes that decode to a value in [0, 1024)."""
        if len(data) < 8:
            return False
        header_len = struct.unpack("<q", data[:8])[0]
        return 0 <= header_len < 1024

    def _ensure_stream(self):
        """Lazily wrap the raw stream in a CRC-stripping stream."""
        if self._current_stream is not None:
            return
        self._current_stream = _CRCStrippingStream(self._raw_stream)
        # Detect whether the server sent protocol-framed data or raw blob data.
        # For a single blob the server may omit the framing header; for
        # multiple blobs it always includes [HeaderLen][Header]... frames.
        # We only need the first 8 bytes for the heuristic; peek does not
        # consume them.
        peek = self._current_stream.peek(8)
        self._framed = self._is_framed(peek)

    def _consume_previous(self):
        """Consume the footer of the previous blob if any data remains unread."""
        if self._current_stream is None:
            return
        # If there was a previous blob, skip any remaining data and read footer
        footer_len = _read_le_long(self._current_stream)
        if footer_len is not None and footer_len > 0:
            _read_exact(self._current_stream, footer_len)

    def __iter__(self):
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration

        self._ensure_stream()

        if self._framed is True:
            return self._next_framed()
        else:
            return self._next_raw()

    def _next_raw(self):
        """Yield the entire decompressed payload as a single blob.

        Used when the server omits protocol framing (e.g. single-blob responses).
        """
        if self._first:
            self._first = False
            # Read the entire remaining stream as a single blob
            data = self._current_stream.read()
            self._finished = True
            if not data:
                raise StopIteration
            return data, None
        self._finished = True
        raise StopIteration

    def _next_framed(self):
        """Parse one protocol-framed blob from the stream.

        Wire format per blob:
            [8-byte LE HeaderLen][Header JSON][8-byte LE DataLen][Data]
            [8-byte LE FooterLen][Footer]
        """
        if not self._first:
            self._consume_previous()
        self._first = False

        # Read header
        header_len = _read_le_long(self._current_stream)
        if header_len is None:
            self._finished = True
            raise StopIteration

        header_bytes = _read_exact(self._current_stream, header_len)
        mime_type = None
        if header_bytes:
            try:
                header = json.loads(header_bytes.decode("utf-8"))
                mime_type = header.get("ContentType")
            except (ValueError, UnicodeDecodeError):
                pass

        # Read data length
        data_len = _read_le_long(self._current_stream)
        if data_len is None:
            self._finished = True
            raise StopIteration

        # Read data
        data = _read_exact(self._current_stream, data_len)
        return data, mime_type

    def _parse_next_frame_header(self):
        """Parse the next frame header, leaving the stream positioned at the data.

        Returns (mime_type, data_len) or raises StopIteration if no more frames.
        Advances past header + data_len prefix, so the stream is ready for
        data reads.

        The caller must ensure the previous frame's footer has already been
        consumed before calling this method.
        """
        self._first = False

        # Read header
        header_len = _read_le_long(self._current_stream)
        if header_len is None:
            self._finished = True
            raise StopIteration

        header_bytes = _read_exact(self._current_stream, header_len)
        mime_type = None
        if header_bytes:
            try:
                header = json.loads(header_bytes.decode("utf-8"))
                mime_type = header.get("ContentType")
            except (ValueError, UnicodeDecodeError):
                pass

        # Read data length
        data_len = _read_le_long(self._current_stream)
        if data_len is None:
            self._finished = True
            raise StopIteration

        return mime_type, data_len

    def read_data(self, size=-1):
        """Read up to *size* bytes of the current blob's data from the stream.

        This is used by BlobStreamReader for chunked reads. The stream
        position must be within the current blob's data region.

        Parameters
        ----------
        size : int
            Maximum bytes to read. ``-1`` reads all remaining data of
            the current blob.
        """
        return self._current_stream.read(size)

    def skip_remaining_data_and_footer(self, remaining_bytes):
        """Skip *remaining_bytes* of unread data and the trailing footer.

        Parameters
        ----------
        remaining_bytes : int
            Number of data bytes still unread in the current frame.
        """
        if remaining_bytes > 0:
            self._current_stream.read(remaining_bytes)
        # Now consume the footer
        self._consume_previous()


class BlobStreamReader:
    """File-like reader for streaming blob data from a :class:`BlobDataIterator`.

    Provides ``read(size)`` for incremental reads of the current blob,
    a ``mime_type`` property, and a ``next()`` method to advance to the
    next blob.

    Unlike iterating over :class:`BlobDataIterator` which materializes
    each blob entirely in memory, this reader reads data from the
    underlying stream in chunks, avoiding buffering the entire blob.

    Calling ``next()`` before the current blob is fully exhausted raises
    an :class:`IOError`. When all blobs have been read, ``next()``
    returns ``None``.

    Parameters
    ----------
    iterator : BlobDataIterator
        The underlying blob data iterator.

    Examples
    --------
    >>> blob_reader = client.read_blobs(blob_references=refs, stream=True)
    >>> while blob_reader is not None:
    ...     print(f"MIME: {blob_reader.mime_type}")
    ...     chunk = blob_reader.read(4096)
    ...     while chunk:
    ...         process(chunk)
    ...         chunk = blob_reader.read(4096)
    ...     blob_reader = blob_reader.next()
    """

    def __init__(self, iterator):
        self._iterator = iterator
        self._mime_type = None
        self._data_remaining = 0  # bytes still unread in current blob
        self._exhausted = False  # current blob fully read
        self._finished = False  # no more blobs available
        self._loaded = False

    def _ensure_loaded(self):
        """Parse the next blob header and prepare for chunked reads."""
        if self._loaded:
            return
        self._loaded = True
        self._iterator._ensure_stream()

        if self._iterator._finished:
            self._finished = True
            return

        if self._iterator._framed is True:
            try:
                mime_type, data_len = self._iterator._parse_next_frame_header()
                self._mime_type = mime_type
                self._data_remaining = data_len
            except StopIteration:
                self._finished = True
        else:
            # Raw (unframed) mode: single blob, everything remaining is data.
            # Since the CRC-stripping stream is not seekable, we use a
            # sentinel value to indicate "read until the stream is exhausted".
            self._mime_type = None
            self._data_remaining = -1  # unknown length; read until EOF
            self._iterator._finished = True  # raw mode has only one blob

        if self._data_remaining == 0 and not self._finished:
            self._exhausted = True

    @property
    def mime_type(self):
        """str or None: MIME type of the current blob."""
        self._ensure_loaded()
        return self._mime_type

    def read(self, size=-1):
        """Read up to *size* bytes from the current blob.

        Parameters
        ----------
        size : int, optional
            Maximum number of bytes to read. ``-1`` (default) reads
            all remaining bytes of the current blob.

        Returns
        -------
        bytes
            Data read. Empty bytes ``b""`` when the current blob is
            exhausted.
        """
        self._ensure_loaded()
        if self._finished or self._exhausted:
            return b""
        if self._data_remaining == 0:
            self._exhausted = True
            return b""

        if self._data_remaining < 0:
            # Raw (unframed) mode with unknown length: read until EOF
            data = self._iterator.read_data(size)
            if not data:
                self._exhausted = True
            return data

        if size < 0 or size > self._data_remaining:
            size = self._data_remaining

        data = self._iterator.read_data(size)
        self._data_remaining -= len(data)
        if self._data_remaining <= 0:
            self._exhausted = True
        return data

    def next(self):
        """Advance to the next blob in-place and return ``self``.

        Raises ``IOError`` if the current blob has not been fully read.
        Returns ``None`` when there are no more blobs.

        After a successful call, ``self`` is updated to reference the next
        blob's data and ``mime_type``. Do not retain references to the
        reader before calling ``next()`` — the same object is mutated.

        Returns
        -------
        BlobStreamReader or None
        """
        self._ensure_loaded()
        if not self._exhausted and not self._finished:
            raise IOError(
                "Cannot advance to next blob: current blob has not been fully read"
            )
        if self._finished:
            return None

        # Skip any unread data and the footer of the current blob
        self._iterator.skip_remaining_data_and_footer(self._data_remaining)

        # Reset state for the next blob
        self._mime_type = None
        self._data_remaining = 0
        self._exhausted = False
        self._loaded = False

        self._ensure_loaded()
        if self._finished:
            return None
        return self


# ---------------------------------------------------------------------------
# Arrow I/O
# ---------------------------------------------------------------------------


class ArrowReader:
    """Arrow batch reader that wraps a :class:`StreamReader`.

    Methods
    -------
    read()
        Read the next Arrow record batch from the stream.
        Returns None when all batches have been read.
    """

    def __init__(self, stream_reader):
        if pa is None:
            raise ValueError("To use arrow reader you need to install pyarrow")

        self._reader = stream_reader
        self._arrow_stream = None

    def _read_next_batch(self):
        if self._arrow_stream is None:
            self._arrow_stream = pa.ipc.open_stream(self._reader)

        try:
            batch = self._arrow_stream.read_next_batch()
            return batch
        except StopIteration:
            return None

    def read(self):
        if not self._reader.readable():
            logger.error("Reader has been closed")
            return None

        batch = self._read_next_batch()
        if batch is None:
            self._reader.close()

        return batch

    def get_status(self):
        return self._reader.get_status()

    def get_request_id(self):
        return self._reader.get_request_id()


class ArrowWriter:
    """Arrow batch writer that wraps a :class:`StreamWriter`.

    Converts Arrow record batches to Arrow IPC format and writes them
    to the underlying stream.

    Methods
    -------
    write(record_batch)
        Write an Arrow RecordBatch to the stream. Returns True on
        success, False if writer is closed or error occurred.
    finish()
        Finish writing and close the upload. Returns
        (commit_message, success_bool).
    """

    def __init__(self, stream_writer, compression):
        self._arrow_writer = None
        self._compression = compression
        self._sink = stream_writer

    def write(self, record_batch):
        if not self._sink.writable():
            logger.error("Writer has been closed")
            return False

        if self._arrow_writer is None:
            self._arrow_writer = pa.ipc.new_stream(
                self._sink,
                record_batch.schema,
                options=pa.ipc.IpcWriteOptions(
                    compression=self._compression.to_compression_name()
                ),
            )

        self._arrow_writer.write_batch(record_batch)

        if not self._sink.writable():
            logger.error("Writer has been closed as exception occurred")
            return False

        return True

    def finish(self):
        if self._arrow_writer:
            self._arrow_writer.close()
        return self._sink.finish()

    def get_status(self):
        return self._sink.get_status()

    def get_request_id(self):
        return self._sink.get_request_id()

    def get_write_stream_response(self):
        """Get the parsed WriteStreamResponse after finish().

        In Exactly-Once mode, the server response contains
        ExactlyOnceRowOffset which tracks the committed row position.
        Delegates to the underlying StreamWriter.
        """
        return self._sink.get_write_stream_response()
