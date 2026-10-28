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

"""Arrow IPC stream reader for :mod:`odps.maxstorage`.

Wraps the HTTP response's (optionally decompressed) Arrow IPC stream via
``pa.ipc.open_stream``.  Decompression uses
``odps.tunnel.io.stream.get_decompress_stream`` with the reader's
``compress_option`` (default ``None`` = uncompressed).  Reuses struct-timestamp
conversion logic from the tunnel Arrow reader.
"""

import logging
from queue import Empty, Queue
from threading import Lock, Thread

from ...tunnel.io.stream import get_decompress_stream

try:
    import pyarrow as pa
except ImportError:
    pa = None

logger = logging.getLogger(__name__)


def _is_timestamp_struct_type(arrow_type):
    if not isinstance(arrow_type, pa.StructType):
        return False
    if arrow_type.num_fields != 2:
        return False
    f0, f1 = arrow_type[0], arrow_type[1]
    return (
        f0.name == "sec"
        and f0.type == pa.int64()
        and f1.name == "nano"
        and f1.type == pa.int32()
    )


def _convert_struct_timestamps(batch):
    """Convert ``{sec, nano}`` struct columns to ``pa.timestamp("ns")``."""
    changed = False
    new_arrays = []
    new_fields = []
    for i in range(batch.num_columns):
        col = batch.column(i)
        field = batch.schema.field(i)
        if _is_timestamp_struct_type(field.type):
            sec = col.field("sec")
            nano = col.field("nano")
            combined = pa.compute.add(
                pa.compute.multiply(sec, pa.scalar(1_000_000_000, type=pa.int64())),
                pa.compute.cast(nano, pa.int64()),
            )
            new_arrays.append(combined.cast(pa.timestamp("ns")))
            new_fields.append(pa.field(field.name, pa.timestamp("ns")))
            changed = True
        else:
            new_arrays.append(col)
            new_fields.append(field)
    if changed:
        return pa.RecordBatch.from_arrays(new_arrays, schema=pa.schema(new_fields))
    return batch


class _ArrowStreamReaderBase:
    """Shared plumbing for the Arrow IPC stream readers.

    Owns the pyarrow availability check, the (optionally decompressed)
    response stream endpoint, the end-of-stream :meth:`read` helper and
    closing of the response.  Subclasses open their own IPC reader in
    :meth:`_ensure_opened` and release it in :meth:`_close_reader`.
    """

    def __init__(self, raw_response, compress_option=None):
        if pa is None:
            raise ValueError(
                "To use %s you need to install pyarrow" % type(self).__name__
            )

        self._raw_response = raw_response
        self._compress_option = compress_option
        self._opened = False

    def _get_stream(self):
        if self._compress_option is not None:
            return get_decompress_stream(self._raw_response, self._compress_option)
        return self._raw_response.raw

    def _ensure_opened(self):
        """Mark the stream as opened; subclasses open their IPC reader here."""
        self._opened = True

    def __iter__(self):
        return self

    def read(self):
        """Read the next batch, or ``None`` at end of stream."""
        try:
            return next(self)
        except StopIteration:
            return None

    @property
    def metrics_metadata(self):
        """Metrics metadata of extended streams; ``None`` for ordinary ones."""
        return None

    def _close_reader(self):
        """Release the currently open IPC reader, if any."""

    def close(self):
        self._close_reader()
        if self._raw_response is not None:
            if hasattr(self._raw_response, "close"):
                self._raw_response.close()


class ArrowStreamReader(_ArrowStreamReaderBase):
    """Arrow IPC stream reader yielding :class:`pyarrow.RecordBatch`.

    Wraps an optionally decompressed response stream via ``pa.ipc.open_stream``.
    Iterator protocol yields ``RecordBatch``.
    """

    def __init__(self, raw_response, compress_option=None):
        super().__init__(raw_response, compress_option)
        self._arrow_stream = None

    def _ensure_opened(self):
        if self._opened:
            return
        super()._ensure_opened()
        self._arrow_stream = pa.ipc.open_stream(self._get_stream())

    def __next__(self):
        self._ensure_opened()
        return _convert_struct_timestamps(self._arrow_stream.read_next_batch())

    @property
    def arrow_schema(self):
        self._ensure_opened()
        return self._arrow_stream.schema

    def _close_reader(self):
        if self._arrow_stream is not None and hasattr(self._arrow_stream, "close"):
            self._arrow_stream.close()


class ExtendedArrowStreamReader(_ArrowStreamReaderBase):
    """Reads tagged Arrow substreams, keeping metrics frames separate from rows.

    The negotiated extended IPC format concatenates complete Arrow IPC
    streams, each preceded by a one-byte tag:

    * tag ``1`` — a data substream whose batches are table rows;
    * tag ``2`` — a metrics substream; its schema custom metadata is
      collected into :attr:`metrics_metadata` and its batches are never
      surfaced as table rows.

    Duration metrics (``server_processing_duration_us`` /
    ``rate_limit_duration_us``) reported by multiple metrics substreams are
    summed; all other keys keep the last observed value.  The iterator
    protocol yields :class:`pyarrow.RecordBatch` from data substreams only.
    """

    METRICS_STREAM_TAG = 2
    _SUMMING_METRIC_KEYS = ("server_processing_duration_us", "rate_limit_duration_us")

    def __init__(self, raw_response, compress_option=None):
        super().__init__(raw_response, compress_option)
        self._stream = None
        self._sub_reader = None
        self._metrics_metadata = {}
        self._metrics_lock = Lock()
        self._arrow_schema_cache = None

    def _ensure_opened(self):
        if self._opened:
            return
        super()._ensure_opened()
        self._stream = self._get_stream()
        # Resolve the first data substream and its schema eagerly: the schema
        # must be readable without touching the stream, otherwise a consumer
        # peeking at ``arrow_schema`` could open the same substream as a
        # background producer and corrupt the IPC stream.
        self._sub_reader = self._open_next_data_substream()
        if self._sub_reader is not None:
            self._arrow_schema_cache = self._sub_reader.schema

    def _read_tag(self):
        """Read the next substream tag byte, or ``None`` at end of input."""
        tag_byte = self._stream.read(1)
        if not tag_byte:
            return None
        return tag_byte[0]

    def _consume_metrics(self, reader):
        """Exhaust a metrics substream and merge its schema metadata."""
        schema = reader.schema
        for _ in reader:
            # Drain the stream so the underlying input is positioned at
            # the next tag byte.  Metrics batches are never table rows.
            pass
        if schema is None:
            return
        for key, value in (schema.metadata or {}).items():
            key = key.decode() if isinstance(key, bytes) else key
            value = value.decode() if isinstance(value, bytes) else value
            with self._metrics_lock:
                if key in self._SUMMING_METRIC_KEYS and key in self._metrics_metadata:
                    try:
                        self._metrics_metadata[key] = str(
                            int(self._metrics_metadata[key]) + int(value)
                        )
                        continue
                    except (TypeError, ValueError):
                        logger.warning(
                            "Non-numeric duration metric %s=%r; "
                            "keeping the raw value",
                            key,
                            value,
                        )
                self._metrics_metadata[key] = value

    def _open_next_data_substream(self):
        """Advance past any metrics substreams; return the next data reader.

        Returns ``None`` once the tagged input is exhausted.
        """
        while True:
            tag = self._read_tag()
            if tag is None:
                return None
            reader = pa.ipc.open_stream(self._stream)
            if tag == self.METRICS_STREAM_TAG:
                self._consume_metrics(reader)
                continue
            return reader

    def __next__(self):
        self._ensure_opened()
        while True:
            if self._sub_reader is None:
                self._sub_reader = self._open_next_data_substream()
                if self._sub_reader is None:
                    raise StopIteration
            try:
                batch = self._sub_reader.read_next_batch()
            except StopIteration:
                self._sub_reader = None
                continue
            return _convert_struct_timestamps(batch)

    @property
    def metrics_metadata(self):
        """Merged metrics metadata collected from metrics substreams."""
        with self._metrics_lock:
            return dict(self._metrics_metadata)

    @property
    def arrow_schema(self):
        """The schema of the first data substream."""
        self._ensure_opened()
        return self._arrow_schema_cache

    def _close_reader(self):
        if self._sub_reader is not None and hasattr(self._sub_reader, "close"):
            self._sub_reader.close()


_ASYNC_DONE = object()
"""Sentinel placed on the async queue to signal end-of-stream."""


class AsyncArrowStreamReader:
    """Async Arrow IPC stream reader yielding :class:`pyarrow.RecordBatch`.

    A background thread reads batches from a synchronous stream reader and
    places them on a bounded queue.  The consumer iterator takes from the
    queue, overlapping network I/O with batch processing.

    The queue is bounded (default 2) to provide back-pressure — the
    producer blocks when the queue is full, preventing unbounded memory
    growth when the consumer is slow.
    """

    def __init__(self, sync_reader, queue_size=2):
        self._sync_reader = sync_reader
        self._queue = Queue(maxsize=max(1, queue_size))
        self._thread = None
        self._started = False
        self._finished = False
        self._closed = False

        # Open the stream synchronously so schema is available before the
        # producer thread starts — avoids a schema/batch race on the queue.
        self._sync_reader._ensure_opened()

    @property
    def arrow_schema(self):
        return self._sync_reader.arrow_schema

    @property
    def metrics_metadata(self):
        return self._sync_reader.metrics_metadata

    def _start(self):
        if self._started:
            return
        self._started = True
        self._thread = Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self):
        """Background producer: read batches from the IPC stream into the queue."""
        try:
            while True:
                batch = next(self._sync_reader)
                self._queue.put(batch)
            # the sync reader raises StopIteration at EOS — handled below
        except StopIteration:
            self._queue.put(_ASYNC_DONE)
        except Exception as exc:  # noqa: BLE001 - propagate to consumer
            self._queue.put(exc)

    def __iter__(self):
        self._start()
        return self

    def __next__(self):
        # The producer signals end-of-stream once and then exits, so remember
        # it: a further read must not block on an empty queue.
        if self._finished or self._closed:
            raise StopIteration
        self._start()
        item = self._queue.get()
        if item is _ASYNC_DONE:
            self._finished = True
            raise StopIteration
        if isinstance(item, Exception):
            self._finished = True
            raise item
        return item

    def read(self):
        """Read the next batch, or ``None`` at end of stream."""
        try:
            return next(self)
        except StopIteration:
            return None

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._sync_reader.close()
        # Drain the queue to unblock the producer if it's blocked on put
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
