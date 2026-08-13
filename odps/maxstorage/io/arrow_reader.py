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
from threading import Thread

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


class ArrowStreamReader:
    """Arrow IPC stream reader yielding :class:`pyarrow.RecordBatch`.

    Wraps an optionally decompressed response stream via ``pa.ipc.open_stream``.
    Iterator protocol yields ``RecordBatch``.
    """

    def __init__(self, raw_response, compress_option=None):
        if pa is None:
            raise ValueError("To use ArrowStreamReader you need to install pyarrow")

        self._raw_response = raw_response
        self._compress_option = compress_option
        self._arrow_stream = None
        self._opened = False

    def _get_stream(self):
        if self._compress_option is not None:
            return get_decompress_stream(self._raw_response, self._compress_option)
        return self._raw_response.raw

    def _ensure_opened(self):
        if self._opened:
            return
        self._opened = True
        stream = self._get_stream()
        self._arrow_stream = pa.ipc.open_stream(stream)

    def __iter__(self):
        return self

    def __next__(self):
        self._ensure_opened()
        batch = self._arrow_stream.read_next_batch()
        batch = _convert_struct_timestamps(batch)
        return batch

    def read(self):
        """Read the next batch, or ``None`` at end of stream."""
        try:
            return next(self)
        except StopIteration:
            return None

    @property
    def arrow_schema(self):
        self._ensure_opened()
        return self._arrow_stream.schema

    def close(self):
        if self._arrow_stream is not None and hasattr(self._arrow_stream, "close"):
            self._arrow_stream.close()
        if self._raw_response is not None:
            if hasattr(self._raw_response, "close"):
                self._raw_response.close()


_ASYNC_DONE = object()
"""Sentinel placed on the async queue to signal end-of-stream."""


class AsyncArrowStreamReader:
    """Async Arrow IPC stream reader yielding :class:`pyarrow.RecordBatch`.

    A background thread reads batches from the IPC stream and places them
    on a bounded queue.  The consumer iterator takes from the queue,
    overlapping network I/O with batch processing.

    The queue is bounded (default 2) to provide back-pressure — the
    producer blocks when the queue is full, preventing unbounded memory
    growth when the consumer is slow.
    """

    def __init__(self, raw_response, compress_option=None, queue_size=2):
        if pa is None:
            raise ValueError(
                "To use AsyncArrowStreamReader you need to install pyarrow"
            )

        self._sync_reader = ArrowStreamReader(raw_response, compress_option)
        self._queue = Queue(maxsize=max(1, queue_size))
        self._thread = None
        self._started = False
        self._closed = False

        # Open the stream synchronously so schema is available before the
        # producer thread starts — avoids a schema/batch race on the queue.
        self._sync_reader._ensure_opened()

    @property
    def arrow_schema(self):
        return self._sync_reader.arrow_schema

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
                batch = self._sync_reader._arrow_stream.read_next_batch()
                batch = _convert_struct_timestamps(batch)
                self._queue.put(batch)
            # read_next_batch raises StopIteration at EOS — handled below
        except StopIteration:
            self._queue.put(_ASYNC_DONE)
        except Exception as exc:  # noqa: BLE001 - propagate to consumer
            self._queue.put(exc)

    def __iter__(self):
        self._start()
        return self

    def __next__(self):
        self._start()
        item = self._queue.get()
        if item is _ASYNC_DONE:
            raise StopIteration
        if isinstance(item, Exception):
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
