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

"""[Stub module — not for direct import or use]

In the ODPS execution engine, ``BlobWriter`` and ``BlobReference``
are provided by the C++ extension module (``libremote_udf_py*``).
This file contains stub classes solely for IDE type hints and
documentation.

At runtime inside a UDF, the C++ classes replace these stubs automatically.
Do NOT import this module in user code outside the ODPS execution environment.

Usage::

    from odps.udf.blob import BlobWriter, BlobReference

    writer = BlobWriter()
    writer.write(b'data')
    ref = writer.commit()
    data = ref.read(-1)
    ref.close()
"""

__all__ = ["BlobReference", "BlobWriter"]


class BlobReference:
    """[Stub class — do not instantiate directly]

    Represents a readable blob stream. The real implementation is backed by
    the C++ extension and communicates with the execution engine via reverse
    RPC. This stub exists only for type hints.

    A ``BlobReference`` is returned by :meth:`BlobWriter.commit()`.
    It provides a file-like interface for reading blob data from the
    execution engine via reverse RPC.

    Typical usage::

        ref = writer.commit()
        data = ref.read(-1)  # read all remaining data
        ref.close()

    Supports context manager protocol::

        with writer.commit() as ref:
            data = ref.read(-1)
    """

    def __init__(self):
        """Initialize a BlobReference.

        Note: In the execution engine this is constructed internally
        from C++ and not instantiated directly by user code.
        """
        pass

    def read(self, size=-1):
        """Read bytes from the blob stream.

        Args:
            size (int): Number of bytes to read. ``-1`` reads all
                remaining data. Must be positive if not ``-1``.

        Returns:
            bytes: The data read from the stream. Returns empty bytes
                when no more data is available.
        """
        pass

    def seek(self, offset, whence=0):
        """Change the stream position.

        Args:
            offset (int): The offset to seek to (or relative offset
                when ``whence=1``).
            whence (int): ``0`` for SEEK_SET (from beginning),
                ``1`` for SEEK_CUR (from current position).

        Returns:
            int: The new absolute position.

        Raises:
            ValueError: If ``whence`` is not ``0`` or ``1``.
        """
        pass

    def tell(self):
        """Return the current stream position.

        Returns:
            int: The current position in the stream.
        """
        pass

    def readable(self):
        """Return True if the stream is readable (not closed).

        Returns:
            bool: True if the stream is not closed.
        """
        pass

    def seekable(self):
        """Return True if the stream supports seeking.

        Returns:
            bool: True if the stream is not closed.
        """
        pass

    def writable(self):
        """Return whether the stream is writable.

        Returns:
            bool: Always False, as ``BlobReference`` is a read-only stream.
        """
        pass

    @property
    def closed(self):
        """bool: True if the stream has been closed."""
        pass

    def close(self):
        """Close the stream and release resources.

        Idempotent: calling close on an already-closed reference is a no-op.
        """
        pass

    def __enter__(self):
        """Enter the context manager.

        Returns:
            BlobReference: self
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, calling :meth:`close`."""
        pass

    def __iter__(self):
        """Return an iterator that yields 8MB chunks until EOF."""
        pass

    def __next__(self):
        """Return the next 8MB chunk, or raise StopIteration at EOF.

        Raises:
            StopIteration: When the end of the stream is reached.
        """
        pass


class BlobWriter:
    """[Stub class — do not instantiate directly]

    Provides a file-like interface for writing blob data. The real
    implementation is backed by the C++ extension and communicates with
    the execution engine via reverse RPC. This stub exists only for
    type hints.

    A ``BlobWriter`` creates a new blob stream, writes data to it,
    and commits it to produce a :class:`BlobReference`.

    Typical usage::

        writer = BlobWriter()
        writer.write(b'hello')
        writer.write(b' world')
        ref = writer.commit()
        # ref is a BlobReference for reading back the data
        ref.close()

    Supports context manager protocol::

        with BlobWriter() as writer:
            writer.write(b'data')
            ref = writer.commit()
    """

    def __init__(self):
        """Initialize a BlobWriter.

        The actual writer is created lazily on the first call to
        :meth:`write` or :meth:`commit` via reverse RPC to the
        execution engine.
        """
        pass

    def write(self, data):
        """Write bytes to the blob stream.

        Args:
            data (bytes): The data to write.

        Returns:
            int: The number of bytes written.

        Raises:
            ValueError: If the writer has been closed or already committed.
        """
        pass

    def commit(self):
        """Commit the blob stream and return a BlobReference.

        Returns:
            BlobReference: A reference to the committed blob, which can
                be used to read back the written data.

        Raises:
            ValueError: If the writer has been closed or already committed.
        """
        pass

    def close(self):
        """Close the writer without committing.

        Idempotent: calling close on an already-closed writer is a no-op.
        If the writer has not been committed, the uncommitted data is
        discarded.
        """
        pass

    def flush(self):
        """Flush the writer.

        Note: This is a no-op in the current implementation. Data is
        buffered and sent via RPC on write.
        """
        pass

    def writable(self):
        """Return True if the writer can accept more writes.

        Returns:
            bool: True if the writer is neither closed nor committed.
        """
        pass

    @property
    def closed(self):
        """bool: True if the writer has been closed."""
        pass

    def tell(self):
        """Return the current write position (total bytes written).

        Returns:
            int: The total number of bytes written so far.
        """
        pass

    def __enter__(self):
        """Enter the context manager.

        Returns:
            BlobWriter: self
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, calling :meth:`close`."""
        pass
