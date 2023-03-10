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

import hashlib
import os

from .. import compat
from ..compat import six
from ..config import options
from .resource import FileResource

RESOURCE_SIZE_MAX = 512 * 1024 * 1024  # a single resource's size must be at most 512M


class ResourceFile(object):
    __slots__ = (
        'resource', 'mode', '_opened', 'size', '_open_binary', '_encoding', '_overwrite'
    )

    def __init__(self, resource, mode='r', encoding='utf-8', overwrite=None):
        self.resource = resource

        self._open_binary = 'b' in mode
        mode = mode.replace('b', '')
        self.mode = FileResource.Mode(mode)
        self._encoding = encoding

        if self.mode in (FileResource.Mode.WRITE, FileResource.Mode.TRUNCEREADWRITE):
            self.size = 0
        else:
            self.resource._reload_size()
        self._opened = True
        self._overwrite = overwrite

    def _convert(self, content):
        if self._open_binary and isinstance(content, six.text_type):
            return content.encode(self._encoding)
        elif not self._open_binary and isinstance(content, six.binary_type):
            return content.decode(self._encoding)
        return content

    def _read_resource(self, offset=None, read_size=None):
        return self.resource.parent.read_resource(
            self.resource,
            text_mode=not self._open_binary,
            encoding=self._encoding,
            offset=offset,
            read_size=read_size,
        )

    def _new_buffer(self, content=None):
        io_clz = six.BytesIO if self._open_binary else six.StringIO
        return io_clz() if content is None else io_clz(content)

    def read(self, size=-1):
        raise NotImplementedError

    def readline(self, size=-1):
        raise NotImplementedError

    def readlines(self, sizehint=-1):
        raise NotImplementedError

    def write(self, content):
        raise NotImplementedError

    def writelines(self, seq):
        raise NotImplementedError

    def seek(self, pos, whence=compat.SEEK_SET):  # io.SEEK_SET
        raise NotImplementedError

    def seekable(self):
        raise NotImplementedError

    def tell(self):
        raise NotImplementedError

    def truncate(self, size=None):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def _iter(self):
        raise NotImplementedError

    def __iter__(self):
        return self._iter()

    def _next(self):
        raise NotImplementedError

    def __next__(self):
        return self._next()

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class LocalResourceFile(ResourceFile):
    __slots__ = "_fp", "_need_commit"

    def __init__(self, resource, mode='r', encoding='utf-8', overwrite=None):
        super(LocalResourceFile, self).__init__(
            resource, mode=mode, encoding=encoding, overwrite=overwrite
        )

        if self.mode in (FileResource.Mode.WRITE, FileResource.Mode.TRUNCEREADWRITE):
            self._fp = self._new_buffer()
        else:
            self._fp = self._read_resource()
            self._sync_size()

        self._need_commit = False

    def _sync_size(self):
        curr_pos = self.tell()
        self.seek(0, compat.SEEK_END)
        self.size = self.tell()
        self.seek(curr_pos)

    def read(self, size=-1):
        return self._fp.read(size)

    def readline(self, size=-1):
        return self._fp.readline(size)

    def readlines(self, sizehint=-1):
        return self._fp.readlines(sizehint)

    def _check_size(self):
        if self.size > RESOURCE_SIZE_MAX:
            raise IOError(
                "Single resource's max size is %sM" % (RESOURCE_SIZE_MAX / (1024 ** 2))
            )

    def write(self, content):
        content = self._convert(content)

        length = len(content)
        if self.mode in (FileResource.Mode.APPEND, FileResource.Mode.APPENDREADWRITE):
            self.seek(0, compat.SEEK_END)

        if length > 0:
            self._need_commit = True

        res = self._fp.write(content)
        self._sync_size()
        self._check_size()
        return res

    def writelines(self, seq):
        seq = [self._convert(s) for s in seq]

        length = sum(len(s) for s in seq)
        if self.mode in (FileResource.Mode.APPEND, FileResource.Mode.APPENDREADWRITE):
            self.seek(0, compat.SEEK_END)

        if length > 0:
            self._need_commit = True

        res = self._fp.writelines(seq)
        self._sync_size()
        self._check_size()
        return res

    def seek(self, pos, whence=compat.SEEK_SET):
        return self._fp.seek(pos, whence)

    def seekable(self):
        return True

    def tell(self):
        return self._fp.tell()

    def truncate(self, size=None):
        curr_pos = self.tell()
        self._fp.truncate(size)

        self.seek(0, compat.SEEK_END)
        self.size = self.tell()

        self.seek(curr_pos)

        self._need_commit = True

    def flush(self):
        if self._need_commit:
            is_create = self.resource._is_create()

            resources = self.resource.parent

            if is_create:
                resources.create(self=self.resource, file_obj=self._fp)
            else:
                resources.update(obj=self.resource, file_obj=self._fp)

            self._need_commit = False

    def close(self):
        if not self._opened:
            # already closed
            return

        self.flush()

        self._fp = None
        self.size = 0
        self._need_commit = False
        self._opened = False

    def _iter(self):
        return self._fp.__iter__()

    def _next(self):
        return next(self._fp)


class StreamResourceFile(ResourceFile):
    __slots__ = (
        "_md5_digest", "_buffer", "_buffered_size", "_resource_parts",
        "_resource_counter", "_chunk_size", "_is_source_exhausted",
        "_source_offset",
    )

    def __init__(self, resource, mode='r', encoding='utf-8', overwrite=None):
        mode = mode.replace("+", "")

        super(StreamResourceFile, self).__init__(
            resource, mode=mode, encoding=encoding, overwrite=overwrite
        )

        if self.mode not in (FileResource.Mode.READ, FileResource.Mode.WRITE):
            raise compat.UnsupportedOperation(
                "Unsupported access mode %s under streaming mode" % mode
            )

        self._md5_digest = hashlib.md5()
        self._resource_parts = []
        self._resource_counter = 0
        self._buffered_size = 0
        self._chunk_size = options.resource_chunk_size
        self._rebuild_buffer()

        self._is_source_exhausted = False
        self._source_offset = 0

    def _rebuild_buffer(self):
        self._buffer, buffer = self._new_buffer(), getattr(self, "_buffer", None)
        self._buffered_size = 0
        return buffer

    def _build_part_resource_name(self):
        name = "%s.part.tmp.%06d" % (self.resource.name, self._resource_counter)
        self._resource_counter += 1
        return name

    def _load_next_offset(self):
        if self._is_source_exhausted:
            return

        buf = self.resource.parent.read_resource(
            self.resource, offset=self._source_offset, read_size=self._chunk_size
        )
        self._rebuild_buffer()
        self._buffer.write(self._convert(buf.read()))
        self._buffer.seek(0, os.SEEK_SET)
        self._buffered_size = buf.tell()
        self._source_offset += buf.tell()
        self._is_source_exhausted = buf.is_eof

    def read(self, size=-1):
        buf = self._new_buffer()
        size_to_read = size
        while size_to_read != 0:
            if self._buffered_size == 0:
                self._load_next_offset()
            if self._buffered_size == 0:
                break
            res = self._buffer.read(size_to_read)

            buf.write(res)

            res_len = len(res)
            if size_to_read > 0:
                size_to_read -= res_len
            self._buffered_size -= res_len
        return buf.getvalue()

    def _is_line_terminated(self, line):
        terminator = b'\n' if self._open_binary else os.linesep
        return line.endswith(terminator)

    def readline(self, size=-1):
        line_buf = self._new_buffer()
        size_to_read = size
        while size_to_read != 0:
            if self._buffered_size == 0:
                self._load_next_offset()
            if self._buffered_size == 0:
                break

            res = self._buffer.readline(size_to_read)
            # read and concatenate to existing line
            line_buf.write(res)

            res_len = len(res)
            if size_to_read > 0:
                size_to_read -= res_len
            self._buffered_size -= res_len

            if self._is_line_terminated(res):
                break
        return line_buf.getvalue()

    def readlines(self, sizehint=-1):
        if sizehint == 0:
            return []

        lines_buf = []
        lines_to_read = sizehint
        while lines_to_read != 0:
            if self._buffered_size == 0:
                self._load_next_offset()
            if self._buffered_size == 0:
                break

            old_pos = self._buffer.tell()
            lines = self._buffer.readlines(lines_to_read)
            if not lines:
                break

            self._buffered_size -= self._buffer.tell() - old_pos

            if self._is_line_terminated(lines[-1]):
                lines_to_read -= len(lines)
                has_terminator = True
            else:
                # last line not complete
                lines_to_read -= len(lines) - 1
                has_terminator = False
            lines_to_read = max(-1, lines_to_read)
            lines_buf.append((lines, has_terminator))

        # merge fraction of lines
        last_has_terminator = True
        res_line_groups = []
        for lines, has_terminator in lines_buf:
            if not last_has_terminator:
                res_line_groups[-1].append(lines[0])
                lines = lines[1:]
            if lines:
                res_line_groups.extend([l] for l in lines)
            last_has_terminator = has_terminator

        res_lines = []
        sep = b"" if self._open_binary else ""
        for lg in res_line_groups:
            res_lines.append(sep.join(lg))

        return res_lines

    def write(self, content):
        if self._buffered_size >= self._chunk_size:
            self.flush()

        content = self._convert(content)

        last_pos = self._buffer.tell()
        self._buffer.write(content)
        content_size = self._buffer.tell() - last_pos

        self._buffered_size += content_size
        self.size += content_size

    def writelines(self, seq):
        if self._buffered_size >= self._chunk_size:
            self.flush()

        seq = [self._convert(s) for s in seq]

        last_pos = self._buffer.tell()
        self._buffer.writelines(seq)
        content_size = self._buffer.tell() - last_pos

        self._buffered_size += content_size
        self.size += content_size

    def seek(self, pos, whence=compat.SEEK_SET):
        raise compat.UnsupportedOperation("File or stream is not seekable.")

    def seekable(self):
        return False

    def tell(self):
        return self.size

    def truncate(self, size=None):
        raise compat.UnsupportedOperation("File or stream is not seekable.")

    def flush(self):
        value = self._rebuild_buffer()

        if value.tell() > 0:
            res = self.resource.parent.create(
                name=self._build_part_resource_name(), type="file", temp=True, part=True, fileobj=value
            )
            self._resource_parts.append(res)
            if self._open_binary:
                self._md5_digest.update(value.getvalue())
            else:
                self._md5_digest.update(value.getvalue().encode(self._encoding))

    def close(self):
        if self.mode == FileResource.Mode.READ or not self._opened:
            return

        self.flush()
        if self._overwrite is None:
            self._overwrite = not self.resource._is_create()

        self.resource.parent.merge_part_files(
            self.resource,
            self._resource_parts,
            self._md5_digest.hexdigest(),
            overwrite=self._overwrite,
        )
        self._opened = False

    def _iter(self):
        return self

    def _next(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line
