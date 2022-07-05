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

import mmap
import os
import tempfile
from urllib.parse import urlparse

import numpy as np

try:
    from mars.lib.filesystem import FileSystem
except ImportError:
    from mars.filesystem import FileSystem

from ...core import ODPS
from ...compat import futures, BytesIO, OrderedDict
from ...errors import NoSuchObject
from ...tunnel.volumetunnel import VolumeTunnel
from ...utils import to_binary, to_str


DIR_ERROR_MSG = (
    "Argument `path` illegal, expect to be "
    "odps:///**project**/volumes/**volume**/, "
    "got {}"
)

FILE_ERROR_MSG = (
    "Argument `path` illegal, expect to be "
    "odps:///**project**/volumes/**volume**/**file**, "
    "got {}"
)

DEFAULT_CHUNK_SIZE = 512 * 1024**2

META_FILE_NAME = "_meta_"


class _VolumeFileObject(object):
    def __init__(self, volume_parted, mode, temp_dir=None, chunk_size=None):
        self._volume_parted = volume_parted
        self._mode = mode
        self._temp_dir = temp_dir
        self._chunk_size = chunk_size

        # writer
        self._index = 0
        self._writer = None
        self._write_buffer = None
        self._meta = dict()

        # reader
        self._reader = None
        self._read_filename = None

    def _read_multi_volume_files(self):
        # get meta file first
        with self._volume_parted.open_reader(META_FILE_NAME) as meta_reader:
            content = to_str(meta_reader.read())
            meta = OrderedDict()
            length = 0
            for line in content.split("\n"):
                if "," in line:
                    filename, size = line.split(",", 1)
                    size = int(size)
                    meta[filename] = size
                    length += size

        temp_dir = self._temp_dir or tempfile.mkdtemp()
        write_filename = os.path.join(
            temp_dir, self._volume_parted.volume.name, self._volume_parted.name
        )
        os.makedirs(os.path.dirname(write_filename), exist_ok=True)

        with open(write_filename, "wb") as f:
            for size in meta.values():
                f.write(b"\0" * size)

        with open(write_filename, "rb+") as f:
            m = mmap.mmap(f.fileno(), length)
            offsets = [0] + np.cumsum(list(meta.values())).tolist()

            volume_part = self._volume_parted
            volume_tunnel = VolumeTunnel(
                client=volume_part._client,
                project=volume_part.project,
                endpoint=volume_part.project._tunnel_endpoint,
            )

            def _write(filename, start, end):
                with volume_tunnel.create_download_session(
                    volume_part.volume.name, volume_part.name, filename
                ).open() as reader:
                    content_part = reader.read()
                    m[start:end] = content_part

            executor = futures.ThreadPoolExecutor(8)
            fs = []
            for i, filename in enumerate(meta):
                future = executor.submit(_write, filename, offsets[i], offsets[i + 1])
                fs.append(future)
            [f.result() for f in fs]

        self._read_filename = write_filename
        self._reader = open(write_filename, "rb")

    def read(self, size=None):
        self._volume_parted.reload()
        if self._volume_parted.file_number == 1:
            if self._reader is None:
                self._reader = self._volume_parted.open_reader(
                    list(self._volume_parted.files)[0].name
                )
            return self._reader.read(size=size)
        else:
            if self._reader is None:
                self._read_multi_volume_files()
            return self._reader.read(size)

    def _ensure_writer_open(self):
        if self._writer is None:
            self._writer = self._volume_parted.open_writer()
            self._write_buffer = BytesIO()

    def _flush_to_volume_file(self):
        # write buffer full, write with a file
        filename = str(self._index)
        buffer = self._write_buffer
        content_part = buffer.getvalue()
        self._writer.write(filename, content_part)
        self._meta[filename] = buffer.tell()
        self._index += 1
        # clear write buffer
        self._write_buffer = BytesIO()

    def write(self, content):
        self._ensure_writer_open()

        content_left = len(content)
        while content_left > 0:
            buffer_left_size = self._chunk_size - self._write_buffer.tell()
            content_part = content[:buffer_left_size]
            content = content[buffer_left_size:]
            self._write_buffer.write(content_part)
            content_left -= len(content_part)
            if self._write_buffer.tell() >= self._chunk_size:
                # write buffer full
                self._flush_to_volume_file()

    def close(self):
        if self._writer is not None:
            if self._write_buffer.tell() > 0:
                self._flush_to_volume_file()
                self._index = 0
            if len(self._meta) > 1:
                meta_content = "\n".join(
                    "{},{}".format(filename, size)
                    for filename, size in self._meta.items()
                )
                # write meta file if file length > 1
                self._writer.write(META_FILE_NAME, to_binary(meta_content))
            self._writer.__exit__(None, None, None)
        if self._reader is not None:
            if self._read_filename is not None:
                # remove read file name
                os.remove(self._read_filename)
            self._reader.__exit__(None, None, None)

    def __enter__(self):
        if self._writer or self._reader:
            (self._writer or self._reader).__enter__()
        return self

    def __exit__(self, *_):
        self.close()


class VolumeFileSystem(FileSystem):
    """
    Schema follows:

    /**project**/volumes/**volume**
    """

    def __init__(self, odps=None, path=None, temp_dir=None, chunk_size=None):
        if odps is None:
            # try to get from environments
            odps = ODPS.from_environments()
        if odps is None:
            # try to get from global config
            odps = ODPS.from_global()
        if odps is None:
            # still None, raise error
            raise ValueError("`odps` should be specified for VolumeFileSystem")

        self._odps = odps
        self._path = path
        self._temp_dir = temp_dir
        self._chunk_size = chunk_size or DEFAULT_CHUNK_SIZE

    @staticmethod
    def parse_from_path(uri):
        parsed = urlparse(uri)
        options = dict()
        if parsed.path:
            options["path"] = parsed.path.strip("/")
        return options

    @staticmethod
    def _extract_info(path, is_file=False):
        options = VolumeFileSystem.parse_from_path(path)
        error_msg = DIR_ERROR_MSG if not is_file else FILE_ERROR_MSG

        if "path" not in options:
            raise ValueError(error_msg.format(path))

        path = options["path"]
        splits = path.split("/")
        expect_size = 3 if not is_file else 4
        if len(splits) != expect_size:
            raise ValueError(error_msg.format(path))

        volumes = splits[1]
        if volumes.lower() != "volumes":
            raise ValueError(error_msg.format(path))

        return splits[:1] + splits[2:]

    def ls(self, path):
        path = path.strip("/") or self._path
        project, volume = self._extract_info(path)
        try:
            files = self._odps.list_volume_partitions(volume, project=project)
        except NoSuchObject:
            raise FileNotFoundError("File {} not found".format(path))
        return [
            "odps:///{}/volumes/{}/{}".format(project, volume, f.name) for f in files
        ]

    def delete(self, path, recursive: bool = False):
        path = path.strip("/") or self._path
        if "/" not in path:
            path = "{}/{}".format(self._path, path)
        options = VolumeFileSystem.parse_from_path(path)
        fpath = options["path"]

        splits = fpath.split("/")
        if len(splits) == 3:
            is_file = False
        elif len(splits) == 4:
            is_file = True
        else:
            raise ValueError(
                "Argument `path` illegal, expect to be "
                "odps:///**project**/volumes/**volume**/ or"
                "odps:///**project**/volumes/**volume**/**partition**/, "
                "got {}".format(path)
            )

        if is_file:
            project, volume, partition = self._extract_info(path, is_file=True)
            if self._odps.exist_volume_partition(
                volume, partition=partition, project=project
            ):
                self._odps.delete_volume_partition(
                    volume, partition=partition, project=project
                )
        else:
            project, volume = self._extract_info(path)
            if self._odps.exist_volume(volume, project=project):
                self._odps.delete_volume(volume, project=project)

    def open(self, path, mode="rb"):
        if mode not in ("rb", "wb"):
            raise ValueError("`mode` can be `rb` or `wb` only")

        path = path.strip("/") or self._path
        if "/" not in path:
            path = "{}/{}".format(self._path, path)
        project, volume, partition = self._extract_info(path, is_file=True)
        volume_parted = self._odps.get_volume_partition(
            volume, partition=partition, project=project
        )

        return _VolumeFileObject(
            volume_parted, mode, temp_dir=self._temp_dir, chunk_size=self._chunk_size
        )

    def mkdir(self, path, create_parents=True):
        assert create_parents

        path = path.strip("/") or self._path
        project, volume = self._extract_info(path)

        if not self._odps.exist_volume(volume, project=project):
            self._odps.create_parted_volume(volume, project=project)

    def stat(self, path):
        path = path.strip("/") or self._path
        if "/" not in path:
            path = "{}/{}".format(self._path, path)
        options = VolumeFileSystem.parse_from_path(path)
        fpath = options["path"]

        splits = fpath.split("/")
        if len(splits) == 3:
            is_file = False
        elif len(splits) == 4:
            is_file = True
        else:
            raise ValueError(
                "Argument `path` illegal, expect to be "
                "odps:///**project**/volumes/**volume**/ or"
                "odps:///**project**/volumes/**volume**/**partition**/, "
                "got {}".format(path)
            )

        if is_file:
            project, volume, partition = self._extract_info(path, is_file=True)
            volume_parted = self._odps.get_volume_partition(
                volume, partition=partition, project=project
            )
            stat = dict(
                name=path,
                size=volume_parted.length,
                created=volume_parted.creation_time,
            )
        else:
            project, volume = self._extract_info(path)
            vol = self._odps.get_volume(volume, project=project)
            stat = dict(name=path, size=vol.length, created=vol.creation_time)
        return stat

    def _isfilestore(self):
        return True

    def cat(self, path):
        raise NotImplementedError

    def exists(self, path):
        raise NotImplementedError

    def glob(self, path, recursive=False):
        raise NotImplementedError

    def isdir(self, path):
        raise NotImplementedError

    def isfile(self, path):
        raise NotImplementedError

    def rename(self, path, new_path):
        raise NotImplementedError

    def walk(self, path):
        raise NotImplementedError
