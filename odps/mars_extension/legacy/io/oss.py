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

import fnmatch

try:
    import oss2
except ImportError:
    oss2 = None

from ....compat import urlparse


class OSSRandomReader(object):
    mode = "rb"

    def __init__(self, bucket, key):
        self._bucket = bucket
        self._key = key
        self._pos = 0
        self._file_len = None
        self._reader = None
        self._reload_object()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _reload_object(self):
        if self._file_len is None:
            self._file_len = self._bucket.get_object_meta(self._key).content_length
        self._reader = self._bucket.get_object(self._key, (self._pos, None))

    def tell(self):
        return self._pos

    def read(self, amt=None):
        buf = self._reader.read(amt)
        self._pos += len(buf)
        return buf

    def seek(self, offset, whence=0):
        old_pos = self._pos
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos = min(self._file_len, self._pos + offset)
        else:
            self._pos = max(0, self._file_len - offset)
        if self._pos != old_pos:
            self._reload_object()

    def close(self):
        self._bucket = None
        self._reader = None


class OSSWriter(object):
    mode = "wb"

    def __init__(self, bucket, key):
        self._bucket = bucket
        self._key = key

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, content):
        self._bucket.put_object(self._key, content)

    def close(self):
        self._bucket = None


class OSSFileSystem(object):
    sep = "/"

    def __init__(
        self, endpoint=None, access_id=None, secret_access_key=None, bucket_name=None
    ):
        self._bucket = oss2.Bucket(
            oss2.Auth(access_id, secret_access_key), endpoint, bucket_name
        )

    def _normalize_path(self, path):
        parse_result = urlparse(path)
        path = parse_result.path
        if parse_result.netloc:
            path = parse_result.netloc + parse_result.path
        return path

    def glob(self, path):
        path = self._normalize_path(path)

        prefix_pos = 0
        while prefix_pos < len(path):
            if str.isalnum(path[prefix_pos]) or path[prefix_pos] in "_/":
                prefix_pos += 1
            else:
                break
        prefix_path = path[0:prefix_pos]

        for obj_info in oss2.ObjectIterator(self._bucket, prefix_path):
            if fnmatch.fnmatch(obj_info.key, path):
                yield "oss://" + obj_info.key

    def open(self, path, mode="rb"):
        path = self._normalize_path(path)
        if mode == "rb":
            return OSSRandomReader(self._bucket, path)
        elif mode == "wb":
            return OSSWriter(self._bucket, path)
        else:
            raise NotImplementedError("Does not support mode except `rb` and `wb`")


from . import core

if oss2:
    core._file_systems["oss"] = OSSFileSystem
