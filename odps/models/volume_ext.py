#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import contextlib

from .. import serializers
from ..compat import Enum
from ..errors import OSSSignUrlError
from ..lib import requests
from . import cache_parent
from .volume_fs import FSVolumeObject, FSVolumeObjects, FSVolume


class SignUrlMethod(Enum):
    GET = "get"
    PUT = "put"


class ExternalVolumeObject(FSVolumeObject):
    @classmethod
    def _get_base_cls(cls):
        return ExternalVolumeObject

    @classmethod
    def _get_file_cls(cls):
        return ExternalVolumeFile

    @classmethod
    def _get_dir_cls(cls):
        return ExternalVolumeDir

    def get_sign_url(self, method, seconds=None):
        if isinstance(method, SignUrlMethod):
            method = method.value
        params = {'sign_url': method.lower()}
        if seconds:
            params["expire_seconds"] = seconds
        headers = {'x-odps-volume-fs-path': self.path}

        schema_name = self.volume._get_schema_name()
        if schema_name is not None:
            params["curr_schema"] = schema_name
        resp = self._client.get(
            self.parent.resource(), action='meta', params=params, headers=headers
        )
        self.parse(self._client, resp, obj=self)
        return self.sign_url

    def _request_sign_url(self, path, method, *args, **kw):
        if path:
            path = self.path.rstrip('/') + '/' + path.lstrip('/')
        else:
            path = self.path
        vol_rel_path = path[len(self.volume.name) + 1:]
        sign_url = self.volume.get_sign_url(vol_rel_path, method)
        if method == SignUrlMethod.PUT:
            resp = requests.put(sign_url, *args, **kw)
        else:
            resp = requests.get(sign_url, *args, **kw)
        resp.volume_path = path
        self._check_response(resp)
        return resp

    @staticmethod
    def _check_response(resp):
        # when response code is a string, just skip
        if hasattr(resp, "status_code") and resp.status_code >= 400:
            try:
                import oss2.exceptions

                oss_exc = oss2.exceptions.make_exception(resp.raw)
                raise OSSSignUrlError(oss_exc)
            except ImportError:
                raise OSSSignUrlError(resp.content)

    def _delete(self, recursive=False):
        """
        Delete current directory.

        :param recursive: indicate whether a recursive deletion should be performed.
        """
        params = {'recursive': str(recursive).lower()}
        headers = {'x-odps-volume-fs-path': self.path.rstrip("/")}
        self._del_cache(self.path)
        self._client.delete(
            self.parent.resource(),
            params=params,
            headers=headers,
            curr_schema=self.parent._get_schema_name(),
        )

    @contextlib.contextmanager
    def _open_reader(self, path):
        """
        Open a volume file and read contents in it.

        :param str path: file name to be opened
        :return: file reader

        :Example:
        >>> with fs_dir.open_reader('file') as reader:
        >>>     [print(line) for line in reader]
        """
        req = self._request_sign_url(path, SignUrlMethod.GET, stream=True)
        yield req.raw

    @contextlib.contextmanager
    def _open_writer(self, path, **kwargs):
        """
        Open a volume file and write contents into it.

        :param str path: file name to be opened
        :return: file reader

        :Example:
        >>> with fs_dir.open_writer('file') as reader:
        >>>     writer.write('some content')
        """
        if kwargs.pop("replication", None) is not None:  # pragma: no cover
            raise TypeError("External volume does not support replication argument")

        with self._request_sign_url(path, SignUrlMethod.PUT, file_upload=True) as writer:
            yield writer
        self._check_response(writer.result)


@cache_parent
class ExternalVolumeFile(ExternalVolumeObject):
    def __init__(self, **kw):
        super(ExternalVolumeFile, self).__init__(**kw)
        self._isdir = False

    def delete(self):
        """
        Delete current file.
        """
        return self._delete(False)

    def open_reader(self):
        """
        Open current file and read contents in it.
        :return: file reader

        :Example:
        >>> with fs_file.open_reader('file') as reader:
        >>>     [print(line) for line in reader]
        """
        return self._open_reader(None)

    def open_writer(self, **kw):
        return self._open_writer(None, **kw)


@cache_parent
class ExternalVolumeDir(ExternalVolumeObject):
    def __init__(self, **kw):
        super(ExternalVolumeDir, self).__init__(**kw)
        self._isdir = True

    @property
    def objects(self):
        return ExternalVolumeObjects(parent=self, client=self._client)

    def create_dir(self, path):
        """
        Creates and returns a sub-directory under the current directory.
        :param str path: directory name to be created
        :return: directory object
        :rtype: :class:`odps.models.FSVolumeDir`
        """
        path = path.strip("/") + "/"
        resp = self._request_sign_url(path, SignUrlMethod.PUT, b"")
        dir_object = type(self)(path=resp.volume_path, parent=self.parent, client=self._client)
        dir_object.reload()
        return dir_object

    def __contains__(self, item):
        return item in self.objects

    def __iter__(self):
        return self.objects.iterate()

    def __getitem__(self, item):
        return self.objects[item]

    def delete(self, recursive=False):
        """
        Delete current directory.

        :param recursive: indicate whether a recursive deletion should be performed.
        """
        self._delete(recursive=recursive)

    def open_reader(self, path):
        """
        Open a volume file and read contents in it.

        :param str path: file name to be opened
        :return: file reader

        :Example:
        >>> with fs_dir.open_reader('file') as reader:
        >>>     [print(line) for line in reader]
        """
        return self._open_reader(path)

    def open_writer(self, path, **kwargs):
        """
        Open a volume file and write contents into it.

        :param str path: file name to be opened
        :return: file reader

        :Example:
        >>> with fs_dir.open_writer('file') as reader:
        >>>     writer.write('some content')
        """
        return self._open_writer(path, **kwargs)


class ExternalVolumeObjects(FSVolumeObjects):
    objects = serializers.XMLNodesReferencesField(ExternalVolumeObject, 'Item')

    @classmethod
    def _get_single_object_cls(cls):
        return ExternalVolumeObject


class ExternalVolume(FSVolume):
    _dir_cls = ExternalVolumeDir

    def get_sign_url(self, path, method, seconds=None):
        path = "/" + self.name + "/" + path.lstrip("/")
        vol_file = ExternalVolumeFile(path=path, parent=self, client=self._client)
        return vol_file.get_sign_url(method, seconds=seconds)
