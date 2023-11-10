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

from .. import serializers, errors, utils
from ..compat import six, long_type
from .core import LazyLoad, cache, Iterable
from .cache import cache_parent
from .volumes import Volume


class FSVolumeObject(LazyLoad):
    __slots__ = '_volume_fs_tunnel',
    _type_indicator = '_isdir'
    _cache_name_arg = 'path'

    class CreateRequestXML(serializers.XMLSerializableModel):
        _root = 'Item'

        type = serializers.XMLNodeField('Type')
        path = serializers.XMLNodeField('Path')

    class UpdateRequestXML(serializers.XMLSerializableModel):
        _root = 'Item'

        path = serializers.XMLNodeField('Path')
        replication = serializers.XMLNodeField('Replication')

    _project = serializers.XMLNodeField('Project')
    _volume = serializers.XMLNodeField('Volume')
    path = serializers.XMLNodeField('Path')
    _isdir = serializers.XMLNodeField('Isdir', type='bool')
    permission = serializers.XMLNodeField('permission')
    _replication = serializers.XMLNodeField('BlockReplications', parse_callback=int)
    length = serializers.XMLNodeField('Length', parse_callback=long_type)
    quota = serializers.XMLNodeField('Quota', parse_callback=long_type)
    block_size = serializers.XMLNodeField('BlockSize', parse_callback=long_type)
    owner = serializers.XMLNodeField('Owner')
    group = serializers.XMLNodeField('Group')
    creation_time = serializers.XMLNodeField('CreationTime', type='rfc822')
    access_time = serializers.XMLNodeField('AccessTime', type='rfc822')
    last_modified_time = serializers.XMLNodeField('ModificationTime', type='rfc822')
    symlink = serializers.XMLNodeField('Symlink')
    sign_url = serializers.XMLNodeField('URL')

    @classmethod
    def _get_base_cls(cls):
        return FSVolumeObject

    @classmethod
    def _get_dir_cls(cls):
        return FSVolumeDir

    @classmethod
    def _get_file_cls(cls):
        return FSVolumeFile

    @classmethod
    def _get_objects_cls(cls):
        return FSVolumeObjects

    @staticmethod
    def _filter_cache(_, **kwargs):
        isdir = kwargs.get('_isdir')
        return isdir is not None and isdir != 'UNKNOWN'

    @utils.experimental('Volume2 is still experimental. Usage in production environment is strongly opposed.')
    @cache
    def __new__(cls, *args, **kwargs):
        isdir = kwargs.get('_isdir')

        base_cls = cls._get_base_cls()
        if cls is not base_cls and issubclass(cls, base_cls):
            return object.__new__(cls)
        if isdir is not None:
            if isdir == 'UNKNOWN':
                return object.__new__(base_cls)
            return object.__new__(cls._get_dir_cls() if isdir else cls._get_file_cls())

        obj = base_cls(_isdir='UNKNOWN', **kwargs)
        obj.reload()
        return base_cls(**obj.extract())

    def _name(self):
        return self.path

    def _set_state(self, name, parent, client):
        self.__init__(path=name, _parent=parent, _client=client)

    def split(self):
        return self.path.rsplit('/', 1)

    @property
    def basename(self):
        _, fn = self.split()
        return fn

    @property
    def dirname(self):
        dn, _ = self.split()
        return dn

    @property
    def volume(self):
        return self.parent

    @property
    def is_root(self):
        return self.path == '/' + self.parent.name

    def reload(self):
        # check if the volume path is the root
        if self.is_root:
            return

        schema_name = self.parent._get_schema_name()
        params = {}
        if schema_name is not None:
            params["curr_schema"] = schema_name

        resp = self._client.get(
            self.parent.resource(),
            action='meta',
            params=params,
            headers={'x-odps-volume-fs-path': self.path},
        )
        self.parse(self._client, resp, obj=self)

        self._loaded = True

    @staticmethod
    def _normpath(path):
        path = path.rstrip('/')
        i = 0
        parts = []
        start = 0
        while i < len(path):
            if path[i] == '/' or i == len(path) - 1:
                chunk = path[start:i + 1]
                start = i + 1
                if chunk in ['', '/', '.', './']:
                    # do nothing
                    pass
                elif chunk in ['..', '../']:
                    if len(parts):
                        parts = parts[:len(parts) - 1]
                    else:
                        parts.append(chunk)
                else:
                    parts.append(chunk)
            i += 1
        if path.startswith('/'):
            return '/' + ''.join(parts)
        return ''.join(parts)

    def _del_cache(self, path):
        root_objs = self._get_objects_cls()(parent=self.volume.root, client=self._client)
        if not path.startswith('/'):
            path = self.path + '/' + path.lstrip('/')
        del root_objs[path]

    def move(self, new_path, replication=None):
        """
        Move current path to a new location.

        :param new_path: target location of current file / directory
        :param replication: number of replication
        """
        if not new_path.startswith('/'):
            new_path = self._normpath(self.dirname + '/' + new_path)
        else:
            new_path = self._normpath(new_path)
        if new_path == self.path:
            raise ValueError('New path should be different from the original one.')
        update_def = self.UpdateRequestXML(path=new_path)
        if replication:
            update_def.replication = replication
        headers = {
            'Content-Type': 'application/xml',
            'x-odps-volume-fs-path': self.path,
        }

        schema_name = self.parent._get_schema_name()
        params = {}
        if schema_name is not None:
            params["curr_schema"] = schema_name

        self._client.put(
            self.parent.resource(),
            action='meta',
            params=params,
            headers=headers,
            data=update_def.serialize(),
        )

        self._del_cache(self.path)
        self.path = new_path
        self.reload()

    def _create_volume_fs_tunnel(self, endpoint=None, quota_name=None):
        if self._volume_fs_tunnel is not None:
            return self._volume_fs_tunnel

        from ..tunnel import VolumeFSTunnel

        self._volume_fs_tunnel = VolumeFSTunnel(
            client=self._client,
            project=self.project,
            endpoint=endpoint or self.project._tunnel_endpoint,
            quota_name=quota_name,
        )
        return self._volume_fs_tunnel


@cache_parent
class FSVolumeDir(FSVolumeObject):
    """
    VolumeFSDir represents a directory under a file system volume in ODPS.
    You can use ``create_dir`` to create a sub-directory, ``open_reader`` to open a file to read,
    ``open_writer`` to write a file and ``delete`` to remove. Following operations are also supported.

    >>> # iterate over all files and sub-directories
    >>> for o in fs_dir:
    >>>     print(o.path)
    >>> # check if a file exists in current volume
    >>> print(file_name in fs_dir)
    >>> # get a file/directory object
    >>> file_obj = fs_dir[file_name]
    """
    def __init__(self, **kw):
        super(FSVolumeDir, self).__init__(**kw)
        self._isdir = True

    @property
    def objects(self):
        return self._get_objects_cls()(parent=self, client=self._client)

    def create_dir(self, path):
        """
        Creates and returns a sub-directory under the current directory.
        :param str path: directory name to be created
        :return: directory object
        :rtype: :class:`odps.models.FSVolumeDir`
        """
        path = self.path + '/' + path.lstrip('/')
        dir_def = self.CreateRequestXML(type='directory', path=path)
        headers = {'Content-Type': 'application/xml'}
        self._client.post(
            self.parent.resource(),
            headers=headers,
            data=dir_def.serialize(),
            curr_schema=self.parent._get_schema_name(),
        )

        dir_object = type(self)(path=path, parent=self.parent, client=self._client)
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
        params = {'recursive': recursive}
        headers = {'x-odps-volume-fs-path': self.path}
        self._del_cache(self.path)
        self._client.delete(
            self.parent.resource(),
            params=params,
            headers=headers,
            curr_schema=self.parent._get_schema_name(),
        )

    def open_reader(self, path, **kw):
        """
        Open a volume file and read contents in it.

        :param str path: file name to be opened
        :param start: start position
        :param length: length limit
        :param quota_name: name of tunnel quota
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :return: file reader

        :Example:
        >>> with fs_dir.open_reader('file') as reader:
        >>>     [print(line) for line in reader]
        """
        endpoint = kw.pop('endpoint', None)
        quota_name = kw.pop('quota_name', None)
        tunnel = self._create_volume_fs_tunnel(endpoint=endpoint, quota_name=quota_name)
        path = self.path.lstrip('/')[len(self.parent.name):].lstrip('/') + '/' + path.lstrip('/')
        return tunnel.open_reader(self.parent, path, **kw)

    def open_writer(self, path, replication=None, **kw):
        """
        Open a volume file and write contents into it.

        :param str path: file name to be opened
        :param quota_name: name of tunnel quota
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :return: file reader

        :Example:
        >>> with fs_dir.open_writer('file') as reader:
        >>>     writer.write('some content')
        """
        endpoint = kw.pop('endpoint', None)
        quota_name = kw.pop('quota_name', None)
        tunnel = self._create_volume_fs_tunnel(endpoint=endpoint, quota_name=quota_name)
        vol_path = self.path.lstrip('/')[len(self.parent.name):].lstrip('/') + '/' + path.lstrip('/')
        return tunnel.open_writer(self.parent, vol_path, replication=replication, **kw)


@cache_parent
class FSVolumeFile(FSVolumeObject):
    def __init__(self, **kw):
        super(FSVolumeFile, self).__init__(**kw)
        self._isdir = False

    @property
    def replication(self):
        """
        Get / set replication number of the file.
        """
        return self._replication

    @replication.setter
    def replication(self, value):
        update_def = self.UpdateRequestXML(replication=value)
        headers = {
            'Content-Type': 'application/xml',
            'x-odps-volume-fs-path': self.path,
        }

        schema_name = self.parent._get_schema_name()
        params = {}
        if schema_name is not None:
            params["curr_schema"] = schema_name

        self._client.put(
            self.parent.resource(),
            action='meta',
            params=params,
            headers=headers,
            data=update_def.serialize(),
        )
        self.reload()

    def delete(self, **_):
        """
        Delete current file.
        """
        params = {'recursive': False}
        headers = {'x-odps-volume-fs-path': self.path}

        schema_name = self.parent._get_schema_name()
        if schema_name is not None:
            params["curr_schema"] = schema_name

        self._del_cache(self.path)
        self._client.delete(self.parent.resource(), params=params, headers=headers)

    def open_reader(self, **kw):
        """
        Open current file and read contents in it.

        :param start: start position
        :param length: length limit
        :param quota_name: name of tunnel quota
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :return: file reader

        :Example:
        >>> with fs_file.open_reader('file') as reader:
        >>>     [print(line) for line in reader]
        """
        endpoint = kw.pop('endpoint', None)
        quota_name = kw.pop('quota_name', None)
        tunnel = self._create_volume_fs_tunnel(endpoint=endpoint, quota_name=quota_name)
        path = self.path.lstrip('/')[len(self.parent.name):].lstrip('/')
        return tunnel.open_reader(self.parent, path, **kw)

    def open_writer(self, replication=None, **kw):
        endpoint = kw.pop('endpoint', None)
        quota_name = kw.pop('quota_name', None)
        tunnel = self._create_volume_fs_tunnel(endpoint=endpoint, quota_name=quota_name)
        path = self.path.lstrip('/')[len(self.parent.name):].lstrip('/')
        return tunnel.open_writer(self.parent, path, replication=replication, **kw)


class FSVolumeObjects(Iterable):
    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems', parse_callback=int)
    objects = serializers.XMLNodesReferencesField(FSVolumeObject, 'Item')

    @property
    def project(self):
        return self.parent.parent.parent

    @property
    def volume(self):
        return self.parent.parent

    @classmethod
    def _get_single_object_cls(cls):
        return FSVolumeObject

    def _get(self, name):
        path = name
        if not path.startswith(self.parent.path):
            path = self.parent.path + '/' + name.lstrip('/')
        return self._get_single_object_cls()(client=self._client, parent=self.volume, path=path)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            try:
                # as reload() will be done in constructor of VolumeFSObject, we return directly.
                return self._get(item)
            except errors.NoSuchObject:
                return False
        elif isinstance(item, self._get_single_object_cls()):
            obj = item
            try:
                obj.reload()
                return True
            except errors.NoSuchObject:
                return False
        else:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self):
        params = {'expectmarker': 'true'}
        headers = {'x-odps-volume-fs-path': self.parent.path}

        schema_name = self.volume._get_schema_name()
        if schema_name is not None:
            params["curr_schema"] = schema_name

        def _it():
            last_marker = params.get('marker')
            if 'marker' in params and \
                    (last_marker is None or len(last_marker) == 0):
                return

            url = self.volume.resource()
            resp = self._client.get(url, params=params, headers=headers)

            r = type(self).parse(self._client, resp, obj=self, parent=self.volume)
            params['marker'] = r.marker

            return r.objects

        while True:
            objects = _it()
            if objects is None:
                break
            for obj in objects:
                yield obj


@cache_parent
class FSVolume(Volume):
    """
    FSVolume represents the new-fashioned file system volume in ODPS.
    You can use ``create_dir`` to create a directory, ``open_reader`` to open a file to read
    and ``open_writer`` to write a file. Following operations are also supported.

    >>> # iterate over all files and directories
    >>> for o in fs_volume:
    >>>     print(o.path)
    >>> # check if a file exists in current volume
    >>> print(file_name in fs_volume)
    >>> # get a file/directory object
    >>> file_obj = fs_volume[file_name]
    """
    __slots__ = '_root_dir',

    _dir_cls = FSVolumeDir

    def create_dir(self, path):
        """
        Creates and returns a directory under the current volume.
        :param str path: directory name to be created
        :return: directory object
        :rtype: :class:`odps.models.FSVolumeDir`
        """
        return self.root.create_dir(path)

    def __contains__(self, item):
        return item in self.root

    def __iter__(self):
        for it in self.root:
            yield it

    def __getitem__(self, item):
        if not item:
            return self.root
        return self.root[item]

    @property
    def path(self):
        return '/' + self.name

    @property
    def location(self):
        if self.type != Volume.Type.EXTERNAL:
            raise AttributeError
        return self.properties.get(Volume.EXTERNAL_VOLUME_LOCATION_KEY)

    @property
    def rolearn(self):
        if self.type != Volume.Type.EXTERNAL:
            raise AttributeError
        return self.properties.get(Volume.EXTERNAL_VOLUME_ROLEARN_KEY)

    def open_reader(self, path, **kw):
        """
        Open a volume file and read contents in it.

        :param str path: file name to be opened
        :param start: start position
        :param length: length limit
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :return: file reader

        :Example:
        >>> with volume.open_reader('file') as reader:
        >>>     [print(line) for line in reader]
        """
        return self.root.open_reader(path, **kw)

    def open_writer(self, path, replication=None, **kw):
        """
        Open a volume file and write contents into it.

        :param str path: file name to be opened
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :return: file reader

        :Example:
        >>> with volume.open_writer('file') as reader:
        >>>     writer.write('some content')
        """
        return self.root.open_writer(path, replication=replication, **kw)

    def delete(self, path, recursive=False):
        try:
            return self[path].delete(recursive=recursive)
        except TypeError:
            return self[path].delete()

    @property
    def root(self):
        if not self._root_dir:
            self._root_dir = self._dir_cls(path='/' + self.name, parent=self, client=self._client)
        return self._root_dir


# keep renames compatible
VolumeFSDir = FSVolumeDir
VolumeFSFile = FSVolumeFile
