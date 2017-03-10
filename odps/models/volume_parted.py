#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from .core import LazyLoad, Iterable
from .. import serializers, errors, utils
from ..compat import six, Enum
from .volumes import Volume
from .cache import cache_parent


class VolumeFile(serializers.XMLSerializableModel):
    _root = 'VolumeFileModel'

    name = serializers.XMLNodeField('Name')

    @property
    def partition(self):
        return self.parent.parent

    @property
    def project(self):
        return self.partition.project

    @property
    def volume(self):
        return self.partition.volume

    def open_reader(self, **kw):
        return self.partition.open_reader(self.name, **kw)

    @property
    def path(self):
        return '/'.join((self.project.name, 'volumes', self.volume.name, self.partition.name, self.name))


class VolumePartition(LazyLoad):
    """
    Represents a partition in a volume.
    """
    __slots__ = '_volume_tunnel', '_download_id', '_upload_id'

    class Type(Enum):
        NEW = 'NEW'
        OLD = 'OLD'

    _root = 'Meta'
    _type_indicator = 'type'

    name = serializers.XMLNodeField('Name')
    owner = serializers.XMLNodeField('Owner')
    type = serializers.XMLNodeField('Type', parse_callback=lambda t: VolumePartition.Type(t.upper()))
    comment = serializers.XMLNodeField('Comment')
    length = serializers.XMLNodeField('Length', parse_callback=int)
    file_number = serializers.XMLNodeField('FileNumber', parse_callback=int)
    creation_time = serializers.XMLNodeField('CreationTime', parse_callback=utils.parse_rfc822)
    last_modified_time = serializers.XMLNodeField('LastModifiedTime', parse_callback=utils.parse_rfc822)

    def __init__(self, **kwargs):
        super(VolumePartition, self).__init__(**kwargs)
        self._volume_tunnel = None
        self._download_id = None
        self._upload_id = None

    def reload(self):
        resp = self._client.get(self.resource())
        self.parse(self._client, resp, obj=self)

    @property
    def project(self):
        return self.parent.parent.parent

    @property
    def volume(self):
        return self.parent

    class VolumeFiles(serializers.XMLSerializableModel):
        _root = 'Items'
        skip_null = False

        marker = serializers.XMLNodeField('Marker')
        files = serializers.XMLNodesReferencesField(VolumeFile, 'Item')
        max_items = serializers.XMLNodeField('MaxItems', parse_callback=int)

        def __getitem__(self, item):
            for f in self.iterate(name=item):
                if f.name == item:
                    return f
            raise KeyError

        def get(self, item, default=None):
            try:
                return self[item]
            except KeyError:
                return default

        def __iter__(self):
            return self.iterate()

        def iterate(self, name=None, max_items=None):
            """
            :param name: the prefix of volume name name
            :return:
            """
            params = {'expectmarker': 'true', 'path': ''}
            if name is not None:
                params['name'] = name
            if max_items is not None:
                params['maxitems'] = max_items

            def _it():
                last_marker = params.get('marker')
                if 'marker' in params and \
                        (last_marker is None or len(last_marker) == 0):
                    return

                url = self.parent.resource()
                resp = self.parent._client.get(url, params=params)

                v = self.parse(resp, obj=self, parent=self.parent)
                params['marker'] = v.marker

                return v.files

            while True:
                files = _it()
                if files is None:
                    break
                for f in files:
                    yield f

    @property
    def files(self):
        return self.VolumeFiles(_parent=self)

    def list_files(self):
        """
        List files in the partition.

        :return: files
        :rtype: list
        """
        return self.files.iterate()

    def _create_volume_tunnel(self, endpoint=None):
        if self._volume_tunnel is not None:
            return self._volume_tunnel

        from ..tunnel import VolumeTunnel

        self._volume_tunnel = VolumeTunnel(client=self._client, project=self.project,
                                           endpoint=endpoint or self.project._tunnel_endpoint)
        return self._volume_tunnel

    def open_reader(self, file_name, reopen=False, endpoint=None, start=None, length=None, **kwargs):
        """
        Open a volume file for read. A file-like object will be returned which can be used to read contents from
        volume files.

        :param str file_name: name of the file
        :param bool reopen: whether we need to open an existing read session
        :param str endpoint: tunnel service URL
        :param start: start position
        :param length: length limit
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`

        :Example:
        >>> with partition.open_reader('file') as reader:
        >>>     [print(line) for line in reader]
        """
        tunnel = self._create_volume_tunnel(endpoint=endpoint)
        download_id = self._download_id if not reopen else None
        download_session = tunnel.create_download_session(volume=self.volume.name, partition_spec=self.name,
                                                          file_name=file_name, download_id=download_id, **kwargs)
        self._download_id = download_session.id

        open_args = {}
        if start is not None:
            open_args['start'] = start
        if length is not None:
            open_args['length'] = length
        return download_session.open(**open_args)

    def open_writer(self, reopen=False, endpoint=None, **kwargs):
        """
        Open a volume partition to write to. You can use `open` method to open a file inside the volume and write to it,
        or use `write` method to write to specific files.

        :param bool reopen: whether we need to open an existing write session
        :param str endpoint: tunnel service URL
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :Example:
        >>> with partition.open_writer() as writer:
        >>>     writer.open('file1').write('some content')
        >>>     writer.write('file2', 'some content')
        """
        tunnel = self._create_volume_tunnel(endpoint=endpoint)
        upload_id = self._upload_id if not reopen else None
        upload_session = tunnel.create_upload_session(volume=self.volume.name, partition_spec=self.name,
                                                      upload_id=upload_id, **kwargs)
        file_dict = dict()

        class FilesWriter(object):
            @property
            def status(self):
                return upload_session.status

            @staticmethod
            def open(file_name, **kwargs):
                if file_name in file_dict:
                    return file_dict[file_name]
                writer = upload_session.open(file_name, **kwargs)
                file_dict[file_name] = writer
                return writer

            @staticmethod
            def write(file_name, buf, **kwargs):
                writer = FilesWriter.open(file_name, **kwargs)
                writer.write(buf)

            @staticmethod
            def close():
                for w in six.itervalues(file_dict):
                    w.close()

                upload_session.commit(list(six.iterkeys(file_dict)))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.close()

        return FilesWriter()


@cache_parent
class PartedVolume(Volume):
    """
    PartedVolume represents the old-fashioned partitioned volume in ODPS.
    """
    class Partitions(Iterable):
        _root = 'Volume'

        marker = serializers.XMLNodeField('Marker')
        partitions = serializers.XMLNodesReferencesField(VolumePartition, 'Partitions', 'Partition')
        max_items = serializers.XMLNodeField('MaxItems', parse_callback=int)

        def _get(self, item):
            return VolumePartition(client=self._client, parent=self.parent, name=item)

        def __contains__(self, item):
            if isinstance(item, six.string_types):
                partition = self._get(item)
            elif isinstance(item, VolumePartition):
                partition = item
            else:
                return False

            try:
                partition.reload()
                return True
            except errors.NoSuchObject:
                return False

        def __iter__(self):
            return self.iterate()

        def iterate(self, name=None, owner=None):
            """
            :param name: the prefix of volume name name
            :param owner:
            :return:
            """
            params = {'expectmarker': 'true'}
            if name is not None:
                params['name'] = name
            if owner is not None:
                params['owner'] = owner

            def _it():
                last_marker = params.get('marker')
                if 'marker' in params and \
                        (last_marker is None or len(last_marker) == 0):
                    return

                url = self.parent.resource()
                resp = self._client.get(url, params=params)

                v = PartedVolume.Partitions.parse(self._client, resp, obj=self)
                params['marker'] = v.marker

                return v.partitions

            while True:
                partitions = _it()
                if partitions is None:
                    break
                for partition in partitions:
                    yield partition

        def delete(self, name):
            if not isinstance(name, Volume):
                partition = self._get(name)
            else:
                partition = name

            url = partition.resource()
            del self[name]
            self._client.delete(url)

    @property
    def partitions(self):
        return PartedVolume.Partitions(parent=self, client=self._client)

    def list_partitions(self, owner=None):
        """
        List partitions.

        :return: partitions
        :rtype: list
        """
        return self.partitions.iterate(owner=owner)

    def exist_partition(self, name):
        """
        If the volume with given name exists in a partition or not.

        :param str name: partition name
        """
        return name in self.partitions

    def get_partition(self, name):
        """
        Get partition from volume by name

        :param str name: partition name
        :return: partition
        :rtype: :class:`odps.models.VolumePartition`
        """
        return self.partitions[name]

    def delete_partition(self, name):
        """
        Delete partition by given name

        :param str name: partition name
        """
        return self.partitions.delete(name)

    def open_reader(self, partition, file_name, endpoint=None, start=None, length=None, **kwargs):
        """
        Open a volume file for read. A file-like object will be returned which can be used to read contents from
        volume files.

        :param str partition: name of the partition
        :param str file_name: name of the file
        :param str endpoint: tunnel service URL
        :param start: start position
        :param length: length limit
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`

        :Example:
        >>> with volume.open_reader('part', 'file') as reader:
        >>>     [print(line) for line in reader]
        """
        return self.partitions[partition].open_reader(file_name, endpoint=endpoint, start=start, length=length,
                                                      **kwargs)

    def open_writer(self, partition, endpoint=None, **kwargs):
        """
        Open a volume partition to write to. You can use `open` method to open a file inside the volume and write to it,
        or use `write` method to write to specific files.

        :param str partition: name of the partition
        :param str endpoint: tunnel service URL
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :Example:
        >>> with volume.open_writer() as writer:
        >>>     writer.open('file1').write('some content')
        >>>     writer.write('file2', 'some content')
        """
        return self.partitions[partition].open_writer(endpoint=endpoint, **kwargs)
