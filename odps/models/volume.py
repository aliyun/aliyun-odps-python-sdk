#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from .core import LazyLoad
from .volume_partition import VolumePartition
from .. import serializers, utils, errors
from ..compat import Enum, six


class Volume(LazyLoad):
    """
    Volume is the file-accessing object provided by ODPS.
    """

    class Type(Enum):
        NEW = 'NEW'
        OLD = 'OLD'

    _root = 'Meta'
    _type_indicator = 'type'

    name = serializers.XMLNodeField('Name')
    owner = serializers.XMLNodeField('Owner')
    comment = serializers.XMLNodeField('Comment')
    type = serializers.XMLNodeField('Type', parse_callback=lambda t: Volume.Type(t.upper()))
    length = serializers.XMLNodeField('Length', parse_callback=int)
    file_number = serializers.XMLNodeField('FileNumber', parse_callback=int)
    creation_time = serializers.XMLNodeField('CreationTime', parse_callback=utils.parse_rfc822)
    last_modified_time = serializers.XMLNodeField('LastModifiedTime', parse_callback=utils.parse_rfc822)

    def __init__(self, **kwargs):
        typo = kwargs.get('type')
        if isinstance(typo, six.string_types):
            kwargs['type'] = Volume.Type(typo.upper())
        super(Volume, self).__init__(**kwargs)

    @property
    def project(self):
        return self.parent.parent

    def reload(self):
        resp = self._client.get(self.resource())
        self.parse(self._client, resp, obj=self)

    class Partitions(serializers.XMLSerializableModel):
        _root = 'Volume'
        __slots__ = ['volume']

        marker = serializers.XMLNodeField('Marker')
        partitions = serializers.XMLNodesReferencesField(VolumePartition, 'Partitions', 'Partition')
        max_items = serializers.XMLNodeField('MaxItems', parse_callback=int)

        def __init__(self, volume, **kwargs):
            self.volume = volume
            super(Volume.Partitions, self).__init__(**kwargs)

        def _get(self, item):
            return VolumePartition(client=self.volume._client, parent=self.volume, name=item)

        def __getitem__(self, item):
            if isinstance(item, six.string_types):
                return self._get(item)
            raise ValueError('Unsupported getitem value: %s' % item)

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

                url = self.volume.resource()
                resp = self.volume._client.get(url, params=params)

                v = Volume.Partitions.parse(resp, obj=self)
                params['marker'] = v.marker

                return v.partitions

            while True:
                partitions = _it()
                if not partitions:
                    break
                for partition in partitions:
                    yield partition

        def delete(self, name):
            if not isinstance(name, Volume):
                partition = VolumePartition(name=name, parent=self.volume)
            else:
                partition = name

            url = partition.resource()

            self.volume._client.delete(url)

    @property
    def partitions(self):
        return self.Partitions(self)

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
        >>> with volume.open_reader('file') as reader:
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
