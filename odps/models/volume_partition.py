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

import contextlib

from .core import LazyLoad
from .. import serializers, utils
from ..config import options
from ..compat import Enum, six


class VolumeFile(serializers.XMLSerializableModel):
    _root = 'VolumeFileModel'

    name = serializers.XMLNodeField('Name')


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
        __slots__ = ['partition']

        marker = serializers.XMLNodeField('Marker')
        files = serializers.XMLNodesReferencesField(VolumeFile, 'Item')
        max_items = serializers.XMLNodeField('MaxItems', parse_callback=int)

        def __init__(self, partition, **kwargs):
            self.partition = partition
            super(VolumePartition.VolumeFiles, self).__init__(**kwargs)

        def __iter__(self):
            return self.iterate()

        def iterate(self, name=None):
            """

            :param name: the prefix of volume name name
            :param owner:
            :return:
            """

            params = {'expectmarker': 'true', 'path': ''}
            if name is not None:
                params['name'] = name

            def _it():
                last_marker = params.get('marker')
                if 'marker' in params and \
                        (last_marker is None or len(last_marker) == 0):
                    return

                url = self.partition.resource()
                resp = self.partition._client.get(url, params=params)

                v = self.parse(resp, obj=self)
                params['marker'] = v.marker

                return v.files

            while True:
                files = _it()
                if not files:
                    break
                for f in files:
                    yield f

    @property
    def files(self):
        return self.VolumeFiles(self)

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
                                           endpoint=endpoint or options.tunnel_endpoint)
        return self._volume_tunnel

    @contextlib.contextmanager
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
        yield download_session.open(**open_args)

    @contextlib.contextmanager
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

        yield FilesWriter()

        for w in six.itervalues(file_dict):
            w.close()

        upload_session.commit(list(six.iterkeys(file_dict)))
