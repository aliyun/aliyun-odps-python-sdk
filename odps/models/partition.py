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

from datetime import datetime

from .core import LazyLoad, XMLRemoteModel, JSONRemoteModel
from .. import serializers, types


class Partition(LazyLoad):
    """
    A partition is a collection of rows in a table whose partition columns are equal to specific
    values.

    In order to write data into partition, users should call the ``open_writer``
    method with **with statement**. At the same time, the ``open_reader`` method is used
    to provide the ability to read records from a partition. The behavior of these
    methods are the same as those in Table class except that there are no 'partition' params.
    """
    __slots__ = 'spec', 'creation_time', 'last_meta_modified_time', 'last_modified_time', \
                'size', '_is_extend_info_loaded', \
                'is_archived', 'is_exstore', 'lifecycle', \
                'physical_size', 'file_num', 'reserved'

    class Column(XMLRemoteModel):

        name = serializers.XMLNodeAttributeField(attr='Name')
        value = serializers.XMLNodeAttributeField(attr='Value')

    class PartitionMeta(JSONRemoteModel):

        creation_time = serializers.JSONNodeField(
            'createTime', parse_callback=datetime.fromtimestamp, set_to_parent=True)
        last_meta_modified_time = serializers.JSONNodeField(
            'lastDDLTime', parse_callback=datetime.fromtimestamp, set_to_parent=True)
        last_modified_time = serializers.JSONNodeField(
            'lastModifiedTime', parse_callback=datetime.fromtimestamp, set_to_parent=True)
        size = serializers.JSONNodeField(
            'partitionSize', parse_callback=int, set_to_parent=True)

    class PartitionExtendedMeta(PartitionMeta):

        is_archived = serializers.JSONNodeField(
            'IsArchived', parse_callback=bool, set_to_parent=True)
        is_exstore = serializers.JSONNodeField(
            'IsExstore', parse_callback=bool, set_to_parent=True)
        lifecycle = serializers.JSONNodeField(
            'LifeCycle', parse_callback=int, set_to_parent=True)
        physical_size = serializers.JSONNodeField(
            'PhysicalSize', parse_callback=int, set_to_parent=True)
        file_num = serializers.JSONNodeField(
            'FileNum', parse_callback=int, set_to_parent=True)
        reserved = serializers.JSONNodeField(
            'Reserved', type='json', set_to_parent=True)

    columns = serializers.XMLNodesReferencesField(Column, 'Column')
    _schema = serializers.XMLNodeReferenceField(PartitionMeta, 'Schema')
    _extended_schema = serializers.XMLNodeReferenceField(PartitionExtendedMeta, 'Schema')

    def __init__(self, **kwargs):
        self._is_extend_info_loaded = False

        super(Partition, self).__init__(**kwargs)

    def __str__(self):
        return str(self.partition_spec)

    def __repr__(self):
        return '<Partition %s.`%s`(%s)>' % (
            str(self.table.project.name), str(self.table.name), str(self.partition_spec))

    def __getattribute__(self, attr):
        if attr in ('is_archived', 'is_exstore', 'lifecycle',
                    'physical_size', 'file_num', 'reserved'):
            if not self._is_extend_info_loaded:
                self.reload_extend_info()

            return object.__getattribute__(self, attr)

        val = object.__getattribute__(self, attr)
        if val is None and not self._loaded:
            if attr in getattr(Partition.PartitionMeta, '__fields'):
                self.reload()
                return object.__getattribute__(self, attr)

        return super(Partition, self).__getattribute__(attr)

    def _set_state(self, name, parent, client):
        self.__init__(spec=name, _parent=parent, _client=client)

    def _name(self):
        return

    @classmethod
    def get_partition_spec(cls, columns=None, spec=None):
        if spec is not None:
            return spec

        spec = types.PartitionSpec()
        for col in columns:
            spec[col.name] = col.value

        return spec

    @property
    def partition_spec(self):
        return self.get_partition_spec(self._getattr('columns'), self._getattr('spec'))

    @property
    def name(self):
        return str(self.partition_spec)

    @property
    def table(self):
        return self.parent.parent

    @property
    def project(self):
        return self.table.project

    def reload(self):
        url = self.resource()
        params = {'partition': str(self.partition_spec)}
        resp = self._client.get(url, params=params)

        self.parse(self._client, resp, obj=self)

        self._loaded = True

    def reload_extend_info(self):
        url = self.resource()
        params = {'extended': '',
                  'partition': str(self.partition_spec)}
        resp = self._client.get(url, params=params)

        self.parse(self._client, resp, obj=self)
        self._is_extend_info_loaded = True

    def to_df(self):
        """
        Create a PyODPS DataFrame from this partition.

        :return: DataFrame object
        """
        from ..df import DataFrame

        return DataFrame(self.table).filter_parts(self)

    def drop(self, async_=False, if_exists=False, **kw):
        """
        Drop this partition.

        :param async_: run asynchronously if True
        :return: None
        """
        async_ = kw.get('async', async_)
        return self.parent.delete(self, if_exists=if_exists, async_=async_)

    def open_reader(self, **kw):
        """
        Open the reader to read the entire records from this partition.

        :param reopen: the reader will reuse last one, reopen is true means open a new reader.
        :type reopen: bool
        :param endpoint: the tunnel service URL
        :param compress_option: compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param compress_algo: compression algorithm, work when ``compress_option`` is not provided,
                              can be ``zlib``, ``snappy``
        :param compress_level: used for ``zlib``, work when ``compress_option`` is not provided
        :param compress_strategy: used for ``zlib``, work when ``compress_option`` is not provided
        :return: reader, ``count`` means the full size, ``status`` means the tunnel status

        :Example:

        >>> with partition.open_reader() as reader:
        >>>     count = reader.count  # How many records of a partition
        >>>     for record in reader[0: count]:
        >>>         # read all data, actually better to split into reading for many times
        """
        return self.table.open_reader(str(self), **kw)

    def open_writer(self, blocks=None, **kw):
        return self.table.open_writer(str(self), blocks=blocks, **kw)
