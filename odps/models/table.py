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

from datetime import datetime
import contextlib
import collections

import six

from .core import LazyLoad, JSONRemoteModel
from .partitions import Partitions
from ..config import options
from .. import types, serializers, utils, readers


class TableSchema(types.OdpsSchema, JSONRemoteModel):

    class Shard(JSONRemoteModel):

        hub_lifecycle = serializers.JSONNodeField('HubLifecycle')
        shard_num = serializers.JSONNodeField('ShardNum')
        distribute_cols = serializers.JSONNodeField('DistributeCols')
        sort_cols = serializers.JSONNodeField('SortCols')

    class TableColumn(types.Column, JSONRemoteModel):
        name = serializers.JSONNodeField('name')
        type = serializers.JSONNodeField('type', parse_callback=types.validate_data_type)
        comment = serializers.JSONNodeField('comment')
        label = serializers.JSONNodeField('label')

        def __init__(self, **kwargs):
            JSONRemoteModel.__init__(self, **kwargs)

    class TablePartition(types.Partition, TableColumn):
        def __init__(self, **kwargs):
            JSONRemoteModel.__init__(self, **kwargs)

    def __init__(self, **kwargs):
        kwargs['_columns'] = columns = kwargs.pop('columns', None)
        kwargs['_partitions'] = partitions = kwargs.pop('partitions', None)
        JSONRemoteModel.__init__(self, **kwargs)
        types.OdpsSchema.__init__(self, columns=columns, partitions=partitions)

    def load(self):
        self.update(self._columns, self._partitions)

    comment = serializers.JSONNodeField('comment', set_to_parent=True),
    owner = serializers.JSONNodeField('owner', set_to_parent=True),
    creation_time = serializers.JSONNodeField(
        'createTime', parse_callback=datetime.fromtimestamp,
        set_to_parent=True)
    last_modified_time = serializers.JSONNodeField(
        'lastModifiedTime', parse_callback=datetime.fromtimestamp,
        set_to_parent=True)
    last_meta_modified_time = serializers.JSONNodeField(
        'lastDDLTime', parse_callback=datetime.fromtimestamp,
        set_to_parent=True)
    is_virtual_view = serializers.JSONNodeField(
        'isVirtualView', parse_callback=bool, set_to_parent=True)
    lifecycle = serializers.JSONNodeField(
        'lifecycle', parse_callback=int, set_to_parent=True)
    view_text = serializers.JSONNodeField('viewText', set_to_parent=True)
    size = serializers.JSONNodeField("size", parse_callback=int, set_to_parent=True)
    is_archived = serializers.JSONNodeField(
        'IsArchived', parse_callback=bool, set_to_parent=True)
    physical_size = serializers.JSONNodeField(
        'PhysicalSize', parse_callback=int, set_to_parent=True)
    file_num = serializers.JSONNodeField(
        'FileNum', parse_callback=int, set_to_parent=True)
    shard = serializers.JSONNodeReferenceField(
        Shard, 'shardInfo', check_before=['shardExist'], set_to_parent=True)
    table_label = serializers.JSONNodeField(
        'tableLabel', callback=lambda t: t if t != '0' else '', set_to_parent=True)
    _columns = serializers.JSONNodesReferencesField(TableColumn, 'columns')
    _partitions = serializers.JSONNodesReferencesField(TablePartition, 'partitionKeys')


class Table(LazyLoad):
    __slots__ = '_is_extend_info_loaded', 'last_meta_modified_time', 'is_virtual_view', \
                'lifecycle', 'view_text', 'size', \
                'is_archived', 'physical_size', 'file_num', 'shard', \
                '_table_tunnel', '_download_id', '_upload_id'

    name = serializers.XMLNodeField('Name')
    table_id = serializers.XMLNodeField('TableId')
    format = serializers.XMLNodeAttributeField(attr='format')
    schema = serializers.XMLNodeReferenceField(TableSchema, 'Schema')
    comment = serializers.XMLNodeField('Comment')
    owner = serializers.XMLNodeField('Owner')
    table_label = serializers.XMLNodeField('TableLabel')
    creation_time = serializers.XMLNodeField('CreationTime',
                                             parse_callback=utils.parse_rfc822)
    last_modified_time = serializers.XMLNodeField('LastModifiedTime',
                                                  parse_callback=utils.parse_rfc822)

    def __init__(self, **kwargs):
        self._is_extend_info_loaded = False

        super(Table, self).__init__(**kwargs)

    def reload(self):
        url = self.resource()
        resp = self._client.get(url)

        self.parse(self._client, resp, obj=self)
        self.schema.load()
        self._loaded = True

    def reload_extend_info(self):
        params = {'extended': ''}
        resp = self._client.get(self.resource(), params=params)

        self.parse(self._client, resp, obj=self)
        self._is_extend_info_loaded = True

        if not self._loaded:
            self.schema = None

    def __getattribute__(self, attr):
        if attr in ('is_archived', 'physical_size', 'file_num'):
            if not self._is_extend_info_loaded:
                self.reload_extend_info()
                return object.__getattribute__(self, attr)

        val = object.__getattribute__(self, attr)
        if val is None and not self._loaded:
            try:
                schema = object.__getattribute__(self, 'schema')
            except AttributeError:
                schema = None

            if schema is None:
                return super(Table, self).__getattribute__(attr)

            if attr in getattr(schema, '__fields'):
                self.reload()
                return object.__getattribute__(self, attr)

        return super(Table, self).__getattribute__(attr)

    def read(self, limit, partition=None, columns=None):
        if limit <= 0:
            raise ValueError('limit number should >= 0.')

        params = {'data': '', 'linenum': limit}
        if partition is not None:
            if not isinstance(partition, types.PartitionSpec):
                partition = types.PartitionSpec(partition)
            params['partition'] = str(partition)
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, types.Column) else col
            params['cols'] = ','.join(col_name(col) for col in columns)

        resp = self._client.get(self.resource(), params=params, stream=True)
        return readers.RecordReader(self.schema, resp)

    def _create_table_tunnel(self, endpoint=None):
        if self._table_tunnel is not None:
            return self._table_tunnel

        from ..tunnel import TableTunnel

        self._table_tunnel = TableTunnel(client=self._client, project=self.project,
                                         endpoint=endpoint or options.tunnel_endpoint)
        return self._table_tunnel

    @contextlib.contextmanager
    def open_reader(self, partition=None, **kw):
        reopen = kw.pop('reopen', False)
        endpoint = kw.pop('endpoint', None)

        tunnel = self._create_table_tunnel(endpoint=endpoint)
        download_id = self._download_id if not reopen else None
        download_session = tunnel.create_download_session(table=self, partition_spec=partition,
                                                          download_id=download_id, **kw)
        self._download_id = download_session.id

        class RecordReader(object):
            @property
            def count(self):
                return download_session.count

            @property
            def status(self):
                return download_session.status

            def read(self, start=None, count=None, compress=False):
                start = start or 0
                count = count or self.count-start

                for record in download_session.open_record_reader(
                        start, count, compress=compress):
                    yield record

        yield RecordReader()

    @contextlib.contextmanager
    def open_writer(self, partition=None, blocks=None, **kw):
        reopen = kw.pop('reopen', False)
        commit = kw.pop('commit', True)
        endpoint = kw.pop('endpoint', None)

        tunnel = self._create_table_tunnel(endpoint=endpoint)
        upload_id = self._upload_id if not reopen else None
        upload_session = tunnel.create_upload_session(table=self, partition_spec=partition,
                                                      upload_id=upload_id, **kw)
        self._upload_id = upload_session.id

        blocks = blocks or [0, ]
        blocks_writes = [False] * len(blocks)

        class RecordWriter(object):
            @property
            def status(self):
                return upload_session.status

            @classmethod
            def write(cls, *args, **kwargs):
                block_id = kwargs.get('block_id')
                if block_id is None:
                    if isinstance(args[0], six.integer_types):
                        block_id = args[0]
                        args = args[1:]
                    else:
                        block_id = 0

                if len(args) == 1:
                    arg = args[0]
                    if isinstance(arg, collections.Iterable):
                        records = arg
                    else:
                        records = [arg, ]
                elif len(args) > 1:
                    records = args
                else:
                    raise ValueError('Cannot write no records to table.')

                compress = kwargs.get('compress', False)
                with upload_session.open_record_writer(block_id, compress=compress) as w:
                    for record in records:
                        w.write(record)
                    blocks_writes[blocks.index(block_id)] = True

        yield RecordWriter()

        if commit:
            blocks = [block for block, block_write in zip(blocks, blocks_writes) if block_write]
            upload_session.commit(blocks)
            self._upload_id = None

    @property
    def project(self):
        return self.parent.parent

    @property
    def partitions(self):
        return Partitions(parent=self, client=self._client)

    def create_partition(self, partition_spec, if_not_exists=False):
        return self.partitions.create(partition_spec, if_not_exists=if_not_exists)

    def delete_partition(self, partition_spec, if_exists=False):
        return self.partitions.delete(partition_spec, if_exists=if_exists)

    def exist_partition(self, partition_spec):
        return partition_spec in self.partitions

    def iterate_partitions(self, spec=None):
        return self.partitions.iterate_partitions(spec=spec)

    def get_partition(self, partition_spec):
        return self.partitions[partition_spec]

    def drop(self):
        return self.parent.delete(self)

    def new_record(self, values=None):
        return types.Record(schema=self.schema, values=values)
