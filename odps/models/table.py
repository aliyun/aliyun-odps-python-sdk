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

from .core import LazyLoad, JSONRemoteModel
from .partitions import Partitions
from ..config import options
from .. import types, serializers, utils, readers
from ..compat import six


class TableSchema(types.OdpsSchema, JSONRemoteModel):
    """
    Schema includes the columns and partitions information of a :class:`odps.models.Table`.

    There are two ways to initialize a Schema object, first is to provide columns and partitions,
    the second way is to call the class method ``from_lists``. See the examples below:

    :Example:

    >>> columns = [Column(name='num', type='bigint', comment='the column')]
    >>> partitions = [Partition(name='pt', type='string', comment='the partition')]
    >>> schema = Schema(columns=columns, partitions=partitions)
    >>> schema.columns
    [<column num, type bigint>, <partition pt, type string>]
    >>>
    >>> schema = Schema.from_lists(['num'], ['bigint'], ['pt'], ['string'])
    >>> schema.columns
    [<column num, type bigint>, <partition pt, type string>]
    """

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
            if self.type is not None:
                self.type = types.validate_data_type(self.type)

    class TablePartition(types.Partition, TableColumn):
        def __init__(self, **kwargs):
            TableSchema.TableColumn.__init__(self, **kwargs)

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

    def __getstate__(self):
        return self._columns, self._partitions

    def __setstate__(self, state):
        columns, partitions = state
        self.__init__(_columns=columns, _partitions=partitions)


class Table(LazyLoad):
    """
    Table means the same to the RDBMS table, besides, a table can consist of partitions.

    Table's properties are the same to the ones of :class:`odps.models.Project`,
    which will not load from remote ODPS service until users try to get them.

    In order to write data into table, users should call the ``open_writer``
    method with **with statement**. At the same time, the ``open_reader`` method is used
    to provide the ability to read records from a table or its partition.

    :Example:

    >>> table = odps.get_table('my_table')
    >>> table.owner  # first will load from remote
    >>> table.reload()  # reload to update the properties
    >>>
    >>> for record in table.head(5):
    >>>     # check the first 5 records
    >>> for record in table.head(5, partition='pt=test', columns=['my_column'])
    >>>     # only check the `my_column` column from certain partition of this table
    >>>
    >>> with table.open_reader() as reader:
    >>>     count = reader.count  # How many records of a table or its partition
    >>>     for record in record[0: count]:
    >>>         # read all data, actually better to split into reading for many times
    >>>
    >>> with table.open_writer() as writer:
    >>>     writer.write(records)
    >>> with table.open_writer(partition='pt=test', blocks=[0, 1]):
    >>>     writer.write(0, gen_records(block=0))
    >>>     writer.write(1, gen_records(block=1))  # we can do this parallel
    """

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
            if attr in getattr(TableSchema, '__fields'):
                self.reload()
                return object.__getattribute__(self, attr)

        return super(Table, self).__getattribute__(attr)

    def __repr__(self):
        buf = six.StringIO()

        buf.write('odps.Table\n')
        buf.write('  name: {0}.`{1}`\n'.format(self.project.name, self.name))

        name_space = 2 * max(len(col.name) for col in self.schema.columns)
        type_space = 2 * max(len(repr(col.type)) for col in self.schema.columns)

        not_empty = lambda field: field is not None and len(field.strip()) > 0

        buf.write('  schema:\n')
        cols_strs = []
        for col in self.schema._columns:
            cols_strs.append('{0}: {1}{2}'.format(
                col.name.ljust(name_space),
                repr(col.type).ljust(type_space),
                '# {0}'.format(col.comment) if not_empty(col.comment) else ''
            ))
        buf.write(utils.indent('\n'.join(cols_strs), 4))
        buf.write('\n')

        if self.schema._partitions:
            buf.write('  partitions:\n')

            partition_strs = []
            for partition in self.schema._partitions:
                partition_strs.append('{0}: {1}{2}'.format(
                    partition.name.ljust(name_space),
                    repr(partition.type).ljust(type_space),
                    '# {0}'.format(partition.comment) if not_empty(partition.comment) else ''
                ))
            buf.write(utils.indent('\n'.join(partition_strs), 4))

        return buf.getvalue()

    def head(self, limit, partition=None, columns=None):
        """
        Get the head records of a table or its partition.

        :param limit: records' size, 10000 at most
        :param partition: partition of this table
        :param columns: the columns which is subset of the table columns
        :type columns: list
        :return: records
        :rtype: list

        .. seealso:: :class:`odps.models.Record`
        """

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
        with readers.RecordReader(self.schema, resp) as reader:
            for record in reader:
                yield record

    def _create_table_tunnel(self, endpoint=None):
        if self._table_tunnel is not None:
            return self._table_tunnel

        from ..tunnel import TableTunnel

        self._table_tunnel = TableTunnel(client=self._client, project=self.project,
                                         endpoint=endpoint or options.tunnel_endpoint)
        return self._table_tunnel

    @contextlib.contextmanager
    def open_reader(self, partition=None, **kw):
        """
        Open the reader to read the entire records from this table or its partition.

        :param partition: partition of this table
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

        >>> with table.open_reader() as reader:
        >>>     count = reader.count  # How many records of a table or its partition
        >>>     for record in record[0: count]:
        >>>         # read all data, actually better to split into reading for many times
        """

        reopen = kw.pop('reopen', False)
        endpoint = kw.pop('endpoint', None)

        tunnel = self._create_table_tunnel(endpoint=endpoint)
        download_id = self._download_id if not reopen else None
        download_session = tunnel.create_download_session(table=self, partition_spec=partition,
                                                          download_id=download_id, **kw)
        self._download_id = download_session.id

        class RecordReader(readers.AbstractRecordReader):
            def __init__(self):
                self._it = iter(self)

            @property
            def count(self):
                return download_session.count

            @property
            def status(self):
                return download_session.status

            def __iter__(self):
                for record in self.read():
                    yield record

            def __next__(self):
                return next(self._it)

            next = __next__

            def _iter(self, start=None, end=None, step=None):
                count = self._calc_count(start, end, step)
                return self.read(start=start, count=count, step=step)

            def read(self, start=None, count=None, step=None,
                     compress=False, columns=None):
                start = start or 0
                step = step or 1
                count = count*step if count is not None else self.count-start

                with download_session.open_record_reader(
                        start, count, compress=compress, columns=columns) as reader:
                    for record in reader[::step]:
                        yield record

        yield RecordReader()

    @contextlib.contextmanager
    def open_writer(self, partition=None, blocks=None, **kw):
        """
        Open the writer to write records into this table or its partition.

        :param partition: partition of this table
        :param blocks: block ids to open
        :param reopen: the reader will reuse last one, reopen is true means open a new reader.
        :type reopen: bool
        :param endpoint: the tunnel service URL
        :param compress_option: compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param compress_algo: compression algorithm, work when ``compress_option`` is not provided,
                              can be ``zlib``, ``snappy``
        :param compress_level: used for ``zlib``, work when ``compress_option`` is not provided
        :param compress_strategy: used for ``zlib``, work when ``compress_option`` is not provided
        :return: writer, status means the tunnel writer status

        :Example:

        >>> with table.open_writer() as writer:
        >>>     writer.write(records)
        >>> with table.open_writer(partition='pt=test', blocks=[0, 1]):
        >>>     writer.write(0, gen_records(block=0))
        >>>     writer.write(1, gen_records(block=1))  # we can do this parallel
        """

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
                    if isinstance(arg, types.Record):
                        records = [arg, ]
                    else:
                        records = arg
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

    def create_partition(self, partition_spec, if_not_exists=False, async=False):
        return self.partitions.create(partition_spec, if_not_exists=if_not_exists, async=async)

    def delete_partition(self, partition_spec, if_exists=False, async=False):
        return self.partitions.delete(partition_spec, if_exists=if_exists, async=async)

    def exist_partition(self, partition_spec):
        return partition_spec in self.partitions

    def iterate_partitions(self, spec=None):
        return self.partitions.iterate_partitions(spec=spec)

    def get_partition(self, partition_spec):
        return self.partitions[partition_spec]

    def drop(self, async=False):
        """
        drop this table.

        :param async: run asynchronously if True
        :return: None
        """
        return self.parent.delete(self, async=async)

    def new_record(self, values=None):
        """
        Generate a record of the table.

        :param values: the values of this records
        :type values: list
        :return: record
        :rtype: :class:`odps.models.Record`

        :Example:

        >>> table = odps.create_table('test_table', schema=Schema.from_lists(['name', 'id'], ['sring', 'string']))
        >>> record = table.new_record()
        >>> record[0] = 'my_name'
        >>> record[1] = 'my_id'
        >>> record = table.new_record(['my_name', 'my_id'])

        .. seealso:: :class:`odps.models.Record`
        """
        return types.Record(schema=self.schema, values=values)
