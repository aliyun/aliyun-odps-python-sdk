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

import itertools
from datetime import datetime
from types import GeneratorType

try:
    import pyarrow as pa
except (AttributeError, ImportError):
    pa = None

from .. import types as odps_types, serializers, utils, readers
from ..compat import six, dir2
from .core import LazyLoad, JSONRemoteModel
from .readers import TunnelArrowReader, TunnelRecordReader
from .record import Record
from .partitions import Partitions


class TableSchema(odps_types.OdpsSchema, JSONRemoteModel):
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

    class TableColumn(odps_types.Column, JSONRemoteModel):
        name = serializers.JSONNodeField('name')
        type = serializers.JSONNodeField('type', parse_callback=odps_types.validate_data_type)
        comment = serializers.JSONNodeField('comment')
        label = serializers.JSONNodeField('label')

        def __init__(self, **kwargs):
            JSONRemoteModel.__init__(self, **kwargs)
            if self.type is not None:
                self.type = odps_types.validate_data_type(self.type)

    class TablePartition(odps_types.Partition, TableColumn):
        def __init__(self, **kwargs):
            TableSchema.TableColumn.__init__(self, **kwargs)

    def __init__(self, **kwargs):
        kwargs['_columns'] = columns = kwargs.pop('columns', None)
        kwargs['_partitions'] = partitions = kwargs.pop('partitions', None)
        JSONRemoteModel.__init__(self, **kwargs)
        odps_types.OdpsSchema.__init__(self, columns=columns, partitions=partitions)

    def load(self):
        self.update(self._columns, self._partitions)
        self.build_snapshot()

    comment = serializers.JSONNodeField('comment', set_to_parent=True)
    owner = serializers.JSONNodeField('owner', set_to_parent=True)
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
    record_num = serializers.JSONNodeField(
        'recordNum', parse_callback=int, set_to_parent=True)
    location = serializers.JSONNodeField(
        'location', set_to_parent=True)
    storage_handler = serializers.JSONNodeField(
        'storageHandler', set_to_parent=True)
    resources = serializers.JSONNodeField(
        'resources', set_to_parent=True)
    serde_properties = serializers.JSONNodeField(
        'serDeProperties', type='json', set_to_parent=True)
    reserved = serializers.JSONNodeField(
        'Reserved', type='json', set_to_parent=True)
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
        self.__init__(columns=columns, partitions=partitions)

    def __dir__(self):
        return sorted(set(dir2(self)) - set(type(self)._parent_attrs))


class TableRecordReader(TunnelRecordReader):
    def __init__(self, table, download_session, partition_spec=None, columns=None):
        super(TableRecordReader, self).__init__(
            table, download_session, columns=columns
        )
        self._partition_spec = partition_spec

    @property
    def schema(self):
        return self._parent.schema

    def _get_process_split_reader(self):
        rest_client = self._parent._client
        table_name = self._parent.name
        project = self._parent.project.name
        tunnel_endpoint = self._download_session._client.endpoint
        partition_spec = self._partition_spec

        def read_table_split(conn, download_id, start, count, idx):
            # read part data
            from ..tunnel import TableTunnel

            tunnel = TableTunnel(
                client=rest_client, project=project, endpoint=tunnel_endpoint
            )
            session = tunnel.create_download_session(
                table_name, download_id=download_id, partition_spec=partition_spec
            )
            data = session.open_record_reader(start, count).to_pandas()
            conn.send((idx, data))

        return read_table_split


class TableArrowReader(TunnelArrowReader):
    @property
    def schema(self):
        return self._parent.schema


class AbstractTableWriter(object):
    def __init__(
        self,
        table,
        upload_session,
        blocks=None,
        commit=True,
    ):
        self._table = table
        self._upload_session = upload_session
        self._commit = commit
        self._closed = False

        self._blocks = blocks or upload_session.blocks or [0, ]
        self._blocks_writes = [False] * len(self._blocks)
        self._blocks_writers = [None] * len(self._blocks)

        for block in (upload_session.blocks or ()):
            self._blocks_writes[self._blocks.index(block)] = True

    @property
    def upload_id(self):
        return self._upload_session.id

    @property
    def status(self):
        return self._upload_session.status

    def _open_writer(self, block_id, compress):
        raise NotImplementedError

    def _write_contents(self, writer, *args):
        raise NotImplementedError

    def write(self, *args, **kwargs):
        if self._closed:
            raise IOError('Cannot write to a closed writer.')

        block_id = kwargs.get('block_id')
        if block_id is None:
            if type(args[0]) in six.integer_types:
                block_id = args[0]
                args = args[1:]
            else:
                block_id = 0

        compress = kwargs.get('compress', False)
        idx = self._blocks.index(block_id)
        writer = self._blocks_writers[idx]
        if writer is None:
            writer = self._open_writer(block_id, compress)

        self._write_contents(writer, *args)
        self._blocks_writes[idx] = True

    def close(self):
        for writer in self._blocks_writers:
            if writer is not None:
                writer.close()

        if self._commit:
            written_blocks = [
                block
                for block, block_write in zip(self._blocks, self._blocks_writes)
                if block_write
            ]
            self._upload_session.commit(written_blocks)

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if an error occurs inside the with block, we do not commit
        if exc_val is not None:
            return
        self.close()


class TableRecordWriter(AbstractTableWriter):
    def _open_writer(self, block_id, compress):
        writer = self._upload_session.open_record_writer(
            block_id, compress=compress
        )
        self._blocks_writers[block_id] = writer
        return writer

    def _write_contents(self, writer, *args):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, Record):
                records = [arg, ]
            elif isinstance(arg, (list, tuple)):
                if isinstance(arg[0], Record):
                    records = arg
                elif isinstance(arg[0], (list, tuple)):
                    records = (self._table.new_record(vals) for vals in arg)
                else:
                    records = [self._table.new_record(arg), ]
            elif isinstance(arg, GeneratorType):
                try:
                    # peek the first element and then put back
                    next_arg = six.next(arg)
                    chained = itertools.chain((next_arg,), arg)
                    if isinstance(next_arg, Record):
                        records = chained
                    else:
                        records = (self._table.new_record(vals) for vals in chained)
                except StopIteration:
                    records = ()
            else:
                raise ValueError('Unsupported record type.')
        elif len(args) > 1:
            records = args
        else:
            raise ValueError('Cannot write to table without contents.')

        for record in records:
            writer.write(record)


class TableArrowWriter(AbstractTableWriter):
    def _open_writer(self, block_id, compress):
        writer = self._upload_session.open_arrow_writer(block_id)
        self._blocks_writers[block_id] = writer
        return writer

    def _write_contents(self, writer, *args):
        for arg in args:
            writer.write(arg)


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
    >>>     for record in reader[0: count]:
    >>>         # read all data, actually better to split into reading for many times
    >>>
    >>> with table.open_writer() as writer:
    >>>     writer.write(records)
    >>> with table.open_writer(partition='pt=test', blocks=[0, 1]):
    >>>     writer.write(0, gen_records(block=0))
    >>>     writer.write(1, gen_records(block=1))  # we can do this parallel
    """
    _extend_args = 'is_archived', 'physical_size', 'file_num', 'location', \
                   'storage_handler', 'resources', 'serde_properties', \
                   'reserved'
    __slots__ = '_is_extend_info_loaded', 'last_meta_modified_time', 'is_virtual_view', \
                'lifecycle', 'view_text', 'size', 'shard', '_table_tunnel', \
                '_id_thread_local', 'record_num'
    __slots__ += _extend_args

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

    _download_ids = utils.thread_local_attribute('_id_thread_local', dict)
    _upload_ids = utils.thread_local_attribute('_id_thread_local', dict)

    def __init__(self, **kwargs):
        self._is_extend_info_loaded = False
        if "table_schema" in kwargs:
            kwargs["schema"] = kwargs.pop("table_schema")

        super(Table, self).__init__(**kwargs)

        try:
            del self._id_thread_local
        except AttributeError:
            pass

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
        if attr in type(self)._extend_args:
            if not self._is_extend_info_loaded:
                self.reload_extend_info()
                return object.__getattribute__(self, attr)

        val = object.__getattribute__(self, attr)
        if val is None and not self._loaded:
            if attr in getattr(TableSchema, '__fields'):
                self.reload()
                return object.__getattribute__(self, attr)

        return super(Table, self).__getattribute__(attr)

    def _repr(self):
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
                '# {0}'.format(utils.to_str(col.comment)) if not_empty(col.comment) else ''
            ))
        buf.write(utils.indent('\n'.join(cols_strs), 4))
        buf.write('\n')

        if self.schema.partitions:
            buf.write('  partitions:\n')

            partition_strs = []
            for partition in self.schema.partitions:
                partition_strs.append('{0}: {1}{2}'.format(
                    partition.name.ljust(name_space),
                    repr(partition.type).ljust(type_space),
                    '# {0}'.format(utils.to_str(partition.comment)) if not_empty(partition.comment) else ''
                ))
            buf.write(utils.indent('\n'.join(partition_strs), 4))

        return buf.getvalue()

    @property
    def stored_as(self):
        return (self.reserved or dict()).get('StoredAs')

    @property
    def table_schema(self):
        return self.schema

    @staticmethod
    def gen_create_table_sql(table_name, table_schema, comment=None, if_not_exists=False,
                             lifecycle=None, shard_num=None, hub_lifecycle=None,
                             with_column_comments=True, project=None, **kw):
        from ..utils import escape_odps_string

        buf = six.StringIO()
        table_name = utils.to_text(table_name)
        project = utils.to_text(project)
        comment = utils.to_text(comment)

        stored_as = kw.get('stored_as')
        external_stored_as = kw.get('external_stored_as')
        storage_handler = kw.get('storage_handler')

        buf.write(u'CREATE%s TABLE ' % (' EXTERNAL' if storage_handler or external_stored_as else ''))
        if if_not_exists:
            buf.write(u'IF NOT EXISTS ')
        if project is not None:
            buf.write(u'%s.`%s` ' % (project, table_name))
        else:
            buf.write(u'`%s` ' % table_name)

        if isinstance(table_schema, six.string_types):
            buf.write(u'(\n')
            buf.write(table_schema)
            buf.write(u'\n)\n')
            if comment:
                buf.write(u"COMMENT '%s'\n" % escape_odps_string(comment))
        elif isinstance(table_schema, tuple):
            buf.write(u'(\n')
            buf.write(table_schema[0])
            buf.write(u'\n)\n')
            if comment:
                buf.write(u"COMMENT '%s'\n" % escape_odps_string(comment))
            buf.write(u'PARTITIONED BY ')
            buf.write(u'(\n')
            buf.write(table_schema[1])
            buf.write(u'\n)\n')
        else:
            def write_columns(col_array):
                size = len(col_array)
                buf.write(u'(\n')
                for idx, column in enumerate(col_array):
                    buf.write(u'  `%s` %s' % (utils.to_text(column.name), utils.to_text(column.type)))
                    if with_column_comments and column.comment:
                        buf.write(u" COMMENT '%s'" % utils.to_text(column.comment))
                    if idx < size - 1:
                        buf.write(u',\n')
                buf.write(u'\n)\n')

            write_columns(table_schema.simple_columns)
            if comment:
                buf.write(u"COMMENT '%s'\n" % comment)
            if table_schema.partitions:
                buf.write(u'PARTITIONED BY ')
                write_columns(table_schema.partitions)

        serde_properties = kw.get('serde_properties')
        location = kw.get('location')
        resources = kw.get('resources')
        if storage_handler or external_stored_as:
            if storage_handler:
                buf.write("STORED BY '%s'\n" % escape_odps_string(storage_handler))
            else:
                buf.write("STORED AS %s\n" % escape_odps_string(external_stored_as))
            if serde_properties:
                buf.write('WITH SERDEPROPERTIES (\n')
                for idx, k in enumerate(serde_properties):
                    buf.write("  '%s' = '%s'" % (escape_odps_string(k), escape_odps_string(serde_properties[k])))
                    if idx + 1 < len(serde_properties):
                        buf.write(',')
                    buf.write('\n')
                buf.write(')\n')
            if location:
                buf.write("LOCATION '%s'\n" % location)
            if resources:
                buf.write("USING '%s'\n" % resources)
        if stored_as:
            buf.write("STORED AS %s\n" % escape_odps_string(stored_as))
        if lifecycle is not None and lifecycle > 0:
            buf.write(u'LIFECYCLE %s\n' % lifecycle)
        if shard_num is not None:
            buf.write(u'INTO %s SHARDS' % shard_num)
            if hub_lifecycle is not None:
                buf.write(u' HUBLIFECYCLE %s\n' % hub_lifecycle)

        return buf.getvalue().strip()

    def get_ddl(self, with_comments=True, if_not_exists=False):
        """
        Get DDL SQL statement for the given table.

        :param with_comments: append comment for table and each column
        :return: DDL statement
        """
        shard_num = self.shard.shard_num if self.shard is not None else None
        return self.gen_create_table_sql(
            self.name, self.schema, self.comment if with_comments else None,
            if_not_exists=if_not_exists, with_column_comments=with_comments,
            lifecycle=self.lifecycle, shard_num=shard_num, project=self.project.name,
            storage_handler=self.storage_handler, serde_properties=self.serde_properties,
            location=self.location, resources=self.resources,
        )

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
            if not isinstance(partition, odps_types.PartitionSpec):
                partition = odps_types.PartitionSpec(partition)
            params['partition'] = str(partition)
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, odps_types.Column) else col
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
                                         endpoint=endpoint or self.project._tunnel_endpoint)
        return self._table_tunnel

    def open_reader(
        self,
        partition=None,
        reopen=False,
        endpoint=None,
        download_id=None,
        timeout=None,
        arrow=False,
        columns=None,
        **kw
    ):
        """
        Open the reader to read the entire records from this table or its partition.

        :param partition: partition of this table
        :param reopen: the reader will reuse last one, reopen is true means open a new reader.
        :type reopen: bool
        :param endpoint: the tunnel service URL
        :param download_id: use existing download_id to download table contents
        :param arrow: use arrow tunnel to read data
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
        >>>     for record in reader[0: count]:
        >>>         # read all data, actually better to split into reading for many times
        """

        from ..tunnel.tabletunnel import TableDownloadSession

        tunnel = self._create_table_tunnel(endpoint=endpoint)
        download_ids = dict()
        if download_id is None:
            download_ids = self._download_ids
            download_id = download_ids.get(partition) if not reopen else None
        download_session = tunnel.create_download_session(
            table=self, partition_spec=partition, download_id=download_id, timeout=timeout, **kw)

        if download_id and download_session.status != TableDownloadSession.Status.Normal:
            download_session = tunnel.create_download_session(
                table=self, partition_spec=partition, timeout=timeout, **kw)
        download_ids[partition] = download_session.id

        if arrow:
            return TableArrowReader(self, download_session, columns=columns)
        else:
            return TableRecordReader(self, download_session, partition, columns=columns)

    def open_writer(
        self,
        partition=None,
        blocks=None,
        reopen=False,
        create_partition=False,
        commit=True,
        endpoint=None,
        upload_id=None,
        arrow=False,
        **kw
    ):
        """
        Open the writer to write records into this table or its partition.

        :param partition: partition of this table
        :param blocks: block ids to open
        :param reopen: the reader will reuse last one, reopen is true means open a new reader.
        :type reopen: bool
        :param create_partition: if true, the partition will be created if not exist
        :type create_partition: bool
        :param endpoint: the tunnel service URL
        :param upload_id: use existing upload_id to upload data
        :param arrow: use arrow tunnel to write data
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

        from ..tunnel.tabletunnel import TableUploadSession

        if partition and not isinstance(partition, odps_types.PartitionSpec):
            partition = odps_types.PartitionSpec(partition)
        if create_partition and not self.exist_partition(create_partition):
            self.create_partition(partition, if_not_exists=True)

        tunnel = self._create_table_tunnel(endpoint=endpoint)
        if upload_id is None:
            upload_ids = self._upload_ids
            upload_id = upload_ids.get(partition) if not reopen else None
        else:
            upload_ids = dict()
        upload_session = tunnel.create_upload_session(
            table=self, partition_spec=partition, upload_id=upload_id, **kw
        )

        if upload_id and upload_session.status.value != TableUploadSession.Status.Normal.value:
            # check upload session status
            upload_session = tunnel.create_upload_session(table=self, partition_spec=partition, **kw)
            upload_id = None
        upload_ids[partition] = upload_session.id

        writer_cls = TableArrowWriter if arrow else TableRecordWriter

        class _RecordWriter(writer_cls):
            def close(self):
                super(_RecordWriter, self).close()
                if commit:
                    upload_ids[partition] = None

        return _RecordWriter(self, upload_session, blocks, commit)

    @property
    def project(self):
        return self.parent.parent

    @property
    def partitions(self):
        return Partitions(parent=self, client=self._client)

    def create_partition(self, partition_spec, if_not_exists=False, async_=False, **kw):
        """
        Create a partition within the table.

        :param partition_spec: specification of the partition.
        :param if_not_exists:
        :param async_:
        :return: partition object
        :rtype: odps.models.partition.Partition
        """
        async_ = kw.get('async', async_)
        return self.partitions.create(partition_spec, if_not_exists=if_not_exists, async_=async_)

    def delete_partition(self, partition_spec, if_exists=False, async_=False, **kw):
        """
        Delete a partition within the table.

        :param partition_spec: specification of the partition.
        :param if_exists:
        :param async_:
        """
        async_ = kw.get('async', async_)
        return self.partitions.delete(partition_spec, if_exists=if_exists, async_=async_)

    def exist_partition(self, partition_spec):
        """
        Check if a partition exists within the table.

        :param partition_spec: specification of the partition.
        """
        return partition_spec in self.partitions

    def exist_partitions(self, prefix_spec=None):
        """
        Check if partitions with provided conditions exist.

        :param prefix_spec: prefix of partition
        :return: whether partitions exist
        """
        try:
            next(self.partitions.iterate_partitions(spec=prefix_spec))
        except StopIteration:
            return False
        return True

    def iterate_partitions(self, spec=None):
        """
        Create an iterable object to iterate over partitions.

        :param spec: specification of the partition.
        """
        return self.partitions.iterate_partitions(spec=spec)

    def get_partition(self, partition_spec):
        """
        Get a partition with given specifications.

        :param partition_spec: specification of the partition.
        :return: partition object
        :rtype: odps.models.partition.Partition
        """
        return self.partitions[partition_spec]

    def truncate(self, async_=False, **kw):
        """
        truncate this table.

        :param async_: run asynchronously if True
        :return: None
        """
        from .tasks import SQLTask
        async_ = kw.get('async', async_)
        task = SQLTask(name='SQLTruncateTableTask', query='truncate table %s.%s' % (self.project.name, self.name))
        instance = self.project.parent[self._client.project].instances.create(task=task)

        if not async_:
            instance.wait_for_success()
        else:
            return instance

    def drop(self, async_=False, if_exists=False, **kw):
        """
        Drop this table.

        :param async_: run asynchronously if True
        :return: None
        """
        async_ = kw.get('async', async_)
        return self.parent.delete(self, async_=async_, if_exists=if_exists)

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
        return Record(schema=self.schema, values=values)

    def to_df(self):
        """
        Create a PyODPS DataFrame from this table.

        :return: DataFrame object
        """
        from ..df import DataFrame

        return DataFrame(self)
