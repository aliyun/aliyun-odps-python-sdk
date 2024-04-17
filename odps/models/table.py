#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import warnings
from datetime import datetime

try:
    import pyarrow as pa
except (AttributeError, ImportError):
    pa = None

from .. import types as odps_types, serializers, utils, readers
from ..compat import dir2, six, Enum
from ..config import options
from .cluster_info import ClusterInfo, ClusterType
from .core import LazyLoad, JSONRemoteModel
from .partitions import Partitions
from .record import Record
from .storage_tier import StorageTier, StorageTierInfo
from .tableio import (
    TableArrowReader,
    TableArrowWriter,
    TableRecordReader,
    TableRecordWriter,
)


class TableSchema(odps_types.OdpsSchema, JSONRemoteModel):
    """
    Schema includes the columns and partitions information of a :class:`odps.models.Table`.

    There are two ways to initialize a Schema object, first is to provide columns and partitions,
    the second way is to call the class method ``from_lists``. See the examples below:

    :Example:

    >>> columns = [Column(name='num', type='bigint', comment='the column')]
    >>> partitions = [Partition(name='pt', type='string', comment='the partition')]
    >>> schema = TableSchema(columns=columns, partitions=partitions)
    >>> schema.columns
    [<column num, type bigint>, <partition pt, type string>]
    >>>
    >>> schema = TableSchema.from_lists(['num'], ['bigint'], ['pt'], ['string'])
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
        nullable = serializers.JSONNodeField('isNullable')

        def __init__(self, **kwargs):
            kwargs.setdefault("nullable", True)
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
    last_data_modified_time = serializers.JSONNodeField(
        'lastModifiedTime', parse_callback=datetime.fromtimestamp,
        set_to_parent=True)
    last_meta_modified_time = serializers.JSONNodeField(
        'lastDDLTime', parse_callback=datetime.fromtimestamp,
        set_to_parent=True)
    is_virtual_view = serializers.JSONNodeField(
        'isVirtualView', parse_callback=bool, set_to_parent=True)
    is_materialized_view = serializers.JSONNodeField(
        'isMaterializedView', parse_callback=bool, set_to_parent=True)
    is_materialized_view_rewrite_enabled = serializers.JSONNodeField(
        'isMaterializedViewRewriteEnabled',
        parse_callback=lambda x: x is not None and str(x).lower() == "true",
        set_to_parent=True,
    )
    is_materialized_view_outdated = serializers.JSONNodeField(
        'isMaterializedViewOutdated',
        parse_callback=lambda x: x is not None and str(x).lower() == "true",
        set_to_parent=True,
    )
    lifecycle = serializers.JSONNodeField(
        'lifecycle', parse_callback=int, set_to_parent=True)
    view_text = serializers.JSONNodeField('viewText', set_to_parent=True)
    view_expanded_text = serializers.JSONNodeField('viewExpandedText', set_to_parent=True)
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
    _extend_args = (
        'is_archived', 'physical_size', 'file_num', 'location', 'storage_handler',
        'resources', 'serde_properties', 'reserved', 'is_transactional',
        'primary_key', 'storage_tier_info', 'cluster_info'
    )
    __slots__ = (
        '_is_extend_info_loaded', 'last_meta_modified_time', 'is_virtual_view',
        'is_materialized_view', 'is_materialized_view_rewrite_enabled',
        'is_materialized_view_outdated', 'lifecycle', 'view_text',
        'view_expanded_text', 'size', 'shard', 'record_num',
        '_table_tunnel', '_id_thread_local'
    )
    __slots__ += _extend_args

    class Type(Enum):
        MANAGED_TABLE = "MANAGED_TABLE"
        VIRTUAL_VIEW = "VIRTUAL_VIEW"
        EXTERNAL_TABLE = "EXTERNAL_TABLE"
        MATERIALIZED_VIEW = "MATERIALIZED_VIEW"

    name = serializers.XMLNodeField('Name')
    table_id = serializers.XMLNodeField('TableId')
    format = serializers.XMLNodeAttributeField(attr='format')
    table_schema = serializers.XMLNodeReferenceField(TableSchema, 'Schema')
    comment = serializers.XMLNodeField('Comment')
    owner = serializers.XMLNodeField('Owner')
    table_label = serializers.XMLNodeField('TableLabel')
    creation_time = serializers.XMLNodeField(
        'CreationTime', parse_callback=utils.parse_rfc822
    )
    last_data_modified_time = serializers.XMLNodeField(
        'LastModifiedTime', parse_callback=utils.parse_rfc822
    )
    last_access_time = serializers.XMLNodeField(
        'LastAccessTime', parse_callback=utils.parse_rfc822
    )
    type = serializers.XMLNodeField(
        'Type', parse_callback=lambda s: Table.Type(s.upper()) if s is not None else None
    )

    _download_ids = utils.thread_local_attribute('_id_thread_local', dict)
    _upload_ids = utils.thread_local_attribute('_id_thread_local', dict)

    def __init__(self, **kwargs):
        self._is_extend_info_loaded = False
        if 'schema' in kwargs:
            warnings.warn(
                'Argument schema is deprecated and will be replaced by table_schema.',
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs['table_schema'] = kwargs.pop('schema')
        super(Table, self).__init__(**kwargs)

        try:
            del self._id_thread_local
        except AttributeError:
            pass

    def table_resource(self, client=None, endpoint=None, force_schema=False):
        schema_name = self._get_schema_name()
        if force_schema:
            schema_name = schema_name or "default"
        if schema_name is None:
            return self.resource(client=client, endpoint=endpoint)
        return "/".join(
            [
                self.project.resource(client, endpoint=endpoint),
                "schemas",
                schema_name,
                "tables",
                self.name,
            ]
        )

    @property
    def full_table_name(self):
        schema_name = self._get_schema_name()
        if schema_name is None:
            return '{0}.`{1}`'.format(self.project.name, self.name)
        else:
            return '{0}.{1}.`{2}`'.format(self.project.name, schema_name, self.name)

    def reload(self):
        url = self.resource()
        resp = self._client.get(url, curr_schema=self._get_schema_name())

        self.parse(self._client, resp, obj=self)
        self.table_schema.load()
        self._loaded = True

    def reset(self):
        super(Table, self).reset()
        self._is_extend_info_loaded = False
        self.table_schema = None

    @property
    def schema(self):
        warnings.warn(
            'Table.schema is deprecated and will be replaced by Table.table_schema.',
            DeprecationWarning,
            stacklevel=3,
        )
        utils.add_survey_call(".".join([type(self).__module__, type(self).__name__, "schema"]))
        return self.table_schema

    @property
    def last_modified_time(self):
        warnings.warn(
            "Table.last_modified_time is deprecated and will be replaced by "
            "Table.last_data_modified_time.",
            DeprecationWarning,
            stacklevel=3,
        )
        utils.add_survey_call(".".join(
            [type(self).__module__, type(self).__name__, "last_modified_time"]
        ))
        return self.last_data_modified_time

    def _parse_reserved(self):
        if not self.reserved:
            self.is_transactional = None
            self.primary_key = None
            self.storage_tier_info = None
            self.cluster_info = None
            return
        is_transactional = self.reserved.get("Transactional")
        self.is_transactional = (
            is_transactional is not None and is_transactional.lower() == "true"
        )
        self.primary_key = self.reserved.get('PrimaryKey')
        self.storage_tier_info = StorageTierInfo.deserial(self.reserved)
        self.cluster_info = ClusterInfo.deserial(self.reserved)

    def reload_extend_info(self):
        params = {}
        schema_name = self._get_schema_name()
        if schema_name is not None:
            params['curr_schema'] = schema_name
        resp = self._client.get(self.resource(), action='extended', params=params)

        self.parse(self._client, resp, obj=self)
        self._is_extend_info_loaded = True

        if not self._loaded:
            self.table_schema = None

        self._parse_reserved()

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
        buf.write('  name: {0}\n'.format(self.full_table_name))
        if self.type:
            buf.write('  type: {0}\n'.format(self.type.value))

        name_space = 2 * max(len(col.name) for col in self.table_schema.columns)
        type_space = 2 * max(len(repr(col.type)) for col in self.table_schema.columns)

        not_empty = lambda field: field is not None and len(field.strip()) > 0

        buf.write('  schema:\n')
        cols_strs = []
        for col in self.table_schema._columns:
            cols_strs.append('{0}: {1}{2}'.format(
                col.name.ljust(name_space),
                repr(col.type).ljust(type_space),
                '# {0}'.format(utils.to_str(col.comment)) if not_empty(col.comment) else ''
            ))
        buf.write(utils.indent('\n'.join(cols_strs), 4))
        buf.write('\n')

        if self.table_schema.partitions:
            buf.write('  partitions:\n')

            partition_strs = []
            for partition in self.table_schema.partitions:
                partition_strs.append('{0}: {1}{2}'.format(
                    partition.name.ljust(name_space),
                    repr(partition.type).ljust(type_space),
                    '# {0}'.format(utils.to_str(partition.comment)) if not_empty(partition.comment) else ''
                ))
            buf.write(utils.indent('\n'.join(partition_strs), 4))

        if self.view_text:
            buf.write('  view text:\n{0}'.format(utils.indent(self.view_text, 4)))

        return buf.getvalue()

    @property
    def stored_as(self):
        return (self.reserved or dict()).get('StoredAs')

    @classmethod
    def gen_create_table_sql(
        cls, table_name, table_schema, comment=None, if_not_exists=False,
        lifecycle=None, shard_num=None, hub_lifecycle=None,
        with_column_comments=True, transactional=False, primary_key=None,
        storage_tier=None, project=None, schema=None, table_type=None,
        view_text=None, **kw
    ):
        from ..utils import escape_odps_string

        buf = six.StringIO()
        table_name = utils.to_text(table_name)
        project = utils.to_text(project)
        schema = utils.to_text(schema)
        comment = utils.to_text(comment)
        view_text = utils.to_text(view_text)
        table_type = cls.Type(table_type or cls.Type.MANAGED_TABLE)
        is_view = table_type in (cls.Type.VIRTUAL_VIEW, cls.Type.MATERIALIZED_VIEW)
        primary_key = [primary_key] if isinstance(primary_key, six.string_types) else primary_key

        stored_as = kw.get('stored_as')
        external_stored_as = kw.get('external_stored_as')
        storage_handler = kw.get('storage_handler')
        table_properties = kw.get("table_properties") or {}
        cluster_info = kw.get('cluster_info')

        rewrite_enabled = kw.get('rewrite_enabled')
        rewrite_enabled = rewrite_enabled if rewrite_enabled is not None else True

        if table_type == cls.Type.EXTERNAL_TABLE:
            type_str = u'EXTERNAL TABLE'
        elif table_type == cls.Type.VIRTUAL_VIEW:
            type_str = u'VIEW'
        elif table_type == cls.Type.MATERIALIZED_VIEW:
            type_str = u'MATERIALIZED VIEW'
        else:
            type_str = u'EXTERNAL TABLE' if storage_handler or external_stored_as else u'TABLE'

        buf.write(u'CREATE %s ' % type_str)
        if if_not_exists:
            buf.write(u'IF NOT EXISTS ')
        if project is not None:
            buf.write(u'%s.' % project)
        if schema is not None:
            buf.write(u'%s.' % schema)
        buf.write(u'`%s` ' % table_name)

        if is_view and lifecycle is not None and lifecycle > 0:
            buf.write("LIFECYCLE %s " % lifecycle)

        def _write_primary_key(prev=""):
            if not primary_key:
                return
            if not prev.strip().endswith(","):
                buf.write(u",\n")
            buf.write(u"  PRIMARY KEY (%s)" % ", ".join("`%s`" % c for c in primary_key))

        if isinstance(table_schema, six.string_types):
            buf.write(u'(\n')
            buf.write(table_schema)
            _write_primary_key(table_schema)
            buf.write(u'\n)\n')
            if comment:
                buf.write(u"COMMENT '%s'\n" % escape_odps_string(comment))
        elif isinstance(table_schema, tuple):
            buf.write(u'(\n')
            buf.write(table_schema[0])
            _write_primary_key(table_schema[0])
            buf.write(u'\n)\n')
            if comment:
                buf.write(u"COMMENT '%s'\n" % escape_odps_string(comment))
            buf.write(u'PARTITIONED BY ')
            buf.write(u'(\n')
            buf.write(table_schema[1])
            buf.write(u'\n)\n')
        else:
            def write_columns(col_array, with_pk=False):
                size = len(col_array)
                buf.write(u'(\n')
                for idx, column in enumerate(col_array):
                    buf.write(u'  `%s` %s' % (utils.to_text(column.name), utils.to_text(column.type)))
                    if not column.nullable and not options.sql.ignore_fields_not_null:
                        buf.write(u" NOT NULL")
                    if with_column_comments and column.comment:
                        buf.write(u" COMMENT '%s'" % utils.to_text(column.comment))
                    if idx < size - 1:
                        buf.write(u',\n')
                if with_pk:
                    _write_primary_key()
                buf.write(u'\n)\n')

            def write_view_columns(col_array):
                size = len(col_array)
                buf.write(u'(\n')
                for idx, column in enumerate(col_array):
                    buf.write(u'  `%s`' % (utils.to_text(column.name)))
                    if with_column_comments and column.comment:
                        buf.write(u" COMMENT '%s'" % utils.to_text(column.comment))
                    if idx < size - 1:
                        buf.write(u',\n')
                buf.write(u'\n)\n')

            if not is_view:
                write_columns(table_schema.simple_columns, with_pk=True)
            else:
                write_view_columns(table_schema.simple_columns)

            if comment:
                buf.write(u"COMMENT '%s'\n" % comment)
            if table_type == cls.Type.MATERIALIZED_VIEW and not rewrite_enabled:
                buf.write(u'DISABLE REWRITE\n')
            if table_schema.partitions:
                if not is_view:
                    buf.write(u'PARTITIONED BY ')
                    write_columns(table_schema.partitions)
                else:
                    buf.write(u'PARTITIONED ON ')
                    write_view_columns(table_schema.partitions)

        if cluster_info is not None:
            if cluster_info.cluster_type == ClusterType.RANGE:
                cluster_type_str = u"RANGE "
            else:
                cluster_type_str = u""
            cluster_cols = u", ".join(u"`%s`" % col for col in cluster_info.cluster_cols)
            buf.write("%sCLUSTERED BY (%s)" % (cluster_type_str, cluster_cols))
            if cluster_info.sort_cols:
                sort_cols = u", ".join(
                    u"`%s` %s" % (c.name, c.order.value) for c in cluster_info.sort_cols
                )
                buf.write(u" SORTED BY (%s)" % sort_cols)
            if cluster_info.bucket_num:
                buf.write(" INTO %s BUCKETS" % cluster_info.bucket_num)
            buf.write(u"\n")

        if transactional:
            table_properties["transactional"] = "true"
        if storage_tier:
            if isinstance(storage_tier, six.string_types):
                storage_tier = StorageTier(utils.underline_to_camel(storage_tier).lower())
            table_properties["storagetier"] = storage_tier.value

        if table_properties:
            buf.write(u"TBLPROPERTIES (\n")
            for k, v in table_properties.items():
                buf.write(u'  "%s"="%s"' % (k, v))
            buf.write(u'\n)\n')

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
        if not is_view and lifecycle is not None and lifecycle > 0:
            buf.write(u'LIFECYCLE %s\n' % lifecycle)
        if shard_num is not None:
            buf.write(u'INTO %s SHARDS' % shard_num)
            if hub_lifecycle is not None:
                buf.write(u' HUBLIFECYCLE %s\n' % hub_lifecycle)
            else:
                buf.write(u'\n')

        if is_view and view_text:
            buf.write(u'AS %s\n' % view_text)
        return buf.getvalue().strip()

    def get_ddl(self, with_comments=True, if_not_exists=False, force_table_ddl=False):
        """
        Get DDL SQL statement for the given table.

        :param with_comments: append comment for table and each column
        :param if_not_exists: generate `if not exists` code for generated DDL
        :param force_table_ddl: force generate table DDL if object is a view
        :return: DDL statement
        """
        shard_num = self.shard.shard_num if self.shard is not None else None
        storage_tier = (
            self.storage_tier_info.storage_tier.value if self.storage_tier_info else None
        )
        table_type = self.type if not force_table_ddl else self.Type.MANAGED_TABLE
        return self.gen_create_table_sql(
            self.name, self.table_schema, self.comment if with_comments else None,
            if_not_exists=if_not_exists, with_column_comments=with_comments,
            lifecycle=self.lifecycle, shard_num=shard_num, project=self.project.name,
            storage_handler=self.storage_handler, serde_properties=self.serde_properties,
            location=self.location, resources=self.resources, table_type=table_type,
            storage_tier=storage_tier, cluster_info=self.cluster_info,
            transactional=self.is_transactional, primary_key=self.primary_key,
            view_text=self.view_text,
            rewrite_enabled=self.is_materialized_view_rewrite_enabled,
        )

    @utils.survey
    def _head_by_data(self, limit, partition=None, columns=None, timeout=None):
        if limit <= 0:
            raise ValueError('limit number should >= 0.')

        params = {'linenum': limit}
        if partition is not None:
            if not isinstance(partition, odps_types.PartitionSpec):
                partition = odps_types.PartitionSpec(partition)
            params['partition'] = str(partition)
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, odps_types.Column) else col
            params['cols'] = ','.join(col_name(col) for col in columns)

        schema_name = self._get_schema_name()
        if schema_name is not None:
            params['schema_name'] = schema_name

        resp = self._client.get(
            self.resource(), action='data', params=params, stream=True, timeout=timeout
        )
        return readers.CsvRecordReader(self.table_schema, resp)

    def _head_by_preview(self, limit, partition=None, columns=None, compress_algo=None, timeout=None):
        table_tunnel = self._create_table_tunnel()
        return table_tunnel.open_preview_reader(
            self, partition_spec=partition, columns=columns, limit=limit,
            compress_algo=compress_algo, arrow=False, timeout=timeout, read_all=True
        )

    def head(self, limit, partition=None, columns=None, use_legacy=True, timeout=None):
        """
        Get the head records of a table or its partition.

        :param int limit: records' size, 10000 at most
        :param partition: partition of this table
        :param list columns: the columns which is subset of the table columns
        :return: records
        :rtype: list

        .. seealso:: :class:`odps.models.Record`
        """
        try:
            if pa is not None and not use_legacy:
                timeout = timeout if timeout is not None else options.tunnel.legacy_fallback_timeout
                return self._head_by_preview(
                    limit, partition=partition, columns=columns, timeout=timeout
                )
        except:
            # only raises when under tests and
            # use_legacy specified explicitly as False
            if use_legacy is False:
                raise
        return self._head_by_data(
            limit, partition=partition, columns=columns, timeout=timeout
        )

    def _create_table_tunnel(self, endpoint=None, quota_name=None):
        if self._table_tunnel is not None:
            return self._table_tunnel

        from ..tunnel import TableTunnel

        self._table_tunnel = TableTunnel(
            client=self._client,
            project=self.project,
            endpoint=endpoint or self.project._tunnel_endpoint,
            quota_name=quota_name,
        )
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
        quota_name=None,
        async_mode=True,
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
        :param quota_name: name of tunnel quota
        :param async_mode: enable async mode to create tunnels, can set True if session creation
            takes a long time.
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

        if partition and not isinstance(partition, odps_types.PartitionSpec):
            partition = odps_types.PartitionSpec(partition)
        tunnel = self._create_table_tunnel(endpoint=endpoint, quota_name=quota_name)
        download_ids = dict()
        if download_id is None:
            download_ids = self._download_ids
            download_id = download_ids.get(partition) if not reopen else None
        download_session = utils.call_with_retry(
            tunnel.create_download_session,
            table=self, partition_spec=partition, download_id=download_id,
            timeout=timeout, async_mode=async_mode, **kw
        )

        if download_id and download_session.status != TableDownloadSession.Status.Normal:
            download_session = utils.call_with_retry(
                tunnel.create_download_session,
                table=self, partition_spec=partition, timeout=timeout,
                async_mode=async_mode, **kw
            )
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
        quota_name=None,
        mp_context=None,
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
        :param quota_name: name of tunnel quota
        :param bool overwrite: if True, will overwrite existing data
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

        tunnel = self._create_table_tunnel(endpoint=endpoint, quota_name=quota_name)
        if upload_id is None:
            upload_ids = self._upload_ids
            upload_id = upload_ids.get(partition) if not reopen else None
        else:
            upload_ids = dict()
        upload_session = utils.call_with_retry(
            tunnel.create_upload_session,
            table=self, partition_spec=partition, upload_id=upload_id, **kw
        )

        if upload_id and upload_session.status.value != TableUploadSession.Status.Normal.value:
            # check upload session status
            upload_session = utils.call_with_retry(
                tunnel.create_upload_session, table=self, partition_spec=partition, **kw
            )
            upload_id = None
        upload_ids[partition] = upload_session.id
        # as data of partition changed, remove existing download id to avoid TableModified error
        self._download_ids.pop(partition, None)

        writer_cls = TableArrowWriter if arrow else TableRecordWriter

        def _writer_on_close():
            if commit:
                upload_ids[partition] = None

        return writer_cls(
            self, upload_session, blocks, commit, on_close=_writer_on_close, mp_context=mp_context
        )

    @property
    def partitions(self):
        return Partitions(parent=self, client=self._client)

    @utils.with_wait_argument
    def create_partition(self, partition_spec, if_not_exists=False, async_=False, hints=None):
        """
        Create a partition within the table.

        :param partition_spec: specification of the partition.
        :param if_not_exists:
        :param hints:
        :param async_:
        :return: partition object
        :rtype: odps.models.partition.Partition
        """
        return self.partitions.create(
            partition_spec, if_not_exists=if_not_exists, hints=hints, async_=async_
        )

    @utils.with_wait_argument
    def delete_partition(self, partition_spec, if_exists=False, async_=False, hints=None):
        """
        Delete a partition within the table.

        :param partition_spec: specification of the partition.
        :param if_exists:
        :param hints:
        :param async_:
        """
        return self.partitions.delete(
            partition_spec, if_exists=if_exists, hints=hints, async_=async_
        )

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

    def iterate_partitions(self, spec=None, reverse=False):
        """
        Create an iterable object to iterate over partitions.

        :param spec: specification of the partition.
        :param reverse: output partitions in reversed order
        """
        return self.partitions.iterate_partitions(spec=spec, reverse=reverse)

    def get_partition(self, partition_spec):
        """
        Get a partition with given specifications.

        :param partition_spec: specification of the partition.
        :return: partition object
        :rtype: odps.models.partition.Partition
        """
        return self.partitions[partition_spec]

    def get_max_partition(self, spec=None, skip_empty=True, reverse=False):
        """
        Get partition with maximal values within certain spec.

        :param spec: parent partitions. if specified, will return partition with
            maximal value within specified parent partition
        :param skip_empty: if True, will skip partitions without data
        :param reverse: if True, will return minimal value
        :return: Partition
        """
        if not self.table_schema.partitions:
            raise ValueError("Table %r not partitioned" % self.name)
        return self.partitions.get_max_partition(
            spec, skip_empty=skip_empty, reverse=reverse
        )

    @utils.with_wait_argument
    def truncate(self, partition_spec=None, async_=False, hints=None):
        """
        truncate this table.

        :param partition_spec: partition specs
        :param hints:
        :param async_: run asynchronously if True
        :return: None
        """
        from .tasks import SQLTask

        partition_expr = ""
        if partition_spec is not None:
            if not isinstance(partition_spec, (list, tuple)):
                partition_spec = [partition_spec]
            partition_expr = " " + ", ".join(
                "PARTITION (%s)" % spec for spec in partition_spec
            )

        # as data of partition changed, remove existing download id to avoid TableModified error
        for part in (partition_spec or [None]):
            if isinstance(part, six.string_types):
                part = odps_types.PartitionSpec(part)
            self._download_ids.pop(part, None)

        task = SQLTask(
            name='SQLTruncateTableTask',
            query='TRUNCATE TABLE %s%s;' % (self.full_table_name, partition_expr)
        )

        hints = hints or {}
        hints['odps.sql.submit.mode'] = ''
        schema_name = self._get_schema_name()
        if schema_name is not None:
            hints["odps.sql.allow.namespace.schema"] = "true"
            hints["odps.namespace.schema"] = "true"
        task.update_sql_settings(hints)

        instance = self.project.parent[self._client.project].instances.create(task=task)

        if not async_:
            instance.wait_for_success()
        else:
            return instance

    @utils.with_wait_argument
    def drop(self, async_=False, if_exists=False, hints=None):
        """
        Drop this table.

        :param async_: run asynchronously if True
        :param if_exists:
        :param hints:
        :return: None
        """
        return self.parent.delete(self, async_=async_, if_exists=if_exists, hints=hints)

    @utils.with_wait_argument
    def set_storage_tier(self, storage_tier, partition_spec=None, async_=False, hints=None):
        """
        Set storage tier of
        """
        from .tasks import SQLTask

        partition_expr = ""
        if partition_spec is not None:
            if not isinstance(partition_spec, (list, tuple)):
                partition_spec = [partition_spec]
            partition_expr = " " + ", ".join(
                "PARTITION (%s)" % spec for spec in partition_spec
            )

        # as data of partition changed, remove existing download id to avoid TableModified error
        for part in (partition_spec or [None]):
            if isinstance(part, six.string_types):
                part = odps_types.PartitionSpec(part)
            self._download_ids.pop(part, None)

        if isinstance(storage_tier, six.string_types):
            storage_tier = StorageTier(utils.underline_to_camel(storage_tier).lower())

        property_item = "TBLPROPERTIES" if not partition_spec else "PARTITIONPROPERTIES"
        task = SQLTask(
            name='SQLSetStorageTierTask',
            query="ALTER TABLE %s%s SET %s('storagetier'='%s')" % (
                self.full_table_name, partition_expr, property_item, storage_tier.value
            )
        )

        hints = hints or {}
        hints['odps.sql.submit.mode'] = ''
        hints['odps.tiered.storage.enable'] = 'true'
        schema_name = self._get_schema_name()
        if schema_name is not None:
            hints["odps.sql.allow.namespace.schema"] = "true"
            hints["odps.namespace.schema"] = "true"
        task.update_sql_settings(hints)

        instance = self.project.parent[self._client.project].instances.create(task=task)
        self._is_extend_info_loaded = False

        if not async_:
            instance.wait_for_success()
        else:
            return instance

    def new_record(self, values=None):
        """
        Generate a record of the table.

        :param values: the values of this records
        :type values: list
        :return: record
        :rtype: :class:`odps.models.Record`

        :Example:

        >>> table = odps.create_table('test_table', schema=TableSchema.from_lists(['name', 'id'], ['sring', 'string']))
        >>> record = table.new_record()
        >>> record[0] = 'my_name'
        >>> record[1] = 'my_id'
        >>> record = table.new_record(['my_name', 'my_id'])

        .. seealso:: :class:`odps.models.Record`
        """
        return Record(schema=self.table_schema, values=values)

    def to_df(self):
        """
        Create a PyODPS DataFrame from this table.

        :return: DataFrame object
        """
        from ..df import DataFrame

        return DataFrame(self)
