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

import json
from collections import Iterable

from .rest import RestClient
from .config import options
from .tempobj import clean_objects
from .compat import six
from . import models
from . import accounts


DEFAULT_ENDPOINT = 'http://service.odps.aliyun.com/api'
LOG_VIEW_HOST_DEFAULT = 'http://logview.odps.aliyun.com'


class ODPS(object):
    """
    Main entrance to ODPS.

    Convenient operations on ODPS objects are provided.
    Please refer to `ODPS docs <https://docs.aliyun.com/#/pub/odps/basic/definition&project>`_
    to see the details.

    Generally, basic operations such as ``list``, ``get``, ``exist``, ``create``, ``delete``
    are provided for each ODPS object.
    Take the ``Table`` as an example.

    :Example:

    >>> odps = ODPS('**your access id**', '**your access key**', 'default_project')
    >>>
    >>> for table in odps.list_tables():
    >>>    # handle each table
    >>>
    >>> table = odps.get_table('dual')
    >>>
    >>> odps.exist_table('dual') is True
    >>>
    >>> odps.create_table('test_table', schema)
    >>>
    >>> odps.delete_table('test_table')
    """

    def __init__(self, access_id, secret_access_key, project,
                 endpoint=DEFAULT_ENDPOINT, tunnel_endpoint=None):
        """
        Initial ODPS, access_id and access_key is required, and should ensure correctness,
        or ``SignatureNotMatch`` error will throw. If `tunnel_endpoint` is not set,
        the tunnel API will route service URL automatically.

        :param access_id: Aliyun Access ID
        :param secret_access_key: Aliyun Access Key
        :param project: default project name
        :param endpoint: Rest service URL
        :param tunnel_endpoint:  Tunnel service URL
        """

        self.account = self._build_account(access_id, secret_access_key)
        self.endpoint = endpoint
        self.project = project
        self.rest = RestClient(self.account, endpoint, project)

        self._projects = models.Projects(client=self.rest)
        self._project = self.get_project()

        self._tunnel_endpoint = tunnel_endpoint
        options.tunnel_endpoint = self._tunnel_endpoint

        clean_objects(self)

    def get_project(self, name=None):
        """
        Get project by given name.

        :param name: project name, if not provided, will be the default project
        :return: the right project
        :rtype: :class:`odps.models.Project`
        :raise: :class:`odps.errors.NoSuchObject` if not exists

        .. seealso:: :class:`odps.models.Project`
        """

        if name is None:
            return self._projects[self.project]
        elif isinstance(name, models.Project):
            return name
        return self._projects[name]

    def exist_project(self, name):
        """
        If project name which provided exists or not.

        :param name: project name, if not provided, will be the default project
        :return: True if exists or False
        :rtype: bool
        """

        return name in self._projects

    def list_tables(self, project=None, prefix=None, owner=None):
        """
        List all tables of a project.
        If prefix is provided, the listed tables will all start with this prefix.
        If owner is provided, the listed tables will be belong to such owner.

        :param project: project name, if not provided, will be the default project
        :param prefix: the listed tables start with this **prefix**
        :type prefix: str
        :param owner: Aliyun account, the owner which listed tables belong to
        :type owner: str
        :return: tables in this project, filtered by the optional prefix and owner.
        :rtype: generator
        """

        project = self.get_project(name=project)
        return project.tables.iterate(name=prefix, owner=owner)

    def get_table(self, name, project=None):
        """
        Get table by given name.

        :param name: table name
        :param project: project name, if not provided, will be the default project
        :return: the right table
        :rtype: :class:`odps.models.Table`
        :raise: :class:`odps.errors.NoSuchObject` if not exists

        .. seealso:: :class:`odps.models.Table`
        """

        project = self.get_project(name=project)
        return project.tables[name]

    def exist_table(self, name, project=None):
        """
        If the table with given name exists or not.

        :param name: table name
        :param project: project name, if not provided, will be the default project
        :return: True if table exists or False
        :rtype: bool
        """

        project = self.get_project(name=project)
        return name in project.tables

    def create_table(self, name, schema, project=None, comment=None, if_not_exists=False,
                     lifecycle=None, shard_num=None, hub_lifecycle=None, async=False):
        """
        Create an table by given schema and other optional parameters.

        :param name: table name
        :param schema: table schema
        :type schema: :class:`odps.models.Schema`
        :param project: project name, if not provided, will be the default project
        :param comment:  table comment
        :param if_not_exists:  will not create if this table already exists, default False
        :type if_not_exists: bool
        :param lifecycle:  table's lifecycle. If absent, `options.lifecycle` will be used.
        :type lifecycle: int
        :param shard_num:  table's shard num
        :type shard_num: int
        :param hub_lifecycle:  hub lifecycle
        :type hub_lifecycle: int
        :param async: if True, will run asynchronously
        :type async: bool
        :return: the created Table if not async else odps instance
        :rtype: :class:`odps.models.Table` or :class:`odps.models.Instance`

        .. seealso:: :class:`odps.models.Table`, :class:`odps.models.Schema`
        """

        if lifecycle is None and options.lifecycle is not None:
            lifecycle = options.lifecycle
        project = self.get_project(name=project)
        return project.tables.create(name, schema, comment=comment, if_not_exists=if_not_exists,
                                     lifecycle=lifecycle, shard_num=shard_num,
                                     hub_lifecycle=hub_lifecycle, async=async)

    def delete_table(self, name, project=None, if_exists=False, async=False):
        """
        Delete the table with given name

        :param name: table name
        :param project: project name, if not provided, will be the default project
        :param if_exists:  will not delete the table if not exists, default False
        :param async: if True, will run asynchronously
        :type async: bool
        :return: None if not async else odps instance
        """

        project = self.get_project(name=project)
        return project.tables.delete(name, if_exists=if_exists, async=async)

    def read_table(self, name, limit=None, start=0, step=None,
                   project=None, partition=None, **kw):
        """
        Read table's records.

        :param name: table or table name
        :type name: :class:`odps.models.table.Table` or str
        :param limit:  the records' size, if None will read all records from the table
        :param start:  the record where read starts with
        :param step:  default as 1
        :param project: project name, if not provided, will be the default project
        :param partition: the partition of this table to read
        :param columns: the columns' names which are the parts of table's columns
        :type columns: list
        :param compress: if True, the data will be compressed during downloading
        :type compress: bool
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param endpoint: tunnel service URL
        :param reopen: reading the table will reuse the session which opened last time,
                       if set to True will open a new download session, default as False
        :return: records
        :rtype: generator

        :Example:

        >>> for record in odps.read_table('test_table', 100):
        >>>     # deal with such 100 records
        >>> for record in odps.read_table('test_table', partition='pt=test', start=100, limit=100):
        >>>     # read the `pt=test` partition, skip 100 records and read 100 records

        .. seealso:: :class:`odps.models.Record`
        """

        if not isinstance(name, six.string_types):
            name = name.name

        project = self.get_project(name=project)
        table = project.tables[name]

        compress = kw.pop('compress', False)
        columns = kw.pop('columns', None)

        with table.open_reader(partition=partition, **kw) as reader:
            for record in reader.read(start, limit, step=step,
                                      compress=compress, columns=columns):
                yield record

    def write_table(self, name, *block_records, **kw):
        """
        Write records into given table.

        :param name: table or table name
        :type name: :class:`.models.table.Table` or str
        :param block_records: if given records only, the block id will be 0 as default.
        :param project: project name, if not provided, will be the default project
        :param partition: the partition of this table to write
        :param compress: if True, the data will be compressed during uploading
        :type compress: bool
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param endpoint:  tunnel service URL
        :param reopen: writing the table will reuse the session which opened last time,
                       if set to True will open a new upload session, default as False
        :return: None

        :Example:

        >>> odps.write_table('test_table', records)  # write to block 0 as default
        >>>
        >>> odps.write_table('test_table', 0, records)  # write to block 0 explicitly
        >>>
        >>> odps.write_table('test_table', 0, records1, 1, records2)  # write to multi-blocks
        >>>
        >>> odps.write_table('test_table', records, partition='pt=test') # write to certain partition

        .. seealso:: :class:`odps.models.Record`
        """

        if not isinstance(name, six.string_types):
            name = name.name

        project = self.get_project(name=kw.pop('project', None))
        table = project.tables[name]
        partition = kw.pop('partition', None)

        if len(block_records) == 1 and isinstance(block_records[0], Iterable):
            blocks = [0, ]
            records_iterators = block_records
        else:
            blocks = block_records[::2]
            records_iterators = block_records[1::2]

            if len(blocks) != len(records_iterators):
                raise ValueError('Should invoke like '
                                 'odps.write_table(block_id, records, block_id2, records2, ..., **kw)')

        with table.open_writer(partition=partition, blocks=blocks, **kw) as writer:
            for block, records in zip(blocks, records_iterators):
                writer.write(block, records)

    def list_resources(self, project=None):
        """
        List all resources of a project.

        :param project: project name, if not provided, will be the default project
        :return: resources
        :rtype: generator
        """

        project = self.get_project(name=project)
        for resource in project.resources:
            yield resource

    def get_resource(self, name, project=None):
        """
        Get a resource by given name

        :param name: resource name
        :param project: project name, if not provided, will be the default project
        :return: the right resource
        :rtype: :class:`odps.models.Resource`
        :raise: :class:`odps.errors.NoSuchObject` if not exists

        .. seealso:: :class:`odps.models.Resource`
        """

        project = self.get_project(name=project)
        return project.resources[name]

    def exist_resource(self, name, project=None):
        """
        If the resource with given name exists or not.

        :param name: resource name
        :param project: project name, if not provided, will be the default project
        :return: True if exists or False
        :rtype: bool
        """

        project = self.get_project(name=project)
        return name in project.resources

    def open_resource(self, name, project=None, mode='r+', encoding='utf-8'):
        """
        Open a file resource as file-like object.
        This is an elegant and pythonic way to handle file resource.

        The argument ``mode`` stands for the open mode for this file resource.
        It can be binary mode if the 'b' is inside. For instance,
        'rb' means opening the resource as read binary mode
        while 'r+b' means opening the resource as read+write binary mode.
        This is most import when the file is actually binary such as tar or jpeg file,
        so be aware of opening this file as a correct mode.

        Basically, the text mode can be 'r', 'w', 'a', 'r+', 'w+', 'a+'
        just like the builtin python ``open`` method.

        * ``r`` means read only
        * ``w`` means write only, the file will be truncated when opening
        * ``a`` means append only
        * ``r+`` means read+write without constraint
        * ``w+`` will truncate first then opening into read+write
        * ``a+`` can read+write, however the written content can only be appended to the end

        :param name: file resource or file resource name
        :type name: :class:`odps.models.FileResource` or str
        :param project: project name, if not provided, will be the default project
        :param mode: the mode of opening file, described as above
        :param encoding: utf-8 as default
        :return: file-like object

        :Example:

        >>> with odps.open_resource('test_resource', 'r') as fp:
        >>>     fp.read(1)  # read one unicode character
        >>>     fp.write('test')  # wrong, cannot write under read mode
        >>>
        >>> with odps.open_resource('test_resource', 'wb') as fp:
        >>>     fp.readlines() # wrong, cannot read under write mode
        >>>     fp.write('hello world') # write bytes
        >>>
        >>> with odps.open_resource('test_resource') as fp: # default as read-write mode
        >>>     fp.seek(5)
        >>>     fp.truncate()
        >>>     fp.flush()
        """

        from .models import FileResource

        if isinstance(name, FileResource):
            return name.open(mode=mode)
        return self.get_resource(name, project=project).open(mode=mode, encoding=encoding)

    def create_resource(self, name, typo, project=None, **kwargs):
        """
        Create a resource by given name and given type.

        Currently, the resource type can be ``file``, ``jar``, ``py``, ``archive``, ``table``.

        The ``file``, ``jar``, ``py``, ``archive`` can be classified into file resource.
        To init the file resource, you have to provide another parameter which is a file-like object.

        For the table resource, the table name, project name, and partition should be provided
        which the partition is optional.

        :param name: resource name
        :param typo: resource type, now support ``file``, ``jar``, ``py``, ``archive``, ``table``
        :param project: project name, if not provided, will be the default project
        :param kwargs: optional arguments, I will illustrate this in the example below.
        :return: resource depends on the type, if ``file`` will be :class:`odps.models.FileResource` and so on
        :rtype: :class:`odps.models.Resource`'s subclasses

        :Example:

        >>> from odps.models.resource import *
        >>>
        >>> res = odps.create_resource('test_file_resource', 'file', file_obj=open('/to/path/file'))
        >>> assert isinstance(res, FileResource)
        >>> True
        >>>
        >>> res = odps.create_resource('test_py_resource.py', 'py', file_obj=StringIO('import this'))
        >>> assert isinstance(res, PyResource)
        >>> True
        >>>
        >>> res = odps.create_resource('test_table_resource', 'table', table_name='test_table', partition='pt=test')
        >>> assert isinstance(res, TableResource)
        >>> True
        >>>

        .. seealso:: :class:`odps.models.FileResource`, :class:`odps.models.PyResource`,
                     :class:`odps.models.JarResource`, :class:`odps.models.ArchiveResource`,
                     :class:`odps.models.TableResource`
        """

        project = self.get_project(name=project)
        return project.resources.create(name=name, type=typo, **kwargs)

    def delete_resource(self, name, project=None):
        """
        Delete resource by given name.

        :param name: resource name
        :param project: project name, if not provided, will be the default project
        :return: None
        """

        project = self.get_project(name=project)
        return project.resources.delete(name)

    def list_functions(self, project=None):
        """
        List all functions of a project.

        :param project: project name, if not provided, will be the default project
        :return: functions
        :rtype: generator
        """

        project = self.get_project(name=project)
        for function in project.functions:
            yield function

    def get_function(self, name, project=None):
        """
        Get the function by given name

        :param name: function name
        :param project: project name, if not provided, will be the default project
        :return: the right function
        :raise: :class:`odps.errors.NoSuchObject` if not exists

        .. seealso:: :class:`odps.models.Function`
        """

        project = self.get_project(name=project)
        return project.functions[name]

    def exist_function(self, name, project=None):
        """
        If the function with given name exists or not.

        :param name: function name
        :param project: project name, if not provided, will be the default project
        :return: True if the function exists or False
        :rtype: bool
        """

        project = self.get_project(name=project)
        return name in project.functions

    def create_function(self, name, project=None, **kwargs):
        """
        Create a function by given name.

        :param name: function name
        :param project: project name, if not provided, will be the default project
        :param class_type: main class
        :type class_type: str
        :param resources: the resources that function needs to use
        :type resources: list
        :return: the created function
        :rtype: :class:`odps.models.Function`

        :Example:

        >>> res = odps.get_resource('test_func.py')
        >>> func = odps.create_function('test_func', class_type='test_func.Test', resources=[res, ])

        .. seealso:: :class:`odps.models.Function`
        """

        project = self.get_project(name=project)
        return project.functions.create(name=name, **kwargs)

    def delete_function(self, name, project=None):
        """
        Delete a function by given name.

        :param name: function name
        :param project: project name, if not provided, will be the default project
        :return: None
        """

        project = self.get_project(name=project)
        return project.functions.delete(name)

    def list_instances(self, project=None, from_time=None, end_time=None,
                       status=None, only_owner=None):
        """
        List instances of a project by given optional conditions
        including start time, end time, status and if only the owner.

        :param project: project name, if not provided, will be the default project
        :param from_time: the start time of filtered instances
        :type from_time: datetime, int or float
        :param end_time: the end time of filtered instances
        :type end_time: datetime, int or float
        :param status: including ``odps.models.Instance.Status.Running``,
                       ``odps.models.Instance.Status.Suspended``,
                       ``odps.models.Instance.Status.Terminated``
        :param only_owner: True will filter the instances created by current user
        :type only_owner: bool
        :return: instances
        :rtype: list
        """
        project = self.get_project(name=project)
        return project.instances.iterate(
            from_time=from_time, end_time=end_time,
            status=status, only_owner=only_owner)

    def get_instance(self, id_, project=None):
        """
        Get instance by given instance id.

        :param id_: instance id
        :param project: project name, if not provided, will be the default project
        :return: the right instance
        :rtype: :class:`odps.models.Instance`
        :raise: :class:`odps.errors.NoSuchObject` if not exists

        .. seealso:: :class:`odps.models.Instance`
        """

        project = self.get_project(name=project)
        return project.instances[id_]

    def exist_instance(self, id_, project=None):
        """
        If the instance with given id exists or not.

        :param id_: instance id
        :param project: project name, if not provided, will be the default project
        :return: True if exists or False
        :rtype: bool
        """

        project = self.get_project(name=project)
        return id_ in project.instances

    def stop_instance(self, id_, project=None):
        """
        Stop the running instance by given instance id.

        :param id_: instance id
        :param project: project name, if not provided, will be the default project
        :return: None
        """

        project = self.get_project(name=project)
        project.instances[id_].stop()

    stop_job = stop_instance  # to keep compatible

    def execute_sql(self, sql, project=None, priority=None, running_cluster=None,
                    hints=None, **kwargs):
        """
        Run a given SQL statement and block until the SQL executed successfully.

        :param sql: SQL statement
        :type sql: str
        :param project: project name, if not provided, will be the default project
        :param priority: instance priority, 9 as default
        :type priority: int
        :param running_cluster: cluster to run this instance
        :param hints: settings for SQL, e.g. `odps.mapred.map.split.size`
        :type hints: dict
        :return: instance
        :rtype: :class:`odps.models.Instance`

        :Example:

        >>> instance = odps.execute_sql('select * from dual')
        >>> with instance.open_reader() as reader:
        >>>     for record in reader:  # iterate to handle result with schema
        >>>         # handle each record
        >>>
        >>> instance = odps.execute_sql('desc dual')
        >>> with instance.open_reader() as reader:
        >>>     print(reader.raw)  # without schema, just get the raw result

        .. seealso:: :class:`odps.models.Instance`
        """

        async = kwargs.pop('async', False)

        inst = self.run_sql(
            sql, project=project, priority=priority, running_cluster=running_cluster,
            hints=hints, **kwargs)
        if not async:
            inst.wait_for_success()

        return inst

    def run_sql(self, sql, project=None, priority=None, running_cluster=None,
                hints=None, aliases=None, **kwargs):
        """
        Run a given SQL statement asynchronously

        :param sql: SQL statement
        :type sql: str
        :param project: project name, if not provided, will be the default project
        :param priority: instance priority, 9 as default
        :type priority: int
        :param running_cluster: cluster to run this instance
        :param hints: settings for SQL, e.g. `odps.mapred.map.split.size`
        :type hints: dict
        :param aliases:
        :type aliases: dict
        :return: instance
        :rtype: :class:`odps.models.Instance`

        .. seealso:: :class:`odps.models.Instance`
        """

        def update(kv, dest):
            if not kv:
                return
            for k, v in six.iteritems(kv):
                if isinstance(v, bool):
                    dest[k] = 'true' if v else 'false'
                else:
                    dest[k] = str(v)

        task = models.SQLTask(query=sql, **kwargs)
        if hints or options.sql.settings:
            if task.properties is None:
                task.properties = dict()
            settings = dict()

            update(options.sql.settings, settings)
            update(hints, settings)
            task.properties['settings'] = json.dumps(settings)

        if aliases:
            if task.properties is None:
                task.properties = dict()

            d = dict()
            update(aliases, d)
            task.properties['aliases'] = json.dumps(d)

        if options.biz_id:
            if task.properties is None:
                task.properties = dict()
            task.properties['biz_id'] = str(options.biz_id)

        project = self.get_project(name=project)
        return project.instances.create(task=task, priority=priority,
                                        running_cluster=running_cluster)

    def list_volumes(self, project=None, owner=None):
        """
        List volumes of a project.

        :param project: project name, if not provided, will be the default project
        :param owner: Aliyun account
        :type owner: str
        :return: volumes
        :rtype: list
        """
        project = self.get_project(name=project)
        return project.volumes.iterate(owner=owner)

    def create_volume(self, name, project=None, **kwargs):
        """
        Create a volume in a project.

        :param str name: volume name name
        :param str project: project name, if not provided, will be the default project
        :return: volume
        :rtype: :class:`odps.models.Volume`
        """
        project = self.get_project(name=project)
        return project.volumes.create(name=name, **kwargs)

    def exist_volume(self, name, project=None):
        """
        If the volume with given name exists or not.

        :param str name: volume name
        :param str project: project name, if not provided, will be the default project
        :return: True if exists or False
        :rtype: bool
        """
        project = self.get_project(name=project)
        return name in project.volumes

    def get_volume(self, name, project=None):
        """
        Get volume by given name

        :param str name: volume name
        :param str project: project name, if not provided, will be the default project
        :return: volume
        :rtype: :class:`odps.models.Volume`
        """
        project = self.get_project(name=project)
        return project.volumes[name]

    def delete_volume(self, name, project=None):
        """
        Delete volume by given name.

        :param name: volume name
        :param project: project name, if not provided, will be the default project
        :return: None
        """
        project = self.get_project(name=project)
        return project.volumes.delete(name)

    def list_volume_partitions(self, volume, project=None):
        """
        List partitions of a volume.

        :param str volume: volume name
        :param str project: project name, if not provided, will be the default project
        :return: partitions
        :rtype: list
        """
        volume = self.get_volume(volume, project)
        return volume.partitions.iterate

    def get_volume_partition(self, volume, partition, project=None):
        """
        Get partition in a volume by given name

        :param str volume: volume name
        :param str partition: partition name
        :param str project: project name, if not provided, will be the default project
        :return: partitions
        :rtype: :class:`odps.models.VolumePartition`
        """
        volume = self.get_volume(volume, project)
        return volume.partitions[partition]

    def exist_volume_partition(self, volume, partition, project=None):
        """
        If the volume with given name exists in a partition or not.

        :param str volume: volume name
        :param str partition: partition name
        :param str project: project name, if not provided, will be the default project
        """
        volume = self.get_volume(volume, project)
        return partition in volume.partitions

    def delete_volume_partition(self, volume, partition, project=None):
        """
        Delete partition in a volume by given name

        :param str volume: volume name
        :param str partition: partition name
        :param str project: project name, if not provided, will be the default project
        """
        volume = self.get_volume(volume, project)
        return volume.delete_partition(partition)

    def list_volume_files(self, volume, partition, project=None):
        """
        List files in a partition of a volume.

        :param str volume: volume name
        :param str partition: partition name
        :param str project: project name, if not provided, will be the default project
        :return: files
        :rtype: list
        """
        volume = self.get_volume(volume, project)
        return volume.partitions[partition].files.iterate

    def open_volume_reader(self, volume, partition, file_name, project=None, endpoint=None,
                           start=None, length=None, **kwargs):
        """
        Open a volume file for read. A file-like object will be returned which can be used to read contents from
        volume files.

        :param str volume: name of the volume
        :param str partition: name of the partition
        :param str file_name: name of the file
        :param str project: project name, if not provided, will be the default project
        :param str endpoint: tunnel service URL
        :param start: start position
        :param length: length limit
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`

        :Example:
        >>> with odps.open_volume_reader('volume', 'partition', 'file') as reader:
        >>>     [print(line) for line in reader]
        """
        volume = self.get_volume(volume, project)
        return volume.partitions[partition].open_reader(file_name, endpoint=endpoint, start=start, length=length,
                                                        **kwargs)

    def open_volume_writer(self, volume, partition, project=None, endpoint=None, **kwargs):
        """
        Open a volume partition to write to. You can use `open` method to open a file inside the volume and write to it,
        or use `write` method to write to specific files.

        :param str volume: name of the volume
        :param str partition: name of the partition
        :param str project: project name, if not provided, will be the default project
        :param str endpoint: tunnel service URL
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :Example:
        >>> with odps.open_volume_writer('volume', 'partition') as writer:
        >>>     writer.open('file1').write('some content')
        >>>     writer.write('file2', 'some content')
        """
        volume = self.get_volume(volume, project)
        return volume.partitions[partition].open_writer(endpoint=endpoint, **kwargs)

    def list_xflows(self, project=None, owner=None):
        """
        List xflows of a project which can be filtered by the xflow owner.

        :param project: project name, if not provided, will be the default project
        :param owner: Aliyun account
        :type owner: str
        :return: xflows
        :rtype: list
        """

        project = self.get_project(name=project)
        return project.xflows.iterate(owner=owner)

    def get_xflow(self, name, project=None):
        """
        Get xflow by given name

        :param name: xflow name
        :param project: project name, if not provided, will be the default project
        :return: xflow
        :rtype: :class:`odps.models.XFlow`
        :raise: :class:`odps.errors.NoSuchObject` if not exists

        .. seealso:: :class:`odps.models.XFlow`
        """

        project = self.get_project(name=project)
        return project.xflows[name]

    def exist_xflow(self, name, project=None):
        """
        If the xflow with given name exists or not.

        :param name: xflow name
        :param project: project name, if not provided, will be the default project
        :return: True if exists or False
        :rtype: bool
        """

        project = self.get_project(name=project)
        return name in project.xflows

    def run_xflow(self, xflow_name, xflow_project=None, parameters=None, project=None):
        """
        Run xflow by given name, xflow project, paremeters asynchronously.

        :param xflow_name: XFlow name
        :type xflow_name: str
        :param xflow_project: the project XFlow deploys
        :type xflow_project: str
        :param parameters: parameters
        :type parameters: dict
        :param project: project name, if not provided, will be the default project
        :return: instance
        :rtype: :class:`odps.models.Instance`

        .. seealso:: :class:`odps.models.Instance`
        """

        project = self.get_project(name=project)
        xflow_project = xflow_project or project
        if isinstance(xflow_project, models.Project):
            xflow_project = xflow_project.name
        return project.xflows.execute_xflow(
            xflow_name=xflow_name, xflow_project=xflow_project, project=project, parameters=parameters)

    def execute_xflow(self, xflow_name, xflow_project=None, parameters=None, project=None):
        """
        Run xflow by given name, xflow project, paremeters, block until xflow executed successfully.

        :param xflow_name: XFlow name
        :type xflow_name: str
        :param xflow_project: the project XFlow deploys
        :type xflow_project: str
        :param parameters: parameters
        :type parameters: dict
        :param project: project name, if not provided, will be the default project
        :return: instance
        :rtype: :class:`odps.models.Instance`

        .. seealso:: :class:`odps.models.Instance`
        """

        inst = self.run_xflow(
            xflow_name, xflow_project=xflow_project, parameters=parameters, project=project)
        inst.wait_for_success()
        return inst

    def get_xflow_results(self, instance, project=None):
        """
        The result given the instance of xflow

        :param instance: instance of xflow
        :type instance: :class:`odps.models.Instance`
        :param project: project name, if not provided, will be the default project
        :return: xflow result
        :rtype: dict
        """

        project = self.get_project(name=project)

        from .models import Instance
        if not isinstance(instance, Instance):
            instance = project.instances[instance]

        return project.xflows.get_xflow_results(instance)

    def delete_xflow(self, name, project=None):
        """
        Delete xflow by given name.

        :param name: xflow name
        :param project: project name, if not provided, will be the default project
        :return: None
        """

        project = self.get_project(name=project)

        return project.xflows.delete(name)

    def list_offline_models(self, project=None, prefix=None, owner=None):
        """
        List offline models of project by optional filter conditions including prefix and owner.

        :param project: project name, if not provided, will be the default project
        :param prefix: prefix of offline model's name
        :param owner: Aliyun account
        :return: offline models
        :rtype: list
        """

        project = self.get_project(name=project)
        return project.offline_models.iterate(name=prefix, owner=owner)

    def get_offline_model(self, name, project=None):
        """
        Get offline model by given name

        :param name: offline model name
        :param project: project name, if not provided, will be the default project
        :return: offline model
        :rtype: :class:`odps.models.ml.OfflineModel`
        :raise: :class:`odps.errors.NoSuchObject` if not exists
        """

        project = self.get_project(name=project)
        return project.offline_models[name]

    def exist_offline_model(self, name, project=None):
        """
        If the offline model with given name exists or not.

        :param name: offline model's name
        :param project: project name, if not provided, will be the default project
        :return: True if offline model exists else False
        :rtype: bool
        """

        project = self.get_project(name=project)
        return name in project.offline_models

    def delete_offline_model(self, name, project=None):
        """
        Delete the offline model by given name.

        :param name: offline model's name
        :param project: project name, if not provided, will be the default project
        :return: None
        """

        project = self.get_project(name=project)
        return project.offline_models.delete(name)

    def get_logview_address(self, instance_id, hours=None, project=None):
        """
        Get logview address by given instance id and hours.

        :param instance_id: instance id
        :param hours:
        :param project: project name, if not provided, will be the default project
        :return: logview address
        :rtype: str
        """
        hours = hours or options.log_view_hours
        inst = self.get_instance(instance_id, project=project)
        return inst.get_logview_address(hours=hours)

    @classmethod
    def _build_account(cls, access_id, secret_access_key):
        return accounts.AliyunAccount(access_id, secret_access_key)

    def to_global(self):
        options.access_id = self.account.access_id
        options.access_key = self.account.secret_access_key
        options.default_project = self.project
        options.end_point = self.endpoint

try:
    from odps.internal.core import *
except ImportError:
    pass

options.log_view_host = LOG_VIEW_HOST_DEFAULT
