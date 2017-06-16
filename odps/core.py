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

import os
import json
import re
import warnings
from collections import Iterable

from .rest import RestClient
from .config import options
from .errors import NoSuchObject
from .tempobj import clean_stored_objects
from .compat import six
from . import models, accounts, errors, utils


DEFAULT_ENDPOINT = 'http://service.odps.aliyun.com/api'
DEFAULT_PREDICT_ENDPOINT = 'http://prediction.odps.aliyun.com'
LOG_VIEW_HOST_DEFAULT = 'http://logview.odps.aliyun.com'

DROP_TABLE_REGEX = re.compile('^\s*drop\s+table\s*(|if\s+exists)\s+(?P<table_name>[^\s;]+)', re.I)


@utils.attach_internal
class ODPS(object):
    """
    Main entrance to ODPS.

    Convenient operations on ODPS objects are provided.
    Please refer to `ODPS docs <https://docs.aliyun.com/#/pub/odps/basic/definition&project>`_
    to see the details.

    Generally, basic operations such as ``list``, ``get``, ``exist``, ``create``, ``delete``
    are provided for each ODPS object.
    Take the ``Table`` as an example.

    To create an ODPS instance, access_id and access_key is required, and should ensure correctness,
    or ``SignatureNotMatch`` error will throw. If `tunnel_endpoint` is not set, the tunnel API will
    route service URL automatically.

    :param access_id: Aliyun Access ID
    :param secret_access_key: Aliyun Access Key
    :param project: default project name
    :param endpoint: Rest service URL
    :param tunnel_endpoint:  Tunnel service URL

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
                 endpoint=None, **kw):
        """
        """

        account = kw.pop('account', None)
        if account is None:
            self.account = self._build_account(access_id, secret_access_key)
        else:
            self.account = account
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        self.project = project
        self.rest = RestClient(self.account, self.endpoint, project)

        self._tunnel_endpoint = kw.pop('tunnel_endpoint', None)
        if self._tunnel_endpoint is not None:
            options.tunnel.endpoint = self._tunnel_endpoint

        self._projects = models.Projects(client=self.rest)
        self._project = self.get_project()

        self._predict_endpoint = kw.pop('predict_endpoint', DEFAULT_PREDICT_ENDPOINT)
        if self._predict_endpoint is not None:
            options.predict_endpoint = self._predict_endpoint

        self._seahawks_url = None
        if kw.get('seahawks_url'):
            self._seahawks_url = kw.pop('seahawks_url', None)
            options.seahawks_url = self._seahawks_url

        clean_stored_objects(self)

    @classmethod
    def _from_account(cls, account, project, endpoint=DEFAULT_ENDPOINT,
                      tunnel_endpoint=None, **kwargs):
        return cls(None, None, project, endpoint=endpoint, tunnel_endpoint=tunnel_endpoint,
                   account=account, **kwargs)

    @property
    def projects(self):
        return self._projects

    def list_projects(self, owner=None, user=None, group=None, prefix=None, max_items=None):
        """
        List projects.

        :param owner: Aliyun account, the owner which listed projects belong to
        :param user: name of the user who has access to listed projects
        :param group: name of the group listed projects belong to
        :param prefix: prefix of names of listed projects
        :param max_items: the maximal size of result set
        :return: projects in this endpoint.
        :rtype: generator
        """
        return self.projects.iterate(owner=owner, user=user, group=group, max_items=max_items, name=prefix)

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
            name = self.project
        elif isinstance(name, models.Project):
            return name
        proj = self._projects[name]
        proj._tunnel_endpoint = self._tunnel_endpoint
        return proj

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
        :param schema: table schema. Can be an instance of :class:`odps.models.Schema` or a string like 'col1 string, col2 bigint'
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
        :param if_exists:  will not raise errors when the table does not exist, default False
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

    def list_instances(self, project=None, start_time=None, end_time=None,
                       status=None, only_owner=None, quota_index=None, **kw):
        """
        List instances of a project by given optional conditions
        including start time, end time, status and if only the owner.

        :param project: project name, if not provided, will be the default project
        :param start_time: the start time of filtered instances
        :type start_time: datetime, int or float
        :param end_time: the end time of filtered instances
        :type end_time: datetime, int or float
        :param status: including 'Running', 'Suspended', 'Terminated'
        :param only_owner: True will filter the instances created by current user
        :type only_owner: bool
        :param quota_index:
        :type quota_index: str
        :return: instances
        :rtype: list
        """
        if 'from_time' in kw:
            start_time = kw['from_time']
            warnings.warn('The keyword argument `from_time` has been replaced by `start_time`.')

        project = self.get_project(name=project)
        return project.instances.iterate(
            start_time=start_time, end_time=end_time,
            status=status, only_owner=only_owner, quota_index=quota_index)

    def list_instance_queueing_infos(self, project=None, status=None, only_owner=None,
                                     quota_index=None):
        """
        List instance queueing information.

        :param project: project name, if not provided, will be the default project
        :param status: including 'Running', 'Suspended', 'Terminated'
        :param only_owner: True will filter the instances created by current user
        :type only_owner: bool
        :param quota_index:
        :type quota_index: str
        :return: instance queueing infos
        :rtype: list
        """

        project = self.get_project(name=project)
        return project.instance_queueing_infos.iterate(
            status=status, only_owner=only_owner, quota_index=quota_index)

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

        priority = priority or options.priority
        if priority is None and options.get_priority is not None:
            priority = options.get_priority(self)
        on_instance_create = kwargs.pop('on_instance_create', None)

        drop_table_match = DROP_TABLE_REGEX.match(sql)
        if drop_table_match:
            drop_table_name = drop_table_match.group('table_name').strip('`')
            del self.get_project(project).tables[drop_table_name]

        task = models.SQLTask(query=utils.to_text(sql), **kwargs)
        if options.sql.settings:
            task.update_settings(options.sql.settings)
        if hints:
            task.update_settings(hints)
        if aliases:
            task.update_aliases(aliases)

        if options.biz_id:
            if task.properties is None:
                task.properties = dict()
            task.properties['biz_id'] = str(options.biz_id)

        project = self.get_project(name=project)
        return project.instances.create(task=task, priority=priority,
                                        running_cluster=running_cluster,
                                        create_callback=on_instance_create)

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

    @utils.deprecated('`create_volume` is deprecated. Use `created_parted_volume` instead.')
    def create_volume(self, name, project=None, **kwargs):
        self.create_parted_volume(name, project=project, **kwargs)

    def create_parted_volume(self, name, project=None, **kwargs):
        """
        Create an old-fashioned partitioned volume in a project.

        :param str name: volume name name
        :param str project: project name, if not provided, will be the default project
        :return: volume
        :rtype: :class:`odps.models.PartedVolume`

        .. seealso:: :class:`odps.models.PartedVolume`
        """
        project = self.get_project(name=project)
        return project.volumes.create_parted(name=name, **kwargs)

    def create_fs_volume(self, name, project=None, **kwargs):
        """
        Create a new-fashioned file system volume in a project.

        :param str name: volume name name
        :param str project: project name, if not provided, will be the default project
        :return: volume
        :rtype: :class:`odps.models.FSVolume`

        .. seealso:: :class:`odps.models.FSVolume`
        """
        project = self.get_project(name=project)
        return project.volumes.create_fs(name=name, **kwargs)

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
        Get volume by given name.

        :param str name: volume name
        :param str project: project name, if not provided, will be the default project
        :return: volume object. Return type depends on the type of the volume.
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
        return volume.partitions.iterate()

    def get_volume_partition(self, volume, partition=None, project=None):
        """
        Get partition in a parted volume by given name.

        :param str volume: volume name
        :param str partition: partition name
        :param str project: project name, if not provided, will be the default project
        :return: partitions
        :rtype: :class:`odps.models.VolumePartition`
        """
        if partition is None:
            if not volume.startswith('/') or '/' not in volume.lstrip('/'):
                raise ValueError('You should provide a partition name or use partition path instead.')
            volume, partition = volume.lstrip('/').split('/', 1)
        volume = self.get_volume(volume, project)
        return volume.partitions[partition]

    def exist_volume_partition(self, volume, partition=None, project=None):
        """
        If the volume with given name exists in a partition or not.

        :param str volume: volume name
        :param str partition: partition name
        :param str project: project name, if not provided, will be the default project
        """
        if partition is None:
            if not volume.startswith('/') or '/' not in volume.lstrip('/'):
                raise ValueError('You should provide a partition name or use partition path instead.')
            volume, partition = volume.lstrip('/').split('/', 1)
        try:
            volume = self.get_volume(volume, project)
        except errors.NoSuchObject:
            return False
        return partition in volume.partitions

    def delete_volume_partition(self, volume, partition=None, project=None):
        """
        Delete partition in a volume by given name

        :param str volume: volume name
        :param str partition: partition name
        :param str project: project name, if not provided, will be the default project
        """
        if partition is None:
            if not volume.startswith('/') or '/' not in volume.lstrip('/'):
                raise ValueError('You should provide a partition name or use partition path instead.')
            volume, partition = volume.lstrip('/').split('/', 1)
        volume = self.get_volume(volume, project)
        return volume.delete_partition(partition)

    def list_volume_files(self, volume, partition=None, project=None):
        """
        List files in a volume. In partitioned volumes, the function returns files under specified partition.
        In file system volumes, the function returns files under specified path.

        :param str volume: volume name
        :param str partition: partition name for partitioned volumes, and path for file system volumes.
        :param str project: project name, if not provided, will be the default project
        :return: files
        :rtype: list

        :Example:
        >>> # List files under a partition in a partitioned volume. Two calls are equivalent.
        >>> odps.list_volume_files('parted_volume', 'partition_name')
        >>> odps.list_volume_files('/parted_volume/partition_name')
        >>> # List files under a path in a file system volume. Two calls are equivalent.
        >>> odps.list_volume_files('fs_volume', 'dir1/dir2')
        >>> odps.list_volume_files('/fs_volume/dir1/dir2')
        """
        from .models import PartedVolume
        if partition is None:
            if not volume.startswith('/'):
                raise ValueError('You should provide a partition name or use partition / path instead.')
            volume = volume.lstrip('/')
            if '/' in volume:
                volume, partition = volume.split('/', 1)
        volume = self.get_volume(volume, project)
        if isinstance(volume, PartedVolume):
            if not partition:
                raise ValueError('Malformed partition url.')
            return volume.partitions[partition].files.iterate()
        else:
            return volume[partition].objects.iterate()

    def create_volume_directory(self, volume, path=None, project=None):
        """
        Create a directory under a file system volume.

        :param str volume: name of the volume.
        :param str path: path of the directory to be created.
        :param str project: project name, if not provided, will be the default project.
        :return: directory object.
        """
        from .models import PartedVolume
        if path is None:
            if not volume.startswith('/'):
                raise ValueError('You should provide a valid path.')
            volume = volume.lstrip('/')
            if '/' in volume:
                volume, path = volume.split('/', 1)
        volume = self.get_volume(volume, project)
        if isinstance(volume, PartedVolume):
            raise ValueError('Only supported under file system volumes.')
        else:
            return volume.create_dir(path)

    def get_volume_file(self, volume, path=None, project=None):
        """
        Get a file under a partition of a parted volume, or a file / directory object under a file system volume.

        :param str volume: name of the volume.
        :param str path: path of the directory to be created.
        :param str project: project name, if not provided, will be the default project.
        :return: directory object.
        """
        from .models import PartedVolume
        if path is None:
            if not volume.startswith('/'):
                raise ValueError('You should provide a valid path.')
            volume = volume.lstrip('/')
            if '/' in volume:
                volume, path = volume.split('/', 1)
        volume = self.get_volume(volume, project)
        if isinstance(volume, PartedVolume):
            if '/' not in path:
                raise ValueError('Partition/File format malformed.')
            part, file_name = path.split('/', 1)
            return volume.get_partition(part).files[file_name]
        else:
            return volume[path]

    def move_volume_file(self, old_path, new_path, replication=None, project=None):
        """
        Move a file / directory object under a file system volume to another location in the same volume.

        :param str old_path: old path of the volume file.
        :param str new_path: target path of the moved file.
        :param int replication: file replication.
        :param str project: project name, if not provided, will be the default project.
        :return: directory object.
        """
        from .models import PartedVolume

        if not new_path.startswith('/'):
            # make relative path absolute
            old_root, _ = old_path.rsplit('/', 1)
            new_path = old_root + '/' + new_path

        if not old_path.startswith('/'):
            raise ValueError('You should provide a valid path.')
        old_volume, old_path = old_path.lstrip('/').split('/', 1)

        new_volume, _ = new_path.lstrip('/').split('/', 1)

        if old_volume != new_volume:
            raise ValueError('Moving between different volumes is not supported.')

        volume = self.get_volume(old_volume, project)
        if isinstance(volume, PartedVolume):
            raise ValueError('Only supported under file system volumes.')
        else:
            volume[old_path].move(new_path, replication=replication)

    def delete_volume_file(self, volume, path=None, recursive=False, project=None):
        """
        Delete a file / directory object under a file system volume.

        :param str volume: name of the volume.
        :param str path: path of the directory to be created.
        :param bool recursive: if True, recursively delete files
        :param str project: project name, if not provided, will be the default project.
        :return: directory object.
        """
        from .models import PartedVolume
        if path is None:
            if not volume.startswith('/'):
                raise ValueError('You should provide a valid path.')
            volume = volume.lstrip('/')
            if '/' in volume:
                volume, path = volume.split('/', 1)
        volume = self.get_volume(volume, project)
        if isinstance(volume, PartedVolume):
            raise ValueError('Only supported under file system volumes.')
        else:
            volume[path].delete(recursive=recursive)

    def open_volume_reader(self, volume, partition=None, file_name=None, project=None,
                           start=None, length=None, **kwargs):
        """
        Open a volume file for read. A file-like object will be returned which can be used to read contents from
        volume files.

        :param str volume: name of the volume
        :param str partition: name of the partition
        :param str file_name: name of the file
        :param str project: project name, if not provided, will be the default project
        :param start: start position
        :param length: length limit
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: CompressOption

        :Example:
        >>> with odps.open_volume_reader('parted_volume', 'partition', 'file') as reader:
        >>>     [print(line) for line in reader]
        """
        from .models import PartedVolume
        if partition is None:
            if not volume.startswith('/'):
                raise ValueError('You should provide a partition name or use partition / path instead.')
            volume = volume.lstrip('/')
            volume, partition = volume.split('/', 1)
            if '/' in partition:
                partition, file_name = partition.rsplit('/', 1)
            else:
                partition, file_name = None, partition
        volume = self.get_volume(volume, project)
        if isinstance(volume, PartedVolume):
            if not partition:
                raise ValueError('Malformed partition url.')
            return volume.partitions[partition].open_reader(file_name, start=start, length=length, **kwargs)
        else:
            return volume[partition].open_reader(file_name, start=start, length=length, **kwargs)

    def open_volume_writer(self, volume, partition=None, project=None, **kwargs):
        """
        Write data into a volume. This function behaves differently under different types of volumes.

        Under partitioned volumes, all files under a partition should be uploaded in one submission. The method
        returns a writer object with whose `open` method you can open a file inside the volume and write to it,
        or you can use `write` method to write to specific files.

        Under file system volumes, the method returns a file-like object.

        :param str volume: name of the volume
        :param str partition: partition name for partitioned volumes, and path for file system volumes.
        :param str project: project name, if not provided, will be the default project
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`

        :Example:
        >>> # Writing to partitioned volumes
        >>> with odps.open_volume_writer('parted_volume', 'partition') as writer:
        >>>     # both write methods are acceptable
        >>>     writer.open('file1').write('some content')
        >>>     writer.write('file2', 'some content')
        >>> # Writing to file system volumes
        >>> with odps.open_volume_writer('/fs_volume/dir1/file_name') as writer:
        >>>     writer.write('some content')
        """
        from .models import PartedVolume
        if partition is None:
            if not volume.startswith('/'):
                raise ValueError('You should provide a partition name or use partition / path instead.')
            volume = volume.lstrip('/')
            volume, partition = volume.split('/', 1)
        volume = self.get_volume(volume, project)
        if isinstance(volume, PartedVolume):
            return volume.partitions[partition].open_writer(**kwargs)
        else:
            if '/' in partition:
                partition, file_name = partition.rsplit('/', 1)
            else:
                partition, file_name = None, partition
            return volume[partition].open_writer(file_name, **kwargs)

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
        return project.xflows.run_xflow(
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

    def delete_offline_model(self, name, project=None, if_exists=False):
        """
        Delete the offline model by given name.

        :param name: offline model's name
        :param if_exists:  will not raise errors when the offline model does not exist, default False
        :param project: project name, if not provided, will be the default project
        :return: None
        """

        project = self.get_project(name=project)
        try:
            return project.offline_models.delete(name)
        except NoSuchObject:
            if not if_exists:
                raise

    def create_online_model(self, name, *args, **kwargs):
        """
        Create an online model based on offline model names or custom pipeline.

        :param str name: model name
        :param str project: model name
        :param async: whether to wait for creation completed

        :return: online model

        This method can be used to create online models based on offline models or customized
        pipelines. For offline models, the method call should be

        >>> model = odps.create_online_model('**online_model_name**', '**offline_model_name**')

        For customized models, the method call should be

        >>> from odps.models.ml import ModelPredictor, CustomProcessor
        >>> processor = CustomProcessor(class_name='**class**', lib='**library name**',
                                        resources=['**resource name**', ],  config='**configuration**')
        >>> predictor = ModelPredictor(runtime='Jar or Native', instance_num=5, pipeline=[processor, ],
                                        target_name='**target name**')
        >>> model = odps.create_online_model('**online_model_name**', predictor)

        .. seealso:: :class:`odps.models.ml.OnlineModel`

        """
        # def create_online_model(self, name, offline_model, project=None, qos=100, offline_model_project=None,
        #                         wait=True, interval=1):
        from odps.models.ml.onlinemodel import ModelPredictor
        offline_model = kwargs.get('offline_model', None)
        if args and isinstance(args[0], six.string_types):
            offline_model = offline_model or args[0]
        predictor = kwargs.get('predictor', None)
        if args and isinstance(args[0], ModelPredictor):
            predictor = args[0]
        if predictor is not None and offline_model is not None:
            raise ValueError('You cannot supply both offline_model and predictor at the same time.')
        elif predictor is None and offline_model is None:
            raise ValueError('You should supply at least one argument in offline_model and predictor.')

        project = kwargs.pop('project', None)
        project = self.get_project(name=project)
        if offline_model:
            from .models.ml import OfflineModel
            if isinstance(offline_model, OfflineModel):
                kwargs['offline_model_project'] = offline_model.project.name
                offline_model = offline_model.name
            if 'offline_model_project' not in kwargs:
                kwargs['offline_model_project'] = self.project
            return project.online_models.create(name=name, offline_model_name=offline_model, runtime='Native', **kwargs)
        elif predictor:
            return project.online_models.create(name=name, predictor=predictor, **kwargs)
        else:
            raise ValueError('Cannot determine which type of model to create.')

    def list_online_models(self, project=None, prefix=None, owner=None):
        """
        List offline models of project by optional filter conditions including prefix and owner.

        :param project: project name, if not provided, will be the default project
        :param prefix: prefix of offline model's name
        :param owner: Aliyun account
        :return: offline models
        :rtype: list
        """

        project = self.get_project(name=project)
        return project.online_models.iterate(name=prefix, owner=owner)

    def get_online_model(self, name, project=None):
        """
        Get online model by given name

        :param name: name of the online model
        :param project: project name, if not provided, will be the default project
        :return: offline model
        :rtype: :class:`odps.models.ml.OfflineModel`
        :raise: :class:`odps.errors.NoSuchObject` if not exists
        """

        project = self.get_project(name=project)
        return project.online_models[name]

    def exist_online_model(self, name, project=None):
        """
        If the online model with given name exists or not.

        :param name: name of the online model
        :param project: project name, if not provided, will be the default project
        :return: True if offline model exists else False
        :rtype: bool
        """

        project = self.get_project(name=project)
        return name in project.online_models

    def predict_online_model(self, name, data, schema=None, project=None, endpoint=None):
        """
        Get prediction result with provided data on specified online model.

        :param name: name of the online model
        :param data: data to be predicted
        :param schema: schema of input data
        :param project: name of the project where the online model is located
        :param endpoint: endpoint of predict service
        :return: prediction result

        The input data can be an :class:`odps.models.Record` instance or a dictionary providing column names as keys
        and values as values. In such cases, the parameter `schema` can be neglected. Otherwise a list of column names
        should be provided.

        Multiple values organized in lists or tuples is also acceptable.

        The output result object has three fields. The `label` field provides predicted label, the `value` field
        provides predicted value, while the `scores` field provides a dictionary listing scores for every possible
        labels.
        """
        endpoint = endpoint or self._predict_endpoint or options.predict_endpoint
        return self.get_online_model(name, project=project).predict(data, schema=schema, endpoint=endpoint)

    def config_online_model_ab_test(self, name, *args, **kwargs):
        """
        Config AB-Test percentages of the online model.

        :param name: name of the online model

        This method should be called like

        >>> result = odps.config_online_model_ab_test('**online_model_name**', model1, percentage1, model2, percentage2, ...)

        where `modelx` can be model names or :class:`odps.models.ml.OnlineModel` instances, while `percentagex` should
        be percentage of `modelx` in AB-Test, ranging from 0 to 100.
        """
        return self.get_online_model(name, project=kwargs.get('project')).config_ab_test(*args)

    def delete_online_model(self, name, project=None, async=False):
        """
        Delete the online model by given name.

        :param name: name of the online model
        :param project: project name, if not provided, will be the default project
        :param async: whether to wait for deletion completed
        :return: None
        """

        project = self.get_project(name=project)
        return project.online_models.delete(name, async=async)

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

    def get_project_policy(self, project=None):
        """
        Get policy of a project

        :param project: project name, if not provided, will be the default project
        :return: JSON object
        """
        project = self.get_project(name=project)
        return project.policy

    def set_project_policy(self, policy, project=None):
        """
        Set policy of a project

        :param policy: name of policy.
        :param project: project name, if not provided, will be the default project
        :return: JSON object
        """
        project = self.get_project(name=project)
        project.policy = policy

    def create_role(self, name, project=None):
        """
        Create a role in a project

        :param name: name of the role to create
        :param project: project name, if not provided, will be the default project
        :return: role object created
        """
        project = self.get_project(name=project)
        return project.roles.create(name)

    def list_roles(self, project=None):
        """
        List all roles in a project

        :param project: project name, if not provided, will be the default project
        :return: collection of role objects
        """
        project = self.get_project(name=project)
        return project.roles

    def exist_role(self, name, project=None):
        """
        Check if a role exists in a project

        :param name: name of the role
        :param project: project name, if not provided, will be the default project
        """
        project = self.get_project(name=project)
        return name in project.roles

    def delete_role(self, name, project=None):
        """
        Delete a role in a project

        :param name: name of the role to delete
        :param project: project name, if not provided, will be the default project
        """
        project = self.get_project(name=project)
        project.roles.delete(name)

    def get_role_policy(self, name, project=None):
        """
        Get policy object of a role

        :param name: name of the role
        :param project: project name, if not provided, will be the default project
        :return: JSON object
        """
        project = self.get_project(name=project)
        return project.roles[name].policy

    def set_role_policy(self, name, policy, project=None):
        """
        Get policy object of project

        :param name: name of the role
        :param policy: policy string or JSON object
        :param project: project name, if not provided, will be the default project
        """
        project = self.get_project(name=project)
        project.roles[name].policy = policy

    def list_role_users(self, name, project=None):
        """
        List users who have the specified role.

        :param name: name of the role
        :param project: project name, if not provided, will be the default project
        :return: collection of User objects
        """
        project = self.get_project(name=project)
        return project.roles[name].users

    def create_user(self, name, project=None):
        """
        Add a user into the project

        :param name: user name
        :param project: project name, if not provided, will be the default project
        :return: user created
        """
        project = self.get_project(name=project)
        return project.users.create(name)

    def list_users(self, project=None):
        """
        List users in the project

        :param project: project name, if not provided, will be the default project
        :return: collection of User objects
        """
        project = self.get_project(name=project)
        return project.users

    def exist_user(self, name, project=None):
        """
        Check if a user exists in the project

        :param name: user name
        :param project: project name, if not provided, will be the default project
        """
        project = self.get_project(name=project)
        return name in project.users

    def delete_user(self, name, project=None):
        """
        Delete a user from the project

        :param name: user name
        :param project: project name, if not provided, will be the default project
        """
        project = self.get_project(name=project)
        project.users.delete(name)

    def list_user_roles(self, name, project=None):
        """
        List roles of the specified user

        :param name: user name
        :param project: project name, if not provided, will be the default project
        :return: collection of Role object
        """
        project = self.get_project(name=project)
        return project.users[name].roles

    def get_security_options(self, project=None):
        """
        Get all security options of a project

        :param project: project name, if not provided, will be the default project
        :return: SecurityConfiguration object
        """
        project = self.get_project(name=project)
        return project.security_options

    def get_security_option(self, option_name, project=None):
        """
        Get one security option of a project

        :param option_name: name of the security option. Please refer to ODPS options for more details.
        :param project: project name, if not provided, will be the default project
        :return: option value
        """
        option_name = utils.camel_to_underline(option_name)
        sec_options = self.get_security_options(project=project)
        if not hasattr(sec_options, option_name):
            raise ValueError('Option does not exists.')
        return getattr(sec_options, option_name)

    def set_security_option(self, option_name, value, project=None):
        """
        Set a security option of a project

        :param option_name: name of the security option. Please refer to ODPS options for more details.
        :param value: value of security option to be set.
        :param project: project name, if not provided, will be the default project.
        """
        option_name = utils.camel_to_underline(option_name)
        sec_options = self.get_security_options(project=project)
        if not hasattr(sec_options, option_name):
            raise ValueError('Option does not exists.')
        setattr(sec_options, option_name, value)
        sec_options.update()

    def run_security_query(self, query, project=None, token=None):
        """
        Run a security query to grant / revoke / query privileges

        :param query: query text
        :param project: project name, if not provided, will be the default project
        :return: a JSON object representing the result.
        """
        project = self.get_project(name=project)
        return project.run_security_query(query, token=token)

    @classmethod
    def _build_account(cls, access_id, secret_access_key):
        return accounts.AliyunAccount(access_id, secret_access_key)

    def to_global(self):
        options.account = self.account
        options.default_project = self.project
        options.end_point = self.endpoint
        options.tunnel.endpoint = self._tunnel_endpoint

    @classmethod
    def from_global(cls):
        if options.account is not None and options.default_project is not None:
            return cls._from_account(options.account, options.default_project, endpoint=options.end_point)
        else:
            return None

def _get_odps_from_model(self):
    client = self._client
    account = client.account
    endpoint = client.endpoint
    project = client.project

    return ODPS._from_account(account, project, endpoint=endpoint)

models.RestModel.odps = property(fget=_get_odps_from_model)
del _get_odps_from_model

try:
    from .internal.core import *
except ImportError:
    pass

options.log_view_host = LOG_VIEW_HOST_DEFAULT
if 'PYODPS_ENDPOINT' in os.environ:
    DEFAULT_ENDPOINT = os.environ.get('PYODPS_ENDPOINT')
if 'PYODPS_PREDICT_ENDPOINT' in os.environ:
    DEFAULT_PREDICT_ENDPOINT = os.environ.get('PYODPS_PREDICT_ENDPOINT')
