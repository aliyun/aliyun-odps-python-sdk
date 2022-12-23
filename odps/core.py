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

import os
import json  # noqa: F401
import re
import time
import random
import warnings

from .rest import RestClient
from .config import options
from .errors import NoSuchObject
from .errors import ODPSError
from .tempobj import clean_stored_objects
from .utils import split_quoted
from .compat import six, Iterable
from . import models, accounts, errors, utils


DEFAULT_ENDPOINT = 'http://service.odps.aliyun.com/api'
LOGVIEW_HOST_DEFAULT = 'http://logview.odps.aliyun.com'

DROP_TABLE_REGEX = re.compile(r'^\s*drop\s+table\s*(|if\s+exists)\s+(?P<table_name>[^\s;]+)', re.I)


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
    :param tunnel_endpoint: Tunnel service URL
    :param logview_host: Logview host URL
    :param app_account: Application account, instance of `odps.accounts.AppAccount` used for dual authentication

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
    def __init__(self, access_id=None, secret_access_key=None, project=None,
                 endpoint=None, app_account=None, logview_host=None, **kw):
        self._init(access_id=access_id, secret_access_key=secret_access_key, project=project,
                   endpoint=endpoint, app_account=app_account, logview_host=logview_host, **kw)
        clean_stored_objects(self)

    def _init(self, access_id=None, secret_access_key=None, project=None,
              endpoint=None, **kw):
        account = kw.pop('account', None)
        self.app_account = kw.pop('app_account', None)

        if account is None:
            if access_id is not None:
                self.account = self._build_account(access_id, secret_access_key)
            elif options.account is not None:
                self.account = options.account
            else:
                raise TypeError('`access_id` and `secret_access_key` should be provided.')
        else:
            self.account = account
        self.endpoint = endpoint or options.endpoint or DEFAULT_ENDPOINT
        self.project = project or options.project
        self.rest = RestClient(self.account, self.endpoint, project, app_account=self.app_account,
                               proxy=options.api_proxy)

        self._tunnel_endpoint = kw.pop('tunnel_endpoint', options.tunnel.endpoint)

        self._logview_host = (
            kw.pop("logview_host", None)
            or self.get_logview_host()
            or options.logview_host
        )

        self._projects = models.Projects(client=self.rest)
        if project:
            self._project = self.get_project()

        self._seahawks_url = kw.pop("seahawks_url", None)
        if self._seahawks_url:
            options.seahawks_url = self._seahawks_url

        self._default_session = None
        self._default_session_name = ""

        if kw:
            raise TypeError(
                "Argument %s not acceptable, please check your spellings"
                % ", ".join(kw.keys()),
            )
        # Make instance to global
        if options.is_global_account_overwritable:
            self.to_global(overwritable=True)

    def __getstate__(self):
        params = dict(
            project=self.project,
            endpoint=self.endpoint,
            tunnel_endpoint=self._tunnel_endpoint,
            logview_host=self._logview_host,
            seahawks_url=self._seahawks_url
        )
        if isinstance(self.account, accounts.AliyunAccount):
            params.update(dict(access_id=self.account.access_id,
                               secret_access_key=self.account.secret_access_key))
        return params

    def __setstate__(self, state):
        if 'secret_access_key' in state:
            # if `secret_access_key` in state
            if os.environ.get('ODPS_ENDPOINT', None) is not None:
                state['endpoint'] = os.environ['ODPS_ENDPOINT']
            self._init(**state)
            return
        try:
            bearer_token = os.environ['ODPS_BEARER_TOKEN']
            state['project'] = os.environ['ODPS_PROJECT_NAME']
            state['endpoint'] = os.environ.get('ODPS_RUNTIME_ENDPOINT') or os.environ['ODPS_ENDPOINT']
            account = accounts.BearerTokenAccount(bearer_token)
            state.pop('access_id', None)
            state.pop('secret_access_key', None)
            self._init(None, None, account=account, **state)
        except KeyError:
            self._init(**state)

    def __mars_tokenize__(self):
        return self.__getstate__()

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

    @property
    def logview_host(self):
        return self._logview_host

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
        proj._logview_host = self._logview_host
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

        if isinstance(name, six.string_types):
            name = name.strip()
            if '.' in name:
                project, name = name.split('.', 1)

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

        if isinstance(name, six.string_types):
            name = name.strip()
            if '.' in name:
                project, name = name.split('.', 1)

        project = self.get_project(name=project)
        return name in project.tables

    def create_table(self, name, schema=None, project=None, comment=None, if_not_exists=False,
                     lifecycle=None, shard_num=None, hub_lifecycle=None, async_=False, **kw):
        """
        Create a table by given schema and other optional parameters.

        :param name: table name
        :param schema: table schema. Can be an instance of :class:`odps.models.TableSchema` or a string like 'col1 string, col2 bigint'
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
        :param async_: if True, will run asynchronously
        :type async_: bool
        :return: the created Table if not async else odps instance
        :rtype: :class:`odps.models.Table` or :class:`odps.models.Instance`

        .. seealso:: :class:`odps.models.Table`, :class:`odps.models.TableSchema`
        """

        if "table_schema" in kw:
            schema = kw["table_schema"]

        if isinstance(name, six.string_types):
            name = name.strip()
            if '.' in name:
                project, name = name.split('.', 1)

        if lifecycle is None and options.lifecycle is not None:
            lifecycle = options.lifecycle
        project = self.get_project(name=project)
        async_ = kw.pop('async', async_)
        return project.tables.create(name, schema, comment=comment, if_not_exists=if_not_exists,
                                     lifecycle=lifecycle, shard_num=shard_num,
                                     hub_lifecycle=hub_lifecycle, async_=async_, **kw)

    def delete_table(self, name, project=None, if_exists=False, async_=False, **kw):
        """
        Delete the table with given name

        :param name: table name
        :param project: project name, if not provided, will be the default project
        :param if_exists:  will not raise errors when the table does not exist, default False
        :param async_: if True, will run asynchronously
        :type async_: bool
        :return: None if not async else odps instance
        """

        if isinstance(name, six.string_types):
            name = name.strip()
            if '.' in name:
                project, name = name.split('.', 1)

        project = self.get_project(name=project)
        return project.tables.delete(name, if_exists=if_exists, async_=kw.get('async', async_))

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

        if isinstance(name, six.string_types):
            name = name.strip()
            if '.' in name:
                project, name = name.split('.', 1)
        if not isinstance(name, six.string_types):
            project, name = name.project.name, name.name

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

        project = None
        if isinstance(name, six.string_types):
            name = name.strip()
            if '.' in name:
                project, name = name.split('.', 1)
        if not isinstance(name, six.string_types):
            project, name = name.project.name, name.name

        project = self.get_project(name=kw.pop('project', project))
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

    def list_resources(self, project=None, prefix=None, owner=None):
        """
        List all resources of a project.

        :param project: project name, if not provided, will be the default project
        :return: resources
        :rtype: generator
        """

        project = self.get_project(name=project)
        for resource in project.resources.iterate(name=prefix, owner=owner):
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

    def create_resource(self, name, type=None, project=None, **kwargs):
        """
        Create a resource by given name and given type.

        Currently, the resource type can be ``file``, ``jar``, ``py``, ``archive``, ``table``.

        The ``file``, ``jar``, ``py``, ``archive`` can be classified into file resource.
        To init the file resource, you have to provide another parameter which is a file-like object.

        For the table resource, the table name, project name, and partition should be provided
        which the partition is optional.

        :param name: resource name
        :param type: resource type, now support ``file``, ``jar``, ``py``, ``archive``, ``table``
        :param project: project name, if not provided, will be the default project
        :param kwargs: optional arguments, I will illustrate this in the example below.
        :return: resource depends on the type, if ``file`` will be :class:`odps.models.FileResource` and so on
        :rtype: :class:`odps.models.Resource`'s subclasses

        :Example:

        >>> from odps.models.resource import *
        >>>
        >>> res = odps.create_resource('test_file_resource', 'file', fileobj=open('/to/path/file'))
        >>> assert isinstance(res, FileResource)
        >>> True
        >>>
        >>> res = odps.create_resource('test_py_resource.py', 'py', fileobj=StringIO('import this'))
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

        type_ = kwargs.get('typo') or type
        project = self.get_project(name=project)
        return project.resources.create(name=name, type=type_, **kwargs)

    def delete_resource(self, name, project=None):
        """
        Delete resource by given name.

        :param name: resource name
        :param project: project name, if not provided, will be the default project
        :return: None
        """

        project = self.get_project(name=project)
        return project.resources.delete(name)

    def list_functions(self, project=None, prefix=None, owner=None):
        """
        List all functions of a project.

        :param project: project name, if not provided, will be the default project
        :return: functions
        :rtype: generator
        """

        project = self.get_project(name=project)
        for function in project.functions.iterate(name=prefix, owner=owner):
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

        async_ = kwargs.pop('async_', kwargs.pop('async', False))

        inst = self.run_sql(
            sql, project=project, priority=priority, running_cluster=running_cluster,
            hints=hints, **kwargs)
        if not async_:
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
        task.update_sql_settings(hints)
        if aliases:
            task.update_aliases(aliases)

        project = self.get_project(name=project)
        return project.instances.create(task=task, priority=priority,
                                        running_cluster=running_cluster,
                                        create_callback=on_instance_create)

    def execute_sql_cost(self, sql, project=None, hints=None, **kwargs):
        """

        :param sql: SQL statement
        :type sql: str
        :param project: project name, if not provided, will be the default project
        :param hints: settings for SQL, e.g. `odps.mapred.map.split.size`
        :type hints: dict
        :return: cost info in dict format
        :rtype: cost: dict

        :Example:

        >>> sql_cost = odps.execute_sql_cost('select * from dual')
        >>> sql_cost.udf_num
        0
        >>> sql_cost.complexity
        1.0
        >>> sql_cost.input_size
        100

        """
        task = models.SQLCostTask(query=utils.to_text(sql), **kwargs)
        task.update_sql_cost_settings(hints)
        project = self.get_project(name=project)
        inst = project.instances.create(task=task)
        inst.wait_for_success()
        return inst.get_sql_task_cost()

    def _parse_partition_string(self, partition):
        parts = []
        for p in split_quoted(partition, ','):
            kv = [pp.strip() for pp in split_quoted(p, '=')]
            if len(kv) != 2:
                raise ValueError('Partition representation malformed.')
            if not kv[1].startswith('"') and not kv[1].startswith("'"):
                kv[1] = repr(kv[1])
            parts.append('%s=%s' % tuple(kv))
        return ','.join(parts)

    def execute_merge_files(self, table, partition=None, project=None, hints=None,
                            priority=None):
        """
        Execute a task to merge multiple files in tables and wait for termination.

        :param table: name of the table to optimize
        :param partition: partition to optimize
        :param project: project name, if not provided, will be the default project
        :param hints: settings for merge task.
        :param priority: instance priority, 9 as default
        :return: instance
        :rtype: :class:`odps.models.Instance`
        """
        inst = self.run_merge_files(table, partition=partition, project=project, hints=hints,
                                    priority=priority)
        inst.wait_for_success()
        return inst

    def run_merge_files(self, table, partition=None, project=None, hints=None,
                        priority=None):
        """
        Start running a task to merge multiple files in tables.

        :param table: name of the table to optimize
        :param partition: partition to optimize
        :param project: project name, if not provided, will be the default project
        :param hints: settings for merge task.
        :param priority: instance priority, 9 as default
        :return: instance
        :rtype: :class:`odps.models.Instance`
        """
        from .models.tasks import MergeTask

        if partition:
            table += " partition(%s)" % (self._parse_partition_string(partition))

        priority = priority or options.priority
        if priority is None and options.get_priority is not None:
            priority = options.get_priority(self)

        task = MergeTask(table=table)
        task.update_settings(hints)

        project = self.get_project(name=project)
        return project.instances.create(task=task, priority=priority)

    def execute_archive_table(self, table, partition=None, project=None, hints=None,
                              priority=None):
        """
        Execute a task to archive tables and wait for termination.

        :param table: name of the table to archive
        :param partition: partition to archive
        :param project: project name, if not provided, will be the default project
        :param hints: settings for table archive task.
        :param priority: instance priority, 9 as default
        :return: instance
        :rtype: :class:`odps.models.Instance`
        """
        inst = self.run_archive_table(table, partition=partition, project=project, hints=hints,
                                      priority=priority)
        inst.wait_for_success()
        return inst

    def run_archive_table(self, table, partition=None, project=None, hints=None, priority=None):
        """
        Start running a task to archive tables.

        :param table: name of the table to archive
        :param partition: partition to archive
        :param project: project name, if not provided, will be the default project
        :param hints: settings for table archive task.
        :param priority: instance priority, 9 as default
        :return: instance
        :rtype: :class:`odps.models.Instance`
        """
        from .models.tasks import MergeTask

        if partition:
            table += " partition(%s)" % (self._parse_partition_string(partition))

        priority = priority or options.priority
        if priority is None and options.get_priority is not None:
            priority = options.get_priority(self)

        name = 'archive_task_{0}_{1}'.format(int(time.time()), random.randint(100000, 999999))
        task = MergeTask(name=name, table=table)
        task.update_settings(hints)
        task._update_property_json('archiveSettings', {'odps.merge.archive.flag': True})

        project = self.get_project(name=project)
        return project.instances.create(task=task, priority=priority)

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

    def run_xflow(self, xflow_name, xflow_project=None, parameters=None, project=None,
                  hints=None, priority=None):
        """
        Run xflow by given name, xflow project, paremeters asynchronously.

        :param xflow_name: XFlow name
        :type xflow_name: str
        :param xflow_project: the project XFlow deploys
        :type xflow_project: str
        :param parameters: parameters
        :type parameters: dict
        :param project: project name, if not provided, will be the default project
        :param hints: execution hints
        :type hints: dict
        :param priority: instance priority, 9 as default
        :type priority: int
        :return: instance
        :rtype: :class:`odps.models.Instance`

        .. seealso:: :class:`odps.models.Instance`
        """
        project = self.get_project(name=project)
        xflow_project = xflow_project or project
        if isinstance(xflow_project, models.Project):
            xflow_project = xflow_project.name
        return project.xflows.run_xflow(
            xflow_name=xflow_name, xflow_project=xflow_project, project=project, parameters=parameters,
            hints=hints, priority=priority)

    def execute_xflow(self, xflow_name, xflow_project=None, parameters=None, project=None,
                      hints=None, priority=None):
        """
        Run xflow by given name, xflow project, paremeters, block until xflow executed successfully.

        :param xflow_name: XFlow name
        :type xflow_name: str
        :param xflow_project: the project XFlow deploys
        :type xflow_project: str
        :param parameters: parameters
        :type parameters: dict
        :param project: project name, if not provided, will be the default project
        :param hints: execution hints
        :type hints: dict
        :param priority: instance priority, 9 as default
        :type priority: int
        :return: instance
        :rtype: :class:`odps.models.Instance`

        .. seealso:: :class:`odps.models.Instance`
        """

        inst = self.run_xflow(
            xflow_name, xflow_project=xflow_project, parameters=parameters, project=project,
            hints=hints, priority=priority)
        inst.wait_for_success()
        return inst

    def get_xflow_results(self, instance, project=None):
        """
        The result given the results of xflow

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

    def get_xflow_sub_instances(self, instance, project=None):
        """
        The result iterates the sub instance of xflow

        :param instance: instance of xflow
        :type instance: :class:`odps.models.Instance`
        :param project: project name, if not provided, will be the default project
        :return: sub instances dictionary
        """
        project = self.get_project(name=project)
        return project.xflows.get_xflow_sub_instances(instance)

    def iter_xflow_sub_instances(self, instance, interval=1, project=None):
        """
        The result iterates the sub instance of xflow and will wait till instance finish

        :param instance: instance of xflow
        :type instance: :class:`odps.models.Instance`
        :param interval: time interval to check
        :param project: project name, if not provided, will be the default project
        :return: sub instances dictionary
        """
        project = self.get_project(name=project)
        return project.xflows.iter_xflow_sub_instances(instance, interval=interval)

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

    def copy_offline_model(self, name, new_name, project=None, new_project=None, async_=False):
        """
        Copy current model into a new location.

        :param new_name: name of the new model
        :param new_project: new project name. if absent, original project name will be used
        :param async_: if True, return the copy instance. otherwise return the newly-copied model
        """
        return self.get_offline_model(name, project=project) \
            .copy(new_name, new_project=new_project, async_=async_)

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

    def get_logview_host(self):
        """
        Get logview host address.
        :return: logview host address
        """
        try:
            logview_host = self.rest.get(self.endpoint + '/logview/host').text
        except:
            logview_host = None
        if not logview_host:
            logview_host = LOGVIEW_HOST_DEFAULT
        return logview_host

    def get_logview_address(self, instance_id, hours=None, project=None):
        """
        Get logview address by given instance id and hours.

        :param instance_id: instance id
        :param hours:
        :param project: project name, if not provided, will be the default project
        :return: logview address
        :rtype: str
        """
        hours = hours or options.logview_hours
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

    def run_security_query(self, query, project=None, token=None, output_json=True):
        """
        Run a security query to grant / revoke / query privileges. If the query is `install package`
        or `uninstall package`, return a waitable AuthQueryInstance object, otherwise returns
        the result string or json value.

        :param str query: query text
        :param str project: project name, if not provided, will be the default project
        :param bool output_json: parse json for the output
        :return: result string / json object
        """
        project = self.get_project(name=project)
        return project.run_security_query(query, token=token, output_json=output_json)

    def execute_security_query(self, query, project=None, token=None, output_json=True):
        """
        Execute a security query to grant / revoke / query privileges and returns
        the result string or json value.

        :param str query: query text
        :param str project: project name, if not provided, will be the default project
        :param bool output_json: parse json for the output
        :return: result string / json object
        """
        from .models import Project

        instance_or_result = self.run_security_query(
            query, project=project, token=token, output_json=output_json
        )
        if not isinstance(instance_or_result, Project.AuthQueryInstance):
            return instance_or_result
        return instance_or_result.wait_for_success()

    @utils.deprecated('You no longer have to manipulate session instances to use MaxCompute QueryAcceleration. Try `run_sql_interactive`.')
    def attach_session(self, session_name, taskname=None, hints=None):
        """
        Attach to an existing session.

        :param session_name: The session name.
        :param taskname: The created sqlrt task name. If not provided, the default value is used. Mostly doesn't matter, default works.
        :return: A SessionInstance you may execute select tasks within.
        """
        if not taskname:
            taskname = models.session.DEFAULT_TASK_NAME
        task = models.tasks.SQLRTTask(name=taskname)
        task.update_settings(hints)
        task.update_settings({"odps.sql.session.share.id": session_name,
                 "odps.sql.submit.mode": "script"})
        project = self.get_project()
        return project.instances.create(task=task,
                                        session_project=project,
                                        session_name=session_name)

    @utils.deprecated('You no longer have to manipulate session instances to use MaxCompute QueryAcceleration. Try `run_sql_interactive`.')
    def default_session(self):
        """
        Attach to the default session of your project.

        :return: A SessionInstance you may execute select tasks within.
        """
        return self.attach_session(models.session.PUBLIC_SESSION_NAME)

    @utils.deprecated('You no longer have to manipulate session instances to use MaxCompute QueryAcceleration. Try `run_sql_interactive`.')
    def create_session(self, session_worker_count, session_worker_memory,
            session_name=None, worker_spare_span=None, taskname=None, hints=None):
        """
        Create session.

        :param session_worker_count: How much workers assigned to the session.
        :param session_worker_memory: How much memory each worker consumes.
        :param session_name: The session name. Not specifying to use its ID as name.
        :param worker_spare_span: format "00-24", allocated workers will be reduced during this time. Not specifying to disable this.
        :param taskname: The created sqlrt task name. If not provided, the default value is used. Mostly doesn't matter, default works.
        :param hints: Extra hints provided to the session. Parameters of this method will override certain hints.
        :return: A SessionInstance you may execute select tasks within.
        """
        if not taskname:
            taskname = models.session.DEFAULT_TASK_NAME
        session_hints = {}
        session_hints["odps.sql.session.worker.count"] = str(session_worker_count)
        session_hints["odps.sql.session.worker.memory"] = str(session_worker_memory)
        session_hints["odps.sql.submit.mode"] = "script"
        if session_name:
            session_hints["odps.sql.session.name"] = session_name
        if worker_spare_span:
            session_hints["odps.sql.session.worker.sparespan"] = worker_spare_span
        task = models.tasks.SQLRTTask(name=taskname)
        task.update_settings(hints)
        task.update_settings(session_hints)
        project = self.get_project()
        return project.instances.create(task=task,
                                        session_project=project,
                                        session_name=session_name)

    def run_sql_interactive(self, sql, hints=None, **kwargs):
        """
        Run SQL query in interactive mode (a.k.a MaxCompute QueryAcceleration).
        Won't fallback to offline mode automatically if query not supported or fails
        :param sql: the sql query.
        :param hints: settings for sql query.
        :return: instance.
        """
        cached_is_running = False
        service_name = kwargs.pop('service_name', models.session.PUBLIC_SESSION_NAME)
        service_startup_timeout = kwargs.pop('service_startup_timeout', 60)
        force_reattach = kwargs.pop('force_reattach', False)
        if self._default_session != None:
            try:
                cached_is_running = self._default_session.is_running()
            except BaseException:
                pass
        if force_reattach or not cached_is_running or self._default_session_name != service_name:
            # should reattach, for whatever reason(timed out, terminated, never created, forced using another session)
            self._default_session = self.attach_session(service_name)
            self._default_session.wait_for_startup(0.1, service_startup_timeout)
            self._default_session_name = service_name
        return self._default_session.run_sql(sql, hints, **kwargs)

    def run_sql_interactive_with_fallback(self, sql, hints=None, **kwargs):
        """
        Run SQL query in interactive mode (a.k.a MaxCompute QueryAcceleration).
        If query is not supported or fails, will fallback to offline mode automatically
        :param sql: the sql query.
        :param hints: settings for sql query.
        :return: instance.
        """
        inst = None
        try:
            if inst is None:
                inst = self.run_sql_interactive(self, sql, hints=hints, **kwargs)
            else:
                inst.wait_for_success(interval=0.2)
            rd = inst.open_reader(tunnel=True, limit=False)
            if not rd:
                raise ODPSError('Get sql result fail')
            return inst
        except:
            return self.execute_sql(sql, hints=hints)

    @classmethod
    def _build_account(cls, access_id, secret_access_key):
        return accounts.AliyunAccount(access_id, secret_access_key)

    def to_global(self, overwritable=False):
        options.is_global_account_overwritable = overwritable
        options.account = self.account
        options.default_project = self.project
        options.endpoint = self.endpoint
        options.logview_host = self.logview_host
        options.tunnel.endpoint = self._tunnel_endpoint
        options.app_account = self.app_account

    @classmethod
    def from_global(cls):
        if options.account is not None and options.default_project is not None:
            return cls._from_account(
                options.account,
                options.default_project,
                endpoint=options.endpoint,
                tunnel_endpoint=options.tunnel.endpoint,
                logview_host=options.logview_host,
                app_account=options.app_account,
            )
        else:
            return None

    @classmethod
    def from_environments(cls):
        try:
            bearer_token = os.environ['ODPS_BEARER_TOKEN']
            project = os.environ['ODPS_PROJECT_NAME']
            endpoint = os.environ['ODPS_ENDPOINT']
            account = accounts.BearerTokenAccount(bearer_token)
            return cls(None, None, account=account, project=project, endpoint=endpoint)
        except KeyError:
            return None


def _get_odps_from_model(self):
    cur = self
    while cur is not None and not isinstance(cur, models.Project):
        cur = cur.parent
    return cur.odps if cur else None


models.RestModel.odps = property(fget=_get_odps_from_model)
del _get_odps_from_model

try:
    from .internal.core import *  # noqa: F401
except ImportError:
    pass

try:
    from .mars_extension import create_mars_cluster
    setattr(ODPS, 'create_mars_cluster', create_mars_cluster)
except ImportError:
    pass
try:
    from .mars_extension import persist_mars_dataframe
    setattr(ODPS, 'persist_mars_dataframe', persist_mars_dataframe)
except ImportError:
    pass
try:
    from .mars_extension import to_mars_dataframe
    setattr(ODPS, 'to_mars_dataframe', to_mars_dataframe)
except ImportError:
    pass
try:
    from .mars_extension import run_script_in_mars
    setattr(ODPS, 'run_script_in_mars', run_script_in_mars)
except ImportError:
    pass
try:
    from .mars_extension import run_mars_job
    setattr(ODPS, 'run_mars_job', run_mars_job)
except ImportError:
    pass
try:
    from .mars_extension import list_mars_instances
    setattr(ODPS, 'list_mars_instances', list_mars_instances)
except ImportError:
    pass
try:
    from .mars_extension import sql_to_mars_dataframe
    setattr(ODPS, 'sql_to_mars_dataframe', sql_to_mars_dataframe)
except ImportError:
    pass


if 'PYODPS_ENDPOINT' in os.environ:
    DEFAULT_ENDPOINT = os.environ.get('PYODPS_ENDPOINT')
