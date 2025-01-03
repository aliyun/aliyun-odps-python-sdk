# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import functools
import inspect
import json  # noqa: F401
import os
import re
import warnings
import weakref

from . import accounts, errors, models, utils
from .compat import six, urlparse
from .config import options
from .rest import RestClient
from .tempobj import clean_stored_objects

DEFAULT_ENDPOINT = "http://service.odps.aliyun.com/api"
DEFAULT_REGION_NAME = "cn"
LOGVIEW_HOST_DEFAULT = "http://logview.aliyun.com"
JOB_INSIGHT_HOST_DEFAULT = "https://maxcompute.console.aliyun.com"

_ALTER_TABLE_REGEX = re.compile(
    r"^\s*(drop|alter)\s+table\s*(|if\s+exists)\s+(?P<table_name>[^\s;]+)", re.I
)
_ENDPOINT_HOST_WITH_REGION_REGEX = re.compile(
    r"service\.([^\.]+)\.(odps|maxcompute)\.aliyun(|-inc)\.com", re.I
)

_logview_host_cache = dict()


def _wrap_model_func(func):
    @six.wraps(func)
    def wrapped(self, *args, **kw):
        return func(self, *args, **kw)

    # keep method signature to avoid doc issue
    if hasattr(inspect, "signature"):
        wrapped.__signature__ = inspect.signature(func)
    return wrapped


@utils.attach_internal
class ODPS(object):
    """
    Main entrance to ODPS.

    Convenient operations on ODPS objects are provided.
    Please refer to `ODPS docs <https://help.aliyun.com/document_detail/27818.html>`_
    for more details.

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

    def __init__(
        self,
        access_id=None,
        secret_access_key=None,
        project=None,
        endpoint=None,
        schema=None,
        app_account=None,
        logview_host=None,
        tunnel_endpoint=None,
        region_name=None,
        quota_name=None,
        namespace=None,
        **kw
    ):
        # avoid polluted copy sources :(
        access_id = utils.strip_if_str(access_id)
        secret_access_key = utils.strip_if_str(secret_access_key)
        project = utils.strip_if_str(project)
        endpoint = utils.strip_if_str(endpoint)
        schema = utils.strip_if_str(schema)
        logview_host = utils.strip_if_str(logview_host)
        tunnel_endpoint = utils.strip_if_str(tunnel_endpoint)
        region_name = utils.strip_if_str(region_name)
        quota_name = utils.strip_if_str(quota_name)
        namespace = utils.strip_if_str(namespace)

        if isinstance(access_id, accounts.BaseAccount):
            assert (
                secret_access_key is None
            ), "Cannot supply secret_access_key with an account"
            kw["account"], access_id = access_id, None

        self._init(
            access_id=access_id,
            secret_access_key=secret_access_key,
            project=project,
            endpoint=endpoint,
            schema=schema,
            app_account=app_account,
            logview_host=logview_host,
            tunnel_endpoint=tunnel_endpoint,
            region_name=region_name,
            quota_name=quota_name,
            namespace=namespace,
            **kw
        )
        clean_stored_objects(self)

    def _init(
        self,
        access_id=None,
        secret_access_key=None,
        project=None,
        endpoint=None,
        schema=None,
        region_name=None,
        namespace=None,
        **kw
    ):
        self._property_update_callbacks = set()

        account = kw.pop("account", None)
        self.app_account = kw.pop("app_account", None)

        if account is None:
            if access_id is not None:
                self.account = self._build_account(access_id, secret_access_key)
            elif options.account is not None:
                self.account = options.account
            else:
                self.account = accounts.from_environments()
                if self.account is None:
                    raise TypeError(
                        "`access_id` and `secret_access_key` should be provided."
                    )
        else:
            self.account = account
        self.endpoint = (
            endpoint
            or options.endpoint
            or os.getenv("ODPS_ENDPOINT")
            or DEFAULT_ENDPOINT
        )
        self.project = (
            project or options.default_project or os.getenv("ODPS_PROJECT_NAME")
        )
        self.region_name = region_name or self._get_region_from_endpoint(self.endpoint)
        self.namespace = (
            namespace or options.default_namespace or os.getenv("ODPS_NAMESPACE")
        )
        self._quota_name = kw.pop("quota_name", None)
        self._schema = schema

        rest_client_cls = kw.pop("rest_client_cls", None) or RestClient
        rest_client_kwargs = kw.pop("rest_client_kwargs", {})
        self.rest = rest_client_cls(
            self.account,
            self.endpoint,
            project,
            schema,
            app_account=self.app_account,
            proxy=options.api_proxy,
            region_name=self.region_name,
            namespace=self.namespace,
            tag="ODPS",
            **rest_client_kwargs
        )

        self._tunnel_endpoint = (
            kw.pop("tunnel_endpoint", None)
            or options.tunnel.endpoint
            or os.getenv("ODPS_TUNNEL_ENDPOINT")
        )

        self._logview_host = (
            kw.pop("logview_host", None)
            or options.logview_host
            or os.getenv("ODPS_LOGVIEW_HOST")
            or self.get_logview_host()
        )
        self._job_insight_host = (
            JOB_INSIGHT_HOST_DEFAULT
            if utils.is_job_insight_available(self.endpoint)
            or options.use_legacy_logview is False
            else None
        )

        self._default_tenant = models.Tenant(client=self.rest)

        self._projects = models.Projects(client=self.rest, _odps_ref=weakref.ref(self))
        if project:
            self._project = self.get_project()

        self._quotas = models.Quotas(client=self.rest)
        if self._quota_name:
            self._quota = self.get_quota()

        self._seahawks_url = kw.pop("seahawks_url", None)
        if self._seahawks_url:
            options.seahawks_url = self._seahawks_url

        self._default_session = None
        self._default_session_name = ""

        # Make instance to global
        overwrite_global = kw.pop("overwrite_global", True)
        if overwrite_global and options.is_global_account_overwritable:
            self.to_global(overwritable=True)

        if kw:
            raise TypeError(
                "Argument %s not acceptable, please check your spellings"
                % ", ".join(kw.keys()),
            )

    @staticmethod
    def _get_region_from_endpoint(endpoint):
        parsed = urlparse(endpoint)
        match = _ENDPOINT_HOST_WITH_REGION_REGEX.search(parsed.hostname or "")
        if match is None:
            return DEFAULT_REGION_NAME
        return match.group(1)

    def __getstate__(self):
        params = dict(
            project=self.project,
            endpoint=self.endpoint,
            tunnel_endpoint=self._tunnel_endpoint,
            logview_host=self._logview_host,
            schema=self.schema,
            seahawks_url=self._seahawks_url,
        )
        if utils.str_to_bool(os.environ.get("PYODPS_PICKLE_ACCOUNT") or "false"):
            params.update(dict(account=self.account))
        elif isinstance(self.account, accounts.AliyunAccount):
            params.update(
                dict(
                    access_id=self.account.access_id,
                    secret_access_key=self.account.secret_access_key,
                )
            )
        return params

    def __setstate__(self, state):
        if "secret_access_key" in state:
            if os.environ.get("ODPS_ENDPOINT", None) is not None:
                state["endpoint"] = os.environ["ODPS_ENDPOINT"]
            self._init(**state)
            return

        bearer_token_account = accounts.BearerTokenAccount.from_environments()
        if bearer_token_account is not None:
            state["project"] = os.environ.get("ODPS_PROJECT_NAME")
            state["endpoint"] = (
                os.environ.get("ODPS_RUNTIME_ENDPOINT") or os.environ["ODPS_ENDPOINT"]
            )
            state.pop("access_id", None)
            state.pop("secret_access_key", None)
            state["account"] = bearer_token_account
            self._init(None, None, **state)
        else:
            self._init(**state)

    def as_account(
        self,
        access_id=None,
        secret_access_key=None,
        account=None,
        app_account=None,
        namespace=None,
    ):
        """
        Creates a new ODPS entry object with a new account information

        :param access_id: Aliyun Access ID of the new account
        :param secret_access_key: Aliyun Access Key of the new account
        :param account: new account object, if `access_id` and `secret_access_key` not supplied
        :param app_account: Application account, instance of `odps.accounts.AppAccount`
            used for dual authentication
        :param namespace: namespace of the new account to be created
        :return:
        """
        if access_id is not None and secret_access_key is not None:
            assert account is None
            account = accounts.AliyunAccount(access_id, secret_access_key)

        params = dict(
            project=self.project,
            endpoint=self.endpoint,
            tunnel_endpoint=self._tunnel_endpoint,
            logview_host=self._logview_host,
            schema=self.schema,
            seahawks_url=self._seahawks_url,
            account=account or self.account,
            app_account=app_account or self.app_account,
            namespace=namespace,
            overwrite_global=False,
        )
        return ODPS(**params)

    def __mars_tokenize__(self):
        return self.__getstate__()

    @classmethod
    def _from_account(
        cls, account, project, endpoint=DEFAULT_ENDPOINT, tunnel_endpoint=None, **kwargs
    ):
        return cls(
            None,
            None,
            project,
            endpoint=endpoint,
            tunnel_endpoint=tunnel_endpoint,
            account=account,
            **kwargs
        )

    def is_schema_namespace_enabled(self, settings=None):
        settings = settings or {}
        setting = str(
            settings.get("odps.namespace.schema")
            or (options.sql.settings or {}).get("odps.namespace.schema")
            or ("true" if options.enable_schema else None)
            or self.default_tenant.get_parameter("odps.namespace.schema")
            or "false"
        )
        return setting.lower() == "true"

    @property
    def region_id(self):
        return self.region_name

    @property
    def default_tenant(self):
        return self._default_tenant

    @property
    def projects(self):
        return self._projects

    @property
    def schema(self):
        """
        Get or set default schema name of the ODPS object
        """
        default_schema = "default" if self.is_schema_namespace_enabled() else None
        return self._schema or default_schema

    @schema.setter
    def schema(self, value):
        self._schema = value
        for cb in self._property_update_callbacks:
            cb(self)

    @property
    def quota_name(self):
        return self._quota_name or options.quota_name or os.getenv("QUOTA_NAME")

    @quota_name.setter
    def quota_name(self, value):
        self._quota_name = value
        for cb in self._property_update_callbacks:
            cb(self)

    @property
    def quotas(self):
        return self._quotas

    @property
    def tunnel_endpoint(self):
        """
        Get or set tunnel endpoint of the ODPS object
        """
        return self._tunnel_endpoint

    @tunnel_endpoint.setter
    def tunnel_endpoint(self, value):
        self._tunnel_endpoint = value
        for cb in self._property_update_callbacks:
            cb(self)

    def list_projects(
        self,
        owner=None,
        user=None,
        group=None,
        prefix=None,
        max_items=None,
        region_id=None,
        tenant_id=None,
    ):
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
        return self.projects.iterate(
            owner=owner,
            user=user,
            group=group,
            max_items=max_items,
            name=prefix,
            region_id=region_id,
            tenant_id=tenant_id,
        )

    @property
    def logview_host(self):
        return self._logview_host

    @property
    def job_insight_host(self):
        return self._job_insight_host

    def get_quota(self, name=None, tenant_id=None):
        """
        Get quota by name

        :param str name: quota name, if not provided, will be the name in ODPS entry
        """
        if name is None:
            name = name or self.quota_name
        if name is None:
            raise TypeError("Need to provide quota name")
        return self._quotas.get(name, tenant_id=tenant_id)

    def exist_quota(self, name):
        """
        If quota name which provided exists or not.

        :param name: quota name
        :return: True if exists or False
        :rtype: bool
        """
        return name in self._quotas

    def list_quotas(self, region_id=None):
        """
        List quotas by region id

        :param str region_id: Region ID
        :return: quotas
        """
        return self._quotas.iterate(region_id=region_id)

    def get_project(self, name=None, default_schema=None):
        """
        Get project by given name.

        :param str name: project name, if not provided, will be the default project
        :param str default_schema: default schema name, if not provided, will be
            the schema specified in ODPS object
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
        # use _schema to avoid requesting for tenant options
        proj._default_schema = default_schema or self._schema
        proj._quota_name = self._quota_name

        proj_ref = weakref.ref(proj)

        def project_update_callback(odps, update_schema=True):
            proj_obj = proj_ref()
            if proj_obj:
                if update_schema:
                    proj_obj._default_schema = odps.schema
                    proj_obj._quota_name = odps._quota_name
                proj_obj._tunnel_endpoint = odps.tunnel_endpoint
            else:
                self._property_update_callbacks.difference_update(
                    [project_update_callback]
                )

        # we need to update default schema value on the project
        self._property_update_callbacks.add(
            functools.partial(
                project_update_callback, update_schema=default_schema is None
            )
        )
        return proj

    def exist_project(self, name):
        """
        If project name which provided exists or not.

        :param name: project name
        :return: True if exists or False
        :rtype: bool
        """

        return name in self._projects

    def list_schemas(self, project=None, prefix=None, owner=None):
        """
        List all schemas of a project.

        :param project: project name, if not provided, will be the default project
        :param str prefix: the listed schemas start with this **prefix**
        :param str owner: Aliyun account, the owner which listed tables belong to
        :return: schemas
        """
        project = self.get_project(name=project)
        return project.schemas.iterate(name=prefix, owner=owner)

    def get_schema(self, name=None, project=None):
        """
        Get the schema by given name.

        :param name: schema name, if not provided, will be the default schema
        :param project: project name, if not provided, will be the default project
        :return: the Schema object
        """
        project = self.get_project(name=project)
        return project.schemas[name or self.schema]

    def exist_schema(self, name, project=None):
        """
        If schema name which provided exists or not.

        :param name: schema name
        :param project: project name, if not provided, will be the default project
        :return: True if exists or False
        :rtype: bool
        """
        project = self.get_project(name=project)
        return name in project.schemas

    @utils.with_wait_argument
    def create_schema(self, name, project=None, async_=False):
        """
        Create a schema with given name

        :param name: schema name
        :param project: project name, if not provided, will be the default project
        :param async_: if True, will run asynchronously
        :return: if async_ is True, return instance, otherwise return Schema object.
        """
        project = self.get_project(name=project)
        return project.schemas.create(name, async_=async_)

    @utils.with_wait_argument
    def delete_schema(self, name, project=None, async_=False):
        """
        Delete the schema with given name

        :param name: schema name
        :param project: project name, if not provided, will be the default project
        :param async_: if True, will run asynchronously
        :type async_: bool
        """
        project = self.get_project(name=project)
        return project.schemas.delete(name, async_=async_)

    def _get_project_or_schema(self, project=None, schema=None):
        if self.is_schema_namespace_enabled():
            schema = schema or "default"
        if schema is not None:
            return self.get_schema(schema, project=project)
        else:
            return self.get_project(project)

    def _split_object_dots(self, name):
        parts = [x.strip() for x in name.split(".")]
        if len(parts) == 1:
            project, schema, name = None, None, parts[0]
        elif len(parts) == 2:
            if self.is_schema_namespace_enabled():
                schema, name = parts
                project = None
            else:
                project, name = parts
                schema = None
        else:
            project, schema, name = parts
        if name.startswith("`") and name.endswith("`"):
            name = name.strip("`")
        return project, schema, name

    def list_tables(
        self,
        project=None,
        prefix=None,
        owner=None,
        schema=None,
        type=None,
        extended=False,
    ):
        """
        List all tables of a project.
        If prefix is provided, the listed tables will all start with this prefix.
        If owner is provided, the listed tables will belong to such owner.

        :param str project: project name, if not provided, will be the default project
        :param str prefix: the listed tables start with this **prefix**
        :param str owner: Aliyun account, the owner which listed tables belong to
        :param str schema: schema name, if not provided, will be the default schema
        :param str type: type of the table
        :param bool extended: if True, load extended information for table
        :return: tables in this project, filtered by the optional prefix and owner.
        :rtype: generator
        """
        parent = self._get_project_or_schema(project, schema)
        return parent.tables.iterate(
            name=prefix, owner=owner, type=type, extended=extended
        )

    def get_table(self, name, project=None, schema=None):
        """
        Get table by given name.

        :param name: table name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :return: the right table
        :rtype: :class:`odps.models.Table`
        :raise: :class:`odps.errors.NoSuchObject` if not exists

        .. seealso:: :class:`odps.models.Table`
        """

        if isinstance(name, six.string_types) and "." in name:
            project, schema, name = self._split_object_dots(name)

        parent = self._get_project_or_schema(project, schema)
        return parent.tables[name]

    def exist_table(self, name, project=None, schema=None):
        """
        If the table with given name exists or not.

        :param name: table name
        :param project: project name, if not provided, will be the default project
        :param schema: schema name, if not provided, will be the default schema
        :type schema: str
        :return: True if table exists or False
        :rtype: bool
        """

        if isinstance(name, six.string_types) and "." in name:
            project, schema, name = self._split_object_dots(name)

        parent = self._get_project_or_schema(project, schema)
        return name in parent.tables

    @utils.with_wait_argument
    def create_table(
        self,
        name,
        table_schema=None,
        project=None,
        schema=None,
        comment=None,
        if_not_exists=False,
        lifecycle=None,
        shard_num=None,
        hub_lifecycle=None,
        hints=None,
        transactional=False,
        primary_key=None,
        storage_tier=None,
        async_=False,
        **kw
    ):
        """
        Create a table by given schema and other optional parameters.

        :param name: table name
        :param table_schema: table schema. Can be an instance
            of :class:`odps.models.TableSchema` or a string like 'col1 string, col2 bigint'
        :param project: project name, if not provided, will be the default project
        :param comment: table comment
        :param str schema: schema name, if not provided, will be the default schema
        :param bool if_not_exists: will not create if this table already exists, default False
        :param int lifecycle: table's lifecycle. If absent, `options.lifecycle` will be used.
        :param int shard_num: table's shard num
        :param int hub_lifecycle: hub lifecycle
        :param dict hints: hints for the task
        :param bool transactional: make table transactional
        :param list primary_key: primary key of the table, only for transactional tables
        :param str storage_tier: storage tier of the table
        :param bool async_: if True, will run asynchronously
        :return: the created Table if not async else odps instance
        :rtype: :class:`odps.models.Table` or :class:`odps.models.Instance`

        .. seealso:: :class:`odps.models.Table`, :class:`odps.models.TableSchema`
        """
        from .types import OdpsSchema

        if table_schema is None and schema:
            if (
                isinstance(schema, OdpsSchema)
                or isinstance(schema, tuple)
                or (isinstance(schema, six.string_types) and " " in schema)
            ):
                table_schema, schema = schema, None
                warnings.warn(
                    "`schema` is renamed as `table_schema` in `create_table`, "
                    "the original parameter now represents schema name. Please "
                    "change your code.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                utils.add_survey_call("ODPS.create_table(schema='schema_name')")

        if table_schema is None:
            raise TypeError("`table_schema` argument not filled")

        if isinstance(name, six.string_types) and "." in name:
            project, schema, name = self._split_object_dots(name)

        if lifecycle is None and options.lifecycle is not None:
            lifecycle = options.lifecycle

        parent = self._get_project_or_schema(project, schema)
        return parent.tables.create(
            name,
            table_schema,
            comment=comment,
            if_not_exists=if_not_exists,
            lifecycle=lifecycle,
            shard_num=shard_num,
            hub_lifecycle=hub_lifecycle,
            hints=hints,
            transactional=transactional,
            primary_key=primary_key,
            storage_tier=storage_tier,
            async_=async_,
            **kw
        )

    def _delete_table(
        self,
        name,
        project=None,
        if_exists=False,
        schema=None,
        hints=None,
        async_=False,
        table_type=None,
    ):
        if isinstance(name, six.string_types) and "." in name:
            project, schema, name = self._split_object_dots(name)

        parent = self._get_project_or_schema(project, schema)
        return parent.tables.delete(
            name, if_exists=if_exists, hints=hints, async_=async_, table_type=table_type
        )

    @utils.with_wait_argument
    def delete_table(
        self, name, project=None, if_exists=False, schema=None, hints=None, async_=False
    ):
        """
        Delete the table with given name

        :param name: table name
        :param project: project name, if not provided, will be the default project
        :param bool if_exists:  will not raise errors when the table does not exist, default False
        :param str schema: schema name, if not provided, will be the default schema
        :param dict hints: hints for the task
        :param bool async_: if True, will run asynchronously
        :return: None if not async else odps instance
        """
        return self._delete_table(
            name,
            project=project,
            if_exists=if_exists,
            schema=schema,
            hints=hints,
            async_=async_,
            table_type="managed_table",
        )

    @utils.with_wait_argument
    def delete_view(
        self, name, project=None, if_exists=False, schema=None, hints=None, async_=False
    ):
        """
        Delete the view with given name

        :param name: view name
        :param project: project name, if not provided, will be the default project
        :param bool if_exists:  will not raise errors when the view does not exist, default False
        :param str schema: schema name, if not provided, will be the default schema
        :param dict hints: hints for the task
        :param bool async_: if True, will run asynchronously
        :return: None if not async else odps instance
        """
        return self._delete_table(
            name,
            project=project,
            if_exists=if_exists,
            schema=schema,
            hints=hints,
            async_=async_,
            table_type="virtual_view",
        )

    @utils.with_wait_argument
    def delete_materialized_view(
        self, name, project=None, if_exists=False, schema=None, hints=None, async_=False
    ):
        """
        Delete the materialized view with given name

        :param name: materialized view name
        :param project: project name, if not provided, will be the default project
        :param bool if_exists:  will not raise errors when the materialized view
            does not exist, default False
        :param str schema: schema name, if not provided, will be the default schema
        :param dict hints: hints for the task
        :param bool async_: if True, will run asynchronously
        :return: None if not async else odps instance
        """
        return self._delete_table(
            name,
            project=project,
            if_exists=if_exists,
            schema=schema,
            hints=hints,
            async_=async_,
            table_type="materialized_view",
        )

    read_table = _wrap_model_func(models.TableIOMethods.read_table)
    write_table = _wrap_model_func(models.TableIOMethods.write_table)

    def list_resources(self, project=None, prefix=None, owner=None, schema=None):
        """
        List all resources of a project.

        :param project: project name, if not provided, will be the default project
        :param str prefix: the listed resources start with this **prefix**
        :param str owner: Aliyun account, the owner which listed tables belong to
        :param str schema: schema name, if not provided, will be the default schema
        :return: resources
        :rtype: generator
        """

        parent = self._get_project_or_schema(project, schema)
        for resource in parent.resources.iterate(name=prefix, owner=owner):
            yield resource

    def get_resource(self, name, project=None, schema=None):
        """
        Get a resource by given name

        :param name: resource name
        :param project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :return: the right resource
        :rtype: :class:`odps.models.Resource`
        :raise: :class:`odps.errors.NoSuchObject` if not exists

        .. seealso:: :class:`odps.models.Resource`
        """

        parent = self._get_project_or_schema(project, schema)
        return parent.resources[name]

    def exist_resource(self, name, project=None, schema=None):
        """
        If the resource with given name exists or not.

        :param name: resource name
        :param schema: schema name, if not provided, will be the default schema
        :type schema: str
        :param project: project name, if not provided, will be the default project
        :return: True if exists or False
        :rtype: bool
        """

        parent = self._get_project_or_schema(project, schema)
        return name in parent.resources

    def open_resource(
        self,
        name,
        project=None,
        mode="r+",
        encoding="utf-8",
        schema=None,
        type="file",
        stream=False,
        comment=None,
        temp=False,
    ):
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
        :param str schema: schema name, if not provided, will be the default schema
        :param str mode: the mode of opening file, described as above
        :param str encoding: utf-8 as default
        :param str type: resource type, can be "file", "archive", "jar" or "py"
        :param bool stream: if True, use stream to upload, False by default
        :param str comment: comment of the resource
        :return: file-like object

        :Example:

        >>> with odps.open_resource('test_resource', mode='r') as fp:
        >>>     fp.read(1)  # read one unicode character
        >>>     fp.write('test')  # wrong, cannot write under read mode
        >>>
        >>> with odps.open_resource('test_resource', mode='wb') as fp:
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

        parent = self._get_project_or_schema(project, schema)
        return parent.resources.get_typed(
            name, type=type, comment=comment, temp=temp
        ).open(mode=mode, encoding=encoding, stream=stream)

    def create_resource(self, name, type=None, project=None, schema=None, **kwargs):
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
        :param schema: schema name, if not provided, will be the default schema
        :type schema: str
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

        type_ = kwargs.get("typo") or type
        parent = self._get_project_or_schema(project, schema)
        return parent.resources.create(name=name, type=type_, **kwargs)

    def delete_resource(self, name, project=None, schema=None):
        """
        Delete resource by given name.

        :param name: resource name
        :param project: project name, if not provided, will be the default project
        :param schema: schema name, if not provided, will be the default schema
        :type schema: str
        :return: None
        """

        parent = self._get_project_or_schema(project, schema)
        return parent.resources.delete(name)

    def list_functions(self, project=None, prefix=None, owner=None, schema=None):
        """
        List all functions of a project.

        :param str project: project name, if not provided, will be the default project
        :param str prefix: the listed functions start with this **prefix**
        :param str owner: Aliyun account, the owner which listed tables belong to
        :param str schema: schema name, if not provided, will be the default schema
        :return: functions
        :rtype: generator
        """

        parent = self._get_project_or_schema(project, schema)
        for function in parent.functions.iterate(name=prefix, owner=owner):
            yield function

    def get_function(self, name, project=None, schema=None):
        """
        Get the function by given name

        :param name: function name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :return: the right function
        :raise: :class:`odps.errors.NoSuchObject` if not exists

        .. seealso:: :class:`odps.models.Function`
        """

        parent = self._get_project_or_schema(project, schema)
        return parent.functions[name]

    def exist_function(self, name, project=None, schema=None):
        """
        If the function with given name exists or not.

        :param str name: function name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :return: True if the function exists or False
        :rtype: bool
        """

        parent = self._get_project_or_schema(project, schema)
        return name in parent.functions

    def create_function(self, name, project=None, schema=None, **kwargs):
        """
        Create a function by given name.

        :param name: function name
        :param project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param str class_type: main class
        :param list resources: the resources that function needs to use
        :return: the created function
        :rtype: :class:`odps.models.Function`

        :Example:

        >>> res = odps.get_resource('test_func.py')
        >>> func = odps.create_function('test_func', class_type='test_func.Test', resources=[res, ])

        .. seealso:: :class:`odps.models.Function`
        """

        parent = self._get_project_or_schema(project, schema)
        return parent.functions.create(name=name, **kwargs)

    def delete_function(self, name, project=None, schema=None):
        """
        Delete a function by given name.

        :param name: function name
        :param project: project name, if not provided, will be the default project
        :param schema: schema name, if not provided, will be the default schema
        :type schema: str
        :return: None
        """

        parent = self._get_project_or_schema(project, schema)
        return parent.functions.delete(name)

    def list_instances(
        self,
        project=None,
        start_time=None,
        end_time=None,
        status=None,
        only_owner=None,
        quota_index=None,
        **kw
    ):
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
        if "from_time" in kw:
            start_time = kw["from_time"]
            warnings.warn(
                "The keyword argument `from_time` has been replaced by `start_time`.",
                DeprecationWarning,
            )

        project = self.get_project(name=project)
        return project.instances.iterate(
            start_time=start_time,
            end_time=end_time,
            status=status,
            only_owner=only_owner,
            quota_index=quota_index,
        )

    def list_instance_queueing_infos(
        self, project=None, status=None, only_owner=None, quota_index=None
    ):
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
            status=status, only_owner=only_owner, quota_index=quota_index
        )

    def get_instance(self, id_, project=None, quota_name=None):
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
        return project.instances.get(id_, quota_name=quota_name or self.quota_name)

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

    def execute_sql(
        self,
        sql,
        project=None,
        priority=None,
        running_cluster=None,
        hints=None,
        quota_name=None,
        **kwargs
    ):
        """
        Run a given SQL statement and block until the SQL executed successfully.

        :param str sql: SQL statement
        :param project: project name, if not provided, will be the default project
        :param int priority: instance priority, 9 as default
        :param str running_cluster: cluster to run this instance
        :param dict hints: settings for SQL, e.g. `odps.mapred.map.split.size`
        :param str quota_name: name of quota to use for SQL job
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

        async_ = kwargs.pop("async_", kwargs.pop("async", False))

        inst = self.run_sql(
            sql,
            project=project,
            priority=priority,
            running_cluster=running_cluster,
            hints=hints,
            quota_name=quota_name,
            **kwargs
        )
        if not async_:
            inst.wait_for_success()
        return inst

    def run_sql(
        self,
        sql,
        project=None,
        priority=None,
        running_cluster=None,
        hints=None,
        aliases=None,
        default_schema=None,
        quota_name=None,
        **kwargs
    ):
        """
        Run a given SQL statement asynchronously

        :param str sql: SQL statement
        :param str project: project name, if not provided, will be the default project
        :param int priority: instance priority, 9 as default
        :param str running_cluster: cluster to run this instance
        :param dict hints: settings for SQL, e.g. `odps.mapred.map.split.size`
        :param dict aliases:
        :param str quota_name: name of quota to use for SQL job
        :return: instance
        :rtype: :class:`odps.models.Instance`

        .. seealso:: :class:`odps.models.Instance`
        """
        on_instance_create = kwargs.pop("on_instance_create", None)
        sql = utils.to_text(sql)

        alter_table_match = _ALTER_TABLE_REGEX.match(sql)
        if alter_table_match:
            drop_table_name = alter_table_match.group("table_name")
            sql_project, sql_schema, sql_name = self._split_object_dots(drop_table_name)
            sql_project = sql_project or project
            sql_schema = sql_schema or default_schema
            del self._get_project_or_schema(sql_project, sql_schema).tables[sql_name]

        merge_instance = models.MergeTask.submit_alter_table_instance(
            self,
            sql,
            project=project,
            schema=default_schema,
            priority=priority,
            running_cluster=running_cluster,
            hints=hints,
            quota_name=quota_name,
            create_callback=on_instance_create,
        )
        if merge_instance is not None:
            return merge_instance

        priority = priority if priority is not None else options.priority
        if priority is None and options.get_priority is not None:
            priority = options.get_priority(self)

        task = models.SQLTask(query=sql, **kwargs)
        task.update_sql_settings(hints)

        schema_hints = {}
        default_schema = default_schema or self.schema
        if self.is_schema_namespace_enabled(hints) or default_schema is not None:
            schema_hints = {
                "odps.sql.allow.namespace.schema": "true",
                "odps.namespace.schema": "true",
            }
        if default_schema is not None:
            schema_hints["odps.default.schema"] = default_schema
        task.update_sql_settings(schema_hints)

        if quota_name or self.quota_name:
            quota_hints = {"odps.task.wlm.quota": quota_name or self.quota_name}
            task.update_sql_settings(quota_hints)

        if aliases:
            task.update_aliases(aliases)

        project = self.get_project(name=project)
        try:
            return project.instances.create(
                task=task,
                priority=priority,
                running_cluster=running_cluster,
                create_callback=on_instance_create,
            )
        except errors.ParseError as ex:
            ex.statement = sql
            raise

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

    @staticmethod
    def _parse_partition_string(partition):
        parts = []
        for p in utils.split_quoted(partition, ","):
            kv = [pp.strip() for pp in utils.split_quoted(p, "=")]
            if len(kv) != 2:
                raise ValueError("Partition representation malformed.")
            if not kv[1].startswith('"') and not kv[1].startswith("'"):
                kv[1] = repr(kv[1])
            parts.append("%s=%s" % tuple(kv))
        return ",".join(parts)

    def list_volumes(self, project=None, schema=None, owner=None):
        """
        List volumes of a project.

        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param str owner: Aliyun account
        :return: volumes
        :rtype: list
        """
        parent = self._get_project_or_schema(project, schema)
        return parent.volumes.iterate(owner=owner)

    @utils.deprecated(
        "`create_volume` is deprecated. Use `created_parted_volume` instead."
    )
    def create_volume(self, name, project=None, **kwargs):
        self.create_parted_volume(name, project=project, **kwargs)

    def create_parted_volume(self, name, project=None, schema=None, **kwargs):
        """
        Create an old-fashioned partitioned volume in a project.

        :param str name: volume name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :return: volume
        :rtype: :class:`odps.models.PartedVolume`

        .. seealso:: :class:`odps.models.PartedVolume`
        """
        parent = self._get_project_or_schema(project, schema)
        return parent.volumes.create_parted(name=name, **kwargs)

    def create_fs_volume(self, name, project=None, schema=None, **kwargs):
        """
        Create a new-fashioned file system volume in a project.

        :param str name: volume name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :return: volume
        :rtype: :class:`odps.models.FSVolume`

        .. seealso:: :class:`odps.models.FSVolume`
        """
        parent = self._get_project_or_schema(project, schema)
        return parent.volumes.create_fs(name=name, **kwargs)

    def create_external_volume(
        self,
        name,
        project=None,
        schema=None,
        location=None,
        rolearn=None,
        auto_create_dir=False,
        accelerate=False,
        **kwargs
    ):
        """
        Create a file system volume based on external storage (for instance, OSS) in a project.

        :param str name: volume name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param str location: location of OSS dir, should be oss://endpoint/bucket/path
        :param str rolearn: role arn of the account hosting the OSS bucket
        :param bool auto_create_dir: if True, will create directory automatically
        :param bool accelerate: if True, will accelerate transfer of large volumes
        :return: volume
        :rtype: :class:`odps.models.FSVolume`

        .. seealso:: :class:`odps.models.FSVolume`
        """
        parent = self._get_project_or_schema(project, schema)
        return parent.volumes.create_external(
            name=name,
            location=location,
            rolearn=rolearn,
            auto_create_dir=auto_create_dir,
            accelerate=accelerate,
            **kwargs
        )

    def exist_volume(self, name, schema=None, project=None):
        """
        If the volume with given name exists or not.

        :param str name: volume name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :return: True if exists or False
        :rtype: bool
        """
        parent = self._get_project_or_schema(project, schema)
        return name in parent.volumes

    def get_volume(self, name, project=None, schema=None):
        """
        Get volume by given name.

        :param str name: volume name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :return: volume object. Return type depends on the type of the volume.
        :rtype: :class:`odps.models.Volume`
        """
        parent = self._get_project_or_schema(project, schema)
        return parent.volumes[name]

    def delete_volume(
        self, name, project=None, schema=None, auto_remove_dir=False, recursive=False
    ):
        """
        Delete volume by given name.

        :param name: volume name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param bool auto_remove_dir: if True, directory created by external volume will be deleted
        :param bool recursive: if True, directory deletion should be recursive
        :return: None
        """
        parent = self._get_project_or_schema(project, schema)
        return parent.volumes.delete(
            name, auto_remove_dir=auto_remove_dir, recursive=recursive
        )

    def list_volume_partitions(self, volume, project=None, schema=None):
        """
        List partitions of a volume.

        :param str volume: volume name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :return: partitions
        :rtype: list
        """
        volume = self.get_volume(volume, project, schema=schema)
        return volume.partitions.iterate()

    def get_volume_partition(self, volume, partition=None, project=None, schema=None):
        """
        Get partition in a parted volume by given name.

        :param str volume: volume name
        :param str partition: partition name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :return: partitions
        :rtype: :class:`odps.models.VolumePartition`
        """
        if partition is None:
            if not volume.startswith("/") or "/" not in volume.lstrip("/"):
                raise ValueError(
                    "You should provide a partition name or use partition path instead."
                )
            volume, partition = volume.lstrip("/").split("/", 1)
        volume = self.get_volume(volume, project, schema=schema)
        return volume.partitions[partition]

    def exist_volume_partition(self, volume, partition=None, project=None, schema=None):
        """
        If the volume with given name exists in a partition or not.

        :param str volume: volume name
        :param str partition: partition name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        """
        if partition is None:
            if not volume.startswith("/") or "/" not in volume.lstrip("/"):
                raise ValueError(
                    "You should provide a partition name or use partition path instead."
                )
            volume, partition = volume.lstrip("/").split("/", 1)
        try:
            volume = self.get_volume(volume, project, schema=schema)
        except errors.NoSuchObject:
            return False
        return partition in volume.partitions

    def delete_volume_partition(
        self, volume, partition=None, project=None, schema=None
    ):
        """
        Delete partition in a volume by given name

        :param str volume: volume name
        :param str partition: partition name
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        """
        if partition is None:
            if not volume.startswith("/") or "/" not in volume.lstrip("/"):
                raise ValueError(
                    "You should provide a partition name or use partition path instead."
                )
            volume, partition = volume.lstrip("/").split("/", 1)
        volume = self.get_volume(volume, project, schema=schema)
        return volume.delete_partition(partition)

    def list_volume_files(self, volume, partition=None, project=None, schema=None):
        """
        List files in a volume. In partitioned volumes, the function returns files under specified partition.
        In file system volumes, the function returns files under specified path.

        :param str volume: volume name
        :param str partition: partition name for partitioned volumes, and path for file system volumes.
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
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
        if partition is None:
            if not volume.startswith("/"):
                raise ValueError(
                    "You should provide a partition name or use partition / path instead."
                )
            volume = volume.lstrip("/")
            if "/" in volume:
                volume, partition = volume.split("/", 1)
        volume = self.get_volume(volume, project, schema=schema)
        if isinstance(volume, models.PartedVolume):
            if not partition:
                raise ValueError("Malformed partition url.")
            return volume.partitions[partition].files.iterate()
        else:
            return volume[partition].objects.iterate()

    def create_volume_directory(self, volume, path=None, project=None, schema=None):
        """
        Create a directory under a file system volume.

        :param str volume: name of the volume.
        :param str path: path of the directory to be created.
        :param str project: project name, if not provided, will be the default project.
        :param str schema: schema name, if not provided, will be the default schema
        :return: directory object.
        """
        if path is None:
            if not volume.startswith("/"):
                raise ValueError("You should provide a valid path.")
            volume = volume.lstrip("/")
            if "/" in volume:
                volume, path = volume.split("/", 1)
        volume = self.get_volume(volume, project, schema=schema)
        if isinstance(volume, models.PartedVolume):
            raise ValueError("Only supported under file system volumes.")
        else:
            return volume.create_dir(path)

    def get_volume_file(self, volume, path=None, project=None, schema=None):
        """
        Get a file under a partition of a parted volume, or a file / directory object under a file system volume.

        :param str volume: name of the volume.
        :param str path: path of the directory to be created.
        :param str project: project name, if not provided, will be the default project.
        :param str schema: schema name, if not provided, will be the default schema
        :return: directory object.
        """
        if path is None:
            if not volume.startswith("/"):
                raise ValueError("You should provide a valid path.")
            volume = volume.lstrip("/")
            if "/" in volume:
                volume, path = volume.split("/", 1)
        volume = self.get_volume(volume, project, schema=schema)
        if isinstance(volume, models.PartedVolume):
            if "/" not in path:
                raise ValueError("Partition/File format malformed.")
            part, file_name = path.split("/", 1)
            return volume.get_partition(part).files[file_name]
        else:
            return volume[path]

    def move_volume_file(
        self, old_path, new_path, replication=None, project=None, schema=None
    ):
        """
        Move a file / directory object under a file system volume to another location in the same volume.

        :param str old_path: old path of the volume file.
        :param str new_path: target path of the moved file.
        :param int replication: file replication.
        :param str project: project name, if not provided, will be the default project.
        :param str schema: schema name, if not provided, will be the default schema
        :return: directory object.
        """
        if not new_path.startswith("/"):
            # make relative path absolute
            old_root, _ = old_path.rsplit("/", 1)
            new_path = old_root + "/" + new_path

        if not old_path.startswith("/"):
            raise ValueError("You should provide a valid path.")
        old_volume, old_path = old_path.lstrip("/").split("/", 1)

        new_volume, _ = new_path.lstrip("/").split("/", 1)

        if old_volume != new_volume:
            raise ValueError("Moving between different volumes is not supported.")

        volume = self.get_volume(old_volume, project, schema=schema)
        if isinstance(volume, models.PartedVolume):
            raise ValueError("Only supported under file system volumes.")
        else:
            volume[old_path].move(new_path, replication=replication)

    def delete_volume_file(
        self, volume, path=None, recursive=False, project=None, schema=None
    ):
        """
        Delete a file / directory object under a file system volume.

        :param str volume: name of the volume.
        :param str path: path of the directory to be created.
        :param bool recursive: if True, recursively delete files
        :param str project: project name, if not provided, will be the default project.
        :param str schema: schema name, if not provided, will be the default schema
        :return: directory object.
        """
        if path is None:
            if not volume.startswith("/"):
                raise ValueError("You should provide a valid path.")
            volume = volume.lstrip("/")
            if "/" in volume:
                volume, path = volume.split("/", 1)
        volume = self.get_volume(volume, project, schema=schema)
        if isinstance(volume, models.PartedVolume):
            raise ValueError("Only supported under file system volumes.")
        else:
            volume[path].delete(recursive=recursive)

    def open_volume_reader(
        self,
        volume,
        partition=None,
        file_name=None,
        project=None,
        schema=None,
        start=None,
        length=None,
        **kwargs
    ):
        """
        Open a volume file for read. A file-like object will be returned which can be used to read contents from
        volume files.

        :param str volume: name of the volume
        :param str partition: name of the partition
        :param str file_name: name of the file
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param start: start position
        :param length: length limit
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: CompressOption

        :Example:
        >>> with odps.open_volume_reader('parted_volume', 'partition', 'file') as reader:
        >>>     [print(line) for line in reader]
        """
        if partition is None:
            if not volume.startswith("/"):
                raise ValueError(
                    "You should provide a partition name or use partition / path instead."
                )
            volume = volume.lstrip("/")
            volume, partition = volume.split("/", 1)
            if "/" in partition:
                partition, file_name = partition.rsplit("/", 1)
            else:
                partition, file_name = None, partition
        volume = self.get_volume(volume, project, schema=schema)
        if isinstance(volume, models.PartedVolume):
            if not partition:
                raise ValueError("Malformed partition url.")
            return volume.partitions[partition].open_reader(
                file_name, start=start, length=length, **kwargs
            )
        else:
            return volume[partition].open_reader(
                file_name, start=start, length=length, **kwargs
            )

    def open_volume_writer(
        self, volume, partition=None, project=None, schema=None, **kwargs
    ):
        """
        Write data into a volume. This function behaves differently under different types of volumes.

        Under partitioned volumes, all files under a partition should be uploaded in one submission. The method
        returns a writer object with whose `open` method you can open a file inside the volume and write to it,
        or you can use `write` method to write to specific files.

        Under file system volumes, the method returns a file-like object.

        :param str volume: name of the volume
        :param str partition: partition name for partitioned volumes, and path for file system volumes.
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
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
        if partition is None:
            if not volume.startswith("/"):
                raise ValueError(
                    "You should provide a partition name or use partition / path instead."
                )
            volume = volume.lstrip("/")
            volume, partition = volume.split("/", 1)
        volume = self.get_volume(volume, project, schema=schema)
        if isinstance(volume, models.PartedVolume):
            return volume.partitions[partition].open_writer(**kwargs)
        else:
            if "/" in partition:
                partition, file_name = partition.rsplit("/", 1)
            else:
                partition, file_name = None, partition
            return volume[partition].open_writer(file_name, **kwargs)

    def list_xflows(self, project=None, owner=None):
        """
        List xflows of a project which can be filtered by the xflow owner.

        :param str project: project name, if not provided, will be the default project
        :param str owner: Aliyun account
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

    def run_xflow(
        self,
        xflow_name,
        xflow_project=None,
        parameters=None,
        project=None,
        hints=None,
        priority=None,
    ):
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
            xflow_name=xflow_name,
            xflow_project=xflow_project,
            project=project,
            parameters=parameters,
            hints=hints,
            priority=priority,
        )

    def execute_xflow(
        self,
        xflow_name,
        xflow_project=None,
        parameters=None,
        project=None,
        hints=None,
        priority=None,
    ):
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
            xflow_name,
            xflow_project=xflow_project,
            parameters=parameters,
            project=project,
            hints=hints,
            priority=priority,
        )
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

    def iter_xflow_sub_instances(self, instance, interval=1, project=None, check=False):
        """
        The result iterates the sub instance of xflow and will wait till instance finish

        :param instance: instance of xflow
        :type instance: :class:`odps.models.Instance`
        :param interval: time interval to check
        :param project: project name, if not provided, will be the default project
        :param bool check: check if the instance is successful
        :return: sub instances dictionary
        """
        project = self.get_project(name=project)
        return project.xflows.iter_xflow_sub_instances(
            instance, interval=interval, check=check
        )

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

    @utils.with_wait_argument
    def copy_offline_model(
        self, name, new_name, project=None, new_project=None, async_=False
    ):
        """
        Copy current model into a new location.

        :param new_name: name of the new model
        :param new_project: new project name. if absent, original project name will be used
        :param async_: if True, return the copy instance. otherwise return the newly-copied model
        """
        return self.get_offline_model(name, project=project).copy(
            new_name, new_project=new_project, async_=async_
        )

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
        except errors.NoSuchObject:
            if not if_exists:
                raise

    def get_logview_host(self):
        """
        Get logview host address.
        :return: logview host address
        """
        if self.endpoint in _logview_host_cache:
            return _logview_host_cache[self.endpoint]

        try:
            logview_host = utils.to_str(
                self.rest.get(self.endpoint + "/logview/host").content
            )
        except:
            logview_host = None
        if not logview_host:
            logview_host = utils.get_default_logview_endpoint(
                LOGVIEW_HOST_DEFAULT, self.endpoint
            )
        _logview_host_cache[self.endpoint] = logview_host
        return logview_host

    def get_logview_address(
        self, instance_id, hours=None, project=None, use_legacy=None
    ):
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
        return inst.get_logview_address(hours=hours, use_legacy=use_legacy)

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
            raise ValueError("Option does not exists.")
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
            raise ValueError("Option does not exists.")
        setattr(sec_options, option_name, value)
        sec_options.update()

    def run_security_query(
        self, query, project=None, schema=None, token=None, hints=None, output_json=True
    ):
        """
        Run a security query to grant / revoke / query privileges. If the query is `install package`
        or `uninstall package`, return a waitable AuthQueryInstance object, otherwise returns
        the result string or json value.

        :param str query: query text
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param bool output_json: parse json for the output
        :return: result string / json object
        """
        project = self.get_project(name=project)
        schema = schema or self.schema
        return project.run_security_query(
            query, schema=schema, token=token, hints=hints, output_json=output_json
        )

    def execute_security_query(
        self, query, project=None, schema=None, token=None, hints=None, output_json=True
    ):
        """
        Execute a security query to grant / revoke / query privileges and returns
        the result string or json value.

        :param str query: query text
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param bool output_json: parse json for the output
        :return: result string / json object
        """
        from .models import Project

        instance_or_result = self.run_security_query(
            query,
            project=project,
            schema=schema,
            token=token,
            hints=hints,
            output_json=output_json,
        )
        if not isinstance(instance_or_result, Project.AuthQueryInstance):
            return instance_or_result
        return instance_or_result.wait_for_success()

    @classmethod
    def _build_account(cls, access_id, secret_access_key):
        return accounts.AliyunAccount(access_id, secret_access_key)

    def to_global(self, overwritable=False):
        options.is_global_account_overwritable = overwritable
        options.account = self.account
        options.default_project = self.project
        # use _schema to avoid requesting for tenant options
        options.default_schema = self._schema
        options.endpoint = self.endpoint
        options.logview_host = self.logview_host
        options.tunnel.endpoint = self._tunnel_endpoint
        options.app_account = self.app_account
        options.region_name = self.region_name
        options.default_namespace = self.namespace

    @classmethod
    def from_global(cls):
        if options.account is not None and options.default_project is not None:
            return cls._from_account(
                options.account,
                options.default_project,
                endpoint=options.endpoint,
                schema=options.default_schema,
                tunnel_endpoint=options.tunnel.endpoint,
                logview_host=options.logview_host,
                app_account=options.app_account,
                region_name=options.region_name,
                namespace=options.default_namespace,
            )
        else:
            return None

    @classmethod
    def from_environments(cls):
        try:
            project = os.getenv("ODPS_PROJECT_NAME")
            endpoint = os.environ["ODPS_ENDPOINT"]
            tunnel_endpoint = os.getenv("ODPS_TUNNEL_ENDPOINT")
            namespace = os.getenv("ODPS_NAMESPACE")
            return cls(
                None,
                None,
                account=accounts.from_environments(),
                project=project,
                endpoint=endpoint,
                tunnel_endpoint=tunnel_endpoint,
                namespace=namespace,
            )
        except KeyError:
            return None

    _attach_mcqa_session = _wrap_model_func(models.SessionMethods._attach_mcqa_session)
    attach_session = _wrap_model_func(models.SessionMethods.attach_session)
    _create_mcqa_session = _wrap_model_func(models.SessionMethods._create_mcqa_session)
    create_session = _wrap_model_func(models.SessionMethods.create_session)
    default_session = _wrap_model_func(models.SessionMethods.default_session)
    _get_default_mcqa_session = _wrap_model_func(
        models.SessionMethods._get_default_mcqa_session
    )
    run_sql_interactive_with_fallback = _wrap_model_func(
        models.SessionMethods.run_sql_interactive_with_fallback
    )
    run_sql_interactive = _wrap_model_func(models.SessionMethods.run_sql_interactive)
    execute_sql_interactive = _wrap_model_func(
        models.SessionMethods.execute_sql_interactive
    )

    run_merge_files = _wrap_model_func(models.MergeTask.run_merge_files)
    execute_merge_files = _wrap_model_func(models.MergeTask.execute_merge_files)
    run_archive_table = _wrap_model_func(models.MergeTask.run_archive_table)
    execute_archive_table = _wrap_model_func(models.MergeTask.execute_archive_table)
    run_freeze_command = _wrap_model_func(models.MergeTask.run_freeze_command)
    execute_freeze_command = _wrap_model_func(models.MergeTask.execute_freeze_command)


def _get_odps_from_model(self):
    cur = self
    while cur is not None and not isinstance(cur, models.Project):
        cur = cur.parent
    return cur.odps if cur else None


models.RestModel.odps = property(fget=_get_odps_from_model)
del _get_odps_from_model

try:
    from .internal.core import *  # noqa: F401
except ImportError:  # pragma: no cover
    pass

try:
    from . import mars_extension

    for _mars_attr in (
        "create_mars_cluster",
        "persist_mars_dataframe",
        "to_mars_dataframe",
        "run_script_in_mars",
        "run_mars_job",
        "list_mars_instances",
        "sql_to_mars_dataframe",
    ):
        setattr(ODPS, _mars_attr, getattr(mars_extension, _mars_attr))
except ImportError:
    pass


DEFAULT_ENDPOINT = os.getenv(
    "ODPS_ENDPOINT", os.getenv("PYODPS_ENDPOINT", DEFAULT_ENDPOINT)
)
del _wrap_model_func
