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

import contextlib
import itertools
import sys
import threading
import time

from sqlalchemy import types as sa_types
from sqlalchemy.engine import Engine, default
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.sql import compiler, sqltypes

try:
    from sqlalchemy.dialects import mysql
except ImportError:
    # for low sqlalchemy versions
    from sqlalchemy.databases import mysql

from . import options, types
from .compat import six
from .core import DEFAULT_ENDPOINT, ODPS
from .errors import BaseODPSError, InternalServerError, NoSuchObject
from .models import Table
from .models.session.v1 import PUBLIC_SESSION_NAME
from .utils import to_str, to_text

test_setting = threading.local()
test_setting.get_tables_filter = None


@contextlib.contextmanager
def update_test_setting(**kw):
    old_values = {}
    for k in kw:
        old_values[k] = getattr(test_setting, k)

    for k, v in six.iteritems(kw):
        setattr(test_setting, k, v)

    yield

    # set back value
    for k, v in six.iteritems(old_values):
        setattr(test_setting, k, v)


_odps_type_to_sqlalchemy_type = {
    types.Boolean: sa_types.Boolean,
    types.Tinyint: mysql.TINYINT,
    types.Smallint: sa_types.SmallInteger,
    types.Int: sa_types.Integer,
    types.Bigint: sa_types.BigInteger,
    types.Float: sa_types.Float,
    types.Double: sa_types.Float,
    types.String: sa_types.String,
    types.Varchar: sa_types.String,
    types.Char: sa_types.String,
    types.Date: sa_types.Date,
    types.Datetime: sa_types.DateTime,
    types.Timestamp: sa_types.TIMESTAMP,
    types.Binary: sa_types.BINARY,
    types.Array: sa_types.String,
    types.Map: sa_types.String,
    types.Struct: sa_types.String,
    types.Decimal: sa_types.DECIMAL,
    types.Json: sa_types.String,
    types.TimestampNTZ: sa_types.TIMESTAMP,
}

_sqlalchemy_global_reusable_odps = {}

_sqlalchemy_obj_list_cache = {}


class ObjectCache(object):
    def __init__(self, expire=24 * 3600):
        self._expire_time = expire
        self._items = dict()
        self._cache_time = dict()

    def __getitem__(self, key):
        if self._cache_time[key] < time.time() - self._expire_time:
            self._cache_time.pop(key, None)
            self._items.pop(key, None)
            raise KeyError(key)
        return self._items[key]

    def __setitem__(self, key, value):
        self._items[key] = value
        self._cache_time[key] = time.time()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class ODPSIdentifierPreparer(compiler.IdentifierPreparer):
    # Just quote everything to make things simpler / easier to upgrade
    reserved_words = compiler.RESERVED_WORDS.copy()
    keywords = [
        "ADD",
        "ALL",
        "ALTER",
        "AND",
        "AS",
        "ASC",
        "BETWEEN",
        "BIGINT",
        "BOOLEAN",
        "BY",
        "CASE",
        "CAST",
        "COLUMN",
        "COMMENT",
        "CREATE",
        "DESC",
        "DISTINCT",
        "DISTRIBUTE",
        "DOUBLE",
        "DROP",
        "ELSE",
        "FALSE",
        "FROM",
        "FULL",
        "GROUP",
        "IF",
        "IN",
        "INSERT",
        "INTO",
        "IS",
        "JOIN",
        "LEFT",
        "LIFECYCLE",
        "LIKE",
        "LIMIT",
        "MAPJOIN",
        "NOT",
        "NULL",
        "ON",
        "OR",
        "ORDER",
        "OUTER",
        "OVERWRITE",
        "PARTITION",
        "RENAME",
        "REPLACE",
        "RIGHT",
        "RLIKE",
        "SELECT",
        "SORT",
        "STRING",
        "TABLE",
        "TABLESAMPLE",
        "TBLPROPERTIES",
        "THEN",
        "TOUCH",
        "TRUE",
        "UNION",
        "VIEW",
        "WHEN",
        "WHERE",
    ]
    reserved_words.update(keywords)
    reserved_words.update([s.lower() for s in keywords])

    def __init__(self, dialect):
        super(ODPSIdentifierPreparer, self).__init__(
            dialect,
            initial_quote="`",
            escape_quote="`",
        )

    def quote(self, ident, force=None):
        return to_str(super(ODPSIdentifierPreparer, self).quote(ident, force=force))


class ODPSCompiler(compiler.SQLCompiler):
    def visit_column(self, *args, **kwargs):
        result = super(ODPSCompiler, self).visit_column(*args, **kwargs)
        dot_count = result.count(".")
        assert dot_count in (0, 1, 2), "Unexpected visit_column result {}".format(
            result
        )
        if dot_count == 2:
            # we have something of the form schema.table.column
            # hive doesn't like the schema in front, so chop it out
            result = result[result.index(".") + 1 :]
        return result

    def visit_char_length_func(self, fn, **kw):
        return "length{}".format(self.function_argspec(fn, **kw))

    def __unicode__(self):
        return to_text(self)


class ODPSTypeCompiler(compiler.GenericTypeCompiler):
    def visit_INTEGER(self, type_):
        return "INT"

    def visit_NUMERIC(self, type_):
        return "DECIMAL"

    def visit_CHAR(self, type_):
        return "STRING"

    def visit_VARCHAR(self, type_):
        return "STRING"

    def visit_NCHAR(self, type_):
        return "STRING"

    def visit_TEXT(self, type_):
        return "STRING"

    def visit_CLOB(self, type_):
        return "STRING"

    def visit_BLOB(self, type_):
        return "BINARY"

    def visit_TIME(self, type_):
        return "TIMESTAMP"


if hasattr(sqltypes.String, "RETURNS_UNICODE"):
    _return_unicode_str = sqltypes.String.RETURNS_UNICODE
else:
    _return_unicode_str = True


class ODPSPingError(BaseODPSError):
    pass


class ODPSDialect(default.DefaultDialect):
    name = "odps"
    driver = "rest"
    preparer = ODPSIdentifierPreparer
    statement_compiler = ODPSCompiler
    supports_views = True
    supports_alter = True
    supports_pk_autoincrement = False
    supports_default_values = False
    supports_empty_insert = False
    supports_native_decimal = True
    supports_native_boolean = True
    supports_unicode_statements = True
    supports_unicode_binds = True
    returns_unicode_strings = _return_unicode_str
    description_encoding = None
    supports_multivalues_insert = True
    type_compiler = ODPSTypeCompiler
    supports_sane_rowcount = False
    supports_statement_cache = False
    _reused_odps = None
    default_schema_name = "default"

    @classmethod
    def dbapi(cls):
        return cls.import_dbapi()

    @classmethod
    def import_dbapi(cls):
        from . import dbapi

        return dbapi

    def create_connect_args(self, url):
        url_string = str(url)
        project = url.host
        if project is None and options.default_project:
            project = options.default_project
        access_id = url.username
        secret_access_key = url.password
        logview_host = options.logview_host
        endpoint = None
        session_name = None
        sqa_type = False
        quota_name = None
        reuse_odps = False
        project_as_schema = False
        fallback_policy = ""
        cache_names = False
        cache_seconds = 24 * 3600
        hints = {}
        if url.query:
            query = dict(url.query)
            if endpoint is None:
                endpoint = query.pop("endpoint", None)
            if logview_host is None:
                logview_host = query.pop("logview_host", query.pop("logview", None))
            if session_name is None:
                session_name = query.pop("session", None)
            if quota_name is None:
                quota_name = query.pop("quota_name", None)
            if sqa_type is False:
                sqa_type = query.pop("interactive_mode", "false").lower()
                if sqa_type == "true":
                    sqa_type = "v1"
                elif sqa_type == "false":
                    sqa_type = False
            if reuse_odps is False:
                reuse_odps = query.pop("reuse_odps", "false").lower() != "false"
            if query.get("project_as_schema", None) is not None:
                project_as_schema = (
                    query.pop("project_as_schema", "false").lower() != "false"
                )
            if fallback_policy == "":
                fallback_policy = query.pop("fallback_policy", "default")
            if cache_names is False:
                cache_names = query.pop("cache_names", "false").lower() != "false"
            cache_seconds = int(query.pop("cache_seconds", cache_seconds))
            hints = query

        if endpoint is None:
            endpoint = options.endpoint or DEFAULT_ENDPOINT
        if session_name is None:
            session_name = PUBLIC_SESSION_NAME

        kwargs = {
            "access_id": access_id,
            "secret_access_key": secret_access_key,
            "project": project,
            "endpoint": endpoint,
            "session_name": session_name,
            "use_sqa": sqa_type,
            "fallback_policy": fallback_policy,
            "project_as_schema": project_as_schema,
            "hints": hints,
        }
        if quota_name is not None:
            kwargs["quota_name"] = quota_name
        if access_id is None:
            kwargs.pop("access_id", None)
            kwargs.pop("secret_access_key", None)
            kwargs["account"] = options.account

        for k, v in six.iteritems(kwargs):
            if v is None:
                raise ValueError(
                    "{} should be provided to create connection, "
                    "you can either specify in connection string as format: "
                    '"odps://<access_id>:<access_key>@<project_name>", '
                    "or create an ODPS object and call `.to_global()` "
                    "to set it to global".format(k)
                )
        if logview_host is not None:
            kwargs["logview_host"] = logview_host

        if cache_names:
            _sqlalchemy_obj_list_cache[url_string] = ObjectCache(expire=cache_seconds)

        if reuse_odps:
            # the odps object can only be reused only if it will be identical
            if (
                url_string in _sqlalchemy_global_reusable_odps
                and _sqlalchemy_global_reusable_odps.get(url_string) is not None
            ):
                kwargs["odps"] = _sqlalchemy_global_reusable_odps.get(url_string)
                kwargs["access_id"] = None
                kwargs["secret_access_key"] = None
            else:
                _sqlalchemy_global_reusable_odps[url_string] = ODPS(
                    access_id=access_id,
                    secret_access_key=secret_access_key,
                    project=project,
                    endpoint=endpoint,
                    logview_host=logview_host,
                )

        return [], kwargs

    def get_odps_from_url(self, url):
        _, kwargs = self.create_connect_args(url)
        if "odps" in kwargs:
            return kwargs["odps"]
        odps_kw = kwargs.copy()
        odps_kw.pop("session_name", None)
        odps_kw.pop("use_sqa", None)
        odps_kw.pop("fallback_policy", None)
        odps_kw.pop("hints", None)
        odps_kw.pop("project_as_schema", None)
        odps_kw["overwrite_global"] = False
        return ODPS(**odps_kw)

    @classmethod
    def get_list_cache(cls, url, key):
        url = str(url)
        if url not in _sqlalchemy_obj_list_cache:
            return None
        return _sqlalchemy_obj_list_cache[url].get(key)

    @classmethod
    def put_list_cache(cls, url, key, value):
        url = str(url)
        if url not in _sqlalchemy_obj_list_cache:
            return
        _sqlalchemy_obj_list_cache[url][key] = value

    def get_schema_names(self, connection, **kw):
        conn = self._get_dbapi_connection(connection)
        if getattr(conn, "_project_as_schema", False):
            fields = ["owner", "user", "group", "prefix"]
            if (conn.odps.project is None) or (kw.pop("listall", None) is not None):
                kwargs = {f: kw.get(f) for f in fields}
                return [proj.name for proj in conn.odps.list_projects(**kwargs)]
            else:
                return [conn.odps.project]
        else:
            try:
                return [schema.name for schema in conn.odps.list_schemas()]
            except:
                return ["default"]

    def has_table(self, connection, table_name, schema=None, **kw):
        conn = self._get_dbapi_connection(connection)
        schema_kw = self._get_schema_kw(connection, schema=schema)
        return conn.odps.exist_table(table_name, **schema_kw)

    @classmethod
    def _get_dbapi_connection(cls, sa_connection):
        if isinstance(sa_connection, Engine):
            sa_connection = sa_connection.connect()
        return sa_connection.connection.connection

    @classmethod
    def _get_schema_kw(cls, connection, schema=None):
        db_conn = cls._get_dbapi_connection(connection)
        if getattr(db_conn, "_project_as_schema", False):
            return dict(project=schema)
        else:
            return dict(schema=schema)

    def get_columns(self, connection, table_name, schema=None, **kw):
        conn = self._get_dbapi_connection(connection)
        schema_kw = self._get_schema_kw(connection, schema=schema)
        table = conn.odps.get_table(table_name, **schema_kw)
        result = []
        try:
            for col in table.table_schema.columns:
                col_type = _odps_type_to_sqlalchemy_type[type(col.type)]
                result.append(
                    {
                        "name": col.name,
                        "type": col_type,
                        "nullable": True,
                        "default": None,
                        "comment": col.comment,
                    }
                )
        except NoSuchObject as e:
            # convert ODPSError to SQLAlchemy NoSuchTableError
            raise NoSuchTableError(str(e))
        return result

    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        # ODPS has no support for foreign keys.
        return []

    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        # ODPS has no support for primary keys.
        return []

    def get_indexes(self, connection, table_name, schema=None, **kw):
        # ODPS has no support for indexes
        return []

    def _iter_tables(self, connection, schema=None, types=None, **kw):
        cache_key = ("tables", schema, tuple(types))
        cached = self.get_list_cache(connection.engine.url, cache_key)
        if cached is not None:
            return cached

        conn = self._get_dbapi_connection(connection)
        filter_ = getattr(test_setting, "get_tables_filter", None)
        if filter_ is None:
            filter_ = lambda x: True
        schema_kw = self._get_schema_kw(connection, schema=schema)

        if not types:
            it = conn.odps.list_tables(**schema_kw)
        else:
            its = []
            for table_type in types:
                list_kw = schema_kw.copy()
                list_kw["type"] = table_type
                its.append(conn.odps.list_tables(**list_kw))
            it = itertools.chain(*its)

        result = [t.name for t in it if filter_(t.name)]
        self.put_list_cache(connection.engine.url, cache_key, result)
        return result

    def get_table_names(self, connection, schema=None, **kw):
        return self._iter_tables(
            connection,
            schema=schema,
            types=[Table.Type.MANAGED_TABLE, Table.Type.EXTERNAL_TABLE],
            **kw
        )

    def get_view_names(self, connection, schema=None, **kw):
        return self._iter_tables(
            connection,
            schema=schema,
            types=[Table.Type.VIRTUAL_VIEW, Table.Type.MATERIALIZED_VIEW],
            **kw
        )

    def get_table_comment(self, connection, table_name, schema=None, **kw):
        conn = self._get_dbapi_connection(connection)
        schema_kw = self._get_schema_kw(connection, schema=schema)
        comment = conn.odps.get_table(table_name, **schema_kw).comment
        return {"text": comment}

    @classmethod
    def _is_stack_superset(cls, tb):
        try:
            cur_frame = tb.tb_frame
            while cur_frame is not None:
                if "superset" in cur_frame.f_code.co_filename:
                    return True
                cur_frame = cur_frame.f_back
            return False
        except:  # pragma: no cover
            return False

    def do_ping(self, dbapi_connection):
        """Stop raising RuntimeError when ping by Superset"""
        try:
            return super(ODPSDialect, self).do_ping(dbapi_connection)
        except InternalServerError:
            raise
        except BaseException as ex:
            _, _, tb = sys.exc_info()
            if not self._is_stack_superset(tb):
                raise
            new_err = ODPSPingError(ex.args[0])
            for attr in (
                "request_id",
                "instance_id",
                "code",
                "host_id",
                "endpoint",
                "tag",
            ):
                setattr(new_err, attr, getattr(ex, attr))
            six.reraise(ODPSPingError, new_err, tb)

    def do_rollback(self, dbapi_connection):
        # No transactions for ODPS
        pass

    def _check_unicode_returns(self, connection, additional_tests=None):
        # We decode everything as UTF-8
        return True

    def _check_unicode_description(self, connection):
        # We decode everything as UTF-8
        return True
