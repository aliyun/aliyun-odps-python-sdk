# -*- coding: utf-8 -*-
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
import decimal
import json
import sys
from collections import OrderedDict

import mock
import pytest

from .. import ODPS, errors, options
from ..accounts import BearerTokenAccount
from ..dbapi import connect as dbapi_connect
from ..df import DataFrame
from ..tests.core import tn
from ..utils import to_str

try:
    import numpy as np
    import pandas as pd
    import sqlalchemy
    from sqlalchemy import types
    from sqlalchemy.engine import create_engine, reflection
    from sqlalchemy.exc import NoSuchTableError
    from sqlalchemy.schema import Column, MetaData, Table
    from sqlalchemy.sql import expression, text

    from ..sqlalchemy_odps import ODPSPingError, update_test_setting

    _ONE_ROW_COMPLEX_CONTENTS = [
        True,
        127,
        32767,
        2147483647,
        9223372036854775807,
        0.5,
        0.25,
        "a string",
        pd.Timestamp(1970, 1, 1, 8),
        b"123",
        [1, 2],
        OrderedDict({1: 2, 3: 4}),
        OrderedDict({"a": 1, "b": 2}),
        decimal.Decimal("0.1"),
    ]
except ImportError:
    dependency_installed = False
else:
    dependency_installed = True


def create_one_row(o):
    table = "one_row"
    if not o.exist_table(table):
        o.execute_sql("CREATE TABLE one_row (number_of_rows INT);")
        o.execute_sql("INSERT INTO TABLE one_row VALUES (1);")


def create_one_row_complex(o):
    need_writes = [False] * 2
    for i, table in enumerate(["one_row_complex", "one_row_complex_null"]):
        if o.exist_table(table):
            continue

        ddl = """
        CREATE TABLE {} (
            `boolean` BOOLEAN,
            `tinyint` TINYINT,
            `smallint` SMALLINT,
            `int` INT,
            `bigint` BIGINT,
            `float` FLOAT,
            `double` DOUBLE,
            `string` STRING,
            `timestamp` TIMESTAMP,
            `binary` BINARY,
            `array` ARRAY<int>,
            `map` MAP<int, int>,
            `struct` STRUCT<a: int, b: int>,
            `decimal` DECIMAL(10, 1)
        );
        """.format(
            table
        )
        o.execute_sql(ddl)
        need_writes[i] = True

    if need_writes[0]:
        o.execute_sql(
            """
        INSERT OVERWRITE TABLE one_row_complex SELECT
            true,
            CAST(127 AS TINYINT),
            CAST(32767 AS SMALLINT),
            2147483647,
            9223372036854775807,
            CAST(0.5 AS FLOAT),
            0.25,
            'a string',
            CAST(CAST(0 AS BIGINT) AS TIMESTAMP),
            CAST('123' AS BINARY),
            array(1, 2),
            map(1, 2, 3, 4),
            named_struct('a', 1, 'b', 2),
            0.1
        FROM one_row;
        """
        )

    if need_writes[1]:
        o.execute_sql(
            """
        INSERT OVERWRITE TABLE one_row_complex_null SELECT
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            CAST(null AS STRUCT<a: int, b: int>),
            null
        FROM one_row;
        """
        )


def create_many_rows(o):
    table = "many_rows"
    if not o.exist_table(table):
        df = pd.DataFrame({"a": np.arange(10000, dtype=np.int32)})
        o.execute_sql(
            """
        CREATE TABLE many_rows (
            a INT
        ) PARTITIONED BY (
            b STRING
        )
        """
        )
        DataFrame(df).persist("many_rows", partition="b='blah'", odps=o)


def create_test(o):
    table = "dummy_table"
    if not o.exist_table(table):
        o.create_table(table, "a int")


@pytest.fixture
def engine(odps, request):
    engine_url = "odps://{}:{}@{}/?endpoint={}&SKYNET_PYODPS_HINT=hint".format(
        odps.account.access_id,
        odps.account.secret_access_key,
        odps.project,
        odps.endpoint,
    )
    if getattr(request, "param", None):
        engine_url += "&" + request.param
        # create an engine to enable cache
        create_engine(engine_url)

    try:
        yield create_engine(engine_url)
    finally:
        from .. import sqlalchemy_odps

        sqlalchemy_odps._sqlalchemy_global_reusable_odps.clear()
        sqlalchemy_odps._sqlalchemy_obj_list_cache.clear()


@pytest.fixture
def connection(engine):
    try:
        with contextlib.closing(engine.connect()) as connection:
            yield connection
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def setup(odps):
    if not dependency_installed:
        pytest.skip("dependency for sqlalchemy_odps not installed")

    options.sql.use_odps2_extension = True
    options.sql.settings = {"odps.sql.decimal.odps2": True}

    # create test tables
    create_many_rows(odps)
    create_one_row(odps)
    create_one_row_complex(odps)
    create_test(odps)

    # patch to make methods compatible
    if not hasattr(reflection.Inspector, "reflect_table"):
        reflection.Inspector.reflect_table = reflection.Inspector.reflecttable

    try:
        yield
    finally:
        options.sql.use_odps2_extension = None
        options.sql.settings = None


@pytest.fixture
def bearer_token_odps(odps):
    policy = {
        "Statement": [{"Action": ["*"], "Effect": "Allow", "Resource": "acs:odps:*:*"}],
        "Version": "1",
    }
    token = odps.get_project().generate_auth_token(policy, "bearer", 2)
    bearer_token_account = BearerTokenAccount(token=token)
    try:
        yield ODPS(
            None,
            None,
            odps.project,
            odps.endpoint,
            account=bearer_token_account,
            overwrite_global=False,
        )
    finally:
        options.account = options.default_project = options.endpoint = None


def _get_sa_table(table_name, engine, *args, **kw):
    try:
        # sqlalchemy 1.x
        metadata = MetaData(bind=engine)
    except TypeError:
        metadata = MetaData()
        if kw.pop("autoload", None):
            kw["autoload_with"] = engine
    return Table(table_name, metadata, *args, **kw)


def test_dbapi_bearer_token(bearer_token_odps, setup):
    conn = dbapi_connect(bearer_token_odps)
    cursor = conn.cursor()
    cursor.execute("select * from one_row")
    rows = list(cursor)
    assert len(rows) == 1
    assert len(rows[0]) == 1


def test_query_with_bearer_token(bearer_token_odps, setup):
    bearer_token_odps.to_global()
    engine = create_engine("odps://")
    connection = engine.connect()
    result = connection.execute(text("SELECT * FROM one_row"))
    rows = result.fetchall()
    assert len(rows) == 1
    assert rows[0].number_of_rows == 1  # number_of_rows is the column name
    assert len(rows[0]) == 1


@pytest.mark.parametrize("engine", ["project_as_schema=false"], indirect=True)
def test_query_param(engine, connection):
    assert connection.connection.dbapi_connection._project_as_schema is False


@pytest.mark.parametrize("engine", [None, "reuse_odps=true"], indirect=True)
def test_basic_query(engine, connection):
    result = connection.execute(text("SELECT * FROM one_row"))
    instance = result.cursor._instance

    rows = result.fetchall()
    assert len(rows) == 1
    assert rows[0].number_of_rows == 1  # number_of_rows is the column name
    assert len(rows[0]) == 1

    settings = json.loads(instance.tasks[0].properties["settings"])
    assert settings["SKYNET_PYODPS_HINT"] == "hint"


@pytest.mark.parametrize("mode", ["false", "true", "v1", "v2"])
def test_interactive_modes(mode, odps, odps_with_mcqa2):
    from ..tunnel.instancetunnel import InstanceTunnel

    raw_create_download_session = InstanceTunnel.create_download_session

    def _create_download_session_with_check(*args, **kw):
        res = raw_create_download_session(*args, **kw)
        if mode == "false":
            assert not res._sessional
            assert not res._instance.id.endswith("_mcqa")
        elif mode in ("true", "v1"):
            assert res._sessional
            assert res._session_subquery_id >= 0
        elif mode == "v2":
            assert not res._sessional
            assert res._instance.id.endswith("_mcqa")
        return res

    url_suffix = "&interactive_mode=" + mode
    if mode == "v2":
        odps = odps_with_mcqa2
        url_suffix += "&quota_name=" + odps.quota_name

    create_one_row(odps)

    engine_url = "odps://{}:{}@{}/?endpoint={}{}".format(
        odps.account.access_id,
        odps.account.secret_access_key,
        odps.project,
        odps.endpoint,
        url_suffix,
    )
    engine = create_engine(engine_url)
    connection = engine.connect()
    with mock.patch(
        "odps.tunnel.instancetunnel.InstanceTunnel.create_download_session",
        new=_create_download_session_with_check,
    ):
        result = connection.execute(text("SELECT * FROM one_row"))
        rows = result.fetchall()
    assert len(rows) == 1
    assert rows[0].number_of_rows == 1  # number_of_rows is the column name
    assert len(rows[0]) == 1


def test_one_row_complex_null(engine, connection):
    one_row_complex_null = _get_sa_table("one_row_complex_null", engine, autoload=True)
    rows = connection.execute(one_row_complex_null.select()).fetchall()
    assert len(rows) == 1
    assert list(rows[0]) == [None] * len(rows[0])


def test_reflect_no_such_table(engine, connection):
    """reflecttable should throw an exception on an invalid table"""
    pytest.raises(
        NoSuchTableError,
        lambda: _get_sa_table("this_does_not_exist", engine, autoload=True),
    )
    pytest.raises(
        NoSuchTableError,
        lambda: _get_sa_table(
            "this_does_not_exist", engine, schema="also_does_not_exist", autoload=True
        ),
    )


def test_reflect_include_columns(engine, connection):
    """When passed include_columns, reflecttable should filter out other columns"""
    one_row_complex = _get_sa_table("one_row_complex", engine)
    insp = reflection.Inspector.from_engine(engine)
    insp.reflect_table(one_row_complex, include_columns=["int"], exclude_columns=[])
    assert len(one_row_complex.c) == 1
    assert one_row_complex.c.int is not None
    pytest.raises(AttributeError, lambda: one_row_complex.c.tinyint)


def test_reflect_with_schema(odps, engine, connection):
    dummy = _get_sa_table("dummy_table", engine, schema=odps.project, autoload=True)
    assert len(dummy.c) == 1
    assert dummy.c.a is not None


@pytest.mark.filterwarnings(
    "default:Omitting:sqlalchemy.exc.SAWarning" if dependency_installed else "default"
)
def test_reflect_partitions(engine, connection):
    """reflecttable should get the partition column as an index"""
    many_rows = _get_sa_table("many_rows", engine, autoload=True)
    assert len(many_rows.c) == 2

    insp = reflection.Inspector.from_engine(engine)

    many_rows = _get_sa_table("many_rows", engine)
    insp.reflect_table(many_rows, include_columns=["a"], exclude_columns=[])
    assert len(many_rows.c) == 1

    many_rows = _get_sa_table("many_rows", engine)
    insp.reflect_table(many_rows, include_columns=["b"], exclude_columns=[])
    assert len(many_rows.c) == 1


@pytest.mark.skipif(sys.version_info[0] < 3, reason="Need Python 3 to run the test")
def test_unicode(engine, connection):
    """Verify that unicode strings make it through SQLAlchemy and the backend"""
    unicode_str = "中文"
    one_row = _get_sa_table("one_row", engine)
    returned_str = connection.execute(
        sqlalchemy.select(expression.bindparam("好", unicode_str)).select_from(one_row)
    ).scalar()
    assert to_str(returned_str) == unicode_str


@pytest.mark.parametrize("engine", ["project_as_schema=true"], indirect=True)
def test_reflect_schemas_with_project(odps, engine, connection):
    insp = sqlalchemy.inspect(engine)
    schemas = insp.get_schema_names()
    assert odps.project in schemas


def test_reflect_schemas(odps, engine, connection):
    insp = sqlalchemy.inspect(engine)
    schemas = insp.get_schema_names()
    assert "default" in schemas


@pytest.mark.parametrize("engine", [None, "cache_names=true"], indirect=True)
def test_get_table_names(odps, engine, connection):
    def _new_list_tables(*_, **__):
        yield odps.get_table("one_row")
        yield odps.get_table("one_row_complex")
        yield odps.get_table("dummy_table")

    with mock.patch(
        "odps.core.ODPS.list_tables", new=_new_list_tables
    ), update_test_setting(
        get_tables_filter=lambda x: x.startswith("one_row")
        or x.startswith("dummy_table")
    ):
        meta = MetaData()
        meta.reflect(bind=engine)
        assert "one_row" in meta.tables
        assert "one_row_complex" in meta.tables

        insp = sqlalchemy.inspect(engine)
        assert "dummy_table" in insp.get_table_names(schema=odps.project)
        # make sure cache works well
        assert "dummy_table" in insp.get_table_names(schema=odps.project)


def test_has_table(engine, connection):
    insp = reflection.Inspector.from_engine(engine)
    assert insp.has_table("one_row") is True
    assert insp.has_table("this_table_does_not_exist") is False


def test_char_length(engine, connection):
    one_row_complex = _get_sa_table("one_row_complex", engine, autoload=True)
    result = connection.execute(
        sqlalchemy.select(sqlalchemy.func.char_length(one_row_complex.c.string))
    ).scalar()
    assert result == len("a string")


def test_reflect_select(engine, connection):
    """reflecttable should be able to fill in a table from the name"""
    one_row_complex = _get_sa_table("one_row_complex", engine, autoload=True)
    assert len(one_row_complex.c) == 14
    assert isinstance(one_row_complex.c.string, Column)
    row = connection.execute(one_row_complex.select()).fetchone()
    assert list(row) == _ONE_ROW_COMPLEX_CONTENTS

    # TODO some of these types could be filled in better
    assert isinstance(one_row_complex.c.boolean.type, types.Boolean)
    assert isinstance(one_row_complex.c.tinyint.type, types.Integer)
    assert isinstance(one_row_complex.c.smallint.type, types.Integer)
    assert isinstance(one_row_complex.c.int.type, types.Integer)
    assert isinstance(one_row_complex.c.bigint.type, types.BigInteger)
    assert isinstance(one_row_complex.c.float.type, types.Float)
    assert isinstance(one_row_complex.c.double.type, types.Float)
    assert isinstance(one_row_complex.c.string.type, types.String)
    assert isinstance(one_row_complex.c.timestamp.type, types.TIMESTAMP)
    assert isinstance(one_row_complex.c.binary.type, types.BINARY)
    assert isinstance(one_row_complex.c.array.type, types.String)
    assert isinstance(one_row_complex.c.map.type, types.String)
    assert isinstance(one_row_complex.c.struct.type, types.String)
    assert isinstance(one_row_complex.c.decimal.type, types.DECIMAL)


def test_type_map(engine, connection):
    """sqlalchemy should use the dbapi_type_map to infer types from raw queries"""
    row = connection.execute(text("SELECT * FROM one_row_complex")).fetchone()
    assert list(row) == _ONE_ROW_COMPLEX_CONTENTS


def test_reserved_words(engine, connection):
    """odps uses backticks"""
    # Use keywords for the table/column name
    fake_table = _get_sa_table(
        "select", engine, Column("sort", sqlalchemy.types.String)
    )
    query = str(
        fake_table.select().where(fake_table.c.sort == "a").compile(bind=engine)
    )
    assert "`select`" in query
    assert "`sort`" in query
    assert '"select"' not in query
    assert '"sort"' not in query


def test_lots_of_types(engine, connection):
    # take type list from sqlalchemy.types
    types = [
        "INT",
        "CHAR",
        "VARCHAR",
        "NCHAR",
        "TEXT",
        "Text",
        "FLOAT",
        "NUMERIC",
        "DECIMAL",
        "TIMESTAMP",
        "DATETIME",
        "CLOB",
        "BLOB",
        "BOOLEAN",
        "SMALLINT",
        "DATE",
        "TIME",
        "String",
        "Integer",
        "SmallInteger",
        "Numeric",
        "Float",
        "DateTime",
        "Date",
        "Time",
        "LargeBinary",
        "Boolean",
        "Unicode",
        "UnicodeText",
    ]
    cols = []
    for i, t in enumerate(types):
        cols.append(Column(str(i), getattr(sqlalchemy.types, t)))
    table = _get_sa_table("test_table", engine, *cols)
    table.drop(bind=engine, checkfirst=True)
    table.create(bind=engine)
    table.drop(bind=engine)


def test_insert_select(engine, connection):
    one_row = _get_sa_table("one_row", engine, autoload=True)
    table = _get_sa_table("insert_test", engine, Column("a", sqlalchemy.types.Integer))
    table.drop(bind=engine, checkfirst=True)
    table.create(bind=engine)
    connection.execute(table.insert().from_select(["a"], one_row.select()))

    result = connection.execute(table.select()).fetchall()
    expected = [(1,)]
    assert result == expected


def test_insert_values(engine, connection):
    table = _get_sa_table("insert_test", engine, Column("a", sqlalchemy.types.Integer))
    table.drop(bind=engine, checkfirst=True)
    table.create(bind=engine)
    connection.execute(table.insert().values([{"a": 1}, {"a": 2}]))

    result = connection.execute(table.select()).fetchall()
    expected = [(1,), (2,)]
    assert result == expected


def test_supports_san_rowcount(engine, connection):
    assert engine.dialect.supports_sane_rowcount_returning is False


def test_desc_sql(engine, connection):
    sql = "desc one_row"
    result = connection.execute(text(sql)).fetchall()
    assert len(result) == 1
    assert len(result[0]) == 1


def test_table_comment(engine, connection):
    insp = sqlalchemy.inspect(engine)
    assert insp.get_table_comment("one_row")["text"] == ""


def test_do_ping(engine, connection):
    engine.dialect.do_ping(engine.raw_connection())
    err_data = None

    def _patched_do_ping(*_, **__):
        raise err_data

    with mock.patch(
        "sqlalchemy.engine.default.DefaultDialect.do_ping", new=_patched_do_ping
    ):
        err_data = errors.InternalServerError("InternalServerError")
        with pytest.raises(errors.InternalServerError):
            engine.dialect.do_ping(engine.raw_connection())

        err_data = errors.NoSuchObject("no_such_obj")
        with pytest.raises(errors.NoSuchObject):
            engine.dialect.do_ping(engine.raw_connection())

        with pytest.raises(ODPSPingError) as ex_data, mock.patch(
            "odps.sqlalchemy_odps.ODPSDialect._is_stack_superset",
            new=lambda *_: True,
        ):
            engine.dialect.do_ping(engine.raw_connection())
        assert not isinstance(ex_data.value, RuntimeError)


def test_orm_with_binary(engine):
    from sqlalchemy import Column, Integer, LargeBinary, String
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker

    Base = declarative_base()

    try:

        class User(Base):
            __tablename__ = tn("test_sa_orm_with_binary")
            id = Column(Integer, primary_key=True)
            name = Column(String)
            pickle = Column(LargeBinary)

        Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        sample_data = [
            User(id=1, name="Alice", pickle=b"pickle_data_alice"),
        ]
        session.add_all(sample_data)
        session.commit()

        result = session.query(User).filter_by(name="Alice").first()
        assert result.id == 1
        assert result.name == "Alice"
        assert result.pickle == b"pickle_data_alice"
    finally:
        Base.metadata.drop_all(engine)
