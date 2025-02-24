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

import itertools
from collections import namedtuple

import mock
import pytest

from .. import dbapi
from ..core import ODPS
from ..superset_odps import ODPSEngineSpec, SupersetException
from ..tests.core import tn


@pytest.fixture
def ss_db_inspector(odps, request):
    pytest.importorskip("sqlalchemy")

    import sqlalchemy as sa

    engine_url = "odps://{}:{}@{}/?endpoint={}&SKYNET_PYODPS_HINT=hint".format(
        odps.account.access_id,
        odps.account.secret_access_key,
        odps.project,
        odps.endpoint,
    )
    if getattr(request, "param", None):
        engine_url += "&" + request.param
        # create an engine to enable cache
        sa.create_engine(engine_url)
    sa_engine = sa.create_engine(engine_url)
    inspector = sa.inspect(sa_engine)

    db = mock.MagicMock()
    db.get_sqla_engine_with_context.return_value = sa_engine

    try:
        yield db, inspector
    finally:
        from .. import sqlalchemy_odps

        sqlalchemy_odps._sqlalchemy_global_reusable_odps.clear()
        sqlalchemy_odps._sqlalchemy_obj_list_cache.clear()


def test_get_table_names(ss_db_inspector):
    db, inspector = ss_db_inspector

    _raw_list_tables = ODPS.list_tables

    def _new_list_tables(*args, **kwargs):
        it = _raw_list_tables(*args, **kwargs)
        return itertools.islice(it, 5)

    spec = ODPSEngineSpec()
    with mock.patch("odps.core.ODPS.list_tables", new=_new_list_tables):
        tables = list(spec.get_table_names(db, inspector, "default"))
    assert len(tables) > 0
    assert tables[0] is not None


@pytest.mark.parametrize(
    "ss_db_inspector", [None, "reuse_odps=true&cache_names=true"], indirect=True
)
def test_get_function_names(ss_db_inspector):
    db, inspector = ss_db_inspector

    spec = ODPSEngineSpec()
    functions = list(spec.get_function_names(db))
    assert len(functions) > 0
    # make sure cache works
    functions = list(spec.get_function_names(db))
    assert len(functions) > 0


def test_get_catalog_names(odps, ss_db_inspector):
    db, inspector = ss_db_inspector

    def mock_list_projects(self, *args, **kw):
        yield odps.get_project()

    spec = ODPSEngineSpec()
    with mock.patch("odps.core.ODPS.list_projects", new=mock_list_projects):
        catalogs = spec.get_catalog_names(db, inspector)
    assert odps.project in catalogs


def test_execute_sql(odps):
    table_name = tn("test_ss_create_table")
    odps.delete_table(table_name, if_exists=True)

    conn = dbapi.connect(
        odps.account.access_id,
        odps.account.secret_access_key,
        odps.project,
        odps.endpoint,
    )
    cursor = conn.cursor()

    spec = ODPSEngineSpec()
    spec.execute(cursor, "create table %s (col string) lifecycle 1" % table_name)
    assert odps.exist_table(table_name)
    odps.delete_table(table_name)


def test_latest_partition(odps, ss_db_inspector):
    import sqlalchemy as sa

    SSTable = namedtuple("SSTable", "table schema")
    db, inspector = ss_db_inspector
    spec = ODPSEngineSpec()

    table_name = tn("test_latest_partition")
    pt_value = "20230415"
    odps.delete_table(table_name, if_exists=True)
    tb = odps.create_table(table_name, ("col string", "pt string"), lifecycle=1)
    tb.create_partition("pt=" + pt_value)

    ss_cols = [dict(name=s) for s in ["col", "pt"]]
    sa_query = sa.select("*").select_from(sa.text(table_name))

    try:
        query = spec.where_latest_partition(
            db, SSTable(table_name, None), sa_query, columns=ss_cols
        )
        # no data, no partition returned
        assert query is None

        with tb.open_writer(partition="pt=" + pt_value) as writer:
            writer.write([["abcd"]])

        query = spec.where_latest_partition(
            db, SSTable(table_name, None), sa_query, columns=ss_cols
        )
        compiled = query.compile(dialect=inspector.dialect)
        # make sure the latest partition is selected
        assert pt_value in set(compiled.params.values())
    finally:
        tb.drop()


def test_df_to_sql(odps, ss_db_inspector):
    db, _inspector = ss_db_inspector
    SSTable = namedtuple("SSTable", "table schema")
    spec = ODPSEngineSpec()

    try:
        import pandas as pd
    except ImportError:
        pytest.skip("Need pandas to run the test")

    data = pd.DataFrame([["abcde"], ["fghij"]], columns=["col"])

    table_name = tn("test_ss_df_to_sql")
    odps.delete_table(table_name, if_exists=True)

    odps.create_table(table_name, "col string", lifecycle=1)

    try:
        with pytest.raises(SupersetException):
            spec.df_to_sql(
                db, SSTable(table_name, None), data, to_sql_kwargs={"if_exists": "fail"}
            )

        spec.df_to_sql(
            db, SSTable(table_name, None), data, to_sql_kwargs={"if_exists": "replace"}
        )
        with odps.get_table(table_name).open_reader() as reader:
            pd.testing.assert_frame_equal(reader.to_pandas(), data)
    finally:
        odps.delete_table(table_name)
