#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
import functools
import decimal
import unittest

from odps import options
from odps.compat import OrderedDict
from odps.df import DataFrame
from odps.tests.core import TestBase
from odps.utils import to_str

try:
    import pytest
    import numpy as np
    import pandas as pd
    import sqlalchemy
    from sqlalchemy import types
    from sqlalchemy.engine import create_engine
    from sqlalchemy.exc import NoSuchTableError
    from sqlalchemy.schema import Column, MetaData, Table
    from sqlalchemy.sql import expression

    from odps.sqlalchemy_odps import update_test_setting

    _ONE_ROW_COMPLEX_CONTENTS = [
        True,
        127,
        32767,
        2147483647,
        9223372036854775807,
        0.5,
        0.25,
        'a string',
        pd.Timestamp(1970, 1, 1, 8),
        b'123',
        [1, 2],
        OrderedDict({1: 2, 3: 4}),
        OrderedDict({"a": 1, "b": 2}),
        decimal.Decimal('0.1'),
    ]
except ImportError:
    dependency_installed = False
else:
    dependency_installed = True


def create_one_row(o):
    table = 'one_row'
    if not o.exist_table(table):
        o.execute_sql("CREATE TABLE one_row (number_of_rows INT);")
        o.execute_sql("INSERT INTO TABLE one_row VALUES (1);")


def create_one_row_complex(o):
    need_writes = [False] * 2
    for i, table in enumerate(['one_row_complex', 'one_row_complex_null']):
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
        """.format(table)
        o.execute_sql(ddl)
        need_writes[i] = True

    if need_writes[0]:
        o.execute_sql("""
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
        """)

    if need_writes[1]:
        o.execute_sql("""
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
        """)


def create_many_rows(o):
    table = 'many_rows'
    if not o.exist_table(table):
        df = pd.DataFrame({'a': np.arange(10000, dtype=np.int32)})
        o.execute_sql("""
        CREATE TABLE many_rows (
            a INT
        ) PARTITIONED BY (
            b STRING
        )
        """)
        DataFrame(df).persist('many_rows', partition="b='blah'",
                              odps=o)


def create_test(o):
    table = 'dummy_table'
    if not o.exist_table(table):
        o.create_table(table, 'a int')


def with_engine_connection(fn):
    """Pass a connection to the given function and handle cleanup.
    The connection is taken from ``self.create_engine()``.
    """
    @functools.wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        engine = self.create_engine()
        try:
            with contextlib.closing(engine.connect()) as connection:
                fn(self, engine, connection, *args, **kwargs)
        finally:
            engine.dispose()
    return wrapped_fn


@unittest.skipIf(not dependency_installed, 'dependency for sqlalchemy_odps not installed')
class Test(TestBase):
    def setup(self):
        options.sql.use_odps2_extension = True
        self.old_endpoint = options.endpoint
        options.endpoint = self.odps.endpoint
        options.sql.settings = {
            'odps.sql.decimal.odps2': True
        }

        # create test tables
        create_many_rows(self.odps)
        create_one_row(self.odps)
        create_one_row_complex(self.odps)
        create_test(self.odps)

    def teardown(self):
        options.sql.use_odps2_extension = False
        options.endpoint = self.old_endpoint
        options.sql.settings = None

    def create_engine(self):
        return create_engine('odps://{}:{}@{}'.format(
            self.odps.account.access_id, self.odps.account.secret_access_key,
            self.odps.project))

    @with_engine_connection
    def test_basic_query(self, engine, connection):
        rows = connection.execute('SELECT * FROM one_row').fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].number_of_rows, 1)  # number_of_rows is the column name
        self.assertEqual(len(rows[0]), 1)

    @with_engine_connection
    def test_one_row_complex_null(self, engine, connection):
        one_row_complex_null = Table('one_row_complex_null', MetaData(bind=engine), autoload=True)
        rows = one_row_complex_null.select().execute().fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(list(rows[0]), [None] * len(rows[0]))

    @with_engine_connection
    def test_reflect_no_such_table(self, engine, connection):
        """reflecttable should throw an exception on an invalid table"""
        self.assertRaises(
            NoSuchTableError,
            lambda: Table('this_does_not_exist', MetaData(bind=engine), autoload=True))
        self.assertRaises(
            NoSuchTableError,
            lambda: Table('this_does_not_exist', MetaData(bind=engine),
                          schema='also_does_not_exist', autoload=True))

    @with_engine_connection
    def test_reflect_include_columns(self, engine, connection):
        """When passed include_columns, reflecttable should filter out other columns"""
        one_row_complex = Table('one_row_complex', MetaData(bind=engine))
        engine.dialect.reflecttable(
            connection, one_row_complex, include_columns=['int'],
            exclude_columns=[], resolve_fks=True)
        self.assertEqual(len(one_row_complex.c), 1)
        self.assertIsNotNone(one_row_complex.c.int)
        self.assertRaises(AttributeError, lambda: one_row_complex.c.tinyint)

    @with_engine_connection
    def test_reflect_with_schema(self, engine, connection):
        dummy = Table('dummy_table', MetaData(bind=engine), schema=self.odps.project,
                      autoload=True)
        self.assertEqual(len(dummy.c), 1)
        self.assertIsNotNone(dummy.c.a)

    @pytest.mark.filterwarnings('default:Omitting:sqlalchemy.exc.SAWarning')
    @with_engine_connection
    def test_reflect_partitions(self, engine, connection):
        """reflecttable should get the partition column as an index"""
        many_rows = Table('many_rows', MetaData(bind=engine), autoload=True)
        self.assertEqual(len(many_rows.c), 2)

        many_rows = Table('many_rows', MetaData(bind=engine))
        engine.dialect.reflecttable(
            connection, many_rows, include_columns=['a'],
            exclude_columns=[], resolve_fks=True)
        self.assertEqual(len(many_rows.c), 1)

        many_rows = Table('many_rows', MetaData(bind=engine))
        engine.dialect.reflecttable(
            connection, many_rows, include_columns=['b'],
            exclude_columns=[], resolve_fks=True)
        self.assertEqual(len(many_rows.c), 1)

    @with_engine_connection
    def test_unicode(self, engine, connection):
        """Verify that unicode strings make it through SQLAlchemy and the backend"""
        unicode_str = "中文"
        one_row = Table('one_row', MetaData(bind=engine))
        returned_str = sqlalchemy.select(
            [expression.bindparam("好", unicode_str)],
            from_obj=one_row,
        ).scalar()
        self.assertEqual(to_str(returned_str), unicode_str)

    @with_engine_connection
    def test_reflect_schemas(self, engine, connection):
        insp = sqlalchemy.inspect(engine)
        schemas = insp.get_schema_names()
        self.assertIn(self.odps.project, schemas)

    @with_engine_connection
    def test_get_table_names(self, engine, connection):
        with update_test_setting(
                get_tables_filter=lambda x: x.startswith('one_row') or
                                            x.startswith('dummy_table')):
            meta = MetaData()
            meta.reflect(bind=engine)
            self.assertIn('one_row', meta.tables)
            self.assertIn('one_row_complex', meta.tables)

            insp = sqlalchemy.inspect(engine)
            self.assertIn(
                'dummy_table',
                insp.get_table_names(schema=self.odps.project),
            )

    @with_engine_connection
    def test_has_table(self, engine, connection):
        self.assertTrue(Table('one_row', MetaData(bind=engine)).exists())
        self.assertFalse(Table('this_table_does_not_exist', MetaData(bind=engine)).exists())

    @with_engine_connection
    def test_char_length(self, engine, connection):
        one_row_complex = Table('one_row_complex', MetaData(bind=engine), autoload=True)
        result = sqlalchemy.select([
            sqlalchemy.func.char_length(one_row_complex.c.string)
        ]).execute().scalar()
        self.assertEqual(result, len('a string'))

    @with_engine_connection
    def test_reflect_select(self, engine, connection):
        """reflecttable should be able to fill in a table from the name"""
        one_row_complex = Table('one_row_complex', MetaData(bind=engine), autoload=True)
        self.assertEqual(len(one_row_complex.c), 14)
        self.assertIsInstance(one_row_complex.c.string, Column)
        row = one_row_complex.select().execute().fetchone()
        self.assertEqual(list(row), _ONE_ROW_COMPLEX_CONTENTS)

        # TODO some of these types could be filled in better
        self.assertIsInstance(one_row_complex.c.boolean.type, types.Boolean)
        self.assertIsInstance(one_row_complex.c.tinyint.type, types.Integer)
        self.assertIsInstance(one_row_complex.c.smallint.type, types.Integer)
        self.assertIsInstance(one_row_complex.c.int.type, types.Integer)
        self.assertIsInstance(one_row_complex.c.bigint.type, types.BigInteger)
        self.assertIsInstance(one_row_complex.c.float.type, types.Float)
        self.assertIsInstance(one_row_complex.c.double.type, types.Float)
        self.assertIsInstance(one_row_complex.c.string.type, types.String)
        self.assertIsInstance(one_row_complex.c.timestamp.type, types.TIMESTAMP)
        self.assertIsInstance(one_row_complex.c.binary.type, types.String)
        self.assertIsInstance(one_row_complex.c.array.type, types.String)
        self.assertIsInstance(one_row_complex.c.map.type, types.String)
        self.assertIsInstance(one_row_complex.c.struct.type, types.String)
        self.assertIsInstance(one_row_complex.c.decimal.type, types.DECIMAL)

    @with_engine_connection
    def test_type_map(self, engine, connection):
        """sqlalchemy should use the dbapi_type_map to infer types from raw queries"""
        row = connection.execute('SELECT * FROM one_row_complex').fetchone()
        self.assertListEqual(list(row), _ONE_ROW_COMPLEX_CONTENTS)

    @with_engine_connection
    def test_reserved_words(self, engine, connection):
        """odps uses backticks"""
        # Use keywords for the table/column name
        fake_table = Table('select', MetaData(bind=engine), Column('sort', sqlalchemy.types.String))
        query = str(fake_table.select(fake_table.c.sort == 'a'))
        self.assertIn('`select`', query)
        self.assertIn('`sort`', query)
        self.assertNotIn('"select"', query)
        self.assertNotIn('"sort"', query)

    @with_engine_connection
    def test_lots_of_types(self, engine, connection):
        # take type list from sqlalchemy.types
        types = [
            'INT', 'CHAR', 'VARCHAR', 'NCHAR', 'TEXT', 'Text', 'FLOAT',
            'NUMERIC', 'DECIMAL', 'TIMESTAMP', 'DATETIME', 'CLOB', 'BLOB',
            'BOOLEAN', 'SMALLINT', 'DATE', 'TIME',
            'String', 'Integer', 'SmallInteger',
            'Numeric', 'Float', 'DateTime', 'Date', 'Time', 'LargeBinary',
            'Boolean', 'Unicode', 'UnicodeText',
        ]
        cols = []
        for i, t in enumerate(types):
            cols.append(Column(str(i), getattr(sqlalchemy.types, t)))
        table = Table('test_table', MetaData(bind=engine), *cols)
        table.drop(checkfirst=True)
        table.create()
        table.drop()

    @with_engine_connection
    def test_insert_select(self, engine, connection):
        one_row = Table('one_row', MetaData(bind=engine), autoload=True)
        table = Table('insert_test', MetaData(bind=engine),
                      Column('a', sqlalchemy.types.Integer))
        table.drop(checkfirst=True)
        table.create()
        connection.execute(table.insert().from_select(['a'], one_row.select()))

        result = table.select().execute().fetchall()
        expected = [(1,)]
        self.assertEqual(result, expected)

    @with_engine_connection
    def test_insert_values(self, engine, connection):
        table = Table('insert_test', MetaData(bind=engine),
                      Column('a', sqlalchemy.types.Integer))
        table.drop(checkfirst=True)
        table.create()
        connection.execute(table.insert([{'a': 1}, {'a': 2}]))

        result = table.select().execute().fetchall()
        expected = [(1,), (2,)]
        self.assertEqual(result, expected)

    @with_engine_connection
    def test_supports_san_rowcount(self, engine, connection):
        self.assertFalse(engine.dialect.supports_sane_rowcount_returning)
