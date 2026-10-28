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

import importlib.util
import sys

from ..models import Table
from ..models.partition import Partition
from ..types import CompositeDataType
from .expr.utils import get_attrs
from .expr.expressions import CollectionExpr
from .types import DynamicSchema, validate_data_type
from .backends.odpssql.types import odps_schema_to_df_schema, odps_type_to_df_type
from .backends.pd.types import pd_to_df_schema, df_type_to_np_type, cast_composite_sequence

try:
    import pandas as pd
    has_pandas = True
except (ImportError, ValueError):
    has_pandas = False

_sqlalchemy_installed_cache = None


def _get_imported_sqlalchemy():
    """Return the sqlalchemy module only if it has already been imported."""
    # correctness: a sqlalchemy object implies its module was imported, so a
    # None here means no isinstance() against sqlalchemy types can succeed.
    return sys.modules.get("sqlalchemy")


def _is_sqlalchemy_table(data):
    sa = _get_imported_sqlalchemy()
    return sa is not None and isinstance(data, sa.Table)


def _sqlalchemy_installed():
    """Whether sqlalchemy is findable, without importing it.

    "Findable", not "importable": a broken install still reports True. The flag
    is advisory - dispatch uses `sys.modules` via `_get_imported_sqlalchemy()`,
    so patching `has_sqlalchemy` does not change behaviour.
    """
    global _sqlalchemy_installed_cache
    if sys.modules.get("sqlalchemy") is not None:
        return True
    if _sqlalchemy_installed_cache:
        return True
    try:
        installed = importlib.util.find_spec("sqlalchemy") is not None
    except (ImportError, ValueError):
        # Transient probe failure (e.g. sys.path being mutated); do not latch
        # False, so the next call can still discover an installed package.
        return False
    if installed:
        _sqlalchemy_installed_cache = True
    return installed


def __getattr__(name):
    if name == "has_sqlalchemy":
        # Advisory: findable is not importable, and dispatch uses `sys.modules`.
        return _sqlalchemy_installed()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | {"has_sqlalchemy"})


def _df_type_of(col_type):
    """Return the DataFrame type of a storage type, or the type itself."""
    if hasattr(col_type, 'CLASS_NAME'):
        return col_type
    try:
        return odps_type_to_df_type(col_type)
    except KeyError:
        # No DataFrame counterpart (exotic storage type); keep it so the frame
        # can still be constructed.
        return col_type


def _convert_column(col):
    """Return ``col`` with a DataFrame type, or ``col`` itself if it has one."""
    df_type = _df_type_of(col.type)
    if df_type is col.type:
        return col

    # Keyword arguments only: the model column classes
    # (``TableSchema.TableColumn``/``TablePartition``) accept nothing else.
    return type(col)(
        name=col.name,
        type=df_type,
        comment=col.comment,
        label=col.label,
        nullable=col.nullable,
        generate_expression=col._generate_expression,
    )


def _normalize_local_schema(schema):
    """Convert storage types in a user-supplied schema into DataFrame types.

    ``DataFrame(pd_df, schema=TableSchema.from_lists(...))`` accepts an
    :class:`odps.models.TableSchema`, whose columns carry ``odps.types``
    (storage) instances. Expressions need the ``odps.df.types`` counterparts:
    those provide ``CLASS_NAME``, which field access builds its class from.

    Only the types change: column classes, comments, labels, nullability, the
    partition split and, for dynamic schemas, the default type are carried over.
    """
    columns = [_convert_column(col) for col in schema.simple_columns]
    partitions = [_convert_column(part) for part in schema.partitions or []]

    if all(
        new.type is old.type
        for new, old in zip(columns + partitions, schema.columns)
    ):
        # Nothing to convert; keep the caller's schema object as it is.
        return schema

    kwargs = {}
    if isinstance(schema, DynamicSchema):
        kwargs['default_type'] = schema.default_type
    return type(schema)(columns=columns, partitions=partitions or None, **kwargs)


class DataFrame(CollectionExpr):
    """
    Main entrance of PyODPS DataFrame.

    Users can initial a DataFrame by :class:`odps.models.Table`.

    :param data: ODPS table or pandas DataFrame
    :type data: :class:`odps.models.Table` or pandas DataFrame

    :Example:

    >>> df = DataFrame(o.get_table('my_example_table'))
    >>> df.dtypes
    odps.Schema {
      movie_id                            int64
      title                               string
      release_date                        string
      video_release_date                  string
      imdb_url                            string
      user_id                             int64
      rating                              int64
      unix_timestamp                      int64
      age                                 int64
      sex                                 string
      occupation                          string
      zip_code                            string
    }
    >>> df.count()
    100000
    >>>
    >>> # Do the `groupby`, aggregate the `movie_id` by count, then sort the count in a reversed order
    >>> # Finally we get the top 25 results
    >>> df.groupby('title').agg(count=df.movie_id.count()).sort('count', ascending=False)[:25]
    >>>
    >>> # We can use the `value_counts` to reach the same goal
    >>> df.movie_id.value_counts()[:25]
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            data = args[0]
        else:
            data = kwargs.pop('_source_data', None)
            if data is None:
                raise ValueError('ODPS Table or pandas DataFrame should be provided.')

        if isinstance(data, Table):
            if '_schema' not in kwargs:
                kwargs['_schema'] = odps_schema_to_df_schema(data.table_schema)
            super(DataFrame, self).__init__(_source_data=data, **kwargs)
        elif isinstance(data, Partition):
            if '_schema' not in kwargs:
                kwargs['_schema'] = odps_schema_to_df_schema(data.table.table_schema)
            super(DataFrame, self).__init__(_source_data=data.parent.parent, **kwargs)
            self._proxy = self.copy().filter_parts(data)
        elif has_pandas and isinstance(data, pd.DataFrame):
            if 'schema' in kwargs and kwargs['schema']:
                schema = _normalize_local_schema(kwargs.pop('schema'))
            elif '_schema' in kwargs:
                schema = _normalize_local_schema(kwargs.pop('_schema'))
            else:
                unknown_as_string = kwargs.pop('unknown_as_string', False)
                as_type = kwargs.pop('as_type', None)
                if as_type:
                    data = data.copy()
                    data.is_copy = False
                    as_type = dict((k, validate_data_type(v)) for k, v in as_type.items())

                    if not isinstance(as_type, dict):
                        raise TypeError('as_type must be dict')
                    for col_name, df_type in as_type.items():
                        if col_name not in data:
                            raise ValueError('col(%s) does not exist in pd.DataFrame' % col_name)
                        try:
                            if isinstance(df_type, CompositeDataType):
                                data[col_name] = cast_composite_sequence(data[col_name], df_type)
                            else:
                                pd_type = df_type_to_np_type(df_type)
                                data[col_name] = data[col_name][data[col_name].notnull()].astype(pd_type)
                        except TypeError:
                            raise TypeError('Cannot cast col(%s) to data type: %s' % (col_name, df_type))
                schema = pd_to_df_schema(data, as_type=as_type,
                                         unknown_as_string=unknown_as_string)
            super(DataFrame, self).__init__(_source_data=data, _schema=schema, **kwargs)
        elif _is_sqlalchemy_table(data):
            if '_schema' not in kwargs:
                from .backends.sqlalchemy.types import sqlalchemy_to_df_schema
                kwargs['_schema'] = sqlalchemy_to_df_schema(data.c)
            super(DataFrame, self).__init__(_source_data=data, **kwargs)
        else:
            raise ValueError('Unknown type: %s' % data)

    def __setstate__(self, state):
        kv = dict(state)
        source_data = kv.pop('_source_data')
        kv.pop('_schema', None)
        self.__init__(source_data, **kv)

    def view(self):
        kv = dict((attr, getattr(self, attr)) for attr in get_attrs(self))
        data = kv.pop('_source_data')
        kv.pop('_schema', None)
        return type(self)(data, **kv)

    @staticmethod
    def batch_persist(dfs, tables, *args, **kwargs):
        """
        Persist multiple DataFrames into ODPS.

        :param dfs: DataFrames to persist.
        :param tables: Table names to persist to. Use (table, partition) tuple to store to a table partition.
        :param args: args for Expr.persist
        :param kwargs: kwargs for Expr.persist

        :Examples:
        >>> DataFrame.batch_persist([df1, df2], ['table_name1', ('table_name2', 'partition_name2')], lifecycle=1)
        """
        from .delay import Delay

        if 'async' in kwargs:
            kwargs['async_'] = kwargs['async']
        execute_keys = ('ui', 'async_', 'n_parallel', 'timeout', 'close_and_notify')
        execute_kw = dict((k, v) for k, v in kwargs.items() if k in execute_keys)
        persist_kw = dict((k, v) for k, v in kwargs.items() if k not in execute_keys)

        delay = Delay()
        persist_kw['delay'] = delay

        for df, table in zip(dfs, tables):
            if isinstance(table, tuple):
                table, partition = table
            else:
                partition = None
            df.persist(table, partition=partition, *args, **persist_kw)

        return delay.execute(**execute_kw)

    @property
    def data(self):
        return self._source_data
