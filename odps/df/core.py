#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import functools

from ..models import Table
from ..models.partition import Partition
from ..compat import six, izip
from .expr.utils import get_attrs
from .expr.expressions import CollectionExpr
from .types import validate_data_type
from .backends.odpssql.types import odps_schema_to_df_schema
from .backends.pd.types import pd_to_df_schema,  df_type_to_np_type
from .backends.sqlalchemy.types import sqlalchemy_to_df_schema

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

try:
    import sqlalchemy
    has_sqlalchemy = True
except ImportError:
    has_sqlalchemy = False


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
                kwargs['_schema'] = odps_schema_to_df_schema(data.schema)
            super(DataFrame, self).__init__(_source_data=data, **kwargs)
        elif isinstance(data, Partition):
            if '_schema' not in kwargs:
                kwargs['_schema'] = odps_schema_to_df_schema(data.parent.parent.schema)
            super(DataFrame, self).__init__(_source_data=data.parent.parent, **kwargs)
            self._proxy = self.copy().filter_partition(data)
        elif has_pandas and isinstance(data, pd.DataFrame):
            if 'schema' in kwargs and kwargs['schema']:
                schema = kwargs.pop('schema')
            elif '_schema' in kwargs:
                schema = kwargs.pop('_schema')
            else:
                unknown_as_string = kwargs.pop('unknown_as_string', False)
                as_type = kwargs.pop('as_type', None)
                if as_type:
                    data = data.copy()
                    data.is_copy = False
                    as_type = dict((k, validate_data_type(v)) for k, v in six.iteritems(as_type))

                    if not isinstance(as_type, dict):
                        raise TypeError('as_type must be dict')
                    for col_name, df_type in six.iteritems(as_type):
                        pd_type = df_type_to_np_type(df_type)
                        if col_name not in data:
                            raise ValueError('col(%s) does not exist in pd.DataFrame' % col_name)
                        try:
                            data[col_name] = data[col_name][data[col_name].notnull()].astype(pd_type)
                        except TypeError:
                            raise TypeError('Cannot cast col(%s) to data type: %s' % (col_name, df_type))
                schema = pd_to_df_schema(data, as_type=as_type,
                                         unknown_as_string=unknown_as_string)
            super(DataFrame, self).__init__(_source_data=data, _schema=schema, **kwargs)
        elif has_sqlalchemy and isinstance(data, sqlalchemy.Table):
            if '_schema' not in kwargs:
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

        execute_keys = ('ui', 'async', 'n_parallel', 'timeout', 'close_and_notify')
        execute_kw = dict((k, v) for k, v in six.iteritems(kwargs) if k in execute_keys)
        persist_kw = dict((k, v) for k, v in six.iteritems(kwargs) if k not in execute_keys)

        delay = Delay()
        persist_kw['delay'] = delay

        for df, table in izip(dfs, tables):
            if isinstance(table, tuple):
                table, partition = table
            else:
                partition = None
            df.persist(table, partition=partition, *args, **persist_kw)

        return delay.execute(**execute_kw)

    @property
    def data(self):
        return self._source_data

