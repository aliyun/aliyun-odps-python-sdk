#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

from ..models import Table
from .expr.utils import get_attrs
from .expr.expressions import CollectionExpr
from .backends.odpssql.types import odps_schema_to_df_schema
from .backends.pd.types import pd_to_df_schema


class DataFrame(CollectionExpr):
    """
    Main entrance of PyOdps DataFrame.

    Users can initial a DataFrame by :class:`odps.models.Table`.

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

    def __init__(self, data, **kwargs):
        """
        :param data: ODPS table
        :type data: :class:`odps.models.Table`
        """
        if isinstance(data, Table):
            schema = odps_schema_to_df_schema(data.schema)
            super(DataFrame, self).__init__(_source_data=data, _schema=schema, **kwargs)
        elif has_pandas and isinstance(data, pd.DataFrame):
            schema = pd_to_df_schema(data)
            super(DataFrame, self).__init__(_source_data=data, _schema=schema, **kwargs)
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

    @property
    def data(self):
        return self._source_data

