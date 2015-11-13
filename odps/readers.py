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

import csv

from requests import Response

from . import types


class RecordReader(object):
    NULL_TOKEN = '\\N'

    def __init__(self, schema, stream, **kwargs):
        self._schema = schema
        self._columns = None
        self._fp = stream
        if isinstance(self._fp, Response):
            content = self._fp.content
        else:
            content = self._fp
        self._csv = csv.reader(content.splitlines())

    def _readline(self):
        try:
            return [item if item != self.NULL_TOKEN else None
                    for item in self._csv.next()]
        except StopIteration:
            return

    def __next__(self):
        self._load_columns()

        values = self._readline()
        if values is None:
            raise StopIteration
        return types.Record(self._columns, values=values)

    next = __next__

    def __iter__(self):
        return self

    reads = __iter__  # compatible

    def read(self):
        try:
            return next(self)
        except StopIteration:
            return

    def read_raw(self):
        self._load_columns()

        return self._readline()

    def _load_columns(self):
        if self._columns is not None:
            return

        values = self._readline()
        self._columns = []
        for value in values:
            if self._schema.is_partition(value):
                self._columns.append(self._schema.get_partition(value))
            else:
                self._columns.append(self._schema.get_column(value))

    def close(self):
        if hasattr(self._fp, 'close'):
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()