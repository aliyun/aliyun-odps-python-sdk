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
import math

from requests import Response

from . import types
from .compat import six


class AbstractRecordReader(object):

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    next = __next__

    @classmethod
    def _calc_count(cls, start, end, step):
        if end is None:
            return end
        return int(math.ceil(float(end - start) / step))

    @classmethod
    def _get_slice(cls, item):
        if isinstance(item, six.integer_types):
            start = item
            end = start + 1
            step = 1
        elif isinstance(item, slice):
            start = item.start or 0
            end = item.stop
            step = item.step or 1
        else:
            raise ValueError('Reader only supports index and slice operation.')

        return start, end, step

    def __getitem__(self, item):
        start, end, step = self._get_slice(item)
        count = self._calc_count(start, end, step)

        if start < 0 or (count is not None and count <= 0) or step < 0:
            raise ValueError('start, count, or step cannot be negative')

        it = self._iter(start=start, end=end, step=step)
        if isinstance(item, six.integer_types):
            try:
                return next(it)
            except StopIteration:
                raise IndexError('Index out of range: %s' % item)
        return it

    def _iter(self, start=None, end=None, step=None):
        start = start or 0
        step = step or 1
        curr = start

        for _ in range(start):
            try:
                next(self)
            except StopIteration:
                return

        while True:
            for i in range(step):
                try:
                    record = next(self)
                except StopIteration:
                    return
                if i == 0:
                    yield record
                curr += 1
                if end is not None and curr >= end:
                    return


class RecordReader(AbstractRecordReader):
    NULL_TOKEN = '\\N'

    def __init__(self, schema, stream, **kwargs):
        self._schema = schema
        self._columns = None
        self._fp = stream
        if isinstance(self._fp, Response):
            self.raw = self._fp.content if six.PY2 else self._fp.text
        else:
            self.raw = self._fp
        self._csv = csv.reader(self.raw.splitlines())

    def _readline(self):
        try:
            values = next(self._csv)
            res = []
            for i, value in enumerate(values):
                if value == self.NULL_TOKEN:
                    res.append(None)
                elif self._columns and self._columns[i].type == types.boolean:
                    if value == 'true':
                        res.append(True)
                    elif value == 'false':
                        res.append(False)
                    else:
                        res.append(value)
                else:
                    res.append(value)
            return res
        except StopIteration:
            return

    def __next__(self):
        self._load_columns()

        values = self._readline()
        if values is None:
            raise StopIteration
        return types.Record(self._columns, values=values)

    next = __next__

    def read(self, start=None, count=None, step=None):
        if count is None:
            end = None
        else:
            start = start or 0
            step = step or 1
            end = start + count * step
        return self._iter(start=start, end=end, step=step)

    def _load_columns(self):
        if self._columns is not None:
            return

        values = self._readline()
        self._columns = []
        for value in values:
            if self._schema is None:
                self._columns.append(types.Column(name=value, typo='string'))
            else:
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