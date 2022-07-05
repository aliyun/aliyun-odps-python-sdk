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

import itertools
from contextlib import contextmanager

from ....errors import ODPSError, NoPermission
from ....utils import write_log as log
from ....types import PartitionSpec
from ....compat import izip
from ....models import Table
from ....tunnel.tabletunnel import TableDownloadSession
from ...expr.arithmetic import And, Equal
from ...expr.reduction import *
from ...expr.element import Func
from ...utils import is_source_collection, is_source_partition
from ..frame import ResultFrame
from . import types


@contextmanager
def _open_reader(t, **kwargs):
    try:
        reader = t.open_reader(**kwargs)
        if reader.status != TableDownloadSession.Status.Normal:
            raise ODPSError('Intentionally reusing')
    except ODPSError:
        reader = t.open_reader(reopen=True, **kwargs)
    yield reader


class TunnelEngine(object):
    def __init__(self, odps):
        self._odps = odps

    @classmethod
    def _is_source_column(cls, expr, table):
        if not isinstance(expr, Column):
            return False

        odps_schema = table.schema
        if odps_schema.is_partition(expr.source_name):
            return False

        return True

    @classmethod
    def _can_propagate(cls, collection, no_filter=False, no_projection=False):
        if not isinstance(collection, CollectionExpr):
            return False

        if is_source_collection(collection) and \
                isinstance(collection._source_data, Table):
            return True
        if isinstance(collection, FilterPartitionCollectionExpr):
            return True
        if not no_filter and cls._filter_on_partition(collection):
            return True
        if not no_projection and cls._projection_on_source(collection):
            return True
        return False

    @classmethod
    def _projection_on_source(cls, expr):
        cols = []

        if isinstance(expr, ProjectCollectionExpr) and \
                cls._can_propagate(expr.input, no_projection=True):
            for col in expr.fields:
                source = next(expr.data_source())
                if not cls._is_source_column(col, source):
                    return False
                cols.append(col.source_name)
            return cols
        elif isinstance(expr, FilterPartitionCollectionExpr):
            return expr.schema.names
        return False

    @classmethod
    def _filter_on_partition(cls, expr):
        if not isinstance(expr, (FilterCollectionExpr, FilterPartitionCollectionExpr)) or \
                not cls._can_propagate(expr.input, no_filter=True):
            return False

        cols = []
        values = []

        def extract(expr):
            if isinstance(expr, Column):
                if is_source_partition(expr, next(expr.data_source())):
                    cols.append(expr.source_name)
                    return True
                else:
                    return False

            if isinstance(expr, And):
                for child in expr.args:
                    if not extract(child):
                        return False
            elif isinstance(expr, Equal) and isinstance(expr._rhs, Scalar) and \
                    not isinstance(expr._rhs, Func):  # skip the internal function
                if extract(expr._lhs):
                    values.append(expr._rhs.value)
                    return True
                else:
                    return False
            else:
                return False

            return True

        if not extract(expr.predicate):
            return False

        if len(cols) == len(values):
            return list(zip(cols, values))
        return False

    @classmethod
    def _to_partition_spec(cls, kv):
        spec = PartitionSpec()
        for k, v in kv:
            spec[k] = v
        return spec

    @classmethod
    def _partition_prefix(cls, all_partitions, filtered_partitions):
        filtered_partitions = sorted(six.iteritems(filtered_partitions.kv),
                                     key=lambda x: all_partitions.index(x[0]))
        if len(filtered_partitions) > len(all_partitions):
            return
        if not all(zip(l == r for l, r in zip(all_partitions, filtered_partitions))):
            return

        return cls._to_partition_spec(filtered_partitions)

    def execute(self, expr, ui=None, progress_proportion=1,
                head=None, tail=None, verify=False, update_progress_count=50):
        if isinstance(expr, (ProjectCollectionExpr, Summary)) and \
                len(expr.fields) == 1 and \
                isinstance(expr.fields[0], Count):
            expr = expr.fields[0]

        columns, partitions, count = (None, ) * 3
        if isinstance(expr, Count):
            if isinstance(expr.input, Column):
                input = expr.input.input
            else:
                input = expr.input
        elif isinstance(expr, SliceCollectionExpr):
            input = expr.input
        else:
            input = expr

        if verify:
            return self._can_propagate(input)
        if not self._can_propagate(input):
            return

        while True:
            if isinstance(input, FilterPartitionCollectionExpr):
                partition_kv = self._filter_on_partition(input)
                if not partition_kv:
                    return
                partitions = self._to_partition_spec(partition_kv)
                if not columns:
                    columns = self._projection_on_source(input)
                break
            else:
                ret = self._filter_on_partition(input)
                if ret:
                    partitions = self._to_partition_spec(ret)
                    input = input.input
                    continue

                ret = self._projection_on_source(input)
                if ret:
                    columns = ret
                    input = input.input
                    continue
                break

        table = next(expr.data_source())
        partition, filter_all_partitions = None, True
        if table.schema.partitions:
            if partitions is not None:
                partition = self._partition_prefix(
                    [p.name for p in table.schema.partitions], partitions)
                if partition is None:
                    return
                if len(table.schema.partitions) != len(partitions):
                    filter_all_partitions = False
            else:
                filter_all_partitions = False

        if isinstance(expr, Count):
            if not filter_all_partitions:
                # if not filter all partitions, fall back to ODPS SQL to calculate count
                return
            try:
                with _open_reader(table, partition=partition) as reader:
                    ui.inc(progress_proportion)
                    return reader.count
            except ODPSError:
                return
        else:
            log('Try to fetch data from tunnel')
            ui.status('Try to download data with tunnel...', clear_keys=True)
            if isinstance(expr, SliceCollectionExpr):
                if expr.start:
                    raise ExpressionError('For ODPS backend, slice\'s start cannot be specified')
                count = expr.stop
            try:
                data = []

                start, size, step = None, None, None
                if head is not None:
                    size = min(head, count) if count is not None else head
                elif tail is not None:
                    if filter_all_partitions:
                        start = None if count is None else max(count - tail, 0)
                        size = tail if count is None else min(count, tail)
                    else:
                        # tail on multi partitions, just fall back to SQL
                        return
                else:
                    size = count

                fetch_partitions = [partition] if filter_all_partitions else \
                    (p.name for p in table.iterate_partitions(partition))
                if tail is not None:
                    fetch_partitions = list(fetch_partitions)[::-1]
                if size is None:
                    fetch_partitions = list(fetch_partitions)

                cum = 0
                last_percent = 0
                for curr_part, partition in izip(itertools.count(1), fetch_partitions):
                    rest = size - cum if size is not None else None
                    finished = False

                    with _open_reader(table, partition=partition) as reader:
                        if tail is not None and start is None:
                            s = max(reader.count - tail, 0)
                            start = s if start is None else max(s, start)

                        unique_columns = list(OrderedDict.fromkeys(columns)) if columns is not None else None
                        for i, r in izip(itertools.count(1),
                                         reader.read(start=start, count=rest, columns=unique_columns)):
                            if size is not None and cum > size - 1:
                                finished = True
                                break
                            cum += 1
                            if cum % update_progress_count == 0:
                                if size is not None:
                                    p = float(cum) / size * progress_proportion
                                    ui.inc(p - last_percent)
                                    last_percent = p
                                else:
                                    p = ((curr_part - 1) / len(fetch_partitions) +
                                         float(i) / reader.count / len(fetch_partitions)) * progress_proportion
                                    ui.inc(p - last_percent)
                                    last_percent = p
                            if partition:
                                spec = PartitionSpec(partition) if not isinstance(partition, PartitionSpec) \
                                    else partition
                                self._fill_back_partition_values(r, table, spec.kv)
                            if columns is None or len(unique_columns) == len(columns):
                                data.append(r.values)
                            else:
                                data.append([r[n] for n in columns])

                    if finished:
                        break

                if last_percent < progress_proportion:
                    ui.inc(progress_proportion - last_percent)
                return ResultFrame(data, schema=expr._schema)
            except NoPermission:
                raise
            except ODPSError:
                return

    @classmethod
    def _fill_back_partition_values(cls, record, table, pkv):
        if pkv:
            for k, v in six.iteritems(pkv):
                if k in record and record[k] is None:
                    # fill back the partition data which is lost in the tunnel
                    record[k] = types.odps_types.validate_value(v, table.schema.get_type(k))