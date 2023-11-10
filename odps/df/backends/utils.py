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

from collections import deque, OrderedDict
import itertools

from ..expr.dynamic import DynamicMixin
from ..expr.expressions import \
    FilterCollectionExpr, FilterPartitionCollectionExpr, Column, Scalar
from ..expr.arithmetic import Or, And, Equal
from ..expr.errors import ExpressionError
from ..utils import is_source_partition
from ...errors import ODPSError
from ... import compat, options


def refresh_dynamic(executed, dag, func=None):
    q = deque()
    q.append(executed)

    while len(q) > 0:
        curr = q.popleft()

        for p in dag.successors(curr):
            if isinstance(p, DynamicMixin):
                q.append(p)

        if isinstance(curr, DynamicMixin):
            deps = set([id(d) for d in curr.deps or []])
            if any(id(c) not in deps and isinstance(c, DynamicMixin) for c in curr.children()):
                continue
            try:
                sub = curr.to_static()
            except ExpressionError:
                # the node is the optimized to disappear which should not be handled here
                continue
            dag.substitute(curr, sub)
            if func:
                func(sub)


def _fetch_partition_size(part, table):
    if len(part.split(',')) < len(table.table_schema.partitions):
        try:
            return sum(p.size for p in table.iterate_partitions(spec=part))
        except ODPSError:
            return
    else:
        try:
            if part not in table.partitions:
                return
        except ODPSError:
            return

        return table.partitions[part].size


def fetch_data_source_size(expr_dag, node, table):
    schema = table.table_schema

    if not schema.partitions:
        # not partitioned table
        return table.size

    size = 0
    for parent in expr_dag.successors(node):
        if isinstance(parent, FilterPartitionCollectionExpr):
            partition_predicates = parent._predicate_string.split('/')
            for p in partition_predicates:
                curr_size = _fetch_partition_size(p, table)
                if curr_size:
                    size += curr_size
                else:
                    return
        elif isinstance(parent, FilterCollectionExpr):
            def gen(n):
                if isinstance(n, Equal):
                    expr = n.lhs
                    if isinstance(expr, Column) and is_source_partition(expr, next(expr.data_source())) \
                            and isinstance(n.rhs, Scalar):
                        return '%s=%s' % (expr.source_name, n.rhs.value)
                elif isinstance(n, And):
                    left_partition_spec = gen(n.lhs)
                    right_partition_spec = gen(n.rhs)
                    if left_partition_spec and right_partition_spec:
                        partition_spec = ','.join((left_partition_spec, right_partition_spec))
                    elif left_partition_spec:
                        partition_spec = left_partition_spec
                    elif right_partition_spec:
                        partition_spec = right_partition_spec
                    else:
                        return
                    return partition_spec

            def walk(predicate):
                if isinstance(predicate, Or):
                    left_size = walk(predicate._lhs)
                    right_size = walk(predicate._rhs)
                    return (left_size or 0) + (right_size or 0)
                elif isinstance(predicate, (And, Equal)):
                    partition_spec = gen(predicate)
                    curr_size = _fetch_partition_size(partition_spec, table) \
                        if partition_spec is not None else None
                    if curr_size is not None:
                        return curr_size

            filter_size = walk(parent._predicate)
            if not filter_size:
                return
            size += filter_size

    return size


def _convert_pd_type(values, table):
    import pandas as pd

    retvals = []
    for val, t in compat.izip(values, table.table_schema.types):
        isnull = pd.isnull(val)
        if isinstance(isnull, bool) and isnull:
            retvals.append(None)
        else:
            retvals.append(val)

    return retvals


def _reorder_pd(frame, table, cast=False):
    import pandas as pd
    from .odpssql.types import df_schema_to_odps_schema, df_type_to_odps_type, odps_type_to_df_type

    from .pd.types import df_type_to_np_type
    from .errors import CompileError

    expr_table_schema = df_schema_to_odps_schema(frame.schema).to_ignorecase_schema()
    for col in expr_table_schema.columns:
        if col.name.lower() not in table.table_schema:
            raise CompileError('Column(%s) does not exist in target table %s, '
                               'writing cannot be performed.' % (col.name, table.name))
        t_col = table.table_schema[col.name.lower()]
        if not cast and not t_col.type.can_implicit_cast(col.type):
            raise CompileError('Cannot implicitly cast column %s from %s to %s.' % (
                col.name, col.type, t_col.type))

    size = len(frame.values)

    df_col_dict = dict((c.name.lower(), c) for c in frame.columns)
    case_dict = dict((c.name.lower(), c.name) for c in frame.columns)

    data_dict = dict()
    for dest_col in table.table_schema.columns:
        if dest_col.name not in df_col_dict:
            data_dict[dest_col.name] = [None] * size
        else:
            src_type = df_type_to_odps_type(
                df_col_dict[dest_col.name].type, project=table.project
            )
            if src_type == dest_col.type:
                data_dict[dest_col.name] = frame.values[case_dict[dest_col.name]]
            elif dest_col.type.can_implicit_cast(src_type) or cast:
                new_np_type = df_type_to_np_type(odps_type_to_df_type(dest_col.type))
                data_dict[dest_col.name] = frame.values[case_dict[dest_col.name]].astype(new_np_type)
            else:
                raise CompileError('Column %s\'s type does not match, expect %s, got %s' % (
                    dest_col.name, src_type, dest_col.type))
    return pd.DataFrame(data_dict, columns=[c.name for c in table.table_schema.columns])


def _get_cached_writer(writer_cache, table, partition=None):
    writer_key = (table.name, partition)
    if writer_key not in writer_cache:
        if len(writer_cache) >= options.df.writer_count_limit:
            writer = writer_cache.popitem(last=False)
            writer.close()

        writer = writer_cache[writer_key] = table.open_writer(
            partition=partition
        )
    else:
        # manually put to end
        writer = writer_cache.pop(writer_key)
        writer_cache[writer_key] = writer
    return writer


def _write_table_no_partitions(
    frame, table, ui, writer_cache, created_partitions, cast=False, overwrite=True,
    partition=None, progress_proportion=1,
):
    def gen():
        df = _reorder_pd(frame, table, cast=cast)
        size = len(df)

        last_percent = 0
        for i, row in zip(itertools.count(), df.values):
            if i % 50 == 0:
                percent = float(i) / size * progress_proportion
                ui.inc(percent - last_percent)
                last_percent = percent

            yield table.new_record(_convert_pd_type(row, table))

        if last_percent < progress_proportion:
            ui.inc(progress_proportion - last_percent)

    if partition not in created_partitions:
        created_partitions.add(partition)
        if overwrite:
            if partition is None:
                table.truncate()
            else:
                table.delete_partition(partition, if_exists=True)
                table.create_partition(partition)

    _get_cached_writer(writer_cache, table, partition).write(gen())


def _write_table_with_partitions(
    frame, table, partitions, ui, writer_cache, created_partitions, cast=False,
    overwrite=True, progress_proportion=1,
):
    df = _reorder_pd(frame, table, cast=cast)
    vals_to_partitions = dict()
    for ps in df[partitions].drop_duplicates().values:
        p = ','.join('='.join([str(n), str(v)]) for n, v in zip(partitions, ps))
        if p not in created_partitions:
            if overwrite:
                table.delete_partition(p, if_exists=True)
            table.create_partition(p, if_not_exists=True)
            created_partitions.add(p)
        vals_to_partitions[tuple(ps)] = p

    size = len(df)
    curr = [0]
    last_percent = [0]
    for name, group in df.groupby(partitions):
        name = name if isinstance(name, tuple) else (name, )
        group = group[[it for it in group.columns.tolist() if it not in partitions]]

        def gen():
            for i, row in zip(itertools.count(), group.values):
                curr[0] += i
                if curr[0] % 50 == 0:
                    percent = float(curr[0]) / size * progress_proportion
                    ui.inc(percent - last_percent[0])
                    last_percent[0] = percent

                yield table.new_record(_convert_pd_type(row, table))

        _get_cached_writer(writer_cache, table, vals_to_partitions[name]).write(gen())

    if last_percent[0] < progress_proportion:
        ui.inc(progress_proportion - last_percent[0])


def write_table(frame, table, ui, cast=False, overwrite=True, partitions=None,
                partition=None, progress_proportion=1):
    from ..backends.frame import ResultFrame

    ui.status('Try to upload to ODPS with tunnel...')
    writers_cache = OrderedDict()
    created_partitions = set()
    for start in range(0, len(frame), options.tunnel.write_row_batch_size):
        subframe = frame[start: start + options.tunnel.write_row_batch_size]
        if not isinstance(subframe, ResultFrame):
            subframe = ResultFrame(subframe, schema=frame.schema)
        batch_proportion = 1.0 * progress_proportion * len(subframe) / len(frame)
        if partitions is None:
            _write_table_no_partitions(
                subframe, table, ui, writers_cache, created_partitions, cast=cast,
                overwrite=overwrite, partition=partition, progress_proportion=batch_proportion,
            )
        else:
            _write_table_with_partitions(
                subframe, table, partitions, ui, writers_cache, created_partitions,
                cast=cast, overwrite=overwrite, progress_proportion=batch_proportion,
            )
    for writer in writers_cache.values():
        writer.close()


def process_persist_kwargs(kw):
    kw.pop('partition', None)
    kw.pop('partitions', None)
    kw.pop('create_table', None)
    kw.pop('drop_table', None)
    kw.pop('create_partition', None)
    kw.pop('drop_partition', None)

    return kw