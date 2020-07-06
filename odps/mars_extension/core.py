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

import re
import os
import logging

import numpy as np
import pandas as pd
from cupid.config import options as cupid_options

from ..df.backends.pd.types import _np_to_df_types
from ..df import types as pd_types
from ..df.backends.odpssql.types import df_type_to_odps_type
from ..models import Schema
from ..types import PartitionSpec


logger = logging.getLogger('mars.worker')


def _check_internal(endpoint):
    from . import INTERNAL_PATTERN

    if INTERNAL_PATTERN and re.search(INTERNAL_PATTERN, endpoint) is not None:
        try:
            from .. import internal
        except ImportError:
            raise EnvironmentError('Please install internal version of PyODPS.')


def create_mars_cluster(odps, worker_num=1, worker_cpu=8, worker_mem=32, cache_mem=None, disk_num=1,
                        min_worker_num=None, resources=None, module_path=None, scheduler_num=1,
                        notebook=None, instance_id=None, **kw):
    """
    :param worker_num: mars cluster worker's number
    :param worker_cpu: number of cpu cores on each mars worker
    :param worker_mem: memory size on each mars worker
    :param cache_mem: cache memory size on each mars worker
    :param disk_num: number of mounted disk
    :param min_worker_num: return if cluster worker's number reach to min_worker
    :param resources: resources name
    :param module_path: user defined module path
    :param scheduler_num: the number of schedulers, default is 0
    :param notebook: whether launch jupyter notebook, defaullt is False
    :param instance_id: existing mars cluster's instance id
    :return: class: `MarsClient`
    """
    from .deploy.client import MarsCupidClient

    _check_internal(odps.endpoint)
    if instance_id is not None:
        inst = odps.get_instance(instance_id)
        client = MarsCupidClient(odps, inst)
    else:
        client = MarsCupidClient(odps)
    return client.submit(worker_num, worker_cpu, worker_mem, disk_num=disk_num, min_worker_num=min_worker_num,
                         cache_mem=cache_mem, resources=resources, module_path=module_path, create_session=True,
                         scheduler_num=scheduler_num, notebook=notebook, **kw)


def to_mars_dataframe(odps, table_name, shape=None, partition=None, chunk_bytes=None,
                      sparse=False, columns=None, add_offset=False, cupid_internal_endpoint=None):
    from .dataframe import read_odps_table

    odps_params = dict(
        project=odps.project, endpoint=cupid_internal_endpoint or cupid_options.cupid.runtime.endpoint)

    data_src = odps.get_table(table_name)

    # get dataframe's shape
    if shape is None:
        shape = (np.nan, len(data_src.schema.simple_columns))

    return read_odps_table(odps.get_table(table_name), shape, partition=partition,
                           chunk_bytes=chunk_bytes, sparse=sparse, columns=columns,
                           odps_params=odps_params, add_offset=add_offset)


def persist_mars_dataframe(odps, df, table_name, overwrite=False, partition=None, write_batch_size=None,
                           unknown_as_string=True, as_type=None, cupid_internal_endpoint=None):
    from .dataframe import write_odps_table

    dtypes = df.dtypes
    odps_types = []
    names = []
    for name, t in zip(dtypes.keys(), list(dtypes.values)):
        names.append(name)
        if as_type and name in as_type:
            odps_types.append(as_type[name])
        else:
            if t in _np_to_df_types:
                df_type = _np_to_df_types[t]
            elif unknown_as_string:
                df_type = pd_types.string
            else:
                raise ValueError('Unknown dtype: {}'.format(t))
            odps_types.append(df_type_to_odps_type(df_type))
    if partition:
        p = PartitionSpec(partition)
        schema = Schema.from_lists(names, odps_types, p.keys, ['string'] * len(p))
    else:
        schema = Schema.from_lists(names, odps_types)
    odps.create_table(table_name, schema, if_not_exists=True, stored_as='aliorc')
    table = odps.get_table(table_name)
    if partition:
        table.create_partition(partition, if_not_exists=True)
    odps_params = dict(project=odps.project,
                       endpoint=cupid_internal_endpoint or cupid_options.cupid.runtime.endpoint)
    if isinstance(df, pd.DataFrame):
        _write_table_in_cupid(odps, df, table, partition=partition, overwrite=overwrite)
    else:
        write_odps_table(df, table, partition=partition, overwrite=overwrite, odps_params=odps_params,
                         write_batch_size=write_batch_size).execute()


def run_script_in_mars(odps, script, mode='exec', n_workers=1, command_argv=None, **kw):
    from .run_script import run_script

    cupid_internal_endpoint = kw.pop('cupid_internal_endpoint', None)
    odps_params = dict(project=odps.project,
                       endpoint=cupid_internal_endpoint or cupid_options.cupid.runtime.endpoint)
    run_script(script, mode=mode, n_workers=n_workers,
               command_argv=command_argv, odps_params=odps_params, **kw)


def run_mars_job(odps, func, args=(), kwargs=None, retry_when_fail=False, n_output=None, **kw):
    from mars.remote import spawn

    if 'notebook' not in kw:
        kw['notebook'] = False
    cupid_internal_endpoint = kw.pop('cupid_internal_endpoint', None)
    client = odps.create_mars_cluster(**kw)
    try:
        r = spawn(func, args=args, kwargs=kwargs, retry_when_fail=retry_when_fail, n_output=n_output)
        r.op.extra_params['project'] = odps.project
        r.op.extra_params['endpoint'] = cupid_internal_endpoint or cupid_options.cupid.runtime.endpoint
        r.execute()
    finally:
        client.stop_server()


def execute_with_odps_context(f):
    def wrapper(ctx, op):
        from cupid import context
        from mars.utils import to_str
        old_envs = os.environ.copy()
        try:
            if context() is None:
                logger.debug('Not in ODPS environment.')
                f(ctx, op)
            else:
                env = os.environ

                logger.debug('Get bearer token from cupid.')
                bearer_token = context().get_bearer_token()
                env['ODPS_BEARER_TOKEN'] = to_str(bearer_token)
                env['ODPS_ENDPOINT'] = str(op.extra_params['endpoint'])
                if 'ODPS_PROJECT_NAME' not in env:
                    env['ODPS_PROJECT_NAME'] = str(op.extra_params['project'])
                f(ctx, op)
                for out in op.outputs:
                    if ctx[out.key] is None:
                        ctx[out.key] = {'status': 'OK'}
        finally:
            os.environ = old_envs
    return wrapper


def _write_table_in_cupid(odps, df, table, partition=None, overwrite=True):
    import pyarrow as pa
    from mars.utils import to_str
    from cupid import CupidSession
    from cupid.io.table.core import BlockWriter

    cupid_session = CupidSession(odps)
    logger.debug('Start creating upload session from cupid.')
    upload_session = cupid_session.create_upload_session(table)
    block_writer = BlockWriter(
        _table_name=table.name,
        _project_name=table.project.name,
        _table_schema=table.schema,
        _partition_spec=partition,
        _block_id='0',
        _handle=to_str(upload_session.handle)
    )
    logger.debug('Start writing table block, block id: 0')
    with block_writer.open_arrow_writer() as cupid_writer:
        sink = pa.BufferOutputStream()

        batch_size = 1024
        schema = pa.RecordBatch.from_pandas(df[:1], preserve_index=False).schema
        arrow_writer = pa.RecordBatchStreamWriter(sink, schema)
        batch_idx = 0
        batch_data = df[batch_size * batch_idx: batch_size * (batch_idx + 1)]
        while len(batch_data) > 0:
            batch = pa.RecordBatch.from_pandas(batch_data, preserve_index=False)
            arrow_writer.write_batch(batch)
            batch_idx += 1
            batch_data = df[batch_size * batch_idx: batch_size * (batch_idx + 1)]
        arrow_writer.close()
        cupid_writer.write(sink.getvalue())
    block_writer.commit()

    upload_session._blocks = {'0': partition}
    upload_session.commit(overwrite=overwrite)
