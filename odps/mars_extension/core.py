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

import hashlib
import json
import uuid
import logging
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from cupid.config import options as cupid_options

from ..df.backends.pd.types import _np_to_df_types
from ..df import types as pd_types
from ..df.backends.odpssql.types import df_type_to_odps_type
from ..models import Schema
from ..types import PartitionSpec
from ..utils import to_binary, write_log


logger = logging.getLogger(__name__)


def _check_internal(endpoint):
    from . import INTERNAL_PATTERN

    if INTERNAL_PATTERN and re.search(INTERNAL_PATTERN, endpoint) is not None:
        try:
            from .. import internal
        except ImportError:
            raise EnvironmentError('Please install internal version of PyODPS.')


def _get_mars_task_name(instance):
    from ..models.tasks import CupidTask

    for task in instance.tasks or []:
        if isinstance(task, CupidTask) and 'settings' in task.properties:
            try:
                hints = json.loads(task.properties['settings'])
            except json.JSONDecodeError:
                continue

            if hints.get('odps.cupid.application.type') == 'mars':
                return task.name


def list_mars_instances(odps, project=None, days=3, return_task_name=False):
    start_time = datetime.now() - timedelta(days=days)
    for instance in odps.list_instances(start_time=start_time, project=project,
                                        status='Running', only_owner=True):
        task_name = _get_mars_task_name(instance)
        if task_name is not None:
            if not return_task_name:
                yield instance
            else:
                yield task_name, instance


def create_mars_cluster(odps, worker_num=1, worker_cpu=8, worker_mem=32, cache_mem=None,
                        min_worker_num=None, disk_num=1, disk_size=100,
                        scheduler_num=1, scheduler_cpu=None, scheduler_mem=None,
                        web_num=1, web_cpu=None, web_mem=None, with_notebook=False,
                        notebook_cpu=None, notebook_mem=None, timeout=None,
                        extra_modules=None, resources=None, instance_id=None, name='default',
                        if_exists='reuse', project=None, **kw):
    """
    Create a Mars cluster and a Mars session as default session,
    then all tasks will be submitted to cluster.

    :param worker_num: mars cluster worker's number
    :param worker_cpu: number of cpu cores on each mars worker
    :param worker_mem: memory size on each mars worker
    :param cache_mem: cache memory size on each mars worker
    :param disk_num: number of mounted disk
    :param min_worker_num: return if cluster worker's number reach to min_worker
    :param resources: resources name
    :param extra_modules: user defined module path
    :param scheduler_num: the number of schedulers, default is 0
    :param with_notebook: whether launch jupyter notebook, defaullt is False
    :param instance_id: existing mars cluster's instance id
    :param name: cluster name, 'default' will be default name
    :param if_exists: 'reuse', 'raise' or 'ignore',
                      if 'reuse', will reuse the first created cluster with the same name,
                      if not created, create a new one;
                      if 'raise', will fail if cluster with same name created already;
                      if 'ignore', will always create a new cluster
    :param project: project name
    :return: class: `MarsClient`
    """
    from .deploy.client import MarsCupidClient

    if kw.get('proxy_endpoint', None) is not None:
        cupid_options.cupid.proxy_endpoint = kw['proxy_endpoint']

    if if_exists not in ('reuse', 'raise', 'ignore'):
        raise ValueError('`if_exists` should be "reuse", "raise", or "ignore"')

    task_name = 'MARS_TASK_{}'.format(hashlib.md5(to_binary(name)).hexdigest())

    _check_internal(odps.endpoint)
    if instance_id is not None:
        inst = odps.get_instance(instance_id, project=project)
        client = MarsCupidClient(odps, inst, project=project)
    elif if_exists in ('reuse', 'raise'):
        client = None

        # need to check the instances before
        for prev_task_name, prev_instance in list_mars_instances(odps, project=project, days=2,
                                                                 return_task_name=True):
            if prev_task_name == task_name:
                # found a instance with the same task name
                if if_exists == 'reuse':
                    write_log('Reusing existing Mars cluster({}), logview address: \n{}'.format(
                        name, prev_instance.get_logview_address()
                    ))
                    client = MarsCupidClient(odps, prev_instance, project=project)
                    break
                else:
                    assert if_exists == 'raise'
                    raise ValueError('Cluster("{}") exists'.format(name))

        if client is None:
            # not create before, just create a new one
            client = MarsCupidClient(odps, project=project)
    else:
        client = MarsCupidClient(odps, project=project)

    worker_mem = int(worker_mem * 1024 ** 3)
    cache_mem = int(cache_mem * 1024 ** 3) if cache_mem else None
    disk_size = int(disk_size * 1024 ** 3)
    scheduler_mem = int(scheduler_mem * 1024 ** 3) if scheduler_mem else None
    web_mem = int(web_mem * 1024 ** 3) if web_mem else None
    notebook_mem = int(notebook_mem * 1024 ** 3) if web_mem else None

    kw = dict(worker_num=worker_num, worker_cpu=worker_cpu, worker_mem=worker_mem,
              worker_cache_mem=cache_mem, min_worker_num=min_worker_num,
              worker_disk_num=disk_num, worker_disk_size=disk_size,
              scheduler_num=scheduler_num, scheduler_cpu=scheduler_cpu,
              scheduler_mem=scheduler_mem, web_num=web_num, web_cpu=web_cpu,
              web_mem=web_mem, with_notebook=with_notebook, notebook_cpu=notebook_cpu,
              notebook_mem=notebook_mem, timeout=timeout, extra_modules=extra_modules,
              resources=resources, task_name=task_name, **kw)
    kw = dict((k, v) for k, v in kw.items() if v is not None)
    return client.submit(**kw)


def to_mars_dataframe(odps, table_name, shape=None, partition=None, chunk_bytes=None,
                      sparse=False, columns=None, add_offset=False, calc_nrows=True,
                      use_arrow_dtype=False, cupid_internal_endpoint=None):
    """
    Read table to Mars DataFrame.

    :param table_name: table name
    :param shape: table shape. A tuple like (1000, 3) which means table count is 1000 and schema length is 3.
    :param partition: partition spec.
    :param chunk_bytes: Bytes to read for each chunk. Default value is '16M'.
    :param sparse: if read as sparse DataFrame.
    :param columns: selected columns.
    :param add_offset: if standardize the DataFrame's index to RangeIndex. False as default.
    :param calc_nrows: if calculate nrows if shape is not specified.
    :param use_arrow_dtype: read to arrow dtype. Reduce memory in some saces.
    :return: Mars DataFrame.
    """
    from cupid import context
    from .dataframe import read_odps_table
    from ..utils import init_progress_ui

    odps_params = dict(
        project=odps.project, endpoint=cupid_internal_endpoint or cupid_options.cupid.runtime.endpoint)

    data_src = odps.get_table(table_name)

    odps_schema = data_src.schema
    if len(odps_schema.partitions) != 0:
        if partition is None:
            raise TypeError('Partition should be specified.')

    for col in columns or []:
        if col not in odps_schema.names:
            raise TypeError("Specific column {} doesn't exist in table".format(col))

    # persist view table to a temp table
    if data_src.is_virtual_view:
        temp_table_name = table_name + '_temp_mars_table_' + str(uuid.uuid4()).replace('-', '_')
        odps.create_table(temp_table_name, schema=data_src.schema, stored_as='aliorc', lifecycle=1)
        data_src.to_df().persist(temp_table_name)
        table_name = temp_table_name
        data_src = odps.get_table(table_name)

    # get dataframe's shape
    if shape is None:
        if calc_nrows and context() is None:
            # obtain count
            if partition is None:
                odps_df = data_src.to_df()
            else:
                odps_df = data_src.get_partition(partition).to_df()
            nrows = odps_df.count().execute(use_tunnel=False, ui=init_progress_ui(mock=True))
        else:
            nrows = np.nan

        shape = (nrows, len(data_src.schema.simple_columns))

    return read_odps_table(odps.get_table(table_name), shape, partition=partition,
                           chunk_bytes=chunk_bytes, sparse=sparse, columns=columns,
                           odps_params=odps_params, add_offset=add_offset,
                           use_arrow_dtype=use_arrow_dtype)


def persist_mars_dataframe(odps, df, table_name, overwrite=False, partition=None, write_batch_size=None,
                           unknown_as_string=True, as_type=None, cupid_internal_endpoint=None):
    """
    Write Mars DataFrame to table.

    :param df: Mars DataFrame.
    :param table_name: table to write.
    :param overwrite: if overwrite the data. False as default.
    :param partition: partition spec.
    :param write_batch_size: batch size of records to write. 1024 as default.
    :param unknown_as_string: set the columns to string type if it's type is Object.
    :param as_type: specify column dtypes. {'a': 'string'} will set column `a` as string type.
    :return: None
    """
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

    if 'with_notebook' not in kw:
        kw['with_notebook'] = False
    task_name = kw.get('name', None)
    if task_name is None:
        kw['name'] = str(uuid.uuid4())
        kw['if_exists'] = 'ignore'

    cupid_internal_endpoint = kw.pop('cupid_internal_endpoint', None)
    client = odps.create_mars_cluster(**kw)
    try:
        r = spawn(func, args=args, kwargs=kwargs, retry_when_fail=retry_when_fail, n_output=n_output)
        r.op.extra_params['project'] = odps.project
        r.op.extra_params['endpoint'] = cupid_internal_endpoint or cupid_options.cupid.runtime.endpoint
        r.execute()
    finally:
        if task_name is None:
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
                if 'endpoint' in op.extra_params:
                    env['ODPS_ENDPOINT'] = str(op.extra_params['endpoint'])
                if ('project' in op.extra_params) and ('ODPS_PROJECT_NAME' not in env):
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
