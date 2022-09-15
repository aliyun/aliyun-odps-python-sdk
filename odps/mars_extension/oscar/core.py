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

import hashlib
import json
import logging
import os
import re
import uuid
import warnings
from datetime import datetime, timedelta

import numpy as np

from ...accounts import AliyunAccount
from ...config import options as odps_options
from ...models import Schema
from ...types import PartitionSpec
from ...utils import to_binary, write_log
from ..utils import (
    use_odps2_type,
    pd_type_to_odps_type,
    rewrite_partition_predicate,
    check_partition_exist,
)

logger = logging.getLogger(__name__)


def _check_internal(endpoint):
    from .. import INTERNAL_PATTERN

    if INTERNAL_PATTERN and re.search(INTERNAL_PATTERN, endpoint) is not None:
        try:
            from ... import internal  # noqa: F401
        except ImportError:
            raise EnvironmentError("Please install internal version of PyODPS.")


def _get_mars_task_name(instance):
    from ...models.tasks import CupidTask

    for task in instance.tasks or []:
        if isinstance(task, CupidTask) and "settings" in task.properties:
            try:
                hints = json.loads(task.properties["settings"])
            except json.JSONDecodeError:
                continue

            if hints.get("odps.cupid.application.type") == "mars":
                return task.name


def list_mars_instances(odps, project=None, days=3, return_task_name=False):
    """
    List all running mars instances in your project.

    :param project:  default project name
    :param days: the days range of filtered instances
    :param return_task_name: If return task name
    :return: Instances.
    """
    start_time = datetime.now() - timedelta(days=days)
    for instance in odps.list_instances(
        start_time=start_time, project=project, status="Running", only_owner=True
    ):
        task_name = _get_mars_task_name(instance)
        if task_name is not None:
            if not return_task_name:
                yield instance
            else:
                yield task_name, instance


def create_mars_cluster(
    odps,
    worker_num=1,
    worker_cpu=8,
    worker_mem=32,
    cache_mem=None,
    min_worker_num=None,
    disk_num=1,
    disk_size=100,
    supervisor_num=1,
    supervisor_cpu=None,
    supervisor_mem=None,
    with_notebook=False,
    notebook_cpu=None,
    notebook_mem=None,
    with_graphscope=False,
    coordinator_cpu=None,
    coordinator_mem=None,
    timeout=None,
    extra_modules=None,
    resources=None,
    instance_id=None,
    name="default",
    if_exists="reuse",
    project=None,
    **kw
):
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
    :param supervisor_num: the number of supervisors, default is 0
    :param with_notebook: whether launch jupyter notebook, default is False
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
    from cupid.config import options as cupid_options
    from .deploy.client import MarsCupidClient

    if kw.get("proxy_endpoint", None) is not None:
        cupid_options.cupid.proxy_endpoint = kw["proxy_endpoint"]

    if if_exists not in ("reuse", "raise", "ignore", "restart"):
        raise ValueError('`if_exists` should be "reuse", "raise, "ignore" or "restart"')

    if min_worker_num is not None and min_worker_num > worker_num:
        raise ValueError("`min_worker` cannot greater than `worker_num`")

    task_name = "MARS_TASK_{}".format(hashlib.md5(to_binary(name)).hexdigest())

    _check_internal(odps.endpoint)
    if instance_id is not None:
        inst = odps.get_instance(instance_id, project=project)
        client = MarsCupidClient(odps, inst, project=project)
    elif if_exists in ("reuse", "raise", "restart"):
        client = None

        # need to check the instances before
        for prev_task_name, prev_instance in list_mars_instances(
            odps, project=project, days=2, return_task_name=True
        ):
            if prev_task_name == task_name:
                # found a instance with the same task name
                if if_exists in ("reuse", "restart"):
                    if if_exists == "reuse":
                        write_log(
                            "Reusing existing Mars cluster({}), logview address: \n{}".format(
                                name, prev_instance.get_logview_address()
                            )
                        )
                    client = MarsCupidClient(odps, prev_instance, project=project)
                    break
                else:
                    assert if_exists == "raise"
                    raise ValueError('Cluster("{}") exists'.format(name))

        if if_exists == "restart" and client is not None:
            # if exists Mars cluster, stop it first
            client.stop_server()
            client = None

        if client is None:
            # not create before, just create a new one
            client = MarsCupidClient(odps, project=project)
    else:
        client = MarsCupidClient(odps, project=project)

    worker_mem = int(worker_mem * 1024**3)
    cache_mem = int(cache_mem * 1024**3) if cache_mem else None
    disk_size = int(disk_size * 1024**3)
    supervisor_mem = int(supervisor_mem * 1024**3) if supervisor_mem else None
    notebook_mem = int(notebook_mem * 1024**3) if notebook_mem else None
    coordinator_mem = int(coordinator_mem * 1024**3) if coordinator_mem else None

    keys_to_pop = [k for k, v in kw.items() if k.startswith("supervisor") and v is None]
    for key in keys_to_pop:
        kw.pop(key)

    kw = dict(
        worker_num=worker_num,
        worker_cpu=worker_cpu,
        worker_mem=worker_mem,
        worker_cache_mem=cache_mem,
        min_worker_num=min_worker_num,
        worker_disk_num=disk_num,
        worker_disk_size=disk_size,
        supervisor_num=supervisor_num,
        supervisor_cpu=supervisor_cpu,
        supervisor_mem=supervisor_mem,
        with_notebook=with_notebook,
        notebook_cpu=notebook_cpu,
        notebook_mem=notebook_mem,
        with_graphscope=with_graphscope,
        coordinator_cpu=coordinator_cpu,
        coordinator_mem=coordinator_mem,
        timeout=timeout,
        extra_modules=extra_modules,
        resources=resources,
        task_name=task_name,
        **kw
    )
    kw = dict((k, v) for k, v in kw.items() if v is not None)
    return client.submit(**kw)


def _get_table_record_count(odps, table_name, partition=None):
    from ...utils import init_progress_ui

    data_src = odps.get_table(table_name)
    if partition is not None:
        if check_partition_exist(data_src, partition):
            data_src = data_src.get_partition(partition)
        else:
            part_cols = [pt.name for pt in data_src.schema.partitions]
            predicate = rewrite_partition_predicate(partition, part_cols)
            odps_df = data_src.to_df().query(predicate)
            return odps_df.count().execute(
                use_tunnel=False, ui=init_progress_ui(mock=True)
            )

    # check if record_num is valid
    if getattr(data_src, "record_num", None) and data_src.record_num > 0:
        return data_src.record_num

    # check if size from table tunnel is valid
    try:
        with data_src.open_reader(timeout=5) as reader:
            return reader.count
    except:
        pass

    # obtain count
    odps_df = data_src.to_df()
    return odps_df.count().execute(use_tunnel=False, ui=init_progress_ui(mock=True))


def to_mars_dataframe(
    odps,
    table_name,
    shape=None,
    partition=None,
    chunk_bytes=None,
    sparse=False,
    columns=None,
    add_offset=None,
    calc_nrows=True,
    index_type="chunk_incremental",
    use_arrow_dtype=False,
    string_as_binary=None,
    chunk_size=None,
    memory_scale=None,
    runtime_endpoint=None,
    append_partitions=False,
    with_split_meta_on_tile=False,
    extra_params=None,
    **kw
):
    """
    Read table to Mars DataFrame.

    :param table_name: table name
    :param shape: table shape. A tuple like (1000, 3) which means table count is 1000 and schema length is 3.
    :param partition: partition spec.
    :param chunk_bytes: Bytes to read for each chunk. Default value is '16M'.
    :param chunk_size: Desired chunk size on rows.
    :param sparse: if read as sparse DataFrame.
    :param columns: selected columns.
    :param add_offset: if standardize the DataFrame's index to RangeIndex. False as default.
    :param index_type: type of retrieved index
    :param calc_nrows: if calculate nrows if shape is not specified.
    :param use_arrow_dtype: read to arrow dtype. Reduce memory in some saces.
    :param string_as_binary: read string columns as binary type.
    :param memory_scale: Scale that real memory occupation divided with raw file size.
    :param append_partitions: append partition name when reading partitioned tables.
    :return: Mars DataFrame.
    """
    from cupid.config import options as cupid_options
    from .dataframe import read_odps_table

    runtime_endpoint = (
        runtime_endpoint
        or kw.pop("cupid_internal_endpoint", None)
        or cupid_options.cupid.runtime.endpoint
        or odps.endpoint
    )
    odps_params = dict(project=odps.project, endpoint=runtime_endpoint)

    if isinstance(odps.account, AliyunAccount):
        odps_params.update(
            dict(
                access_id=odps.account.access_id,
                secret_access_key=odps.account.secret_access_key,
            )
        )

    data_src = odps.get_table(table_name)

    cols = (
        data_src.schema.columns if append_partitions else data_src.schema.simple_columns
    )
    col_names = set(c.name for c in cols)

    for col in columns or []:
        if col not in col_names:
            raise TypeError("Specific column {} doesn't exist in table".format(col))

    # persist view table to a temp table
    if data_src.is_virtual_view:
        temp_table_name = (
            data_src.name + "_temp_mars_table_" + str(uuid.uuid4()).replace("-", "_")
        )
        odps.create_table(
            temp_table_name, schema=data_src.schema, stored_as="aliorc", lifecycle=1
        )
        data_src.to_df().persist(temp_table_name)
        table_name = temp_table_name
        data_src = odps.get_table(table_name)

    # get dataframe's shape
    if shape is None:
        if calc_nrows and "CUPID_SERVICE_SOCKET" not in os.environ:
            # obtain count
            nrows = _get_table_record_count(odps, table_name, partition=partition)
        else:
            nrows = np.nan

        if append_partitions:
            shape = (nrows, len(data_src.schema))
        else:
            shape = (nrows, len(data_src.schema.simple_columns))

    memory_scale = memory_scale or odps_options.mars.to_dataframe_memory_scale
    return read_odps_table(
        odps.get_table(table_name),
        shape,
        partition=partition,
        chunk_bytes=chunk_bytes,
        sparse=sparse,
        columns=columns,
        odps_params=odps_params,
        add_offset=add_offset,
        index_type=index_type,
        chunk_size=chunk_size,
        use_arrow_dtype=use_arrow_dtype,
        string_as_binary=string_as_binary,
        memory_scale=memory_scale,
        append_partitions=append_partitions,
        with_split_meta_on_tile=with_split_meta_on_tile,
        extra_params=extra_params,
    )


@use_odps2_type
def persist_mars_dataframe(
    odps,
    df,
    table_name,
    overwrite=False,
    partition=None,
    write_batch_size=None,
    unknown_as_string=None,
    as_type=None,
    drop_table=False,
    create_table=True,
    drop_partition=False,
    create_partition=None,
    lifecycle=None,
    runtime_endpoint=None,
    **kw
):
    """
    Write Mars DataFrame to table.

    :param df: Mars DataFrame.
    :param table_name: table to write.
    :param overwrite: if overwrite the data. False as default.
    :param partition: partition spec.
    :param write_batch_size: batch size of records to write. 1024 as default.
    :param unknown_as_string: set the columns to string type if it's type is Object.
    :param as_type: specify column dtypes. {'a': 'string'} will set column `a` as string type.
    :param drop_table: drop table if exists, False as default
    :param create_table: create table first if not exits, True as default
    :param drop_partition: drop partition if exists, False as default
    :param create_partition: create partition if not exists, None as default
    :param lifecycle: table lifecycle. If absent, `options.lifecycle` will be used.

    :return: None
    """
    import pandas as pd
    from cupid.config import options as cupid_options
    from .dataframe import write_odps_table

    dtypes = df.dtypes
    odps_types = []
    names = []
    for name, t in zip(dtypes.keys(), list(dtypes.values)):
        names.append(name)
        if as_type and name in as_type:
            odps_types.append(as_type[name])
        else:
            odps_types.append(
                pd_type_to_odps_type(
                    t, name, unknown_as_string=unknown_as_string, project=odps.get_project()
                )
            )
    if partition:
        p = PartitionSpec(partition)
        schema = Schema.from_lists(names, odps_types, p.keys, ["string"] * len(p))
    else:
        schema = Schema.from_lists(names, odps_types)

    if drop_table:
        odps.delete_table(table_name, if_exists=True)

    if partition is None:
        # the non-partitioned table
        if drop_partition:
            raise ValueError("Cannot drop partition for non-partition table")
        if create_partition:
            raise ValueError("Cannot create partition for non-partition table")

        if create_table or (not odps.exist_table(table_name)):
            odps.create_table(
                table_name,
                schema,
                if_not_exists=True,
                stored_as="aliorc",
                lifecycle=lifecycle,
            )
    else:
        if odps.exist_table(table_name) or not create_table:
            t = odps.get_table(table_name)
            table_partition = t.get_partition(partition)
            if drop_partition:
                t.delete_partition(table_partition, if_exists=True)
            if create_partition:
                t.create_partition(table_partition, if_not_exists=True)

        else:
            odps.create_table(
                table_name, schema, stored_as="aliorc", lifecycle=lifecycle
            )

    table = odps.get_table(table_name)

    if len(table.schema.simple_columns) != len(schema.simple_columns):
        raise TypeError(
            "Table column number is %s while input DataFrame has %s columns"
            % (len(table.schema.simple_columns), len(schema.simple_columns))
        )

    for c_left, c_right in zip(table.schema.simple_columns, schema.simple_columns):
        if c_left.name.lower() != c_right.name.lower() or c_left.type != c_right.type:
            raise TypeError(
                "Column type between provided DataFrame and target table"
                " does not agree with each other. DataFrame column %s type is %s,"
                "target table column %s type is %s"
                % (c_right.name, c_right.type, c_left.name, c_left.type)
            )
    if partition:
        table.create_partition(partition, if_not_exists=True)
    runtime_endpoint = (
        runtime_endpoint
        or kw.pop("cupid_internal_endpoint", None)
        or cupid_options.cupid.runtime.endpoint
        or odps.endpoint
    )
    odps_params = dict(project=odps.project, endpoint=runtime_endpoint)
    if isinstance(odps.account, AliyunAccount):
        odps_params.update(
            dict(
                access_id=odps.account.access_id,
                secret_access_key=odps.account.secret_access_key,
            )
        )
    if isinstance(df, pd.DataFrame):
        if "CUPID_SERVICE_SOCKET" not in os.environ:
            raise ValueError("Doesn't support persist pandas dataframe")
        warnings.warn("Persist pandas dataframe directly deprcated", DeprecationWarning)
        _write_table_in_cupid(odps, df, table, partition, overwrite)
    else:
        write_odps_table(
            df,
            table,
            partition=partition,
            overwrite=overwrite,
            odps_params=odps_params,
            unknown_as_string=unknown_as_string,
            write_batch_size=write_batch_size,
        ).execute()


def run_script_in_mars(odps, script, mode="exec", n_workers=1, command_argv=None, **kw):
    from cupid.config import options as cupid_options
    from .run_script import run_script

    runtime_endpoint = kw.pop("runtime_endpoint", None) or kw.pop(
        "cupid_internal_endpoint", None
    )
    odps_params = dict(
        project=odps.project,
        endpoint=runtime_endpoint or cupid_options.cupid.runtime.endpoint or odps.endpoint,
    )
    run_script(
        script,
        mode=mode,
        n_workers=n_workers,
        command_argv=command_argv,
        odps_params=odps_params,
        **kw
    )


def run_mars_job(
    odps, func, args=(), kwargs=None, retry_when_fail=False, n_output=None, **kw
):
    from mars.remote import spawn
    from cupid.config import options as cupid_options

    if "with_notebook" not in kw:
        kw["with_notebook"] = False
    task_name = kw.get("name", None)
    if task_name is None:
        kw["name"] = str(uuid.uuid4())
        kw["if_exists"] = "ignore"

    runtime_endpoint = kw.pop("runtime_endpoint", None) or kw.pop(
        "cupid_internal_endpoint", None
    )
    client = odps.create_mars_cluster(**kw)
    try:
        r = spawn(
            func,
            args=args,
            kwargs=kwargs,
            retry_when_fail=retry_when_fail,
            n_output=n_output,
        )
        r.op.extra_params["project"] = odps.project
        r.op.extra_params["endpoint"] = (
            runtime_endpoint or cupid_options.cupid.runtime.endpoint or odps.endpoint
        )
        r.execute()
    finally:
        if task_name is None:
            client.stop_server()


def sql_to_mars_dataframe(odps, sql, lifecycle=None, **to_mars_df_params):
    from cupid.config import options as cupid_options

    lifecycle = lifecycle or cupid_options.temp_lifecycle
    tmp_table_name = "%s%s" % ("tmp_mars", str(uuid.uuid4()).replace("-", "_"))

    lifecycle_str = "LIFECYCLE {0} ".format(lifecycle) if lifecycle is not None else ""

    format_sql = lambda s: "CREATE TABLE {0} {1}AS \n{2}".format(
        tmp_table_name, lifecycle_str, s
    )

    create_table_sql = format_sql(sql)
    odps.execute_sql(create_table_sql)
    return to_mars_dataframe(odps, tmp_table_name, **to_mars_df_params)


def _write_table_in_cupid(odps, df, table, partition=None, overwrite=True):
    from .cupid_service import CupidServiceClient

    cupid_client = CupidServiceClient()
    odps_params = dict(project=odps.project, endpoint=odps.endpoint)
    table_name = "%s.%s" % (table.project.name, table.name)
    logger.debug("Start creating upload session from cupid.")
    upload_handle = cupid_client.create_table_upload_session(odps_params, table_name)

    writer_config = dict(
        _table_name=table.name,
        _project_name=table.project.name,
        _table_schema=table.schema,
        _partition_spec=partition,
        _block_id="0",
        _handle=upload_handle,
    )
    cupid_client.write_table_data(writer_config, df)

    cupid_client.commit_table_upload_session(
        odps_params, table_name, upload_handle, {"0": partition}, overwrite
    )
