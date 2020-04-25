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

import numpy as np

from ..df.backends.pd.types import _np_to_df_types
from ..df import types as pd_types
from ..df.backends.odpssql.types import df_type_to_odps_type
from ..models import Schema
from ..types import PartitionSpec


def create_mars_cluster(odps, worker_num=1, worker_cpu=8, worker_mem=32, cache_mem=None, disk_num=1,
                        min_worker_num=None,
                        resources=None, module_path=None, scheduler_num=1, instance_id=None, **kw):
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
    :param instance_id: existing mars cluster's instance id
    :return: class: `MarsClient`
    """
    from .deploy.client import MarsCupidClient

    if instance_id is not None:
        inst = odps.get_instance(instance_id)
        client = MarsCupidClient(odps, inst)
    else:
        client = MarsCupidClient(odps)
    return client.submit(worker_num, worker_cpu, worker_mem, disk_num=disk_num, min_worker_num=min_worker_num,
                         cache_mem=cache_mem, resources=resources, module_path=module_path, create_session=True,
                         scheduler_num=scheduler_num, **kw)


def to_mars_dataframe(odps, table_name, shape=None, partition=None, chunk_store_limit=None,
                      sparse=False, add_offset=True, cupid_internal_endpoint=None):
    from .dataframe import read_odps_table

    odps_params = dict(project=odps.project, endpoint=cupid_internal_endpoint or CUPID_INTERNAL_ENDPOINT)

    data_src = odps.get_table(table_name)

    # get dataframe's shape
    if shape is None:
        shape = (np.nan, len(data_src.schema.simple_columns))

    return read_odps_table(odps.get_table(table_name), shape, partition=partition,
                           chunk_store_limit=chunk_store_limit, sparse=sparse,
                           odps_params=odps_params, add_offset=add_offset)


def persist_mars_dataframe(odps, mdf, table_name, overwrite=False, partition=None, write_batch_size=None,
                           unknown_as_string=False, as_type=None, cupid_internal_endpoint=None):
    from .dataframe import write_odps_table

    dtypes = mdf.dtypes
    odps_types = []
    names = []
    for name, t in zip(dtypes.keys(), list(dtypes.values)):
        names.append(name)
        if as_type and name in as_type:
            odps_types.append(as_type[name])
            continue
        if t in _np_to_df_types:
            df_type = _np_to_df_types[t]
        elif unknown_as_string:
            df_type = pd_types.string
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
    odps_params = dict(project=odps.project, endpoint=cupid_internal_endpoint or CUPID_INTERNAL_ENDPOINT)
    return write_odps_table(mdf, table, partition=partition, overwrite=overwrite, odps_params=odps_params,
                            write_batch_size=write_batch_size)


try:
    from ..internal.core import CUPID_INTERNAL_ENDPOINT
except ImportError:
    CUPID_INTERNAL_ENDPOINT = 'http://service.cn.maxcompute.aliyun-inc.com/api'
