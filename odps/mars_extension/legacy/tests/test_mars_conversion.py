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

from itertools import product

import pytest

from ....df import DataFrame
from ....tests.core import tn

try:
    from ..oss import to_mars_tensor_via_oss
except ImportError:
    to_mars_tensor_via_oss = None
    pytestmark = pytest.mark.skip("mars not installed")


def test_sparse_without_shape():
    import pandas as pd

    pdf = pd.DataFrame([[1, 2, 3.1]], columns=["i", "j", "v"])
    df = DataFrame(pdf)
    with pytest.raises(ValueError):
        df.to_mars_tensor_via_oss(["i", "j"], "v", 15, sparse=True, oss_path="fake")


def test_sparse_vector_to_mars(odps, config):
    import pandas as pd
    import numpy as np

    shape = (50,)

    data = np.random.rand(*shape)
    kv = [(i, data[i]) for i in range(shape[0])]
    pdf = pd.DataFrame(kv, columns=["i", "v"])
    df = DataFrame(pdf).persist(
        tn("test_vector_to_mars"), lifecycle=1, odps=odps
    )

    (
        oss_access_id,
        oss_secret_access_key,
        oss_bucket_name,
        oss_endpoint,
    ) = config.oss_config

    t = df.to_mars_tensor_via_oss(
        ["i"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_vector_to_mars"),
        shape=shape,
        sparse=True,
    )

    table_name = tn("test_vector_to_mars_store")
    odps.delete_table(table_name, if_exists=True)
    odps.persist_tensor_via_oss(
        t,
        table_name,
        ["x"],
        "y",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=table_name,
    )

    with odps.get_table(table_name).open_reader() as reader:
        result = sorted([(r["x"], r["y"]) for r in reader], key=lambda x: x[0])
        assert kv == result


def test_no_partition_to_mars(odps, config):
    import numpy as np
    import pandas as pd

    shape = (100, 50)

    data = np.random.rand(*shape)
    kv = [(i, j, data[i, j]) for i, j in product(*[range(s) for s in shape])]
    pdf = pd.DataFrame(kv, columns=["i", "j", "v"])
    df = DataFrame(pdf).persist(
        tn("test_no_partition_dense_to_mars"), lifecycle=1, odps=odps
    )

    (
        oss_access_id,
        oss_secret_access_key,
        oss_bucket_name,
        oss_endpoint,
    ) = config.oss_config

    # test dense
    t = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        shape=shape,
    )

    # test if oss file exist
    t = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        oss_file_exist=True,
    )

    table_name = tn("test_no_partition_dense_to_mars_store")
    odps.delete_table(table_name, if_exists=True)
    odps.persist_tensor_via_oss(
        t,
        table_name,
        ["x", "y"],
        "z",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=table_name,
    )

    with odps.get_table(table_name).open_reader() as reader:
        result = sorted(
            [(r["x"], r["y"], r["z"]) for r in reader], key=lambda x: (x[0], x[1])
        )
        assert kv == result

    # test dense without setting chunks
    t = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        shape=shape,
    )
    table_name = tn("test_no_partition_dense_to_mars_store")
    odps.delete_table(table_name, if_exists=True)
    odps.persist_tensor_via_oss(
        t,
        table_name,
        ["x", "y"],
        "z",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=table_name,
    )

    with odps.get_table(table_name).open_reader() as reader:
        result = sorted(
            [(r["x"], r["y"], r["z"]) for r in reader], key=lambda x: (x[0], x[1])
        )
        assert kv == result

    # test sparse
    t = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        shape=shape,
        sparse=True,
    )
    assert t.issparse() is True
    table_name = tn("test_no_partition_dense_to_mars_store")
    odps.delete_table(table_name, if_exists=True)
    odps.persist_tensor_via_oss(
        t,
        table_name,
        ["x", "y"],
        "z",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=table_name,
    )

    with odps.get_table(table_name).open_reader() as reader:
        result = sorted(
            [(r["x"], r["y"], r["z"]) for r in reader], key=lambda x: (x[0], x[1])
        )
        assert kv == result


def test_no_partition_preprocess_to_mars(odps, config):
    import numpy as np
    import pandas as pd

    shape = (10, 5)

    data = np.random.rand(*shape)
    kv = [(i, j, data[i, j]) for i, j in product(*[range(s) for s in shape])]
    pdf = pd.DataFrame(kv, columns=["i", "j", "v"])
    df = DataFrame(pdf).persist(
        tn("test_no_partition_dense_to_mars"), lifecycle=1, odps=odps
    )

    (
        oss_access_id,
        oss_secret_access_key,
        oss_bucket_name,
        oss_endpoint,
    ) = config.oss_config

    # test preprocess
    preprocess_table_name = tn("test_no_partition_dense_to_mars_preprocess")
    odps.delete_table(preprocess_table_name, if_exists=True)
    t = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        discontinuous_columns=["j"],
        new_table_name=preprocess_table_name,
    )

    # test oss file exist
    t = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        discontinuous_columns=["j"],
        new_table_name=preprocess_table_name,
        oss_file_exist=True,
    )

    table_name = tn("test_no_partition_dense_to_mars_store")
    odps.delete_table(table_name, if_exists=True)
    odps.persist_tensor_via_oss(
        t,
        table_name,
        ["x", "y"],
        "z",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=table_name,
    )

    with odps.get_table(table_name).open_reader() as reader:
        result = sorted(
            [(r["x"], r["y"], r["z"]) for r in reader], key=lambda x: (x[0], x[1])
        )
        assert kv == result


def test_partitions_to_mars(odps, config):
    import numpy as np
    import pandas as pd

    source_table_name = tn("test_partition_dense_to_mars")
    df = odps.create_table(
        source_table_name,
        ("i int, j int, v double", "g string"),
        if_not_exists=True,
    ).to_df()

    shape = (20, 10)
    kvs = []
    for g in (1, 2):
        data = np.random.rand(*shape)
        kv = [(i, j, data[i, j]) for i, j in product(*[range(s) for s in shape])]
        kvs.append(kv)
        pdf = pd.DataFrame(kv, columns=["i", "j", "v"])
        DataFrame(pdf).persist(
            source_table_name, lifecycle=1, partition="g=%s" % g, odps=odps
        )

    (
        oss_access_id,
        oss_secret_access_key,
        oss_bucket_name,
        oss_endpoint,
    ) = config.oss_config

    # test dense
    td = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        partitions=["g"],
        shape={"g=1": shape, "g=2": shape},
    )

    # test oss file exist
    td = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        partitions=["g"],
        oss_file_exist=True,
    )

    out_table_name = "test_partition_dense_to_mars_store"
    odps.delete_table(out_table_name, if_exists=True)
    odps.persist_tensor_via_oss(
        td,
        out_table_name,
        ["x", "y"],
        "z",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=out_table_name,
    )

    for partition, kv in zip(td.keys(), kvs):
        with odps.get_table(out_table_name).get_partition(
            partition
        ).open_reader() as reader:
            result = sorted(
                [(r["x"], r["y"], r["z"]) for r in reader],
                key=lambda x: (x[0], x[1]),
            )
            assert kv == result

    # test sparse
    td = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        partitions=["g"],
        shape={"g=1": shape, "g=2": shape},
        sparse=True,
    )
    ts = list(td.values())
    for t in ts:
        assert t.issparse() is True

    out_table_name = "test_partition_dense_to_mars_store"
    odps.delete_table(out_table_name, if_exists=True)
    odps.persist_tensor_via_oss(
        td,
        out_table_name,
        ["x", "y"],
        "z",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=out_table_name,
    )

    for partition, kv in zip(td.keys(), kvs):
        with odps.get_table(out_table_name).get_partition(
            partition
        ).open_reader() as reader:
            result = sorted(
                [(r["x"], r["y"], r["z"]) for r in reader],
                key=lambda x: (x[0], x[1]),
            )
            assert kv == result


def test_partitions_preprocess_to_mars(odps, config):
    import numpy as np
    import pandas as pd

    source_table_name = tn("test_partition_dense_to_mars")
    df = odps.create_table(
        source_table_name,
        ("i int, j int, v double", "g string"),
        if_not_exists=True,
    ).to_df()

    shape = (20, 10)
    kvs = []
    for g in (1, 2):
        data = np.random.rand(*shape)
        kv = [(i, j, data[i, j]) for i, j in product(*[range(s) for s in shape])]
        kvs.append(kv)
        pdf = pd.DataFrame(kv, columns=["i", "j", "v"])
        DataFrame(pdf).persist(
            source_table_name, lifecycle=1, partition="g=%s" % g, odps=odps
        )

    (
        oss_access_id,
        oss_secret_access_key,
        oss_bucket_name,
        oss_endpoint,
    ) = config.oss_config

    preprocess_table_name = tn("test_partition_dense_to_mars_preprocess")
    odps.delete_table(preprocess_table_name, if_exists=True)
    td = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        partitions=["g"],
        discontinuous_columns=["i"],
        new_table_name=preprocess_table_name,
    )

    # test oss file exist
    td = df.to_mars_tensor_via_oss(
        ["i", "j"],
        "v",
        15,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=tn("test_no_partition_dense_to_mars_oss"),
        partitions=["g"],
        discontinuous_columns=["i"],
        new_table_name=preprocess_table_name,
        oss_file_exist=True,
    )

    out_table_name = "test_partition_dense_to_mars_store"
    odps.persist_tensor_via_oss(
        td,
        out_table_name,
        ["x", "y"],
        "z",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=out_table_name,
    )

    for partition, kv in zip(td.keys(), kvs):
        with odps.get_table(out_table_name).get_partition(
            partition
        ).open_reader() as reader:
            result = sorted(
                [(r["x"], r["y"], r["z"]) for r in reader],
                key=lambda x: (x[0], x[1]),
            )
            assert kv == result


def test_persist_to_tensor(odps, config):
    import mars.tensor as mt
    import numpy as np

    t = mt.ones((10, 10), chunk_size=6)

    (
        oss_access_id,
        oss_secret_access_key,
        oss_bucket_name,
        oss_endpoint,
    ) = config.oss_config

    table_name = "test_persist_to_tensor"
    odps.delete_table(table_name, if_exists=True)
    odps.persist_tensor_via_oss(
        t,
        table_name,
        ["x", "y"],
        "z",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=table_name,
    )
    df = odps.get_table(table_name).to_df()
    t_from_oss = df.to_mars_tensor_via_oss(
        ["x", "y"],
        "z",
        6,
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=table_name,
        oss_file_exist=True,
    )
    np.testing.assert_array_equal(np.ones((10, 10)), t_from_oss.execute())


def test_partitions_persist_to_tensor(odps, config):
    import mars.tensor as mt
    import numpy as np

    t1 = mt.ones((10, 10), chunk_size=6)
    t2 = mt.ones((6, 6), chunk_size=6)

    (
        oss_access_id,
        oss_secret_access_key,
        oss_bucket_name,
        oss_endpoint,
    ) = config.oss_config

    table_name = "test_partitions_persist_to_tensor"
    odps.delete_table(table_name, if_exists=True)
    odps.persist_tensor_via_oss(
        {"g=1": t1, "g=2": t2},
        table_name,
        ["x", "y"],
        "z",
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=table_name,
    )
    df = odps.get_table(table_name).to_df()
    t_from_oss = df.to_mars_tensor_via_oss(
        ["x", "y"],
        "z",
        6,
        partitions=["g"],
        oss_access_id=oss_access_id,
        oss_access_key=oss_secret_access_key,
        oss_bucket_name=oss_bucket_name,
        oss_endpoint=oss_endpoint,
        oss_path=table_name,
        oss_file_exist=True,
    )
    ts = list(t_from_oss.values())
    np.testing.assert_array_equal(np.ones((10, 10)), ts[0].execute())
    np.testing.assert_array_equal(np.ones((6, 6)), ts[1].execute())
