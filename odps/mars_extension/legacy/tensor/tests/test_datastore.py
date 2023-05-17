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

import logging

import pytest

logger = logging.getLogger(__name__)

try:
    import mars.tensor as mt
    from mars.executor import Executor
    from .. import write_coo, read_coo
except ImportError:
    read_coo = None


@pytest.fixture
def executor():
    return Executor()


@pytest.mark.skipif(read_coo is None, reason="mars not installed")
def test_tensor_write_table(executor):
    import tempfile
    import shutil
    import numpy as np

    dir_name = tempfile.mkdtemp(prefix="mars-test-tensor-read")
    try:
        d = mt.ones((100, 100), chunk_size=(20, 10))
        t = write_coo(d, dir_name, ["x", "y"], "val")

        executor.execute_tensor(t)

        t2 = read_coo(
            dir_name + "/*.parquet",
            ["x", "y"],
            "val",
            shape=(100, 100),
            chunk_size=(20, 10),
            sparse=False,
        )
        res = executor.execute_tensor(t2)
        [np.testing.assert_equal(r, np.ones((20, 10))) for r in res]
    finally:
        shutil.rmtree(dir_name)

    dir_name = tempfile.mkdtemp(prefix="mars-test-tensor-read")
    try:
        d = mt.ones(100, chunk_size=20)
        t = write_coo(d, dir_name, ["x"], "val")

        executor.execute_tensor(t)

        t2 = read_coo(
            dir_name + "/*.parquet",
            ["x"],
            "val",
            shape=(100,),
            chunk_size=20,
            sparse=False,
        )
        res = executor.execute_tensor(t2)
        [np.testing.assert_equal(r, np.ones((20,))) for r in res]
    finally:
        shutil.rmtree(dir_name)

    dir_name = tempfile.mkdtemp(prefix="mars-test-tensor-read")
    try:
        d = mt.ones((100, 20), chunk_size=20)
        U, s, V = mt.linalg.svd(d)
        t = write_coo(s, dir_name, ["x"], "val")

        executor.execute_tensor(t)

        t2 = read_coo(
            dir_name + "/*.parquet",
            ["x"],
            "val",
            shape=s.shape,
            chunk_size=20,
            sparse=False,
        )
        res = executor.execute_tensor(t2)[0]
        np.testing.assert_allclose(
            res, np.linalg.svd(np.ones((100, 20)))[1], atol=0.1
        )
    finally:
        shutil.rmtree(dir_name)

    dir_name = tempfile.mkdtemp(prefix="mars-test-tensor-read")
    try:
        d = mt.ones((200, 10), chunk_size=10)
        U, s, V = mt.linalg.svd(d)
        t = write_coo(s, dir_name, ["x", "y"], "val")

        executor.execute_tensor(t)

        t2 = read_coo(
            dir_name + "/*.parquet",
            ["x"],
            "val",
            shape=s.shape,
            chunk_size=20,
            sparse=False,
        )
        res = executor.execute_tensor(t2)[0]
        np.testing.assert_allclose(
            res, np.linalg.svd(np.ones((200, 10)))[1], atol=0.1
        )
    finally:
        shutil.rmtree(dir_name)
