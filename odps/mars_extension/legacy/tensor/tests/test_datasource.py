#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from odps.tests.core import TestBase
from odps.compat import unittest

try:
    from mars.graph import DirectedGraph
    try:
        from mars.executor import Executor
    except:
        from mars.tensor.execution.core import Executor

    from odps.mars_extension.legacy.tensor import read_coo
except ImportError:
    read_coo = None


@unittest.skipIf(read_coo is None, 'mars not installed')
class Test(TestBase):
    def setup(self):
        self.executor = Executor()

    def testTensorReadCOO(self):
        import shutil
        import tempfile
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
        import numpy as np
        import scipy.sparse as sps

        dir_name = tempfile.mkdtemp(prefix='mars-test-tensor-read')
        try:
            data_src = []
            for x in range(2):
                for y in range(2):
                    mat = sps.random(50, 50, 0.1)
                    data_src.append(mat)
                    df = pd.DataFrame(dict(x=mat.row, y=mat.col, val=mat.data),
                                      columns=['x', 'y', 'val'])
                    pq.write_table(pa.Table.from_pandas(df), dir_name + '/' +
                                   'table@%d,%d.parquet' % (x, y))

            t = read_coo(dir_name + '/*.parquet', ['x', 'y'], 'val', shape=(100, 100), chunk_size=50,
                         sparse=True)
            res = self.executor.execute_tensor(t)
            [np.testing.assert_equal(r.toarray(), e.toarray()) for r, e in zip(res, data_src)]

            t = read_coo(dir_name + '/*.parquet', ['x', 'y'], 'val', shape=(100, 100), chunk_size=50,
                         sparse=True)
            DirectedGraph.from_json(t.build_graph(tiled=False).to_json())
            DirectedGraph.from_json(t.build_graph(tiled=False).to_json())
        finally:
            shutil.rmtree(dir_name)
