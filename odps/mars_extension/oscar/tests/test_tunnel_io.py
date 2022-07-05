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

import math

from odps.models import Schema
from odps.compat import unittest
from odps.config import options
from odps.tests.core import TestBase, tn

try:
    import mars
    from odps.mars_extension.oscar.core import to_mars_dataframe  # noqa: F401
except ImportError:
    mars = None


script = """
df = o.to_mars_dataframe('{}', runtime_endpoint='{}').to_pandas()
o.persist_mars_dataframe(df, '{}', unknown_as_string=True, runtime_endpoint='{}')
"""


@unittest.skipIf(mars is None, "mars not installed")
class Test(TestBase):
    def setup(self):
        from cupid.config import options as cupid_options

        options.verbose = True
        cupid_options.cupid.runtime.endpoint = self.odps.endpoint

    def _create_table(self, table_name):
        fields = ["id", "int_num", "float_num", "bool"]
        types = ["string", "bigint", "double", "boolean"]

        self.odps.delete_table(table_name, if_exists=True)
        return self.odps.create_table(
            table_name,
            schema=Schema.from_lists(fields, types),
            stored_as="aliorc",
            lifecycle=1,
        )

    def _gen_data(self):
        return [
            ("hello \x00\x00 world", 2**63 - 1, math.pi, True),
            ("goodbye", 222222, math.e, False),
            ("c" * 300, -(2**63) + 1, -2.222, True),
            ("c" * 20, -(2**11) + 1, 2.222, True),
        ]

    def testArrowTunnel(self):
        import pandas as pd
        import numpy as np
        import mars.dataframe as md

        mars_des_table_name = tn("mars_arrow_tunnel_datastore")
        self.odps.delete_table(mars_des_table_name, if_exists=True)

        data = pd.DataFrame(
            {
                "col1": np.random.rand(
                    1000,
                ),
                "col2": np.random.randint(0, 100, (1000,)),
                "col3": np.random.choice(["a", "b", "c"], size=(1000,)),
            }
        )

        df = md.DataFrame(data, chunk_size=300)
        self.odps.persist_mars_dataframe(
            df, mars_des_table_name, unknown_as_string=True
        )
        expected = self.odps.get_table(mars_des_table_name).to_df().to_pandas()
        pd.testing.assert_frame_equal(
            expected.sort_values("col1").reset_index(drop=True),
            data.sort_values("col1").reset_index(drop=True),
        )

        r = (
            self.odps.to_mars_dataframe(mars_des_table_name, chunk_size=200)
            .execute()
            .to_pandas()
        )
        expected = self.odps.get_table(mars_des_table_name).to_df().to_pandas()
        pd.testing.assert_frame_equal(
            r.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def testArrowTunnelSinglePart(self):
        import pandas as pd
        import numpy as np
        import mars.dataframe as md

        mars_source_table_name = tn("mars_arrow_tunnel_datasource_spart")
        mars_des_table_name = tn("mars_arrow_tunnel_datastore_spart")
        self.odps.delete_table(mars_des_table_name, if_exists=True)
        self.odps.delete_table(mars_source_table_name, if_exists=True)
        table = self.odps.create_table(
            mars_source_table_name,
            schema=("col1 int, col2 string", "pt string"),
            lifecycle=1,
        )
        pt = table.create_partition("pt=test_part")
        with pt.open_writer() as writer:
            writer.write([[1, "test1"], [2, "test2"]])

        r = (
            self.odps.to_mars_dataframe(
                mars_source_table_name, partition="pt=test_part"
            )
            .execute()
            .to_pandas()
        )
        expected = pt.to_df().to_pandas()
        pd.testing.assert_frame_equal(r, expected)

        data = pd.DataFrame(
            {
                "col1": np.random.rand(
                    1000,
                ),
                "col2": np.random.randint(0, 100, (1000,)),
                "col3": np.random.choice(["a", "b", "c"], size=(1000,)),
            }
        )

        df = md.DataFrame(data, chunk_size=300)
        self.odps.persist_mars_dataframe(
            df, mars_des_table_name, partition="pt=test_part", unknown_as_string=True
        )
        expected = (
            self.odps.get_table(mars_des_table_name)
            .get_partition("pt=test_part")
            .to_df()
            .to_pandas()
        )
        pd.testing.assert_frame_equal(
            expected.sort_values("col1").reset_index(drop=True),
            data.sort_values("col1").reset_index(drop=True),
        )

    def testArrowTunnelMultipleParts(self):
        import pandas as pd

        mars_source_table_name = tn("mars_arrow_tunnel_datasource_mpart")
        self.odps.delete_table(mars_source_table_name, if_exists=True)
        table = self.odps.create_table(
            mars_source_table_name,
            schema=("col1 int, col2 string", "pt1 string, pt2 string"),
            lifecycle=1,
        )
        for pid in range(5):
            pt = table.create_partition("pt1=test_part%d,pt2=test_part%d" % (pid, pid))
            with pt.open_writer() as writer:
                writer.write([[1 + pid * 2, "test1"], [2 + pid * 2, "test2"]])

        r = (
            self.odps.to_mars_dataframe(
                mars_source_table_name, append_partitions=True, add_offset=True
            )
            .execute()
            .to_pandas()
        )
        expected = table.to_df().to_pandas()
        pd.testing.assert_frame_equal(r, expected)

        r = (
            self.odps.to_mars_dataframe(
                mars_source_table_name,
                partition="pt1>test_part1",
                append_partitions=True,
                add_offset=True,
            )
            .execute()
            .to_pandas()
        )
        expected = (
            table.to_df().to_pandas().query('pt1>"test_part1"').reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(r, expected)

        r = (
            self.odps.to_mars_dataframe(
                mars_source_table_name,
                partition="pt2=invalid & pt1>test_part1",
                append_partitions=True,
            )
            .execute()
            .to_pandas()
        )
        self.assertEqual(list(r.columns), ["col1", "col2", "pt1", "pt2"])
        self.assertEqual(r.shape[0], 0)

    def testExistedPartition(self):
        import pandas as pd
        import mars.dataframe as md

        mars_source_table_name = tn("mars_existed_partition")
        self.odps.delete_table(mars_source_table_name, if_exists=True)
        table = self.odps.create_table(
            mars_source_table_name,
            schema=("col1 int, col2 string", "pt string"),
            lifecycle=1,
        )
        table.create_partition("pt=test")

        df = md.DataFrame(pd.DataFrame({"col1": [1, 2], "col2": list("ab")}))
        self.odps.persist_mars_dataframe(
            df, mars_source_table_name, partition="pt=test", unknown_as_string=True
        )

    def testEmptyTable(self):
        mars_source_table_name = tn("mars_empty_datasource_tunnel")
        self.odps.delete_table(mars_source_table_name, if_exists=True)
        self.odps.create_table(mars_source_table_name, schema="col1 int, col2 string")

        df = self.odps.to_mars_dataframe(mars_source_table_name)
        result = df.execute().to_pandas()
        self.assertEqual(list(result.columns), ["col1", "col2"])
        self.assertEqual(df.shape[0], 0)
