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
import time
import uuid
from datetime import datetime, date

from odps import DataFrame
from odps.models import Schema, Instance
from odps.compat import unittest
from odps.config import options
from odps.tests.core import TestBase, tn, odps2_typed_case

try:
    import mars
    from odps.mars_extension.oscar.core import (  # noqa: F401
        create_mars_cluster,
        to_mars_dataframe,
        persist_mars_dataframe,
    )
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

    def testCreateMarsCluster(self):
        import pandas as pd

        mars_source_table_name = tn("mars_datasource")
        mars_des_table_name = tn("mars_datastore")
        self._create_table(mars_source_table_name)
        self.odps.delete_table(mars_des_table_name, if_exists=True)
        data = self._gen_data()
        self.odps.write_table(mars_source_table_name, data)

        client = self.odps.create_mars_cluster(1, 4, 8, name=str(uuid.uuid4()))
        try:
            self.assertFalse(client._with_notebook)

            df = self.odps.to_mars_dataframe(
                mars_source_table_name, runtime_endpoint=self.odps.endpoint
            )
            df_head = df.head(2)
            self.odps.persist_mars_dataframe(
                df_head,
                mars_des_table_name,
                unknown_as_string=True,
                runtime_endpoint=self.odps.endpoint,
            )

            des = self.odps.to_mars_dataframe(
                mars_des_table_name, runtime_endpoint=self.odps.endpoint
            )

            expected = (
                self.odps.get_table(mars_source_table_name)
                .to_df()
                .to_pandas()["id"]
                .head(2)
            )
            result = des["id"].to_pandas()
            pd.testing.assert_series_equal(expected, result)
            self.odps.delete_table(mars_source_table_name)
            self.odps.delete_table(mars_des_table_name)
        finally:
            client.stop_server()

    def testMarsDataFrame(self):
        import pandas as pd
        import numpy as np

        client = self.odps.create_mars_cluster(3, 4, 8, name=str(uuid.uuid4()))
        try:
            mars_source_table_name = tn("mars_df")

            data = pd.DataFrame(
                {"c1": np.random.rand(100), "c2": np.random.randint(0, 100, (100,))}
            )
            self.odps.delete_table(mars_source_table_name, if_exists=True)
            self.odps.create_table(
                mars_source_table_name,
                "c1 double, c2 int",
                stored_as="aliorc",
                lifecycle=1,
            )
            DataFrame(data).persist(mars_source_table_name, odps=self.odps)

            df = self.odps.to_mars_dataframe(
                mars_source_table_name, runtime_endpoint=self.odps.endpoint
            )
            result = df.describe().to_pandas()
            expected = data.describe()

            pd.testing.assert_frame_equal(expected, result)
            self.odps.delete_table(mars_source_table_name)
        finally:
            client.stop_server()

    def testMarsKNN(self):
        client = self.odps.create_mars_cluster(
            1, 4, 8, name=str(uuid.uuid4()), supervisor_mem=12, supervisor_cpu=4
        )

        try:
            import numpy as np
            import mars.tensor as mt
            from mars.learn.neighbors import NearestNeighbors
            from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors

            rs = np.random.RandomState(0)
            raw_X = rs.rand(10, 5)
            raw_Y = rs.rand(8, 5)

            X = mt.tensor(raw_X, chunk_size=7)
            Y = mt.tensor(raw_Y, chunk_size=(5, 3))

            nn = NearestNeighbors(n_neighbors=3)
            nn.fit(X)
            ret = nn.kneighbors(Y)

            snn = SkNearestNeighbors(n_neighbors=3)
            snn.fit(raw_X)

            expected = snn.kneighbors(raw_Y)
            result = [r.fetch() for r in ret]
            np.testing.assert_almost_equal(result[0], expected[0])
            np.testing.assert_almost_equal(result[1], expected[1])
        finally:
            client.stop_server()

    def testExtended(self):
        def func():
            import mars.tensor as mt
            from mars.learn.contrib.lightgbm.classifier import LGBMClassifier

            n_rows = 1000
            n_columns = 10
            chunk_size = 50
            rs = mt.random.RandomState(0)
            X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
            y = rs.rand(n_rows, chunk_size=chunk_size)
            y = (y * 10).astype(mt.int32)
            classifier = LGBMClassifier(n_estimators=2)
            classifier.fit(X, y, eval_set=[(X, y)])
            _prediction = classifier.predict(X)  # noqa: F841

        self.odps.run_mars_job(func, image="extended")

    def testRocCurve(self):
        import numpy as np
        import pandas as pd
        import mars.dataframe as md
        from mars.learn.metrics import roc_curve, auc
        from sklearn.metrics import roc_curve as sklearn_roc_curve, auc as sklearn_auc

        client = self.odps.create_mars_cluster(1, 4, 8, name=str(uuid.uuid4()))
        try:
            rs = np.random.RandomState(0)
            raw = pd.DataFrame({"a": rs.randint(0, 10, (10,)), "b": rs.rand(10)})

            df = md.DataFrame(raw)
            y = df["a"].to_tensor().astype("int")
            pred = df["b"].to_tensor().astype("float")
            fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)
            m = auc(fpr, tpr)

            sk_fpr, sk_tpr, sk_threshod = sklearn_roc_curve(
                raw["a"].to_numpy().astype("int"),
                raw["b"].to_numpy().astype("float"),
                pos_label=2,
            )
            expect_m = sklearn_auc(sk_fpr, sk_tpr)
            self.assertAlmostEqual(m.fetch(), expect_m)
        finally:
            client.stop_server()

    def testRunScript(self):
        import pandas as pd
        from io import BytesIO
        from odps.utils import to_binary

        client = self.odps.create_mars_cluster(1, 4, 8, name=str(uuid.uuid4()))
        try:
            mars_source_table_name = tn("mars_script_datasource")
            mars_des_table_name = tn("mars_script_datastore")
            self._create_table(mars_source_table_name)
            self.odps.delete_table(mars_des_table_name, if_exists=True)
            data = self._gen_data()
            self.odps.write_table(mars_source_table_name, data)

            code = BytesIO(
                to_binary(
                    script.format(
                        mars_source_table_name,
                        self.odps.endpoint,
                        mars_des_table_name,
                        self.odps.endpoint,
                    )
                )
            )

            self.odps.run_script_in_mars(code, runtime_endpoint=self.odps.endpoint)
            result = self.odps.get_table(mars_des_table_name).to_df().to_pandas()
            expected = self.odps.get_table(mars_source_table_name).to_df().to_pandas()
            pd.testing.assert_frame_equal(result, expected)
        finally:
            client.stop_server()

    def testRunMarsJob(self):
        import pandas as pd

        odps_entry = self.odps
        mars_source_table_name = tn("mars_script_datasource")
        mars_des_table_name = tn("mars_script_datastore")
        self._create_table(mars_source_table_name)
        self.odps.delete_table(mars_des_table_name, if_exists=True)
        data = self._gen_data()
        self.odps.write_table(mars_source_table_name, data)

        def func(s_name, d_name):
            df = odps_entry.to_mars_dataframe(
                s_name, runtime_endpoint=odps_entry.endpoint
            ).to_pandas()
            odps_entry.persist_mars_dataframe(
                df, d_name, unknown_as_string=True, runtime_endpoint=odps_entry.endpoint
            )

        self.odps.run_mars_job(
            func,
            args=(mars_source_table_name, mars_des_table_name),
            name=str(uuid.uuid4()),
            worker_cpu=4,
            worker_mem=8,
        )

        result = self.odps.get_table(mars_des_table_name).to_df().to_pandas()
        expected = self.odps.get_table(mars_source_table_name).to_df().to_pandas()
        pd.testing.assert_frame_equal(result, expected)

    def testRemote(self):
        import mars.remote as mr

        def add_one(x):
            return x + 1

        def sum_all(xs):
            return sum(xs)

        x_list = []
        for i in range(10):
            x_list.append(mr.spawn(add_one, args=(i,)))

        client = self.odps.create_mars_cluster(1, 4, 8, name=str(uuid.uuid4()))
        try:
            r = mr.spawn(sum_all, args=(x_list,)).execute().fetch()
            self.assertEqual(r, 55)
        finally:
            client.stop_server()

    def testEmptyTable(self):
        mars_source_table_name = tn("mars_empty_datasource")
        self.odps.delete_table(mars_source_table_name, if_exists=True)
        self.odps.create_table(mars_source_table_name, schema="col1 int, col2 string")

        client = self.odps.create_mars_cluster(1, 4, 8, name=str(uuid.uuid4()))
        try:
            df = self.odps.to_mars_dataframe(
                mars_source_table_name, runtime_endpoint=self.odps.endpoint
            )
            result = df.execute().to_pandas()
            self.assertEqual(list(result.columns), ["col1", "col2"])
        finally:
            client.stop_server()

    def testViewTable(self):
        import pandas as pd

        mars_source_table_name = tn("mars_view_datasource")
        self.odps.delete_table(mars_source_table_name, if_exists=True)
        self.odps.create_table(mars_source_table_name, schema="col1 int, col2 string")
        self.odps.write_table(mars_source_table_name, [[1, "test1"], [2, "test2"]])

        mars_view_table_name = tn("mars_view_table")
        self.odps.execute_sql("DROP VIEW IF EXISTS {}".format(mars_view_table_name))
        sql = "create view {} (view_col1, view_col2) as select * from {}".format(
            mars_view_table_name, mars_source_table_name
        )
        self.odps.execute_sql(sql)

        client = self.odps.create_mars_cluster(1, 4, 8, name=str(uuid.uuid4()))
        try:
            df = self.odps.to_mars_dataframe(
                mars_view_table_name, runtime_endpoint=self.odps.endpoint
            )
            result = df.execute().to_pandas()
            expected = pd.DataFrame(
                {"view_col1": [1, 2], "view_col2": ["test1", "test2"]}
            )
            pd.testing.assert_frame_equal(result, expected)
        finally:
            client.stop_server()

    def testClusterTimeout(self):
        client = self.odps.create_mars_cluster(
            1, 4, 8, name=str(uuid.uuid4()), instance_idle_timeout=15
        )
        try:
            self.assertEqual(client._kube_instance.status, Instance.Status.RUNNING)
            time.sleep(60)
            self.assertEqual(client._kube_instance.status, Instance.Status.TERMINATED)
        finally:
            if client._kube_instance.status != Instance.Status.TERMINATED:
                client.stop_server()

    @odps2_typed_case
    def testPersistOdps2Types(self):
        import pandas as pd
        import mars.remote as mr

        mars_source_table_name = tn("mars_odps2_datasource")
        mars_des_table_name = tn("mars_odps2_datastore")
        self.odps.delete_table(mars_source_table_name, if_exists=True)
        self.odps.delete_table(mars_des_table_name, if_exists=True)

        table = self.odps.create_table(
            mars_source_table_name,
            "col1 int, "
            "col2 tinyint,"
            "col3 smallint,"
            "col4 float,"
            "col5 timestamp,"
            "col6 datetime,"
            "col7 date",
            lifecycle=1,
            stored_as="aliorc",
        )

        contents = [
            [
                0,
                1,
                2,
                1.0,
                pd.Timestamp("1998-02-15 23:59:21.943829154"),
                datetime.today(),
                date.today(),
            ],
            [
                0,
                1,
                2,
                1.0,
                pd.Timestamp("1998-02-15 23:59:21.943829154"),
                datetime.today(),
                date.today(),
            ],
            [
                0,
                1,
                2,
                1.0,
                pd.Timestamp("1998-02-15 23:59:21.943829154"),
                datetime.today(),
                date.today(),
            ],
        ]
        self.odps.write_table(table, contents)

        client = self.odps.create_mars_cluster(1, 4, 8, name=str(uuid.uuid4()))

        try:
            df = self.odps.to_mars_dataframe(
                mars_source_table_name, runtime_endpoint=self.odps.endpoint
            )
            df_head = df.head(2)
            self.odps.persist_mars_dataframe(
                df_head,
                mars_des_table_name,
                unknown_as_string=True,
                runtime_endpoint=self.odps.endpoint,
            )

            # test write in remote function
            odps_entry = self.odps
            mars_des_table_name = tn("mars_odps2_datastore_remote")
            self.odps.delete_table(mars_des_table_name, if_exists=True)

            def func(d_name):
                import pandas as pd

                contents = [
                    [
                        0,
                        1,
                        2,
                        1.0,
                        pd.Timestamp("1998-02-15 23:59:21.943829154"),
                        datetime.today(),
                        date.today(),
                    ],
                    [
                        0,
                        1,
                        2,
                        1.0,
                        pd.Timestamp("1998-02-15 23:59:21.943829154"),
                        datetime.today(),
                        date.today(),
                    ],
                    [
                        0,
                        1,
                        2,
                        1.0,
                        pd.Timestamp("1998-02-15 23:59:21.943829154"),
                        datetime.today(),
                        date.today(),
                    ],
                ]
                df = pd.DataFrame(
                    contents, columns=["col" + str(i + 1) for i in range(7)]
                )
                odps_entry.persist_mars_dataframe(
                    df,
                    d_name,
                    unknown_as_string=True,
                    runtime_endpoint=odps_entry.endpoint,
                )

            mr.spawn(func, args=(mars_des_table_name,)).execute()

        finally:
            client.stop_server()

    def testSqlToDataFrame(self):
        import pandas as pd

        mars_source_table_name = tn("mars_sql_datasource")
        self._create_table(mars_source_table_name)
        data = self._gen_data()
        self.odps.write_table(mars_source_table_name, data)

        client = self.odps.create_mars_cluster(2, 4, 8, name=str(uuid.uuid4()))
        try:
            sql = "select count(1) as count from {}".format(mars_source_table_name)
            df = self.odps.sql_to_mars_dataframe(sql)
            r = df.execute().to_pandas()
            pd.testing.assert_frame_equal(r, pd.DataFrame([4], columns=["count"]))

            sql = """
            SELECT
            t1.`id`,
            MAX(t1.`int_num`) AS `int_num_max`,
            MAX(t1.`float_num`) AS `float_num_max`
            FROM cupid_test_release.`{}` t1
            GROUP BY
            t1.`id`
            """.format(
                mars_source_table_name
            )
            df2 = self.odps.sql_to_mars_dataframe(sql)
            r2 = df2.execute().to_pandas()
            expected = self.odps.execute_sql(sql).open_reader().to_pandas()
            pd.testing.assert_frame_equal(r2, expected)

        finally:
            client.stop_server()

    def testFullPartitionedTable(self):
        import pandas as pd

        mars_source_table_name = tn("mars_cupid_datasource_mpart")
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

        client = self.odps.create_mars_cluster(1, 4, 8, name=str(uuid.uuid4()))
        try:
            df = self.odps.to_mars_dataframe(
                mars_source_table_name,
                runtime_endpoint=self.odps.endpoint,
                append_partitions=True,
                add_offset=True,
            )
            result = df.execute().to_pandas()
            expected = table.to_df().to_pandas()
            pd.testing.assert_frame_equal(result, expected)
        finally:
            client.stop_server()

    def testMultipleParts(self):
        import pandas as pd

        mars_source_table_name = tn("mars_arrow_tunnel_datasource_mpart")
        self.odps.delete_table(mars_source_table_name, if_exists=True)
        table = self.odps.create_table(
            mars_source_table_name,
            schema=("col1 int, col2 string", "pt string"),
            lifecycle=1,
        )
        for pid in range(5):
            pt = table.create_partition("pt=test_part%d" % pid)
            with pt.open_writer() as writer:
                writer.write([[1 + pid * 2, "test1"], [2 + pid * 2, "test2"]])

        client = self.odps.create_mars_cluster(1, 4, 8, name=str(uuid.uuid4()))
        try:
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
                    partition="pt>test_part1",
                    append_partitions=True,
                    add_offset=True,
                )
                .execute()
                .to_pandas()
            )
            expected = (
                table.to_df()
                .to_pandas()
                .query('pt>"test_part1"')
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(r, expected)

            table.create_partition("pt=test_part5")
            r = (
                self.odps.to_mars_dataframe(
                    mars_source_table_name,
                    partition="pt=max_pt()",
                    append_partitions=True,
                    add_offset=True,
                )
                .execute()
                .to_pandas()
            )
            expected = (
                table.to_df()
                .to_pandas()
                .query('pt=="test_part4"')
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(r, expected)
        finally:
            client.stop_server()
