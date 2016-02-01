# encoding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from itertools import groupby, product

from odps.utils import load_resource_string


class TestDataMixIn(object):
    def create_ionosphere(self, table_name):
        fields = ','.join('a%02d double' % i for i in range(1, 35)) + ', class bigint'
        self.odps.execute_sql("drop table if exists " + table_name)
        self.odps.execute_sql("create table %s (%s) lifecycle 3" % (table_name, fields))

        upload_ss = self.tunnel.create_upload_session(table_name)
        writer = upload_ss.open_record_writer(0)

        for line in load_resource_string('odps.examples.data', 'ionosphere.txt').splitlines():
            rec = upload_ss.new_record()
            cols = [float(c) if rec._columns[i].type == 'double' else int(c) for i, c in enumerate(line.split(','))]
            [rec.set(i, val) for i, val in enumerate(cols)]
            writer.write(rec)
        writer.close()
        upload_ss.commit([0, ])

    def create_ionosphere_one_part(self, table_name, partition_count=3):
        fields = ','.join('a%02d double' % i for i in range(1, 35)) + ', class bigint'
        self.odps.execute_sql("drop table if exists " + table_name)
        self.odps.execute_sql("create table %s (%s) partitioned by (part bigint) lifecycle 3" % (table_name, fields))
        for part_id in range(partition_count):
            self.odps.execute_sql('alter table %s add if not exists partition (part=%d)' % (table_name, part_id))

        upload_sses = [self.tunnel.create_upload_session(table_name, 'part=%d' % part_id) for part_id in range(partition_count)]
        writers = [session.open_record_writer(0) for session in upload_sses]

        for line_no, line in enumerate(load_resource_string('odps.examples.data', 'ionosphere.txt').splitlines()):
            part_id = line_no % partition_count
            rec = upload_sses[part_id].new_record()
            cols = [float(c) if rec._columns[i].type == 'double' else int(c) for i, c in enumerate(line.split(','))]
            cols.append(part_id)
            [rec.set(i, val) for i, val in enumerate(cols)]
            writers[part_id].write(rec)
        [writer.close() for writer in writers]
        [upload_ss.commit([0, ]) for upload_ss in upload_sses]

    def create_ionosphere_two_parts(self, table_name, partition1_count=2, partition2_count=3):
        fields = ','.join('a%02d double' % i for i in range(1, 35)) + ', class bigint'
        self.odps.execute_sql("drop table if exists " + table_name)
        self.odps.execute_sql("create table %s (%s) partitioned by (part1 bigint, part2 bigint) lifecycle 3" % (table_name, fields))
        for id1, id2 in product(range(partition1_count), range(partition2_count)):
            self.odps.execute_sql('alter table %s add if not exists partition (part1=%d, part2=%d)' % (table_name, id1, id2))

        upload_sses = [[self.tunnel.create_upload_session(table_name, 'part1=%d,part2=%d' % (id1, id2))
                        for id2 in range(partition2_count)] for id1 in range(partition1_count)]
        writers = [[session.open_record_writer(0) for session in sessions] for sessions in upload_sses]

        for line_no, line in enumerate(load_resource_string('odps.examples.data', 'ionosphere.txt').splitlines()):
            id1, id2 = line_no % partition1_count, line_no % partition2_count
            rec = upload_sses[id1][id2].new_record()
            cols = [float(c) if rec._columns[i].type == 'double' else int(c) for i, c in enumerate(line.split(','))]
            cols.extend([id1, id2])
            [rec.set(i, val) for i, val in enumerate(cols)]
            writers[id1][id2].write(rec)
        [writer.close() for ws in writers for writer in ws]
        [upload_ss.commit([0, ]) for upload_sss in upload_sses for upload_ss in upload_sss]

    def create_iris(self, table_name):
        self.odps.execute_sql("drop table if exists " + table_name)
        self.odps.execute_sql(('create table %s (sepal_length double, sepal_width double, petal_length double, '
                               + 'petal_width double, category string) lifecycle 3') % table_name)

        upload_ss = self.tunnel.create_upload_session(table_name)
        writer = upload_ss.open_record_writer(0)

        for line in load_resource_string('odps.examples.data', 'iris.txt').splitlines():
            rec = upload_ss.new_record()
            line_parts = line.split(',')
            cols = [float(c) for c in line_parts[:-1]]
            cols.append(line_parts[4])
            [rec.set(i, val) for i, val in enumerate(cols)]
            writer.write(rec)
        writer.close()
        upload_ss.commit([0, ])

    def create_iris_kv(self, table_name):
        self.odps.execute_sql("drop table if exists " + table_name)
        self.odps.execute_sql('create table %s (content string, category bigint) lifecycle 3' % table_name)

        upload_ss = self.tunnel.create_upload_session(table_name)
        writer = upload_ss.open_record_writer(0)

        for line in load_resource_string('odps.examples.data', 'iris.txt').splitlines():
            rec = upload_ss.new_record()
            line_parts = line.split(',')
            rec.set(0, ','.join('%s:%s' % (idx, c) for idx, c in enumerate(line_parts[:-1])))
            rec.set(1, 0 if 'setosa' in line_parts[-1] else 1)
            writer.write(rec)
        writer.close()
        upload_ss.commit([0, ])

    def create_corpus(self, table_name):
        self.odps.execute_sql("drop table if exists " + table_name)
        self.odps.execute_sql("create table %s (id string, content string) lifecycle 3" % table_name)

        upload_ss = self.tunnel.create_upload_session(table_name)
        writer = upload_ss.open_record_writer(0)

        for line_no, line in enumerate(load_resource_string('odps.examples.data', 'splited_words.txt').splitlines()):
            rec = upload_ss.new_record()
            cols = [line_no + 1, line.replace('####', '')]
            [rec.set(i, val) for i, val in enumerate(cols)]
            writer.write(rec)
        writer.close()
        upload_ss.commit([0, ])

    def create_word_triple(self, table_name):
        self.odps.execute_sql("drop table if exists " + table_name)
        self.odps.execute_sql("create table %s (id string, word string, count bigint) lifecycle 3" % table_name)

        upload_ss = self.tunnel.create_upload_session(table_name)
        writer = upload_ss.open_record_writer(0)

        for line_no, line in enumerate(load_resource_string('odps.examples.data', 'splited_words.txt').splitlines()):
            line = line.strip()
            if not line:
                break
            for word, group in groupby(sorted(line.split('####'))):
                rec = upload_ss.new_record()
                cols = [str(line_no + 1), word, len(list(group))]
                [rec.set(i, val) for i, val in enumerate(cols)]
                writer.write(rec)
        writer.close()
        upload_ss.commit([0, ])

    def create_splited_words(self, table_name):
        self.odps.execute_sql("drop table if exists " + table_name)
        self.odps.execute_sql("create table %s (id string, content string) lifecycle 3" % table_name)

        upload_ss = self.tunnel.create_upload_session(table_name)
        writer = upload_ss.open_record_writer(0)

        for line_no, line in enumerate(load_resource_string('odps.examples.data', 'splited_words.txt').splitlines()):
            if not line.strip():
                break
            for word in line.split('####'):
                rec = upload_ss.new_record()
                cols = [line_no + 1, word]
                [rec.set(i, val) for i, val in enumerate(cols)]
                writer.write(rec)
        writer.close()
        upload_ss.commit([0, ])

    def create_weighted_graph_edges(self, table_name):
        data_rows = [
            ['1', '1', '2', '1', 0.7], ['1', '1', '3', '1', 0.7], ['1', '1', '4', '1', 0.6], ['2', '1', '3', '1', 0.7],
            ['2', '1', '4', '1', 0.6], ['3', '1', '4', '1', 0.6], ['4', '1', '6', '5', 0.3], ['5', '5', '6', '5', 0.6],
            ['5', '5', '7', '5', 0.7], ['5', '5', '8', '5', 0.7], ['6', '5', '7', '5', 0.6], ['6', '5', '8', '5', 0.6],
            ['7', '5', '8', '5', 0.7]
        ]
        self.odps.execute_sql("drop table if exists " + table_name)
        self.odps.execute_sql(("create table %s (flow_out_id string, group_out_id string, flow_in_id string, " +
                               "group_in_id string, edge_weight double) lifecycle 3") % table_name)

        upload_ss = self.tunnel.create_upload_session(table_name)
        writer = upload_ss.open_record_writer(0)

        for rd in data_rows:
            rec = upload_ss.new_record()
            [rec.set(i, val) for i, val in enumerate(rd)]
            writer.write(rec)
        writer.close()
        upload_ss.commit([0, ])

    def create_weighted_graph_vertices(self, table_name):
        data_rows = [
            ['1', '1', 0.7, 1.0], ['2', '1', 0.7, 1.0], ['3', '1', 0.7, 1.0], ['4', '1', 0.5, 1.0], ['5', '5', 0.7, 1.0],
            ['6', '5', 0.5, 1.0], ['7', '5', 0.7, 1.0], ['8', '5', 0.7, 1.0]
        ]
        self.odps.execute_sql("drop table if exists " + table_name)
        self.odps.execute_sql(("create table %s (node string, label string, node_weight double, label_weight double)" +
                               " lifecycle 3") % table_name)

        upload_ss = self.tunnel.create_upload_session(table_name)
        writer = upload_ss.open_record_writer(0)

        for rd in data_rows:
            rec = upload_ss.new_record()
            [rec.set(i, val) for i, val in enumerate(rd)]
            writer.write(rec)
        writer.close()
        upload_ss.commit([0, ])
