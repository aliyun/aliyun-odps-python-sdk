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

import time
import unittest
import ut
from odps.models import Record
from odps.datahub.datahubclient import DatahubClient
from odps.datahub.recordpack import RecordPack
from odps.datahub.errors import DatahubError
from datetime import datetime

TEST_PROJECT = 'test'
TEST_TABLE = 'pyodps_type_test_datahub'

def convert_to_list(one_item_tuple_list):
    ret = []
    for i in one_item_tuple_list:
        ret.append(i[0])
    return ret

def make_record(schema, count):
    record = Record(schema.columns)
    columns = schema._columns
    for i in range(len(columns)):
        type = columns[i].type
        if type == 'string':
            record.set(i, 'sample')
        elif type == 'bigint':
            record.set(i, count)
        elif type == 'double':
            record.set(i, 3.14)
        elif type == 'datetime':
            record.set(i, datetime.today())
        elif type == 'boolean':
            record.set(i, True)
        else :
            raise DatahubError('invalid type')
    return record 

class TestDatahub(ut.TestBase):
    
    def setup(self):
        self.odps.execute_sql("drop table if exists " + TEST_TABLE)
        self.odps.execute_sql("create table if not exists %s (id string, count bigint, f double, b boolean, d datetime)\
                into %d shards hublifecycle 3" % (TEST_TABLE, 1))
        print 'setup'

    def test_datahub_upload(self):
        # Test upload
        client = DatahubClient(self.odps, TEST_PROJECT, TEST_TABLE, self.datahub_endpoint)
        client.load_shard(1)
        client.wait_for_shard_load(120)

        writer = client.open_datahub_writer('0')
        schema = client.get_stream_schema()
        record_pack = RecordPack(schema)
        for i in range(3):
            record = make_record(schema, i)
            record_pack.append(record)
        packid = writer.write(record_pack).get_pack_id()

        reader = client.open_datahub_reader(0, packid)
        while True:
            try:
                record = reader.read()
            except DatahubError:
                break;
            if record is None:
                break
            print record.values

        client.load_shard(0)
        #print 'wait for unload: 10 secs'
        #time.sleep(10)

    def teardown(self):
        self.odps.execute_sql("drop table if exists " + TEST_TABLE)

if __name__ == '__main__':
    unittest.main()
