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

import os
import datetime
import time
import unittest

from odps.models import Schema, Record

CLIENT_READ_PIPE_NUM = 2
CLIENT_WRITE_PIPE_NUM = 2

channel_client = None
channel_server = None


@unittest.skipIf('TEST_SANDBOX' not in os.environ, 'Skipped outside sandbox')
class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        global channel_client

        from cupid.runtime.ctypes_libs import Subprocess_Container_Init, Subprocess_StartFDReceiver
        from cupid.runtime.ctypes_libs import ChannelSlaveClient

        super(Test, self).__init__(*args, **kwargs)

        Subprocess_Container_Init()
        Subprocess_StartFDReceiver()

        if channel_client is None:
            channel_client = ChannelSlaveClient(CLIENT_WRITE_PIPE_NUM, CLIENT_READ_PIPE_NUM, 'channel_slave_client')
            channel_client.start()
        self.client = channel_client

    def testInterfaces(self):
        resp = self.client.sync_call('test', b'111')
        self.assertEqual(b'111_test', resp)

        ri = self.client.create_file_reader('test', b'222')
        self.assertEqual(ri.read(), b'222_test')
        self.assertEqual(ri.result(), b'222_test')
        del ri

        ro = self.client.create_file_writer('test', b'333')
        ro.write(b'444')
        ro.flush()
        self.assertEqual(ro.result(), b'333_444_test')
        del ro

    def testTableIO(self):
        schema = Schema.from_lists(['key', 'value', 'double', 'datetime', 'boolean'],
                                   ['bigint', 'string', 'double', 'datetime', 'boolean'])
        label = self.client.sync_call('test', 'write_label')
        print('Write label: ' + label)
        writer = self.client.create_record_writer(label, schema)
        cur_time = datetime.datetime.now().replace(microsecond=0)
        rec = Record(schema=schema, values=[10, 'abcd', 1.56, cur_time, False])
        for _ in range(10):
            writer.write(rec)
        writer.close()

        time.sleep(3)

        label = channel_client.sync_call('test', 'read_label')
        print('Read label: ' + label)
        reader = channel_client.create_record_reader(label, schema)
        records = list(reader)
        self.assertListEqual(records, [rec] * 20)
        reader.close()


if __name__ == '__main__':
    runner = unittest.main(exit=False)
    os._exit(not runner.result.wasSuccessful())
