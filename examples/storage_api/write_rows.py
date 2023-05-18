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
import sys
from odps.apis.storage_api import *
from util import *
try:
    import pyarrow as pa
except ImportError:
    pa = None

logger = logging.getLogger(__name__)

def write_rows(session_id, batch_count):
    client = get_arrow_client()

    req = WriteRowsRequest(session_id=session_id)

    pylist = list(range(100))

    record_batch = pa.RecordBatch.from_arrays([pa.array(pylist), pa.array(pylist), pa.array(pylist), pa.array(pylist)], ['a', 'b', 'c', 'd'])

    writer = client.write_rows_arrow(req)

    for i in range(0, batch_count):
        suc = writer.write(record_batch)
        if not suc:
            break

    commit_msg, suc = writer.finish()

    if not suc:
        logger.info("Write rows failed")
        return False

    if writer.get_status() != Status.OK:
        logger.info("Write rows failed")
        return False

    if writer.get_request_id() is None:
        logger.info("Missing request id")
        return False

    logger.info(commit_msg)
    return True

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
    if len(sys.argv) != 3:
        raise ValueError("Please provide session id and number of batch to write. Note: one batch contains 100 rows\n")

    session_id = sys.argv[1]
    batch_count = int(sys.argv[2])
    write_rows(session_id, batch_count)