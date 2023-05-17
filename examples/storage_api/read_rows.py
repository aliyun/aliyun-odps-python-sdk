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

logger = logging.getLogger(__name__)

def get_read_session(session_id):
    client = get_arrow_client()

    req = SessionRequest(session_id=session_id)

    resp = client.get_read_session(req)

    return resp

def read_rows(session_id):
    client = get_arrow_client()

    req = ReadRowsRequest(session_id=session_id)

    resp = get_read_session(session_id)

    if resp.status != Status.OK and resp.status != Status.WAIT:
        return False

    if resp.split_count == -1:
        req.row_index = 0
        req.row_count = resp.record_count
    else:
        req.split_index = 0

    reader = client.read_rows_arrow(req)

    total_line = 0
    while True:
        record_batch = reader.read()
        if record_batch is None:
            break

        total_line += record_batch.num_rows

    if reader.get_status() != Status.OK:
        logger.info("Read rows failed")
        return False

    if reader.get_request_id() is None:
        return False

    logger.info("Total line is:" + str(total_line))
    return True

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
    if len(sys.argv) != 2:
        raise ValueError("Please provide session id")

    session_id = sys.argv[1]
    read_rows(session_id)