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

def create_read_session(mode):
    client = get_arrow_client()

    req = TableBatchScanRequest(required_partitions=['pt=test_write_1'])

    if mode == "size":
        req.split_options = SplitOptions.get_default_options(SplitOptions.SplitMode.SIZE)
    elif mode == "row":
        req.split_options = SplitOptions.get_default_options(SplitOptions.SplitMode.ROW_OFFSET)

    resp = client.create_read_session(req)

    if resp.status == Status.WAIT:
        req = SessionRequest(session_id=resp.session_id)
        resp = client.get_read_session(req)

    if resp.status != Status.OK:
        logger.info("Create read session failed")
        return False

    logger.info("Read session id: " + resp.session_id)

    return True

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
    if len(sys.argv) != 2:
        raise ValueError("Please provide split mode: size|row")

    mode = sys.argv[1]
    if mode != "row" and mode != "size":
        raise ValueError("Please provide split mode: size|row")

    create_read_session(mode)