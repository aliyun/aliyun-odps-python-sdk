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
import time
from odps.apis.storage_api import *
from util import *

logger = logging.getLogger(__name__)

def get_read_session(session_id):
    client = get_arrow_client()

    req = SessionRequest(session_id=session_id)

    while True:
        resp = client.get_read_session(req)

        if resp.status != Status.OK:
            logger.info("Get read session failed")
            return False

        if resp.session_status == SessionStatus.INIT:
            logger.info("Wait...")
            time.sleep(1)
            continue

        logger.info("Read session id: " + resp.session_id)
        break

    return True

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
    if len(sys.argv) != 2:
        raise ValueError("Please provide session id")

    session_id = sys.argv[1]
    get_read_session(session_id)