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

def commit_write_session(session_id, commit_msg):
    client = get_arrow_client()

    req = SessionRequest(session_id=session_id)

    resp = client.commit_write_session(req, [commit_msg])

    if resp.status == Status.WAIT:
        resp = client.get_write_session(req)

    if resp.session_status != SessionStatus.COMMITTED:
        logger.info("Fail to commit write session")
        return False

    logger.info("Commit write session success")
    return True

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
    if len(sys.argv) != 3:
        raise ValueError("Please provide session id and commit message\n")

    session_id = sys.argv[1]
    commit_msg = sys.argv[2]
    commit_write_session(session_id, commit_msg)