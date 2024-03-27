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


def check_session_status(session_id):
    client = get_arrow_client()
    req = SessionRequest(session_id=session_id)
    resp = client.get_write_session(req)

    if resp.status != Status.OK:
        logger.info("Get write session failed")
        return

    if resp.session_status == SessionStatus.NORMAL:
        logger.info("Write session id: " + resp.session_id)
    else:
        logger.info("Session status is not expected")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    if len(sys.argv) != 2:
        raise ValueError("Please provide session id")

    session_id = sys.argv[1]
    check_session_status(session_id)
