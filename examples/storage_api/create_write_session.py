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
from odps.apis.storage_api import *
from util import *

logger = logging.getLogger(__name__)

def create_write_session():
    client = get_arrow_client()

    req = TableBatchWriteRequest(partition_spec="pt=test_write_1")

    resp = client.create_write_session(req)

    if resp.status != Status.OK:
        logger.info("Create write session failed")
        return False

    logger.info(resp.session_id)

    return True

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
    create_write_session()