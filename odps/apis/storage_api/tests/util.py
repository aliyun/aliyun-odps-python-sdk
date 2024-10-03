# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import json
import logging
import os
import time

from ..storage_api import *
from .record_batch_generate import generate_base_table, verify_data

logger = logging.getLogger(__name__)


class TestDataItem:
    def __init__(self):
        self.read_session_id = None
        self.write_session_id = None
        self.commit_messages = []
        self.split_count = None
        self.val = None
        self.batch_size = None
        self.batch_count = None
        self.total_count = None
        self.partition = None
        self.data_columns = None
        self.test_data_format = False
        self.data_format = DataFormat()
        self.split_options = SplitOptions.get_default_options(
            SplitOptions.SplitMode.SIZE
        )

        self.has_partition = False

    def push_commit_message(self, commit_message):
        self.commit_messages.append(commit_message)

    def load_conf(self, tag):
        with open(
            os.path.dirname(os.path.abspath(__file__)) + "/data_item.conf", "r"
        ) as conf_file:
            conf = json.load(conf_file)
            tag_conf = conf[tag]
            self.val = tag_conf["val"]
            self.batch_size = tag_conf["batch_size"]
            self.batch_count = tag_conf["batch_count"]
            if tag_conf.get("partition") is not None:
                self.partition = tag_conf["partition"]

        self.total_count = self.batch_size * self.batch_count


def create_write_session(item, storage_api_client):
    req = TableBatchWriteRequest()
    if item.partition is not None:
        req.partition_spec = item.partition

    try:
        resp = storage_api_client.create_write_session(req)
    except Exception as e:
        logger.info(e)
        return False

    if resp.status != Status.OK:
        logger.info("Create write session failed")
        return False

    item.write_session_id = resp.session_id
    item.data_columns = resp.data_schema.data_columns
    if len(resp.data_schema.partition_columns) == 0:
        item.has_partition = True
    else:
        for i in range(0, len(resp.data_schema.partition_columns)):
            item.data_columns.append(resp.data_schema.partition_columns[i])

    return True


def get_write_session(item, storage_api_client):
    req = SessionRequest(session_id=item.write_session_id)

    while True:
        try:
            resp = storage_api_client.get_write_session(req)
        except Exception as e:
            logger.info(e)
            return False

        if resp.status != Status.OK:
            logger.info("get write session failed")
            return False

        if (
            resp.session_status != SessionStatus.NORMAL
            and resp.session_status != SessionStatus.COMMITTED
        ):
            logger.info("Wait...")
            time.sleep(1)
            continue

        break

    return True


def write_rows(item, storage_api_client, compression=Compression.UNCOMPRESSED):
    req = WriteRowsRequest(session_id=item.write_session_id, compression=compression)

    if item.test_data_format:
        req.data_format = item.data_format

    record_batch = generate_base_table(item)

    start = time.time()
    try:
        writer = storage_api_client.write_rows_arrow(req)

        for i in range(0, item.batch_count):
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

        end = time.time()
        logger.info("Write rows cost: " + str(end - start) + "s")
        item.push_commit_message(commit_msg)

        return True
    except Exception as e:
        logger.info(e)
        return False


def commit_write_session(item, storage_api_client):
    req = SessionRequest(session_id=item.write_session_id)

    try:
        resp = storage_api_client.commit_write_session(req, item.commit_messages)
    except Exception as e:
        logger.info(e)
        return False

    if resp.status != Status.OK and resp.status != Status.WAIT:
        logger.info("Fail to commit write session")
        return False

    if resp.status == Status.WAIT:
        return get_write_session(item, storage_api_client)

    return True


def create_read_session(item, storage_api_client):
    req = TableBatchScanRequest()
    req.split_options = item.split_options

    if item.partition is not None:
        req.required_partitions = [item.partition]

    try:
        resp = storage_api_client.create_read_session(req)
    except Exception as e:
        logger.info(e)
        return False

    if resp.status != Status.OK and resp.status != Status.WAIT:
        logger.info("create read session failed")
        return False

    item.read_session_id = resp.session_id
    item.data_columns = resp.data_schema.data_columns
    item.split_count = resp.split_count

    return True


def get_read_session(item, storage_api_client):
    req = SessionRequest(session_id=item.read_session_id)

    while True:
        try:
            resp = storage_api_client.get_read_session(req)
        except Exception as e:
            logger.info(e)
            return False

        if resp.status != Status.OK:
            logger.info("get read session failed")
            return False

        if resp.session_status == SessionStatus.INIT:
            logger.info("Wait...")
            time.sleep(1)
            continue

        item.split_count = resp.split_count
        break

    if item.split_options.split_mode == SplitOptions.SplitMode.ROW_OFFSET:
        if item.total_count != resp.record_count:
            logger.info("Split with row_offset failed, the record_count is not correct")
            return False

    return True


def read_rows(item, storage_api_client, compression=Compression.UNCOMPRESSED):
    req = ReadRowsRequest(
        session_id=item.read_session_id, max_batch_rows=4096, compression=compression
    )

    if item.test_data_format:
        req.data_format = item.data_format

    total_line = 0
    if item.split_options.split_mode == SplitOptions.SplitMode.ROW_OFFSET:
        item.split_count = 1
        req.row_index = 0
        req.row_count = item.total_count

    for i in range(0, item.split_count):
        req.split_index = i
        start = time.time()
        try:
            reader = storage_api_client.read_rows_arrow(req)
        except Exception as e:
            logger.info(e)
            return False

        while True:
            record_batch = reader.read()
            if record_batch is None:
                break
            if not verify_data(record_batch, item, total_line):
                return False

            total_line += record_batch.num_rows

        if reader.get_status() != Status.OK:
            logger.info("Read rows failed")
            return False

        if reader.get_request_id() is None:
            logger.info("Missing request id")
            return False

        end = time.time()
        logger.info("Read rows cost (index " + str(i) + "): " + str(end - start) + "s")

    if total_line != item.total_count:
        logger.info(
            "read rows number incorrect:"
            + str(total_line)
            + " != "
            + str(item.total_count)
        )
        return False

    return True
