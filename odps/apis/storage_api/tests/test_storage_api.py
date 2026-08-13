# Copyright 1999-2026 Alibaba Group Holding Ltd.
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
import time

import mock
import pytest
import requests

try:
    import pyarrow as pa
except ImportError:
    pa = None

from ..storage_api import SessionRequest, StorageApiClient

if pa is not None:
    from ..storage_api import *
    from .util import *
else:
    pytestmark = pytest.mark.skip("Need pyarrow to run this test")

logger = logging.getLogger(__name__)


def _make_mock_storage_api_client():
    client = StorageApiClient.__new__(StorageApiClient)
    client._quota_name = None
    client._tags = []
    client._get_resource = mock.Mock(return_value="http://localhost/commit")
    client._fill_common_headers = mock.Mock(
        side_effect=lambda headers=None: headers or {}
    )
    client._tunnel_rest = mock.Mock()

    response = requests.Response()
    response.status_code = 201
    response._content = b"{}"
    response.headers["x-odps-request-id"] = "request-id"
    client._tunnel_rest.post.return_value = response
    return client


def _get_commit_body(client):
    return json.loads(client._tunnel_rest.post.call_args[1]["data"])


@pytest.mark.parametrize("timeout", [None, 0, 1, 600, 601])
def test_commit_write_session_wait_timeout(timeout):
    """wait_flying_writers_timeout_seconds is omitted when None and sent otherwise;
    out-of-range values pass through for the server to enforce bounds."""
    client = _make_mock_storage_api_client()

    client.commit_write_session(
        SessionRequest("session-id"),
        ["commit-message"],
        wait_flying_writers_timeout_seconds=timeout,
    )

    body = _get_commit_body(client)
    assert body["CommitMessages"] == ["commit-message"]
    if timeout is None:
        assert "WaitFlyingWritersTimeoutSeconds" not in body
    else:
        assert body["WaitFlyingWritersTimeoutSeconds"] == timeout


@pytest.mark.parametrize("timeout", [1.5, "1", True])
def test_commit_write_session_rejects_invalid_wait_timeout(timeout):
    client = _make_mock_storage_api_client()

    with pytest.raises(ValueError):
        client.commit_write_session(
            SessionRequest("session-id"),
            ["commit-message"],
            wait_flying_writers_timeout_seconds=timeout,
        )

    client._tunnel_rest.post.assert_not_called()


def test_storage_api(storage_api_client):
    req = TableBatchWriteRequest(partition_spec="pt=test_write_1")

    resp = storage_api_client.create_write_session(req)

    assert resp.status == Status.OK
    if resp.status != Status.OK:
        raise IOError("Create write session failed")

    req = SessionRequest(session_id=resp.session_id)

    while True:
        resp = storage_api_client.get_write_session(req)

        assert resp.status == Status.OK

        if resp.status != Status.OK:
            raise IOError("Get write session failed")
            return

        if (
            resp.session_status != SessionStatus.NORMAL
            and resp.session_status != SessionStatus.COMMITTED
        ):
            logger.info("Wait...")
            time.sleep(1)
            continue

        break

    req = WriteRowsRequest(session_id=resp.session_id)

    bigint_list = list(range(4096))

    record_batch = pa.RecordBatch.from_arrays(
        [
            pa.array(bigint_list),
            pa.array(bigint_list),
            pa.array(bigint_list),
            pa.array(bigint_list),
        ],
        names=["a", "b", "c", "d"],
    )
    try:
        writer = storage_api_client.write_rows_stream(req)
    except Exception as e:
        logger.info(e)
        return

    start = time.time()
    for i in range(0, 300):
        if i == 0:
            suc = writer.write(record_batch.schema.serialize().to_pybytes())
            if not suc:
                raise IOError("write arrow schema failed")

        suc = writer.write(record_batch.serialize().to_pybytes())
        if not suc:
            raise IOError("write arrow record batch failed")

    # write EOS given https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format
    suc = writer.write(b"\xff\xff\xff\xff\x00\x00\x00\x00")
    if not suc:
        raise IOError("write EOS failed")
    commit_message, suc = writer.finish()

    assert suc is True
    if not suc:
        raise IOError("Write rows failed")
    else:
        end = time.time()
        logger.info("Write rows cost: " + str(end - start) + "s")

    req = SessionRequest(session_id=resp.session_id)

    commit_messages = []
    commit_messages.append(commit_message)
    resp = storage_api_client.commit_write_session(req, commit_messages)

    if resp.status != Status.OK and resp.status != Status.WAIT:
        raise IOError("Fail to commit write session")

    if resp.status == Status.WAIT:
        req = SessionRequest(session_id=resp.session_id)
        while True:
            resp = storage_api_client.get_write_session(req)

            assert resp.status == Status.OK

            if resp.status != Status.OK:
                raise IOError("Get write session failed")

            if (
                resp.session_status != SessionStatus.NORMAL
                and resp.session_status != SessionStatus.COMMITTED
            ):
                logger.info("Wait...")
                time.sleep(1)
                continue

            break

    req = TableBatchScanRequest()

    req.required_partitions = ["pt=test_write_1"]
    req.enable_estimate_stats = True

    resp = storage_api_client.create_read_session(req)

    if resp.status != Status.OK and resp.status != Status.WAIT:
        raise IOError("create read session failed")

    req = SessionRequest(session_id=resp.session_id)

    while True:
        resp = storage_api_client.get_read_session(req)

        if resp.status != Status.OK:
            raise IOError("get read session failed")

        if resp.session_status == SessionStatus.INIT:
            logger.info("Wait...")
            time.sleep(1)
            continue

        split_count = resp.split_count
        break

    assert resp.session_stats is not None
    assert resp.session_stats.estimated_row_count is not None
    assert resp.session_stats.estimated_size is not None

    req = ReadRowsRequest(session_id=resp.session_id, max_batch_rows=4096)

    read_size = 65536
    buf = b""
    for i in range(0, split_count):
        req.split_index = i
        start = time.time()
        reader = storage_api_client.read_rows_stream(req)

        while True:
            data = reader.read(read_size)
            if len(data) == 0:
                break
            buf += data

        reader.close()
        if reader.get_status() != Status.OK:
            raise IOError("Read rows failed")

        end = time.time()
        logger.info("Read rows cost (index " + str(i) + "): " + str(end - start) + "s")

    with pa.ipc.open_stream(buf) as reader:
        schema = reader.schema
        batches = [b for b in reader]
    logger.info(schema)
    logger.info(batches[0])


def _create_ready_write_session(storage_api_client, partition_spec):
    response = storage_api_client.create_write_session(
        TableBatchWriteRequest(partition_spec=partition_spec)
    )
    assert response.status == Status.OK

    request = SessionRequest(session_id=response.session_id)
    for _ in range(60):
        response = storage_api_client.get_write_session(request)
        assert response.status == Status.OK
        if response.session_status == SessionStatus.NORMAL:
            return response.session_id
        time.sleep(1)

    raise AssertionError("Write session did not become NORMAL")


def _single_row_batch(value):
    return pa.RecordBatch.from_arrays(
        [pa.array([value]), pa.array([value]), pa.array([value]), pa.array([value])],
        names=["a", "b", "c", "d"],
    )


def test_commit_with_custom_wait_timeout(storage_api_client):
    """E2E: commit accepts wait_flying_writers_timeout_seconds and succeeds."""
    session_id = _create_ready_write_session(storage_api_client, "pt=commit_timeout")
    writer = storage_api_client.write_rows_arrow(
        WriteRowsRequest(session_id=session_id, block_number=0)
    )
    assert writer.write(_single_row_batch(1))
    commit_message, success = writer.finish()
    assert success is True

    response = storage_api_client.commit_write_session(
        SessionRequest(session_id=session_id),
        [commit_message],
        wait_flying_writers_timeout_seconds=600,
    )

    assert response.status == Status.OK
    assert response.session_status == SessionStatus.COMMITTED
