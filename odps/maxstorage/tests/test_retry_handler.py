# -*- coding: utf-8 -*-
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

"""Unit tests for TunnelRetryHandler adoption in MaxStorage StorageStub."""

import time

import mock
import pytest
import requests

from ...config import options
from ...errors import BaseODPSError
from ...tunnel.retry import (
    REQUEST_TIMEOUT,
    RETRY_INDEX_HEADER,
    RETRY_TRACE_ID_HEADER,
    RetryContext,
    TunnelRetryHandler,
)
from ..models.identifier import InstanceIdentifier, TableIdentifier
from ..models.requests import (
    BatchCompatibleCreateSessionRequest,
    CreateTableReadSessionRequest,
)
from ..stub import StorageStub


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)


def _make_stub():
    rest = mock.MagicMock()
    rest.endpoint = "http://tunnel"
    return StorageStub(rest, "2"), rest


def _make_ok_resp(content=b"{}"):
    resp = requests.Response()
    resp.status_code = 200
    resp._content = content
    resp.headers["x-odps-request-id"] = "fake-req"
    return resp


def _make_tid():
    return TableIdentifier("proj", "table", "schema")


def _flaky(body, fail_n, status_code=502, capture=None):
    """Fake REST method failing *fail_n* times (None = always) then returning OK."""
    state = {"n": 0}

    def fake(*args, **kwargs):
        state["n"] += 1
        if capture is not None:
            capture.append(kwargs.get("headers"))
        if fail_n is None or state["n"] <= fail_n:
            raise BaseODPSError("boom", status_code=status_code)
        return _make_ok_resp(body)

    return fake, state


# ---------------------------------------------------------------------------
# Retry semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status_code,fail_n,expected",
    [
        (502, 2, 3),  # 5xx retried, succeeds on 3rd
        (429, 9, 10),  # 429 infinite, succeeds on 10th
        (REQUEST_TIMEOUT, 1, 2),  # 408 retried
        (503, 7, 8),  # 5xx exhausts at 8 total calls
        (400, 0, 1),  # other 4xx: no retry
    ],
)
def test_stub_post_retry(status_code, fail_n, expected):
    """create_table_read_session retries/propagates per tunnel retry policy."""
    stub, rest = _make_stub()
    body = b'{"SessionId":"s1","Status":"COMMITTED","Schema":{}}'
    rest.post, calls = _flaky(body, None if status_code == 400 else fail_n, status_code)
    if status_code == 400:
        with pytest.raises(BaseODPSError):
            stub.create_table_read_session(_make_tid(), CreateTableReadSessionRequest())
    else:
        stub.create_table_read_session(_make_tid(), CreateTableReadSessionRequest())
    assert calls["n"] == expected


@pytest.mark.parametrize(
    "ok_body,invoke",
    [
        (
            b'{"StreamId":"st1","StreamVersion":1}',
            lambda s, tid: s.write_table(
                tid, "s1", "st1", 1, 10, b"arrow-data", "route-token"
            ),
        ),
        (b"", lambda s, tid: s.read_blobs(["ref1"])),
    ],
)
def test_stub_stream_call_retries(ok_body, invoke):
    """write_table / read_blobs (streaming bodies) retry on 502."""
    stub, rest = _make_stub()
    rest.post, calls = _flaky(ok_body, 1)
    invoke(stub, _make_tid())
    assert calls["n"] == 2


def test_stub_get_retries():
    """get_instance_read_session (GET) retries on 503."""
    stub, rest = _make_stub()
    body = b'{"SessionId":"s1","Status":"COMMITTED"}'
    rest.get, calls = _flaky(body, 2, status_code=503)
    stub.get_instance_read_session(InstanceIdentifier("proj", "inst"), "s1")
    assert calls["n"] == 3


@pytest.mark.parametrize("method", ["table_write_blob", "table_batch_write_blob"])
def test_stub_blob_write_bypasses_retry(method):
    """Blob writes bypass retry (streaming generators not replayable) but inject headers."""
    stub, rest = _make_stub()
    rest.post, calls = _flaky(b"", 1, status_code=502)
    with pytest.raises(BaseODPSError):
        getattr(stub, method)(_make_tid(), {}, b"data")
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "status_code,fail_n,invoke",
    [
        (
            502,
            1,
            lambda s, tid: s.create_batch_compatible_session(
                tid, BatchCompatibleCreateSessionRequest()
            ),
        ),
        (
            429,
            2,
            lambda s, tid: s.write_batch_compatible_block(
                tid, "s1", 0, 0, b"arrow", None, None
            ),
        ),
        (
            503,
            1,
            lambda s, tid: s.commit_batch_compatible_session(tid, "s1", None, []),
        ),
    ],
)
def test_stub_batch_compatible_retries(status_code, fail_n, invoke):
    stub, rest = _make_stub()
    rest.post, calls = _flaky(
        b'{"SessionId":"s1","Status":"COMMITTED"}', fail_n, status_code
    )
    invoke(stub, _make_tid())
    assert calls["n"] == fail_n + 1


# ---------------------------------------------------------------------------
# RetryContext and header injection
# ---------------------------------------------------------------------------


def test_retry_context_sequence_and_headers():
    """Context shares one trace id with increasing indexes; injects headers."""
    root = RetryContext("trace-1", 0)
    contexts = [root] + [root.next() for _ in range(2)]
    headers = {}
    contexts[-1].inject_headers(headers)
    assert len({c.trace_id for c in contexts}) == 1
    assert [c.retry_index for c in contexts] == [0, 1, 2]
    assert headers[RETRY_TRACE_ID_HEADER] == "trace-1"
    assert headers[RETRY_INDEX_HEADER] == "2"


@pytest.mark.parametrize(
    "trace_id,retry_index", [(None, 0), ("", 0), (" \t", 0), ("trace", -1)]
)
def test_retry_context_rejects_invalid_identity(trace_id, retry_index):
    with pytest.raises(ValueError):
        RetryContext(trace_id, retry_index)


def test_handler_ctx_stable_trace_and_increasing_index():
    """execute_with_retry_ctx shares one trace id with increasing indexes."""
    handler = TunnelRetryHandler()
    seen = []
    calls = {"n": 0}

    def action(ctx):
        seen.append((ctx.trace_id, ctx.retry_index))
        calls["n"] += 1
        if calls["n"] < 3:
            raise BaseODPSError("boom", status_code=503)
        return "ok"

    assert handler.execute_with_retry_ctx(action) == "ok"
    assert len({trace for trace, _ in seen}) == 1
    assert [idx for _, idx in seen] == [0, 1, 2]


def test_stub_injects_retry_headers_per_attempt():
    """create_table_read_session stamps trace id + increasing index; caller headers untouched."""
    stub, rest = _make_stub()
    seen_headers = []
    rest.post, calls = _flaky(
        b'{"SessionId":"s1","Schema":{}}', 2, status_code=503, capture=seen_headers
    )
    stub.create_table_read_session(_make_tid(), CreateTableReadSessionRequest())

    assert calls["n"] == 3
    traces = {h.get(RETRY_TRACE_ID_HEADER) for h in seen_headers if h is not None}
    assert len(traces) == 1
    assert [h.get(RETRY_INDEX_HEADER) for h in seen_headers] == ["0", "1", "2"]


@pytest.mark.parametrize("method", ["table_write_blob", "table_batch_write_blob"])
def test_stub_blob_write_injects_fresh_context_headers(method):
    """Blob writes carry a fresh trace id + index 0 (single attempt)."""
    stub, rest = _make_stub()
    seen_headers = []
    rest.post, _ = _flaky(b"", 0, capture=seen_headers)
    getattr(stub, method)(_make_tid(), {}, b"data")

    assert len(seen_headers) == 1
    headers = seen_headers[0]
    assert headers.get(RETRY_TRACE_ID_HEADER)
    assert headers.get(RETRY_INDEX_HEADER) == "0"


def test_stub_retries_non_status_exception_via_options_policy():
    """Non-status exceptions (network IO) retry per options.retry_times."""
    stub, rest = _make_stub()
    calls = {"n": 0}

    def fake(*args, **kwargs):
        calls["n"] += 1
        raise requests.ConnectionError("network down")

    rest.post = fake
    with pytest.raises(requests.ConnectionError):
        stub.create_table_read_session(_make_tid(), CreateTableReadSessionRequest())
    assert calls["n"] == options.retry_times + 1
