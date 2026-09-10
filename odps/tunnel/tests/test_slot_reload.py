#!/usr/bin/env python
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

"""Unit tests for tunnel slot-routing refresh and retry."""

import gc
import json
import threading
import time
import weakref

import mock
import pytest
import requests

from ...errors import BaseODPSError
from ...models import TableSchema
from ..errors import TunnelError
from ..io.writer import Upsert
from ..retry import (
    BAD_GATEWAY,
    GATEWAY_TIMEOUT,
    SLOT_REASSIGNMENT,
    ExponentialWaitRetryPolicy,
    InfiniteExponentialWaitRetryPolicy,
    NoRetryPolicy,
    TunnelRetryHandler,
)
from ..tabletunnel import (
    Slot,
    TableDownloadSession,
    TableStreamUploadSession,
    TableUploadSession,
    TableUpsertSession,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    real_sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    yield real_sleep


# --- Slot -------------------------------------------------------------------


@pytest.mark.parametrize("server", ["invalid-no-colon", ":8000", ""])
def test_slot_invalid_raises(server):
    with pytest.raises(TunnelError):
        Slot("0", server)


def test_slot_equals():
    # Import locally to avoid class inequality caused by reimport. DO NOT HOIST.
    from ..tabletunnel import Slot

    a = Slot("0", "1.2.3.4:8000")
    assert a == Slot("0", "1.2.3.4:8000")
    assert a != Slot("1", "1.2.3.4:8000")
    assert a != "not-a-slot"


# --- TunnelRetryHandler policies --------------------------------------------


@pytest.mark.parametrize(
    "status_code,expected_type,first_non_retriable",
    [
        (SLOT_REASSIGNMENT, InfiniteExponentialWaitRetryPolicy, None),
        (429, InfiniteExponentialWaitRetryPolicy, None),
        (502, ExponentialWaitRetryPolicy, 8),
        (400, NoRetryPolicy, 1),
    ],
)
def test_retry_policy_mapping(status_code, expected_type, first_non_retriable):
    policy = TunnelRetryHandler().get_retry_policy(status_code)
    assert isinstance(policy, expected_type)
    if first_non_retriable is not None:
        assert not policy.should_retry(Exception("x"), first_non_retriable)


@pytest.mark.parametrize(
    "policy_cls,expected",
    [
        (InfiniteExponentialWaitRetryPolicy, [1, 2, 4, 8, 16, 32, 64, 64, 64]),
        (ExponentialWaitRetryPolicy, [1, 2, 4, 8, 16, 32, 64]),
    ],
)
def test_backoff_sequence(policy_cls, expected):
    policy = policy_cls()
    for i, exp in enumerate(expected, start=1):
        assert policy.get_retry_wait_time(i) == exp


@pytest.mark.parametrize(
    "status_code,expected", [(SLOT_REASSIGNMENT, 5), (502, 8), (400, 1)]
)
def test_execute_with_retry(status_code, expected):
    """308 retries infinitely; 5xx exhausts at 8; 4xx propagates."""
    handler = TunnelRetryHandler()
    calls = {"n": 0}

    def action():
        calls["n"] += 1
        if status_code == SLOT_REASSIGNMENT and calls["n"] >= 5:
            return "ok"
        raise BaseODPSError("boom", status_code=status_code)

    if status_code == SLOT_REASSIGNMENT:
        assert handler.execute_with_retry(action) == "ok"
    else:
        with pytest.raises(BaseODPSError):
            handler.execute_with_retry(action)
    assert calls["n"] == expected


def test_execute_with_retry_on_error_status_and_logger():
    seen, logs, calls = [], [], {"n": 0}
    handler = TunnelRetryHandler(
        retry_logger=lambda exc, attempt, wait_ms: logs.append((attempt, wait_ms))
    )

    def action():
        calls["n"] += 1
        if calls["n"] == 1:
            raise BaseODPSError("bg", status_code=429)
        return "ok"

    assert handler.execute_with_retry(action, on_error_status=seen.append) == "ok"
    assert seen == [429]
    assert logs == [(1, 1000)]


@pytest.mark.parametrize(
    "callback,expected",
    [
        (lambda n: (lambda e: n["n"] >= 2), 2),
        (lambda n: (lambda e: True), 1),
    ],
)
def test_retry_handler_on_exception(callback, expected):
    """False continues retry; True raises immediately."""
    handler = TunnelRetryHandler()
    calls = {"n": 0}

    def action():
        calls["n"] += 1
        raise BaseODPSError("bad", status_code=400)

    with pytest.raises(BaseODPSError):
        handler.execute_with_retry(action, on_exception=callback(calls))
    assert calls["n"] == expected


# --- Stream-upload reload ---------------------------------------------------


def _make_reload_resp(slots_data, status="normal", last_batch_id=None):
    body = {
        "session_name": "test-session-id",
        "status": status,
        "slots": [list(s) for s in slots_data],
    }
    if last_batch_id is not None:
        body["last_batch_id"] = str(last_batch_id)
    resp = requests.Response()
    resp.status_code = 200
    resp._content = json.dumps(body).encode()
    return resp


def _init_base_session(sess):
    sess._parent = None
    sess._client = mock.MagicMock()
    sess._table = mock.MagicMock()
    sess._partition_spec = ""
    sess._tags = None
    sess._compress_option = None
    sess._quota_name = None


def _make_stream_session(slots_data=(("0", "1.1.1.1:1000"),)):
    sess = TableStreamUploadSession.__new__(TableStreamUploadSession)
    _init_base_session(sess)
    sess._create_partition = False
    sess._zorder_columns = None
    sess._allow_schema_mismatch = True
    sess.schema_version = None
    sess._schema_version_reloader = None
    sess._slot_num = 0
    sess._dynamic_partition = False
    sess.id = "test-session-id"
    sess.schema = None
    sess.status = "normal"
    sess.quota_name = None
    sess.last_batch_id = None
    sess.last_batch_commit_time = None
    sess.slots = TableStreamUploadSession.Slots(list(slots_data))
    sess._reloading = threading.Lock()
    sess._last_reload_time = 0.0
    sess._reload_throttle = 30
    return sess


@pytest.mark.parametrize(
    "force,throttled,expect_get",
    [(True, True, True), (False, True, False), (False, False, True)],
)
def test_reload_throttle_and_force(monkeypatch, force, throttled, expect_get):
    sess = _make_stream_session()
    gets = []

    def spy_get(*a, **kw):
        gets.append(1)
        return _make_reload_resp((("0", "2.2.2.2:2000"),))

    monkeypatch.setattr(sess._client, "get", spy_get)
    if throttled:
        sess._last_reload_time = time.monotonic()
    sess.reload(force=force)
    assert len(gets) == (1 if expect_get else 0)


def test_reload_readonly_sets_param(monkeypatch):
    sess = _make_stream_session()
    captured = {}

    def spy_get(url, *a, **kw):
        captured["params"] = kw.get("params", {})
        return _make_reload_resp((("0", "1.1.1.1:1000"),))

    monkeypatch.setattr(sess._client, "get", spy_get)
    sess.reload(force=True, readonly=True)
    assert captured["params"].get("read_only") == "true"


def test_reload_non_force_concurrent_dedup(monkeypatch):
    sess = _make_stream_session()
    barrier = threading.Event()
    gets = []

    def slow_get(*a, **kw):
        gets.append(1)
        barrier.wait(timeout=2)
        return _make_reload_resp((("0", "2.2.2.2:2000"),))

    monkeypatch.setattr(sess._client, "get", slow_get)
    results = []
    worker = lambda: results.append(sess.reload(force=False))
    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    time.sleep(0.05)
    t2.start()
    time.sleep(0.1)
    barrier.set()
    t1.join(timeout=2)
    t2.join(timeout=2)
    assert len(gets) == 1


@pytest.mark.parametrize(
    "response_slots,slot_num,new_server,expected_gets,expected_server,expected_count",
    [
        # Slot-count change -> force full reload.
        (
            (("0", "1.1.1.1:1000"), ("1", "2.2.2.2:2000")),
            2,
            "1.1.1.1:1000",
            1,
            "1.1.1.1:1000",
            2,
        ),
        # Same count, routed server changed -> local update, no GET.
        ((("0", "1.1.1.1:1000"),), 1, "9.9.9.9:9999", 0, "9.9.9.9:9999", 1),
        # No change -> noop.
        ((("0", "1.1.1.1:1000"),), 1, "1.1.1.1:1000", 0, "1.1.1.1:1000", 1),
    ],
)
def test_reload_slots(
    monkeypatch,
    response_slots,
    slot_num,
    new_server,
    expected_gets,
    expected_server,
    expected_count,
):
    sess = _make_stream_session((("0", "1.1.1.1:1000"),))
    gets = []

    def spy_get(url, *a, **kw):
        gets.append(1)
        return _make_reload_resp(response_slots)

    monkeypatch.setattr(sess._client, "get", spy_get)
    slot = sess.slots._slots[0]
    sess.reload_slots(slot, new_server, slot_num)
    assert len(gets) == expected_gets
    assert slot.server == expected_server
    assert len(sess.slots) == expected_count


def test_get_last_batch_id_uses_readonly_reload(monkeypatch):
    sess = _make_stream_session()
    captured = {}

    def spy_get(url, *a, **kw):
        captured["params"] = kw.get("params", {})
        return _make_reload_resp((("0", "1.1.1.1:1000"),), last_batch_id=42)

    monkeypatch.setattr(sess._client, "get", spy_get)
    sess._reload_throttle = 0
    assert sess.get_last_batch_id() == 42
    assert captured["params"].get("read_only") == "true"


# --- Upsert update_buckets / keepalive --------------------------------------


def _make_upsert_session(slots_data=(("0", "1.1.1.1:1000", [0, 1]),)):
    sess = TableUpsertSession.__new__(TableUpsertSession)
    _init_base_session(sess)
    sess._slot_num = 1
    sess._commit_timeout = 120
    sess._quota_name = None
    sess._lifecycle = None
    sess.id = "test-upsert-id"
    sess.schema = None
    sess.quota_name = None
    sess.hash_keys = []
    sess.hasher = "java"
    sess.support_partial_update = False
    sess._buckets_lock = threading.RLock()
    sess._keepalive_interval = 30
    sess._keepalive_scheduler = None
    sess._keepalive_lock = threading.Lock()
    sess._keepalive_stopped = False
    sess._retry_handler = TunnelRetryHandler()
    slot_elements = [
        {"slot_id": s[0], "worker_addr": s[1], "buckets": s[2]} for s in slots_data
    ]
    sess.slots = TableUpsertSession.Slots(slot_elements)
    sess.status = TableUpsertSession.Status.Normal
    return sess


def _make_upsert_reload_resp(slots_data, status="NORMAL"):
    body = {
        "id": "test-upsert-id",
        "schema": {"columns": [{"name": "k", "type": "string"}], "partitionKeys": []},
        "hash_key": [],
        "hasher": "java",
        "status": status,
        "slots": [
            {"slot_id": s[0], "worker_addr": s[1], "buckets": s[2]} for s in slots_data
        ],
    }
    resp = requests.Response()
    resp.status_code = 200
    resp._content = json.dumps(body).encode()
    return resp


def _upsert_spy(slots_data, status="NORMAL"):
    """Return (spy_get, counter) that serves upsert reload responses."""
    counter = {"n": 0}

    def spy(url, *a, **kw):
        counter["n"] += 1
        return _make_upsert_reload_resp(slots_data, status=status)

    return spy, counter


@pytest.mark.parametrize(
    "new_server,expect_reload", [("2.2.2.2:2000", False), (None, True)]
)
def test_upsert_update_buckets(monkeypatch, new_server, expect_reload):
    sess = _make_upsert_session(
        (("0", "1.1.1.1:1000", [0]), ("1", "2.2.2.2:2000", [1]))
    )
    spy_get, reloaded = _upsert_spy((("0", "7.7.7.7:7000", [0, 1]),))
    monkeypatch.setattr(sess._client, "get", spy_get)
    monkeypatch.setattr(sess._client, "is_ok", lambda r: True)
    sess.update_buckets(0, new_server)
    assert reloaded["n"] == (1 if expect_reload else 0)
    if expect_reload:
        assert sess.buckets[0].server == "7.7.7.7:7000"


def test_upsert_get_buckets_returns_copy():
    sess = _make_upsert_session()
    snap = sess.get_buckets()
    snap[0] = "mutated"
    assert sess.buckets[0] is not None


@pytest.mark.parametrize("scenario", ["self_stop", "close", "gc"])
def test_upsert_keepalive_stops(monkeypatch, _no_sleep, scenario):
    sess = _make_upsert_session()
    status = "EXPIRED" if scenario == "self_stop" else "NORMAL"
    spy_get, counter = _upsert_spy((("0", "1.1.1.1:1000", [0, 1]),), status=status)
    monkeypatch.setattr(sess._client, "get", spy_get)
    monkeypatch.setattr(sess._client, "is_ok", lambda r: True)

    sess._keepalive_interval = 1 if scenario == "self_stop" else 30
    sess._start_keepalive()
    assert sess._keepalive_scheduler is not None

    if scenario == "self_stop":
        _no_sleep(1.2)
        assert sess._keepalive_scheduler is None
        assert counter["n"] >= 1
    elif scenario == "close":
        sess.close()
        assert sess._keepalive_scheduler is None
        assert sess._keepalive_stopped is True
        sess._start_keepalive()
        assert sess._keepalive_scheduler is None
        n = counter["n"]
        _no_sleep(0.3)
        assert counter["n"] == n
    else:  # gc
        ref = weakref.ref(sess)
        del sess
        gc.collect()
        assert ref() is None


# --- Upsert flush 308 / 502 / 504 handling ----------------------------------


def _make_upsert_stream_for_flush(sess):
    schema = TableSchema.from_lists(["k"], ["string"])
    sess.schema = schema

    def fake_build_bucket_writer(self, slot):
        self._bucket_buffers[slot] = mock.MagicMock(getvalue=lambda: b"data")
        self._bucket_writers[slot] = mock.MagicMock(n_bytes=100, count=1)

    # Construct through the public API so __init__ attribute changes are
    # tracked automatically; patch only the bucket-writer I/O seam and
    # the hasher (sess.hasher='java' isn't a valid RecordHasher input).
    with mock.patch.object(Upsert, "_build_bucket_writer", fake_build_bucket_writer):
        with mock.patch("odps.tunnel.io.writer.RecordHasher", mock.MagicMock()):
            return Upsert(schema, lambda *a, **kw: None, sess)


def _make_tunnel_error(status_code, routed_server=None):
    headers = {}
    if routed_server is not None:
        headers["odps-tunnel-routed-server"] = routed_server
    return BaseODPSError(
        f"err {status_code}", status_code=status_code, response_headers=headers
    )


@pytest.mark.parametrize(
    "routed_server,reload_slots,expected_server",
    [
        ("9.9.9.9:9000", (("0", "1.1.1.1:1000", [0]),), "9.9.9.9:9000"),
        (None, (("0", "8.8.8.8:8000", [0]),), "8.8.8.8:8000"),
    ],
)
def test_upsert_flush_308(monkeypatch, routed_server, reload_slots, expected_server):
    sess = _make_upsert_session((("0", "1.1.1.1:1000", [0]),))
    spy_get, reloaded = _upsert_spy(reload_slots)
    monkeypatch.setattr(sess._client, "get", spy_get)
    monkeypatch.setattr(sess._client, "is_ok", lambda r: True)

    stream = _make_upsert_stream_for_flush(sess)
    calls = {"n": 0}

    def callback(bucket_id, slot, rec_count, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_tunnel_error(SLOT_REASSIGNMENT, routed_server=routed_server)
        assert slot.server == expected_server

    stream._request_callback = callback
    stream.flush(flush_all=True)
    assert calls["n"] == 2
    assert sess.buckets[0].server == expected_server
    assert reloaded["n"] == (0 if routed_server else 1)


@pytest.mark.parametrize(
    "status_code,always_raise,expected,expect_reload",
    [
        (BAD_GATEWAY, False, 3, 2),  # 502 triggers reload per failure, succeeds on 3rd
        (503, True, 8, 0),  # 5xx exhausts at 8 calls
        (400, True, 1, 0),  # 4xx propagates immediately
    ],
)
def test_upsert_flush_retry_and_exhaust(
    monkeypatch, status_code, always_raise, expected, expect_reload
):
    sess = _make_upsert_session((("0", "1.1.1.1:1000", [0]),))
    spy_get, reloaded = _upsert_spy((("0", "6.6.6.6:6000", [0]),))
    monkeypatch.setattr(sess._client, "get", spy_get)
    monkeypatch.setattr(sess._client, "is_ok", lambda r: True)

    stream = _make_upsert_stream_for_flush(sess)
    calls = {"n": 0}

    def callback(bucket_id, slot, rec_count, data):
        calls["n"] += 1
        if always_raise or calls["n"] < 3:
            raise _make_tunnel_error(status_code)
        # 502 triggers a reload; subsequent attempt sees the new server.
        assert slot.server == "6.6.6.6:6000"

    stream._request_callback = callback
    if always_raise:
        with pytest.raises(BaseODPSError):
            stream.flush(flush_all=True)
    else:
        stream.flush(flush_all=True)
    assert calls["n"] == expected
    assert reloaded["n"] == expect_reload


def test_upsert_flush_slot_map_changed_raises():
    sess = _make_upsert_session()
    stream = _make_upsert_stream_for_flush(sess)
    stream._request_callback = lambda *a, **kw: None
    # Extra bucket writer with no matching session bucket trips the size check.
    stream._bucket_writers[99] = mock.MagicMock(n_bytes=100, count=1)
    with pytest.raises(TunnelError, match="session slot map is changed"):
        stream.flush(flush_all=True)


@pytest.mark.parametrize(
    "header,expected",
    [
        ("odps-tunnel-routed-server", "1.1.1.1:1000"),
        ("Content-Length", len(b"payload")),
    ],
)
def test_upsert_upload_block_sends_routing_headers(header, expected):
    """upload_block must set odps-tunnel-routed-server and Content-Length."""
    sess = _make_upsert_session((("0", "1.1.1.1:1000", [0]),))
    sess.schema = TableSchema.from_lists(["k"], ["string"])
    sess.hasher = "default"
    sess._compress_option = None
    captured = {}

    def fake_put(url, data=None, params=None, headers=None):
        captured.update(headers=headers, params=params, data=data)
        resp = requests.Response()
        resp.status_code = 200
        return resp

    sess._client.put = fake_put
    sess._client.is_ok = lambda r: True

    with mock.patch("odps.tunnel.io.writer.RecordHasher", mock.MagicMock()):
        stream = sess.open_upsert_stream(compress=False)
    stream._request_callback(0, sess.buckets[0], 1, b"payload")
    assert captured["headers"][header] == expected


# --- Session retry-handler adoption -----------------------------------------


def _make_upload_session():
    sess = TableUploadSession.__new__(TableUploadSession)
    _init_base_session(sess)
    sess._table.table_resource.return_value = "http://tunnel/test/table"
    sess._create_partition = False
    sess._overwrite = False
    sess.id = "test-upload-id"
    sess.schema = None
    sess.status = "NORMAL"
    sess.quota_name = None
    sess.blocks = []
    sess.max_field_size = None
    return sess


def _make_download_session():
    sess = TableDownloadSession.__new__(TableDownloadSession)
    _init_base_session(sess)
    sess._table.table_resource.return_value = "http://tunnel/test/table"
    sess.id = "test-download-id"
    sess.schema = None
    sess.status = TableDownloadSession.Status.Normal
    sess.count = 0
    sess.quota_name = None
    sess.support_read_by_raw_size = False
    return sess


def _make_ok_resp(content=b"{}"):
    resp = requests.Response()
    resp.status_code = 200
    resp._content = content
    resp.headers["x-odps-request-id"] = "fake-req"
    return resp


@pytest.mark.parametrize(
    "make_session,verb,invoke",
    [
        (
            _make_upload_session,
            "post",
            lambda s: s._create_or_reload_session(reload=False),
        ),
        (_make_download_session, "get", lambda s: s.reload()),
    ],
)
@pytest.mark.parametrize(
    "status_code,fail_n,expected",
    [(502, 1, 2), (503, 2, 3), (400, 0, 1)],
)
def test_session_retry_adoption(
    monkeypatch, make_session, verb, invoke, status_code, fail_n, expected
):
    """Upload/download sessions retry 5xx via retry_handler; 4xx propagates."""
    sess = make_session()
    calls = {"n": 0}

    def fake_call(*args, **kwargs):
        calls["n"] += 1
        if status_code == 400 or calls["n"] <= fail_n:
            raise BaseODPSError("boom", status_code=status_code)
        return _make_ok_resp(
            b'{"UploadID":"test-upload-id","Status":"NORMAL","Schema":{}}'
        )

    monkeypatch.setattr(sess._client, verb, fake_call)
    if status_code == 400:
        with pytest.raises(BaseODPSError):
            invoke(sess)
    else:
        invoke(sess)
    assert calls["n"] == expected


def test_download_read_uses_retry_handler(monkeypatch):
    sess = _make_download_session()
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BaseODPSError("boom", status_code=502)
        resp = _make_ok_resp(b"")
        resp.headers["Content-Encoding"] = ""
        return resp

    monkeypatch.setattr(sess._client, "get", fake_get)
    stream = sess._build_input_stream(0, 10)
    assert calls["n"] == 2
    assert stream is not None


# --- Stream-upload 502/504 reload reroutes PUT -------------------------------


@pytest.mark.parametrize("status_code", [BAD_GATEWAY, GATEWAY_TIMEOUT])
def test_stream_upload_502_reloads_and_reroutes(monkeypatch, status_code):
    """502/504 forces a reload; the retried PUT carries the new routed-server."""
    sess = _make_stream_session((("0", "1.1.1.1:1000"),))
    sess.schema = TableSchema.from_lists(["k"], ["string"])
    put_headers = []

    def fake_put(url, data=None, params=None, headers=None):
        put_headers.append(dict(headers))
        if len(put_headers) == 1:
            raise BaseODPSError("boom", status_code=status_code)
        resp = _make_ok_resp(b"{}")
        resp.headers["odps-tunnel-routed-server"] = "9.9.9.9:9000"
        resp.headers["odps-tunnel-slot-num"] = "1"
        return resp

    monkeypatch.setattr(sess._client, "put", fake_put)
    monkeypatch.setattr(
        sess._client,
        "get",
        lambda *a, **kw: _make_reload_resp((("0", "9.9.9.9:9000"),)),
    )
    monkeypatch.setattr(sess._client, "is_ok", lambda r: True)

    writer = sess._open_writer()
    writer._request_callback(b"data")
    assert len(put_headers) == 2
    assert put_headers[0]["odps-tunnel-routed-server"] == "1.1.1.1:1000"
    assert put_headers[1]["odps-tunnel-routed-server"] == "9.9.9.9:9000"
