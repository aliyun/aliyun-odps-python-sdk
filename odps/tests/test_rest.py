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

from unittest import mock

import pytest
from requests.exceptions import ConnectTimeout as RequestsConnectTimeout

from .. import errors, rest
from ..accounts import AliyunAccount


def _make_client(**kwargs):
    account = kwargs.pop("account", None) or AliyunAccount("id", "key")
    kwargs.setdefault("endpoint", "http://localhost")
    return rest.RestClient(account, **kwargs)


def _fake_response(ok=True, status_code=200):
    return mock.MagicMock(ok=ok, status_code=status_code, headers={}, content=b"")


def _capture_request(client, url="http://localhost/test", method="GET", **kwargs):
    captured = {}

    def fake_send(prepared_req, **kw):
        captured["headers"] = prepared_req.headers
        captured["url"] = prepared_req.url
        return _fake_response()

    with mock.patch.object(client.session, "send", side_effect=fake_send):
        client._request(url, method, **kwargs)
    return captured


@pytest.fixture
def fresh_user_agent(monkeypatch):
    monkeypatch.delenv("MC_PLATFORM_ID", raising=False)
    monkeypatch.setattr(rest, "_default_user_agent", None)
    yield
    monkeypatch.setattr(rest, "_default_user_agent", None)


@pytest.mark.parametrize(
    "platform_id, has_fragment",
    [("qwen-code", True), (None, False)],
)
def test_user_agent_platform_id_and_headers(
    monkeypatch, fresh_user_agent, platform_id, has_fragment
):
    if platform_id is None:
        monkeypatch.delenv("MC_PLATFORM_ID", raising=False)
    else:
        monkeypatch.setenv("MC_PLATFORM_ID", platform_id)
    ua = rest.default_user_agent()
    assert ("Platform:qwen-code" in ua) is has_fragment
    if not has_fragment:
        return
    captured = _capture_request(_make_client())
    assert captured["headers"]["User-Agent"] == _make_client()._user_agent
    assert captured["headers"]["x-odps-user-agent"] == _make_client()._user_agent


def test_default_user_agent_cached(fresh_user_agent):
    assert rest.default_user_agent() is rest.default_user_agent()


@pytest.mark.parametrize(
    "kwargs, attr, expected",
    [
        ({"endpoint": "http://h/"}, "endpoint", "http://h"),
        (
            {"proxy": "http://p:80"},
            "_proxy",
            {"http": "http://p:80", "https": "http://p:80"},
        ),
        ({"user_agent": "ua"}, "_user_agent", "ua"),
        ({"namespace": "ns"}, "namespace", "ns"),
        ({"region_name": "cn"}, "region_name", "cn"),
    ],
)
def test_init(kwargs, attr, expected):
    assert getattr(_make_client(**kwargs), attr) == expected


def test_init_app_account():
    acct = AliyunAccount("a", "b")
    assert _make_client(app_account=acct).app_account is acct


@pytest.mark.parametrize(
    "kwargs, extra, check",
    [
        (
            {"namespace": "ns1"},
            {},
            lambda c: c["headers"]["x-odps-namespace-id"] == "ns1",
        ),
        (
            {"project": "proj", "schema": "sch"},
            {},
            lambda c: "curr_project=proj" in c["url"],
        ),
        (
            {"project": "proj", "schema": "sch"},
            {},
            lambda c: "curr_schema=sch" in c["url"],
        ),
        ({}, {"actions": ["a", "b"]}, lambda c: c["url"].endswith("?a&b")),
        (
            {},
            {"headers": {"X-Custom": "val"}},
            lambda c: c["headers"]["X-Custom"] == "val",
        ),
    ],
)
def test_request_construction(kwargs, extra, check):
    assert check(_capture_request(_make_client(**kwargs), **extra))


def test_request_none_header_raises():
    client = _make_client()
    with mock.patch.object(
        client._account,
        "sign_request",
        lambda req, *a, **kw: req.headers.update({"X-None": None}),
    ):
        with pytest.raises(TypeError, match="cannot be None"):
            client._request("http://localhost/test", "GET")


def test_request_connect_timeout_wrapped():
    client = _make_client()
    with mock.patch.object(client.session, "send", side_effect=RequestsConnectTimeout):
        with pytest.raises(errors.ConnectTimeout):
            client._request("http://localhost/test", "GET")


def test_request_error_response_raises():
    client = _make_client()
    with mock.patch.object(
        client.session, "send", return_value=_fake_response(ok=False, status_code=500)
    ):
        with pytest.raises(errors.ODPSError):
            client._request("http://localhost/test", "GET")


@pytest.mark.parametrize("ok", [True, False])
def test_is_ok(ok):
    assert _make_client().is_ok(_fake_response(ok=ok)) is ok


@pytest.mark.parametrize(
    "method, kwarg, value",
    [
        ("get", "stream", True),
        ("post", "data", b"payload"),
        ("put", "data", b"payload"),
        ("head", "stream", False),
        ("delete", "stream", False),
    ],
)
def test_http_helpers_call_request(method, kwarg, value):
    client = _make_client()
    with mock.patch.object(client, "request", return_value=_fake_response()) as m:
        getattr(client, method)("http://localhost/test", **{kwarg: value})
        assert m.call_args[0][1] == method
