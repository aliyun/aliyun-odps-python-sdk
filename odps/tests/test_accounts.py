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

import base64
import copy
import datetime
import hashlib
import hmac
import json
import os
import pickle
import shutil
import tempfile
import time
import uuid
from urllib.parse import unquote, urlparse

import mock
import pytest
import requests

from .. import ODPS, errors, options, utils
from ..accounts import (
    BearerTokenAccount,
    CloudAccount,
    CredentialProviderAccount,
    SignServer,
    SignServerAccount,
    SignServerError,
    StsAccount,
    _get_v4_signature_prefix,
    from_environments,
)
from ..compat import datetime_utcnow as _real_datetime_utcnow
from ..rest import RestClient
from .core import tn

try:
    from cupid.runtime import context as cupid_context
except ImportError:
    cupid_context = None


@pytest.fixture(autouse=True)
def clear_global_accounts():
    try:
        yield
    finally:
        options.account = options.default_project = options.endpoint = None


@pytest.fixture
def use_legacy_logview(odps):
    from ..core import _jobinsight_host_cache

    old_job_insight_host = odps._job_insight_host
    try:
        _jobinsight_host_cache.clear()
        options.use_legacy_logview = True
        odps._job_insight_host = odps.get_job_insight_host()
        yield
    finally:
        odps._job_insight_host = old_job_insight_host
        options.use_legacy_logview = None


def test_sign_server_account(odps):
    server = SignServer()
    server.accounts[odps.account.access_id] = odps.account.secret_access_key
    try:
        server.start(("127.0.0.1", 0))
        account = SignServerAccount(
            odps.account.access_id, server.server.server_address
        )

        reloaded = pickle.loads(pickle.dumps(account))
        assert reloaded.access_id == account.access_id
        assert reloaded.sign_endpoint == account.sign_endpoint
        assert reloaded.token == account.token

        odps = odps.as_account(account=account)
        odps.delete_table(tn("test_sign_account_table"), if_exists=True)
        t = odps.create_table(tn("test_sign_account_table"), "col string", lifecycle=1)
        assert odps.exist_table(tn("test_sign_account_table")) is True
        t.drop(async_=True)
    finally:
        server.stop()


def test_tokenized_sign_server_account(odps):
    server = SignServer(token=str(uuid.uuid4()))
    server.accounts[odps.account.access_id] = odps.account.secret_access_key
    try:
        server.start(("127.0.0.1", 0))
        account = SignServerAccount(
            odps.account.access_id, server.server.server_address
        )
        odps = ODPS(None, None, odps.project, odps.endpoint, account=account)
        pytest.raises(
            SignServerError,
            lambda: odps.delete_table(tn("test_sign_account_table"), if_exists=True),
        )

        account = SignServerAccount(
            odps.account.access_id, server.server.server_address, token=server.token
        )
        odps = ODPS(None, None, odps.project, odps.endpoint, account=account)
        odps.delete_table(tn("test_sign_account_table"), if_exists=True)
        t = odps.create_table(tn("test_sign_account_table"), "col string", lifecycle=1)
        assert odps.exist_table(tn("test_sign_account_table")) is True
        t.drop(async_=True)
    finally:
        server.stop()


def test_sts_account(odps):
    tmp_path = tempfile.mkdtemp(prefix="tmp_pyodps_")
    req = requests.Request(method="GET", url=odps.get_project().resource())
    try:
        token_account = StsAccount(
            odps.account.access_id, odps.account.secret_access_key, "token"
        )

        reloaded = pickle.loads(pickle.dumps(token_account))
        assert reloaded.access_id == token_account.access_id
        assert reloaded.secret_access_key == token_account.secret_access_key
        assert reloaded.sts_token == token_account.sts_token

        cp_req = copy.deepcopy(req)
        token_account.sign_request(cp_req, odps.endpoint)
        assert "token" == cp_req.headers["authorization-sts-token"]

        os.environ["ODPS_STS_ACCESS_KEY_ID"] = odps.account.access_id
        os.environ["ODPS_STS_ACCESS_KEY_SECRET"] = odps.account.secret_access_key
        os.environ["ODPS_STS_TOKEN"] = "token"
        account = from_environments()
        assert isinstance(account, StsAccount)
        cp_req = copy.deepcopy(req)
        token_account.sign_request(cp_req, odps.endpoint)
        assert "token" == cp_req.headers["authorization-sts-token"]

        os.environ.pop("ODPS_STS_ACCESS_KEY_ID", None)
        os.environ.pop("ODPS_STS_ACCESS_KEY_SECRET", None)
        os.environ.pop("ODPS_STS_TOKEN", None)

        sts_file_name = os.path.join(tmp_path, "sts_file")
        os.environ["ODPS_STS_ACCOUNT_FILE"] = sts_file_name
        exp_time = int(time.time() + 3 * 3600)
        account_data = {
            "accessKeyId": odps.account.access_id,
            "accessKeySecret": odps.account.secret_access_key,
            "securityToken": "token",
            "expiration": datetime.datetime.utcfromtimestamp(exp_time).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        with open(sts_file_name, "w") as out_file:
            out_file.write(json.dumps(account_data))
        account = from_environments()
        assert isinstance(account, StsAccount)
        assert account._expire_time == exp_time

        cp_req = copy.deepcopy(req)
        token_account.sign_request(cp_req, odps.endpoint)
        assert "token" == cp_req.headers["authorization-sts-token"]
    finally:
        shutil.rmtree(tmp_path)
        os.environ.pop("ODPS_STS_ACCESS_KEY_ID", None)
        os.environ.pop("ODPS_STS_ACCESS_KEY_SECRET", None)
        os.environ.pop("ODPS_STS_TOKEN", None)
        os.environ.pop("ODPS_STS_ACCOUNT_FILE", None)


@pytest.mark.skipif(cupid_context is None, reason="cannot import cupid context")
def test_bearer_token_account(odps, use_legacy_logview):
    inst = odps.run_sql("select count(*) from dual")
    inst.wait_for_completion()
    task_name = inst.get_task_names()[0]

    logview_address = inst.get_logview_address()
    token = logview_address[logview_address.find("token=") + len("token=") :]
    bearer_token_account = BearerTokenAccount(token=token)

    reloaded = pickle.loads(pickle.dumps(bearer_token_account))
    assert reloaded.token == bearer_token_account.token

    bearer_token_odps = ODPS(
        None, None, odps.project, odps.endpoint, account=bearer_token_account
    )
    bearer_token_instance = bearer_token_odps.get_instance(inst.id)

    assert inst.get_task_result(task_name) == bearer_token_instance.get_task_result(
        task_name
    )
    assert inst.get_task_summary(task_name) == bearer_token_instance.get_task_summary(
        task_name
    )

    with pytest.raises(errors.NoPermission):
        bearer_token_odps.create_table(
            tn("test_bearer_token_account_table_test1"), "col string", lifecycle=1
        )


def test_fake_bearer_token(odps, use_legacy_logview):
    fake_token_account = BearerTokenAccount(token="fake-token")
    bearer_token_odps = ODPS(
        None,
        None,
        odps.project,
        odps.endpoint,
        account=fake_token_account,
        overwrite_global=False,
    )

    with pytest.raises(errors.ODPSError):
        bearer_token_odps.create_table(
            tn("test_bearer_token_account_table_test2"), "col string", lifecycle=1
        )


def test_bearer_token_load_and_update(odps, use_legacy_logview):
    token = "fake-token"
    tmp_path = tempfile.mkdtemp(prefix="tmp_pyodps_")
    os.environ["ODPS_BEARER_TOKEN_HOURS"] = "0"
    try:
        token_file_name = os.path.join(tmp_path, "token_file")
        with open(token_file_name, "w") as token_file:
            token_file.write(token)
        os.environ["ODPS_BEARER_TOKEN_FILE"] = token_file_name

        create_timestamp = int(time.time())

        options.account = None
        env_odps = ODPS(project=odps.project, endpoint=odps.endpoint)
        assert isinstance(env_odps.account, BearerTokenAccount)
        assert env_odps.account.token == token
        assert env_odps.account._expire_time > create_timestamp

        last_timestamp = env_odps.account._expire_time
        env_odps.account.reload()
        assert env_odps.account._expire_time > last_timestamp

        inst = odps.run_sql("select count(*) from dual")
        logview_address = inst.get_logview_address()
        token = logview_address[logview_address.find("token=") + len("token=") :]
        with open(token_file_name, "w") as token_file:
            token_file.write(token)

        last_timestamp = env_odps.account._expire_time
        env_odps.account.reload()
        assert env_odps.account._expire_time != last_timestamp

        last_timestamp = env_odps.account._expire_time
        env_odps.account.reload()
        assert env_odps.account._expire_time == last_timestamp
    finally:
        shutil.rmtree(tmp_path)
        os.environ.pop("ODPS_BEARER_TOKEN_HOURS", None)
        os.environ.pop("ODPS_BEARER_TOKEN_FILE", None)
        os.environ.pop("ODPS_BEARER_TOKEN_TIMESTAMP_FILE", None)


def test_v4_signature_fallback(odps):
    # Exercise the V4-signature fallback in RestClient.request() directly via
    # rest.get(), avoiding dependence on tenant/project/instance state that
    # varies with test ordering under pytest-randomly.
    endpoint = odps.rest.endpoint
    assert endpoint not in RestClient._endpoints_without_v4_sign

    def _new_is_ok(self, resp):
        if endpoint not in self._endpoints_without_v4_sign:
            raise errors.InvalidParameter("ODPS-0410051: Invalid credentials")
        return resp.ok

    def _new_is_ok2(self, resp):
        if endpoint not in self._endpoints_without_v4_sign:
            raise errors.InternalServerError(
                "ODPS-0010000:System internal error - Error occurred while getting access key for "
                f"'{odps.account.access_id}', CloudV4 request need ak v3 support"
            )
        return resp.ok

    def _new_is_ok3(self, resp):
        if endpoint not in self._endpoints_without_v4_sign:
            raise errors.Unauthorized(
                "The request authorization header is invalid or missing."
            )
        return resp.ok

    url = endpoint + "/projects/" + odps.project
    old_enable_v4_sign = options.enable_v4_sign
    old_region_name = odps.rest._region_name
    try:
        odps.rest._region_name = "mock-region"
        options.enable_v4_sign = True

        for mock_is_ok in (_new_is_ok, _new_is_ok2, _new_is_ok3):
            RestClient._endpoints_without_v4_sign.clear()
            with mock.patch("odps.rest.RestClient.is_ok", new=mock_is_ok):
                odps.rest.get(url)
                assert endpoint in RestClient._endpoints_without_v4_sign
    finally:
        odps.rest._region_name = old_region_name
        RestClient._endpoints_without_v4_sign.discard(endpoint)
        options.enable_v4_sign = old_enable_v4_sign


def test_auth_expire_reload(odps, use_legacy_logview):
    inst = odps.run_sql("select count(*) from dual")
    inst.wait_for_completion()

    tmp_path = tempfile.mkdtemp(prefix="tmp_pyodps_")
    try:
        logview_address = inst.get_logview_address()
        token = logview_address[logview_address.find("token=") + len("token=") :]

        token_file = os.path.join(tmp_path, "token_ts_file")
        os.environ["ODPS_BEARER_TOKEN_FILE"] = token_file
        with open(token_file, "w") as token_file_obj:
            token_file_obj.write("invalid_token")

        token_odps = ODPS(
            account=BearerTokenAccount(), project=odps.project, endpoint=odps.endpoint
        )

        retrial_counts = 0

        def _new_is_ok(self, resp):
            nonlocal retrial_counts
            if not retrial_counts:
                with open(token_file, "w") as token_file_obj:
                    token_file_obj.write(token)
                retrial_counts += 1
                raise errors.AuthenticationRequestExpired("mock auth expired")
            return resp.ok

        with mock.patch("odps.rest.RestClient.is_ok", new=_new_is_ok):
            token_inst = token_odps.get_instance(inst.id)
            token_inst.reload()
            assert retrial_counts == 1
            assert token_odps.account.token is not None
    finally:
        shutil.rmtree(tmp_path)
        os.environ.pop("ODPS_BEARER_TOKEN_FILE", None)


def test_rest_none_header_check(odps):
    old_sign_request = CloudAccount.sign_request

    def new_sign_request(self, req, *args, **kwargs):
        req.headers["x-pyodps-fake-header"] = None
        return old_sign_request(self, req, *args, **kwargs)

    with mock.patch("odps.accounts.CloudAccount.sign_request", new=new_sign_request):
        with pytest.raises(TypeError) as ex_info:
            next(odps.list_tables())
        assert "x-pyodps-fake-header" in str(ex_info.value)


class MockCredentials:
    def __init__(self, odps):
        self._odps = odps

    def get_access_key_id(self):
        return self._odps.account.access_id

    def get_access_key_secret(self):
        return self._odps.account.secret_access_key

    def get_security_token(cls):
        return None  # kept empty to skip sts token check


class MockCredentialProvider:
    def __init__(self, odps):
        self._odps = odps

    def get_credentials(self):
        return MockCredentials(self._odps)


class MockCredentialProvider2:
    def __init__(self, odps):
        self._odps = odps

    def get_credential(self):
        return MockCredentials(self._odps)


@pytest.mark.parametrize(
    "provider_cls", [MockCredentialProvider, MockCredentialProvider2]
)
def test_credential_provider_account(odps, provider_cls):
    account = CredentialProviderAccount(provider_cls(odps))

    reloaded = pickle.loads(pickle.dumps(account))
    assert (
        reloaded.provider._odps.account.access_id
        == account.provider._odps.account.access_id
    )

    cred_odps = ODPS(account, None, odps.project, odps.endpoint)

    table_name = tn("test_bearer_token_account_table")

    cred_odps.delete_table(table_name, if_exists=True)
    t = cred_odps.create_table(table_name, "col string", lifecycle=1)
    with t.open_writer() as writer:
        records = [["val1"], ["val2"], ["val3"]]
        writer.write(records)
    cred_odps.delete_table(table_name)


def _build_canonical(account, url, headers, method="GET"):
    url_components = urlparse(unquote(url), allow_fragments=False)
    req = mock.Mock()
    req.method = method
    req.headers = dict(headers)
    return account._build_canonical_str(url_components, req)


@pytest.mark.parametrize(
    "url,headers,resource_line,extra_assertions",
    [
        # Duplicate query keys collapse to the last value, matching the Java SDK
        # (which stores params in a Map). Previously pyodps raised AssertionError.
        pytest.param(
            "/projects/p?k=1&k=2",
            {"Date": "Wed, 04 Aug 2026 06:00:00 GMT"},
            "/projects/p?k=2",
            {},
            id="dedup-query-keys",
        ),
        # A parameter with an empty value renders as a bare key ("?empty"),
        # matching the Java SDK's buildCanonicalizedResource.
        pytest.param(
            "/projects/p?empty=",
            {"Date": "Wed, 04 Aug 2026 06:00:00 GMT"},
            "/projects/p?empty",
            {},
            id="empty-param-value",
        ),
        # Only headers starting with "x-odps-" are signed; a bare "x-odps" prefix
        # (no trailing dash) must be ignored, matching the Java SDK.
        pytest.param(
            "/projects/p",
            {
                "x-odpsfoo": "IGNORE",
                "x-odps-x": "SIGN",
                "Date": "Wed, 04 Aug 2026 06:00:00 GMT",
            },
            "/projects/p",
            {"x-odpsfoo:IGNORE": False, "x-odps-x:SIGN": True},
            id="header-prefix-dash",
        ),
    ],
)
def test_v4_canonical_string_java_parity(url, headers, resource_line, extra_assertions):
    account = CloudAccount("test_aid", "test_sk")
    canon = _build_canonical(account, url, headers)
    assert canon.split("\n")[-1] == resource_line
    for needle, expected in extra_assertions.items():
        assert (needle in canon) is expected


def test_v4_signature_computes():
    # End-to-end v4 signature is well-formed and carries the credential scope.
    account = CloudAccount("test_aid", "test_sk")
    canon = _build_canonical(
        account, "/projects/p?k=2", {"Date": "Wed, 04 Aug 2026 06:00:00 GMT"}
    )
    auth = account.calc_auth_str(canon, region_name="cn-hangzhou")
    assert auth.startswith("ODPS ")
    assert "aliyun_v4_request" in auth


def test_v4_signature_key_midnight_rollover():
    # The signing-key cache is keyed on the UTC date. It must roll over at UTC
    # midnight and never at local midnight, and must never serve a stale key
    # when the UTC date changes. Runs under a non-UTC TZ (UTC+8) so the date
    # source is exercised against a divergent local clock.
    account = CloudAccount("test_aid", "test_sk")
    region = "cn-hangzhou"
    sig_prefix = _get_v4_signature_prefix()
    canonical = "GET\n\n\nWed, 05 Aug 2026 06:00:00 GMT\n/projects/p?k=2"

    def ref_key(date_str):
        k_secret = utils.to_binary(sig_prefix + "test_sk")
        k_date = hmac.new(k_secret, utils.to_binary(date_str), hashlib.sha256).digest()
        k_region = hmac.new(k_date, utils.to_binary(region), hashlib.sha256).digest()
        k_service = hmac.new(k_region, b"odps", hashlib.sha256).digest()
        return hmac.new(
            k_service, utils.to_binary(sig_prefix + "_request"), hashlib.sha256
        ).digest()

    def parse_auth(auth):
        scope, signature = auth[len("ODPS ") :].rsplit(":", 1)
        return scope.split("/")[1], signature

    # UTC boundary sequence. Under TZ=Asia/Shanghai (UTC+8), local midnight on
    # Aug 6 falls at 16:00 UTC Aug 5 -- the same UTC date -- so the key must
    # be reused there; it must only roll over at UTC midnight (00:00 UTC).
    boundaries = [
        (datetime.datetime(2026, 8, 5, 15, 59), "20260805", "before local midnight"),
        (
            datetime.datetime(2026, 8, 5, 16, 0),
            "20260805",
            "local midnight, UTC unchanged",
        ),
        (datetime.datetime(2026, 8, 5, 23, 59), "20260805", "before UTC midnight"),
        (datetime.datetime(2026, 8, 6, 0, 0), "20260806", "UTC midnight (rollover)"),
    ]

    old_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "Asia/Shanghai"  # UTC+8
        time.tzset()

        # Guard: the real date source is UTC, not local, under this TZ.
        offset = (datetime.datetime.now() - _real_datetime_utcnow()).total_seconds()
        assert 7 * 3600 < offset < 9 * 3600

        prev_date = None
        cached_key = None
        with mock.patch("odps.accounts.datetime_utcnow") as mocked:
            for utc_dt, expected_date, desc in boundaries:
                mocked.return_value = utc_dt
                auth = account.calc_auth_str(canonical, region_name=region)
                date_str, signature = parse_auth(auth)
                assert date_str == expected_date, desc
                # signature must match an independent computation for this UTC
                # date; a stale cached key from the previous date would fail.
                expected_sig = utils.to_str(
                    base64.b64encode(
                        hmac.new(
                            ref_key(date_str), utils.to_binary(canonical), hashlib.sha1
                        ).digest()
                    )
                )
                assert signature == expected_sig, desc
                if date_str == prev_date:
                    # same UTC date -> key reused, cache untouched
                    assert account._last_signature_key == cached_key, desc
                else:
                    assert account._last_signature_date == date_str, desc
                prev_date = date_str
                cached_key = account._last_signature_key
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        time.tzset()
