#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import copy
import datetime
import json
import os
import shutil
import tempfile
import time
import uuid

import mock
import pytest
import requests

from .. import ODPS, errors, options
from ..accounts import (
    AliyunAccount,
    BearerTokenAccount,
    CredentialProviderAccount,
    SignServer,
    SignServerAccount,
    SignServerError,
    StsAccount,
    from_environments,
)
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


def test_sign_server_account(odps):
    server = SignServer()
    server.accounts[odps.account.access_id] = odps.account.secret_access_key
    try:
        server.start(("127.0.0.1", 0))
        account = SignServerAccount(
            odps.account.access_id, server.server.server_address
        )
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
def test_bearer_token_account(odps):
    inst = odps.run_sql("select count(*) from dual")
    inst.wait_for_completion()
    task_name = inst.get_task_names()[0]

    logview_address = inst.get_logview_address()
    token = logview_address[logview_address.find("token=") + len("token=") :]
    bearer_token_account = BearerTokenAccount(token=token)
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


def test_fake_bearer_token(odps):
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


def test_bearer_token_load_and_update(odps):
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
    odps.delete_table(tn("test_sign_account_table"), if_exists=True)
    assert odps.endpoint not in RestClient._endpoints_without_v4_sign

    def _new_is_ok(self, resp):
        if odps.endpoint not in self._endpoints_without_v4_sign:
            raise errors.InvalidParameter("ODPS-0410051: Invalid credentials")
        return resp.ok

    def _new_is_ok2(self, resp):
        if odps.endpoint not in self._endpoints_without_v4_sign:
            raise errors.InternalServerError(
                "ODPS-0010000:System internal error - Error occurred while getting access key for "
                "'%s', AliyunV4 request need ak v3 support" % odps.account.access_id
            )
        return resp.ok

    def _new_is_ok3(self, resp):
        if odps.endpoint not in self._endpoints_without_v4_sign:
            raise errors.Unauthorized(
                "The request authorization header is invalid or missing."
            )
        return resp.ok

    old_enable_v4_sign = options.enable_v4_sign
    try:
        options.enable_v4_sign = True
        RestClient._endpoints_without_v4_sign.clear()
        with mock.patch("odps.rest.RestClient.is_ok", new=_new_is_ok):
            odps.delete_table(tn("test_sign_account_table"), if_exists=True)
            assert odps.endpoint in RestClient._endpoints_without_v4_sign

        RestClient._endpoints_without_v4_sign.clear()
        with mock.patch("odps.rest.RestClient.is_ok", new=_new_is_ok2):
            odps.delete_table(tn("test_sign_account_table"), if_exists=True)
            assert odps.endpoint in RestClient._endpoints_without_v4_sign

        RestClient._endpoints_without_v4_sign.clear()
        with mock.patch("odps.rest.RestClient.is_ok", new=_new_is_ok3):
            odps.delete_table(tn("test_sign_account_table"), if_exists=True)
            assert odps.endpoint in RestClient._endpoints_without_v4_sign
    finally:
        RestClient._endpoints_without_v4_sign.difference_update([odps.endpoint])
        options.enable_v4_sign = old_enable_v4_sign


def test_auth_expire_reload(odps):
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

        retrial_counts = [0]

        def _new_is_ok(self, resp):
            if not retrial_counts[0]:
                with open(token_file, "w") as token_file_obj:
                    token_file_obj.write(token)
                retrial_counts[0] += 1
                raise errors.AuthenticationRequestExpired("mock auth expired")
            return resp.ok

        with mock.patch("odps.rest.RestClient.is_ok", new=_new_is_ok):
            token_inst = token_odps.get_instance(inst.id)
            token_inst.reload()
            assert retrial_counts[0] == 1
            assert token_odps.account.token is not None
    finally:
        shutil.rmtree(tmp_path)
        os.environ.pop("ODPS_BEARER_TOKEN_FILE", None)


def test_rest_none_header_check(odps):
    old_sign_request = AliyunAccount.sign_request

    def new_sign_request(self, req, *args, **kwargs):
        req.headers["x-pyodps-fake-header"] = None
        return old_sign_request(self, req, *args, **kwargs)

    with mock.patch("odps.accounts.AliyunAccount.sign_request", new=new_sign_request):
        with pytest.raises(TypeError) as ex_info:
            next(odps.list_tables())
        assert "x-pyodps-fake-header" in str(ex_info.value)


class MockCredentials(object):
    def __init__(self, odps):
        self._odps = odps

    def get_access_key_id(self):
        return self._odps.account.access_id

    def get_access_key_secret(self):
        return self._odps.account.secret_access_key

    def get_security_token(cls):
        return None  # kept empty to skip sts token check


class MockCredentialProvider(object):
    def __init__(self, odps):
        self._odps = odps

    def get_credentials(self):
        return MockCredentials(self._odps)


class MockCredentialProvider2(object):
    def __init__(self, odps):
        self._odps = odps

    def get_credential(self):
        return MockCredentials(self._odps)


@pytest.mark.parametrize(
    "provider_cls", [MockCredentialProvider, MockCredentialProvider2]
)
def test_credential_provider_account(odps, provider_cls):
    account = CredentialProviderAccount(provider_cls(odps))
    cred_odps = ODPS(account, None, odps.project, odps.endpoint)

    table_name = tn("test_bearer_token_account_table")

    cred_odps.delete_table(table_name, if_exists=True)
    t = cred_odps.create_table(table_name, "col string", lifecycle=1)
    with t.open_writer() as writer:
        records = [["val1"], ["val2"], ["val3"]]
        writer.write(records)
    cred_odps.delete_table(table_name)
