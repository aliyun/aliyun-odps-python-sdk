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

"""A couple of authentication types in ODPS."""

import base64
import calendar
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime

import requests

from . import options, utils
from .compat import cgi, parse_qsl, six, unquote, urlparse

logger = logging.getLogger(__name__)

DEFAULT_TEMP_ACCOUNT_HOURS = 5


class BaseAccount(object):
    def _build_canonical_str(self, url_components, req):
        # Build signing string
        lines = [req.method]
        headers_to_sign = dict()

        canonical_resource = url_components.path
        params = dict()
        if url_components.query:
            params_list = sorted(
                parse_qsl(url_components.query, True), key=lambda it: it[0]
            )
            assert len(params_list) == len(set(it[0] for it in params_list))
            params = dict(params_list)
            convert = lambda kv: kv if kv[1] != "" else (kv[0],)
            params_str = "&".join(["=".join(convert(kv)) for kv in params_list])

            canonical_resource = "%s?%s" % (canonical_resource, params_str)

        headers = req.headers
        logger.debug("headers before signing: %s", headers)
        for k, v in six.iteritems(headers):
            k = k.lower()
            if k in ("content-type", "content-md5") or k.startswith("x-odps"):
                headers_to_sign[k] = v
        for k in ("content-type", "content-md5"):
            if k not in headers_to_sign:
                headers_to_sign[k] = ""
        date_str = headers.get("Date")
        if not date_str:
            req_date = utils.formatdate(usegmt=True)
            headers["Date"] = req_date
            date_str = req_date
        headers_to_sign["date"] = date_str
        for param_key, param_value in six.iteritems(params):
            if param_key.startswith("x-odps-"):
                headers_to_sign[param_key] = param_value

        headers_to_sign = OrderedDict(
            [(k, headers_to_sign[k]) for k in sorted(headers_to_sign)]
        )
        logger.debug("headers to sign: %s", headers_to_sign)
        for k, v in six.iteritems(headers_to_sign):
            if k.startswith("x-odps-"):
                lines.append("%s:%s" % (k, v))
            else:
                lines.append(v)

        lines.append(canonical_resource)
        return "\n".join(lines)

    def sign_request(self, req, endpoint, region_name=None):
        raise NotImplementedError


class AliyunAccount(BaseAccount):
    """
    Account of aliyun.com
    """

    def __init__(self, access_id, secret_access_key):
        self.access_id = access_id
        self.secret_access_key = secret_access_key
        self._last_signature_date = None
        self._last_signature_key = None

    def _get_v4_signature_key(self, date_str, region_name):
        if date_str == self._last_signature_date:
            return self._last_signature_key

        k_secret = utils.to_binary("aliyun_v4" + self.secret_access_key)
        k_date = hmac.new(k_secret, utils.to_binary(date_str), hashlib.sha256).digest()
        k_region = hmac.new(
            k_date, utils.to_binary(region_name), hashlib.sha256
        ).digest()
        k_service = hmac.new(k_region, b"odps", hashlib.sha256).digest()

        self._last_signature_date = date_str
        self._last_signature_key = hmac.new(
            k_service, b"aliyun_v4_request", hashlib.sha256
        ).digest()
        return self._last_signature_key

    def calc_auth_str(self, canonical_str, region_name=None):
        if region_name is None:
            # use legacy v2 sign
            signature = base64.b64encode(
                hmac.new(
                    utils.to_binary(self.secret_access_key),
                    utils.to_binary(canonical_str),
                    hashlib.sha1,
                ).digest()
            )
            return "ODPS %s:%s" % (self.access_id, utils.to_str(signature))
        else:
            # use v4 sign
            date_str = datetime.strftime(datetime.utcnow(), "%Y%m%d")
            credential = "/".join(
                [self.access_id, date_str, region_name, "odps/aliyun_v4_request"]
            )
            sign_key = self._get_v4_signature_key(date_str, region_name)
            signature = base64.b64encode(
                hmac.new(
                    sign_key, utils.to_binary(canonical_str), hashlib.sha1
                ).digest()
            )
            return "ODPS %s:%s" % (credential, utils.to_str(signature))

    def sign_request(self, req, endpoint, region_name=None):
        url = req.url[len(endpoint) :]
        url_components = urlparse(unquote(url), allow_fragments=False)

        canonical_str = self._build_canonical_str(url_components, req)
        logger.debug("canonical string: %s", canonical_str)

        req.headers["Authorization"] = self.calc_auth_str(canonical_str, region_name)
        logger.debug("headers after signing: %r", req.headers)


class AppAccount(BaseAccount):
    """
    Account for applications.
    """

    def __init__(self, access_id, secret_access_key):
        self.access_id = access_id
        self.secret_access_key = secret_access_key

    def sign_request(self, req, endpoint, region_name=None):
        auth_str = req.headers["Authorization"]
        signature = base64.b64encode(
            hmac.new(
                utils.to_binary(self.secret_access_key),
                utils.to_binary(auth_str),
                hashlib.sha1,
            ).digest()
        )
        app_auth_str = (
            "account_provider:%s,signature_method:%s,access_id:%s,signature:%s"
            % ("aliyun", "hmac-sha1", self.access_id, utils.to_str(signature))
        )
        req.headers["application-authentication"] = app_auth_str
        logger.debug("headers after app signing: %r", req.headers)


class SignServer(object):
    class SignServerHandler(six.moves.BaseHTTPServer.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"PyODPS Account Server")

        def do_POST(self):
            try:
                self._do_POST()
            except:
                logger.exception("Failed to sign request on SignServer.")
                self.send_response(500)
                self.end_headers()

        def _do_POST(self):
            ctype, pdict = cgi.parse_header(self.headers.get("content-type"))
            if ctype == "multipart/form-data":
                postvars = cgi.parse_multipart(self.rfile, pdict)
            elif ctype == "application/x-www-form-urlencoded":
                length = int(self.headers.get("content-length"))
                postvars = six.moves.urllib.parse.parse_qs(
                    self.rfile.read(length), keep_blank_values=1
                )
            else:
                self.send_response(400)
                self.end_headers()
                return

            self._sign(postvars)

        def _sign(self, postvars):
            if self.server._token is not None:
                auth = self.headers.get("Authorization")
                if not auth:
                    self.send_response(401)
                    self.end_headers()
                    return
                method, content = auth.split(" ", 1)
                method = method.lower()
                if method == "token":
                    if content != self.server._token:
                        self.send_response(401)
                        self.end_headers()
                        return
                else:
                    self.send_response(401)
                    self.end_headers()
                    return

            assert len(postvars[b"access_id"]) == 1 and len(postvars[b"canonical"]) == 1
            access_id = utils.to_str(postvars[b"access_id"][0])
            canonical = utils.to_str(postvars[b"canonical"][0])
            if b"region_name" not in postvars:
                region_name = None
            else:
                region_name = utils.to_str(postvars[b"region_name"][0])
            secret_access_key = self.server._accounts[access_id]

            account = AliyunAccount(access_id, secret_access_key)
            auth_str = account.calc_auth_str(canonical, region_name)

            self.send_response(200)
            self.send_header("Content-Type", "text/json")
            self.end_headers()
            self.wfile.write(utils.to_binary(auth_str))

        def log_message(self, *args):
            return

    class SignServerCore(
        six.moves.socketserver.ThreadingMixIn, six.moves.BaseHTTPServer.HTTPServer
    ):
        def __init__(self, *args, **kwargs):
            self._accounts = kwargs.pop("accounts", {})
            self._token = kwargs.pop("token", None)
            self._ready = False
            six.moves.BaseHTTPServer.HTTPServer.__init__(self, *args, **kwargs)
            self._ready = True

        def stop(self):
            self.shutdown()
            self.server_close()

    def __init__(self, token=None):
        self._server = None
        self._accounts = dict()
        self._token = token

    @property
    def server(self):
        return self._server

    @property
    def accounts(self):
        return self._accounts

    @property
    def token(self):
        return self._token

    def start(self, endpoint):
        def starter():
            self._server = self.SignServerCore(
                endpoint,
                self.SignServerHandler,
                accounts=self.accounts,
                token=self.token,
            )
            self._server.serve_forever()

        thread = threading.Thread(target=starter)
        thread.daemon = True
        thread.start()
        while self._server is None or not self._server._ready:
            time.sleep(0.05)

    def stop(self):
        self._server.stop()


class SignServerError(Exception):
    def __init__(self, msg, code, content):
        super(SignServerError, self).__init__(msg)
        self.code = code
        self.content = content


class SignServerAccount(BaseAccount):
    _session_local = threading.local()

    def __init__(
        self, access_id, sign_endpoint=None, server=None, port=None, token=None
    ):
        self.access_id = access_id
        self.sign_endpoint = sign_endpoint or (server, port)
        self.token = token

    @property
    def session(self):
        if not hasattr(type(self)._session_local, "_session"):
            adapter_options = dict(
                pool_connections=options.pool_connections,
                pool_maxsize=options.pool_maxsize,
                max_retries=options.retry_times,
            )
            session = requests.Session()
            # mount adapters with retry times
            session.mount("http://", requests.adapters.HTTPAdapter(**adapter_options))
            session.mount("https://", requests.adapters.HTTPAdapter(**adapter_options))

            self._session_local._session = session
        return self._session_local._session

    def sign_request(self, req, endpoint, region_name=None):
        url = req.url[len(endpoint) :]
        url_components = urlparse(unquote(url), allow_fragments=False)

        canonical_str = self._build_canonical_str(url_components, req)
        logger.debug("canonical string: %s", canonical_str)

        headers = dict()
        if self.token:
            headers["Authorization"] = "token " + self.token

        sign_content = dict(access_id=self.access_id, canonical=canonical_str)
        if region_name is not None:
            sign_content["region_name"] = region_name
        resp = self.session.request(
            "post",
            "http://%s:%s" % self.sign_endpoint,
            headers=headers,
            data=sign_content,
        )
        if resp.status_code < 400:
            req.headers["Authorization"] = resp.text
            logger.debug("headers after signing: %r", req.headers)
        else:
            try:
                err_msg = resp_err = resp.text
            except:
                resp_err = resp.content
                err_msg = repr(resp_err)

            raise SignServerError(
                "Sign server returned error code: %d\n%s" % (resp.status_code, err_msg),
                resp.status_code,
                resp_err,
            )


class TempAccountMixin(object):
    def __init__(self, expired_hours=DEFAULT_TEMP_ACCOUNT_HOURS):
        self._last_refresh_time = time.time()
        if expired_hours is not None:
            self._expire_seconds = expired_hours * 3600
            self._expire_time = self._last_refresh_time + self._expire_seconds
        else:
            self._expire_time = self._expire_seconds = None
        self.reload()

    def _is_account_valid(self):
        raise NotImplementedError

    def _reload_account(self):
        raise NotImplementedError

    def _need_update(self):
        if not self._is_account_valid():
            return True
        if self._expire_time is not None and self._expire_seconds is not None:
            min_exp_time = min(
                self._expire_time, self._last_refresh_time + self._expire_seconds
            )
            return time.time() > min_exp_time
        return False

    def reload(self, force=False):
        t = time.time()
        if force or self._need_update():
            self._last_refresh_time = t
            default_expire = t + (
                self._expire_seconds or 3600 * DEFAULT_TEMP_ACCOUNT_HOURS
            )
            self._expire_time = self._reload_account() or default_expire


class StsAccount(TempAccountMixin, AliyunAccount):
    """
    Account of sts
    """

    def __init__(
        self,
        access_id,
        secret_access_key,
        sts_token,
        expired_hours=DEFAULT_TEMP_ACCOUNT_HOURS,
    ):
        self.sts_token = sts_token
        AliyunAccount.__init__(self, access_id, secret_access_key)
        TempAccountMixin.__init__(self, expired_hours=expired_hours)

    @classmethod
    def from_environments(cls):
        expired_hours = int(
            os.getenv("ODPS_STS_TOKEN_HOURS", str(DEFAULT_TEMP_ACCOUNT_HOURS))
        )
        if "ODPS_STS_ACCOUNT_FILE" in os.environ or "ODPS_STS_TOKEN" in os.environ:
            if "ODPS_STS_ACCOUNT_FILE" not in os.environ:
                expired_hours = None
            return cls(None, None, None, expired_hours=expired_hours)
        return None

    def sign_request(self, req, endpoint, region_name=None):
        self.reload()
        super(StsAccount, self).sign_request(req, endpoint, region_name=region_name)
        if self.sts_token:
            req.headers["authorization-sts-token"] = self.sts_token
        if self._last_refresh_time:
            req.headers["x-pyodps-token-timestamp"] = str(self._last_refresh_time)

    def _is_account_valid(self):
        return self.sts_token is not None

    def _resolve_expiration(self, exp_data):
        if exp_data is None or self._expire_seconds is None:
            return None
        try:
            return calendar.timegm(time.strptime(exp_data, "%Y-%m-%dT%H:%M:%SZ"))
        except:
            return None

    def _reload_account(self):
        ts = None
        if "ODPS_STS_ACCOUNT_FILE" in os.environ:
            token_file_name = os.getenv("ODPS_STS_ACCOUNT_FILE")
            if token_file_name and os.path.exists(token_file_name):
                with open(token_file_name, "r") as token_file:
                    token_json = json.load(token_file)
                self.access_id = token_json["accessKeyId"]
                self.secret_access_key = token_json["accessKeySecret"]
                self.sts_token = token_json["securityToken"]
                ts = self._resolve_expiration(token_json.get("expiration"))

                logger.info("STS token reloaded: %s", self.sts_token)
        elif "ODPS_STS_ACCESS_KEY_ID" in os.environ:
            self.access_id = os.getenv("ODPS_STS_ACCESS_KEY_ID")
            self.secret_access_key = os.getenv("ODPS_STS_ACCESS_KEY_SECRET")
            self.sts_token = os.getenv("ODPS_STS_TOKEN")
            logger.info("STS token reloaded: %s", self.sts_token)

        return ts if ts is not None else None


class BearerTokenAccount(TempAccountMixin, BaseAccount):
    def __init__(
        self,
        token=None,
        expired_hours=DEFAULT_TEMP_ACCOUNT_HOURS,
        get_bearer_token_fun=None,
    ):
        self.token = token
        self._custom_bearer_token_func = get_bearer_token_fun
        TempAccountMixin.__init__(self, expired_hours=expired_hours)

    @classmethod
    def from_environments(cls):
        expired_hours = int(
            os.getenv("ODPS_BEARER_TOKEN_HOURS", str(DEFAULT_TEMP_ACCOUNT_HOURS))
        )
        kwargs = {"expired_hours": expired_hours}
        if "ODPS_BEARER_TOKEN_FILE" in os.environ:
            return cls(**kwargs)
        elif "ODPS_BEARER_TOKEN" in os.environ:
            kwargs["expired_hours"] = None
            return cls(os.environ["ODPS_BEARER_TOKEN"], **kwargs)
        return None

    def _get_bearer_token(self):
        if self._custom_bearer_token_func is not None:
            return self._custom_bearer_token_func()

        token_file_name = os.getenv("ODPS_BEARER_TOKEN_FILE")
        if token_file_name and os.path.exists(token_file_name):
            with open(token_file_name, "r") as token_file:
                return token_file.read().strip()
        else:  # pragma: no cover
            from cupid.runtime import RuntimeContext, context

            if not RuntimeContext.is_context_ready():
                return
            cupid_context = context()
            return cupid_context.get_bearer_token()

    def _is_account_valid(self):
        return self.token is not None

    def _reload_account(self):
        token = self._get_bearer_token()
        logger.info("Bearer token reloaded: %s", token)
        self.token = token
        try:
            resolved_token_parts = base64.b64decode(token).decode().split(",")
            return int(resolved_token_parts[2])
        except:
            return None

    def sign_request(self, req, endpoint, region_name=None):
        self.reload()
        url = req.url[len(endpoint) :]
        url_components = urlparse(unquote(url), allow_fragments=False)
        self._build_canonical_str(url_components, req)
        if self.token is None:
            raise TypeError("Cannot sign request with None bearer token")
        req.headers["x-odps-bearer-token"] = self.token
        if self._last_refresh_time:
            req.headers["x-pyodps-token-timestamp"] = str(self._last_refresh_time)
        logger.debug("headers after signing: %r", req.headers)


class CredentialProviderAccount(StsAccount):
    def __init__(self, credential_provider):
        self.provider = credential_provider
        super(CredentialProviderAccount, self).__init__(None, None, None)

    def _refresh_credential(self):
        try:
            credential = self.provider.get_credential()
        except:
            credential = self.provider.get_credentials()

        self.access_id = credential.get_access_key_id()
        self.secret_access_key = credential.get_access_key_secret()
        self.sts_token = credential.get_security_token()

    def sign_request(self, req, endpoint, region_name=None):
        utils.call_with_retry(self._refresh_credential)
        return super(CredentialProviderAccount, self).sign_request(
            req, endpoint, region_name=region_name
        )


def from_environments():
    for account_cls in (StsAccount, BearerTokenAccount):
        account = account_cls.from_environments()
        if account is not None:
            break
    return account
