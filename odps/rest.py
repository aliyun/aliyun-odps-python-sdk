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

"""Restful client enhanced by URL building and request signing facilities.
"""
from __future__ import absolute_import

import json
import logging
import os
import platform
import re
import threading
from string import Template

import requests

try:
    from requests import ConnectTimeout
except ImportError:
    from requests import Timeout as ConnectTimeout
try:
    import requests_unixsocket
except ImportError:
    requests_unixsocket = None

from . import __version__, errors, utils
from .compat import six, urlparse
from .config import options
from .utils import clear_survey_calls, get_package_version, get_survey_calls

try:
    import requests.packages.urllib3.util.ssl_

    requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = "ALL"
    requests.packages.urllib3.disable_warnings()
except ImportError:
    pass

try:
    import urllib3.util.ssl_

    urllib3.util.ssl_.DEFAULT_CIPHERS = "ALL"
    urllib3.disable_warnings()
except ImportError:
    pass


logger = logging.getLogger(__name__)

_default_user_agent = None

_v4_sign_fallback_msgs = [
    "need ak v3 support",
    "accesskey acl denied",
]


def default_user_agent():
    global _default_user_agent
    if _default_user_agent is not None:
        return _default_user_agent

    py_implementation = platform.python_implementation()
    py_version = platform.python_version()
    try:
        py_system = platform.system()
        py_release = platform.release()
    except IOError:
        py_system = "Unknown"
        py_release = "Unknown"

    ua_template = Template(
        options.user_agent_pattern
        or os.getenv("PYODPS_USER_AGENT_PATTERN")
        or "$pyodps_version $mars_version $maxframe_version $python_version $os_version"
    )
    substitutes = dict(
        pyodps_version="%s/%s" % ("pyodps", __version__),
        python_version="%s/%s" % (py_implementation, py_version),
        os_version="%s/%s" % (py_system, py_release),
        mars_version="",
        maxframe_version="",
    )

    try:
        from mars import __version__ as mars_version
    except:
        mars_version = None
    if mars_version:
        substitutes["mars_version"] = "%s/%s" % ("mars", mars_version)

    try:
        maxframe_version = get_package_version("maxframe")
    except:
        maxframe_version = None
    if maxframe_version:
        substitutes["maxframe_version"] = "%s/%s" % ("maxframe", maxframe_version)

    _default_user_agent = ua_template.safe_substitute(**substitutes)
    _default_user_agent = re.sub(" +", " ", _default_user_agent).strip()

    try:
        from .internal.rest import get_internal_user_agent_suffix

        _default_user_agent += " " + get_internal_user_agent_suffix()
    except:
        pass
    return _default_user_agent


class RestClient(object):
    _session_local = threading.local()
    _endpoints_without_v4_sign = set()

    def __init__(
        self,
        account,
        endpoint,
        project=None,
        schema=None,
        user_agent=None,
        region_name=None,
        namespace=None,
        **kwargs
    ):
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
        self._account = account
        self._endpoint = endpoint
        self._region_name = region_name
        self._user_agent = user_agent or default_user_agent()
        self.project = project
        self.schema = schema
        self.namespace = namespace
        self._proxy = kwargs.get("proxy")
        self._app_account = kwargs.get("app_account")
        self._tag = kwargs.get("tag")
        if isinstance(self._proxy, six.string_types):
            self._proxy = dict(http=self._proxy, https=self._proxy)

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def account(self):
        return self._account

    @property
    def app_account(self):
        return self._app_account

    @property
    def region_name(self):
        return self._region_name

    @property
    def session(self):
        try:
            session_cache = type(self)._session_local.session_cache
        except AttributeError:
            session_cache = type(self)._session_local.session_cache = dict()
        try:
            return session_cache[self._endpoint]
        except KeyError:
            pass

        parsed_url = urlparse(self._endpoint)
        adapter_options = dict(
            pool_connections=options.pool_connections,
            pool_maxsize=options.pool_maxsize,
            max_retries=options.retry_times,
        )
        if parsed_url.scheme == "http+unix":
            session = requests_unixsocket.Session()
            session.mount(
                "http+unix://",
                requests_unixsocket.adapters.UnixAdapter(**adapter_options),
            )
        else:
            session = requests.Session()
            # mount adapters with retry times
            session.mount("http://", requests.adapters.HTTPAdapter(**adapter_options))
            session.mount("https://", requests.adapters.HTTPAdapter(**adapter_options))
        session_cache[self._endpoint] = session
        return session

    def request(self, url, method, stream=False, **kwargs):
        sign_region_name = kwargs.get("region_name") or self._region_name
        if (
            self._endpoint in self._endpoints_without_v4_sign
            or not options.enable_v4_sign
        ):
            sign_region_name = None

        auth_expire_retried = False

        while True:
            kwargs["region_name"] = sign_region_name
            try:
                return self._request(url, method, stream=stream, **kwargs)
            except errors.InternalServerError as ex:
                ex_msg = str(ex).lower()
                if sign_region_name is None or all(
                    msg not in ex_msg for msg in _v4_sign_fallback_msgs
                ):
                    raise
                logger.info(
                    "Fallback of V4 signature for %s. Error message: %s", url, ex
                )
                self._endpoints_without_v4_sign.add(self._endpoint)
                sign_region_name = None
            except errors.InvalidParameter as ex:
                if sign_region_name is None or "ODPS-0410051" not in str(ex):
                    # Invalid credentials error not received from server
                    raise
                logger.info(
                    "Fallback of V4 signature for %s. Error message: %s", url, ex
                )
                self._endpoints_without_v4_sign.add(self._endpoint)
                sign_region_name = None
            except errors.AuthorizationRequired as ex:
                if sign_region_name is None or "invalid or missing" not in str(ex):
                    raise
                logger.info(
                    "Fallback of V4 signature for %s. Error message: %s", url, ex
                )
                self._endpoints_without_v4_sign.add(self._endpoint)
                sign_region_name = None
            except errors.AuthenticationRequestExpired:
                if not hasattr(self.account, "reload") or auth_expire_retried:
                    raise
                logger.info(
                    "AuthenticationRequestExpired encountered with %r. "
                    "Will retry with reloaded account.",
                    self.account,
                )
                self.account.reload(True)
                auth_expire_retried = True

    def _request(self, url, method, stream=False, **kwargs):
        self.upload_survey_log()

        region_name = kwargs.pop("region_name", None)

        logger.debug("Start request.")
        logger.debug("%s: %s", method.upper(), url)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            for k, v in kwargs.items():
                logger.debug("%s: %s", k, v)

        # Construct user agent without handling the letter case.
        headers = kwargs.get("headers", {})
        headers = {k: str(v) for k, v in six.iteritems(headers)}
        headers["User-Agent"] = self._user_agent
        if self.namespace:
            headers["x-odps-namespace-id"] = self.namespace
        kwargs["headers"] = headers
        params = kwargs.setdefault("params", {})

        actions = kwargs.pop("actions", None) or kwargs.pop("action", None) or []
        if isinstance(actions, six.string_types):
            actions = [actions]
        if actions:
            separator = "?" if "?" not in url else "&"
            url += separator + "&".join(actions)

        curr_project = kwargs.pop("curr_project", None) or self.project
        if "curr_project" not in params and curr_project is not None:
            params["curr_project"] = curr_project

        curr_schema = kwargs.pop("curr_schema", None) or self.schema
        if "curr_schema" not in params and curr_schema is not None:
            params["curr_schema"] = curr_schema

        timeout = kwargs.pop("timeout", None)
        req = requests.Request(method, url, **kwargs)
        prepared_req = req.prepare()
        logger.debug("request url + params %s", prepared_req.path_url)

        prepared_req.headers.pop("Authorization", None)
        prepared_req.headers.pop("application-authentication", None)
        self._account.sign_request(
            prepared_req, self._endpoint, region_name=region_name
        )
        if getattr(self, "_app_account", None) is not None:
            self._app_account.sign_request(
                prepared_req, self._endpoint, region_name=region_name
            )

        if any(v is None for v in prepared_req.headers.values()):
            none_headers = [k for k, v in prepared_req.headers.items() if v is None]
            raise TypeError(
                "Value of headers %s cannot be None" % ", ".join(none_headers)
            )

        try:
            res = self.session.send(
                prepared_req,
                stream=stream,
                timeout=timeout or (options.connect_timeout, options.read_timeout),
                verify=options.verify_ssl,
                proxies=self._proxy,
            )
        except ConnectTimeout:
            raise errors.ConnectTimeout(
                "Connecting to endpoint %s timeout." % self._endpoint
            )

        logger.debug("response.status_code %d", res.status_code)
        logger.debug("response.headers: \n%s", res.headers)
        if not stream:
            logger.debug("response.content: %s\n", res.content)
        # Automatically detect error
        if not self.is_ok(res):
            errors.throw_if_parsable(res, self._endpoint, self._tag)
        return res

    def get(self, url, stream=False, **kwargs):
        return self.request(url, "get", stream=stream, **kwargs)

    def post(self, url, data=None, **kwargs):
        data = (
            utils.to_binary(data, encoding="utf-8")
            if isinstance(data, six.string_types)
            else data
        )
        return self.request(url, "post", data=data, **kwargs)

    def put(self, url, data=None, **kwargs):
        data = utils.to_binary(data) if isinstance(data, six.string_types) else data
        return self.request(url, "put", data=data, **kwargs)

    def head(self, url, **kwargs):
        return self.request(url, "head", **kwargs)

    def delete(self, url, **kwargs):
        return self.request(url, "delete", **kwargs)

    def upload_survey_log(self):
        try:
            from .models.core import RestModel

            survey = get_survey_calls()
            clear_survey_calls()
            if not survey:
                return
            if self.project is None:
                return
            url = "/".join(
                [self.endpoint, "projects", RestModel._encode(self.project), "logs"]
            )
            self.put(url, json.dumps(survey))
        except:
            pass

    # Misc helper methods
    def is_ok(self, resp):
        return resp.ok
