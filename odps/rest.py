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

"""Restful client enhanced by URL building and request signing facilities.
"""
from __future__ import absolute_import
import json
import logging
import platform
import threading
from contextlib import contextmanager
from string import Template

from . import __version__
from . import compat, errors, utils
from .compat import six
from .config import options
from .lib import requests
from .lib.requests import ConnectTimeout
from .utils import get_survey_calls, clear_survey_calls

try:
    import urllib3.util.ssl_
    urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL'
    urllib3.disable_warnings()
except ImportError:
    pass

if compat.LESS_PY32:
    mv_to_bytes = lambda v: bytes(bytearray(v))
else:
    mv_to_bytes = bytes


logger = logging.getLogger(__name__)

_default_user_agent = None


def default_user_agent():
    global _default_user_agent
    if _default_user_agent is not None:
        return _default_user_agent

    try:
        from mars import __version__ as mars_version
    except ImportError:
        mars_version = "Unknown"

    py_implementation = platform.python_implementation()
    py_version = platform.python_version()
    try:
        py_system = platform.system()
        py_release = platform.release()
    except IOError:
        py_system = 'Unknown'
        py_release = 'Unknown'

    ua_template = Template(
        options.user_agent_pattern or '$pyodps_version $mars_version $python_version $os_version'
    )
    _default_user_agent = ua_template.safe_substitute(
        pyodps_version='%s/%s' % ('pyodps', __version__),
        mars_version='%s/%s' % ('mars', mars_version),
        python_version='%s/%s' % (py_implementation, py_version),
        os_version='%s/%s' % (py_system, py_release)
    )
    return _default_user_agent


class RestUploadWriter(object):
    def __init__(self, ctx, chunk_size=None):
        self._ctx = ctx
        self._writer = None
        self._result = None
        self._chunk_size = chunk_size or options.chunk_size

    @property
    def result(self):
        return self._result

    def open(self):
        self._writer = self._ctx.__enter__()

    def write(self, data):
        if self._writer is None:
            raise IOError("Writer not opened")
        chunk_size = self._chunk_size
        data = memoryview(data)
        while data:
            to_send = mv_to_bytes(data[:chunk_size])
            data = data[chunk_size:]
            self._writer.write(to_send)

    def flush(self):
        pass

    def close(self):
        self.__exit__(None, None, None)
        return self._result

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc_info):
        self._ctx.__exit__(*exc_info)
        self._result = self._writer.result
        self._writer = None


class RestClient(object):
    _session_local = threading.local()

    def __init__(
        self, account, endpoint, project=None, schema=None, user_agent=None, **kwargs
    ):
        if endpoint.endswith('/'):
            endpoint = endpoint[:-1]
        self._account = account
        self._endpoint = endpoint
        self._user_agent = user_agent or default_user_agent()
        self.project = project
        self.schema = schema
        self._proxy = kwargs.get('proxy')
        self._app_account = kwargs.get('app_account')
        self._tag = kwargs.get('tag')
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
    def session(self):
        if not hasattr(type(self)._session_local, '_session'):
            adapter_options = dict(
                pool_connections=options.pool_connections,
                pool_maxsize=options.pool_maxsize,
                max_retries=options.retry_times,
            )
            session = requests.Session()
            # mount adapters with retry times
            session.mount(
                'http://', requests.adapters.HTTPAdapter(**adapter_options))
            session.mount(
                'https://', requests.adapters.HTTPAdapter(**adapter_options))

            self._session_local._session = session
        return self._session_local._session

    def request(self, url, method, stream=False, **kwargs):
        file_upload = kwargs.get("file_upload", False)
        chunk_size = kwargs.pop("chunk_size", None)
        if not file_upload:
            with self._request(url, method, stream=stream, **kwargs) as writer:
                pass
            return writer.result
        else:
            return RestUploadWriter(
                self._request(url, method, stream=stream, **kwargs), chunk_size=chunk_size
            )

    @contextmanager
    def _request(self, url, method, stream=False, **kwargs):
        self.upload_survey_log()

        file_upload = kwargs.get("file_upload", False)

        logger.debug('Start request.')
        logger.debug('url: %s', url)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            for k, v in kwargs.items():
                logger.debug('%s: %s', k, v)

        # Construct user agent without handling the letter case.
        headers = kwargs.get('headers', {})
        headers = {k: str(v) for k, v in six.iteritems(headers)}
        headers['User-Agent'] = self._user_agent
        kwargs['headers'] = headers
        params = kwargs.setdefault('params', {})

        actions = kwargs.pop("actions", None) or kwargs.pop("action", None) or []
        if isinstance(actions, six.string_types):
            actions = [actions]
        if actions:
            separator = "?" if "?" not in url else "&"
            url += separator + "&".join(actions)

        curr_project = kwargs.pop("curr_project", None) or self.project
        if 'curr_project' not in params and curr_project is not None:
            params['curr_project'] = curr_project

        curr_schema = kwargs.pop("curr_schema", None) or self.schema
        if 'curr_schema' not in params and curr_schema is not None:
            params['curr_schema'] = curr_schema

        timeout = kwargs.pop('timeout', None)
        req = requests.Request(method, url, **kwargs)
        prepared_req = req.prepare()
        logger.debug("request url + params %s", prepared_req.path_url)
        self._account.sign_request(prepared_req, self._endpoint)
        if getattr(self, '_app_account', None) is not None:
            self._app_account.sign_request(prepared_req, self._endpoint)

        try:
            res = self.session.send(
                prepared_req,
                stream=stream,
                timeout=timeout or (options.connect_timeout, options.read_timeout),
                verify=options.verify_ssl,
                proxies=self._proxy,
                file_upload=file_upload,
            )
            if not file_upload:
                writer = RestUploadWriter(None)
                yield writer
            else:
                with res as writer:
                    yield writer
                res = writer.result
        except ConnectTimeout:
            raise errors.ConnectTimeout('Connecting to endpoint %s timeout.' % self._endpoint)

        logger.debug('response.status_code %d', res.status_code)
        logger.debug('response.headers: \n%s', res.headers)
        if not stream:
            logger.debug('response.content: %s\n', res.content)
        # Automatically detect error
        if not self.is_ok(res):
            errors.throw_if_parsable(res, self._endpoint, self._tag)
        writer._result = res

    def get(self, url, stream=False, **kwargs):
        return self.request(url, 'get', stream=stream, **kwargs)

    def post(self, url, data=None, **kwargs):
        data = utils.to_binary(data, encoding='utf-8') if isinstance(data, six.string_types) else data
        return self.request(url, 'post', data=data, **kwargs)

    def put(self, url, data=None, **kwargs):
        data = utils.to_binary(data) if isinstance(data, six.string_types) else data
        return self.request(url, 'put', data=data, **kwargs)

    def head(self, url, **kwargs):
        return self.request(url, 'head', **kwargs)

    def delete(self, url, **kwargs):
        return self.request(url, 'delete', **kwargs)

    def upload_survey_log(self):
        try:
            from .models.core import RestModel

            survey = get_survey_calls()
            clear_survey_calls()
            if not survey:
                return
            if self.project is None:
                return
            url = '/'.join([self.endpoint, 'projects', RestModel._encode(self.project), 'logs'])
            self.put(url, json.dumps(survey))
        except:
            pass

    # Misc helper methods
    def is_ok(self, resp):
        return resp.ok
