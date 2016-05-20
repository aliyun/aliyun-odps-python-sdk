# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Restful client enhanced by URL building and request signing facilities.
"""
from __future__ import absolute_import
import json
import logging
import platform

import requests
import requests.packages.urllib3.util.ssl_
requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL'
requests.packages.urllib3.disable_warnings()

from . import __version__
from . import errors, utils
from .config import options
from .utils import get_survey_calls, clear_survey_calls
from .compat import six


LOG = logging.getLogger(__name__)


def default_user_agent():
    py_implementation = platform.python_implementation()
    py_version = platform.python_version()
    try:
        py_system = platform.system()
        py_release = platform.release()
    except IOError:
        py_system = 'Unknown'
        py_release = 'Unknown'
    return ' '.join(['%s/%s' % ('pyodps', __version__),
                     '%s/%s' % (py_implementation, py_version),
                     '%s/%s' % (py_system, py_release)])


class RestClient(object):
    def __init__(self, account, endpoint, project=None, user_agent=None):
        if endpoint.endswith('/'):
            endpoint = endpoint[:-1]
        self._account = account
        self._endpoint = endpoint
        self._user_agent = user_agent or default_user_agent()
        self.project = project

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def account(self):
        return self._account

    def request(self, url, method, stream=False, **kwargs):
        self.upload_survey_log()

        LOG.debug('Start request.')
        LOG.debug('url: ' + url)
        session = requests.Session()
        if LOG.level == logging.DEBUG:
            for k, v in kwargs.items():
                LOG.debug(k + ': ' + utils.to_text(v))

        # mount adapters with retry times
        session.mount(
            'http://', requests.adapters.HTTPAdapter(max_retries=options.retry_times))
        session.mount(
            'https://', requests.adapters.HTTPAdapter(max_retries=options.retry_times))

        # Construct user agent without handling the letter case.
        headers = kwargs.setdefault('headers', {})
        headers['User-Agent'] = self._user_agent
        params = kwargs.setdefault('params', {})
        if 'curr_project' not in params and self.project is not None:
            params['curr_project'] = self.project
        req = requests.Request(method, url, **kwargs)
        prepared_req = req.prepare()
        LOG.debug("request url + params %s" % prepared_req.path_url)
        self._account.sign_request(prepared_req, self._endpoint)

        res = session.send(prepared_req, stream=stream,
                           timeout=(options.connect_timeout, options.read_timeout),
                           verify=False)

        LOG.debug('response.status_code %d' % res.status_code)
        LOG.debug('response.headers: \n%s' % res.headers)
        if not stream: LOG.debug('response.content: %s\n' % (res.content))
        # Automatically detect error
        if not self.is_ok(res):
            errors.throw_if_parsable(res)
        return res

    def get(self, url, stream=False, **kwargs):
        return self.request(url, 'get', stream=stream, **kwargs)

    def post(self, url, data, **kwargs):
        data = utils.to_binary(data) if isinstance(data, six.string_types) else data
        return self.request(url, 'post', data=data, **kwargs)

    def put(self, url, data, **kwargs):
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
