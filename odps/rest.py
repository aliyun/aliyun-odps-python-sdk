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
import logging
import platform

import requests
from . import __version__
from . import errors

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
    """A simple wrapper on requests api, with ODPS signing enabled.
    URLs are constructed by chaining of attributes accessing. 
    Example:
        >>> rest_client.projects.dev.tables.dual.get()
    """

    def __init__(self, account, endpoint):
        if endpoint.endswith('/'):
            endpoint = endpoint[:-1]
        self.__config_dict = {
            'account': account,
            'endpoint': endpoint
        }
        self.__reset()

    def __reset(self):
        self.__cached_tokens = [self.__config_dict['endpoint'],]

    def build_url(self):
        """Build the current resource url.
        Returns a url string and finish this build.
        """
        url = '/'.join(self.__cached_tokens)
        self.__reset()
        return url

    def request(self, method, stream=False, **kwargs):
        """Issue an restful http request and return the response object.
        method: one of restful methods
        """
        url = self.build_url()
        LOG.debug('Start request.')
        LOG.debug('url: ' + url)
        session = requests.Session()
        for k, v in kwargs.viewitems():
            LOG.debug(k + ': ' + str(v))

        # Construct user agent without handling the letter case.
        headers = kwargs.setdefault('headers', {})
        headers['User-Agent'] = default_user_agent()
        req = requests.Request(method, url, **kwargs)
        prepared_req = req.prepare()
        self.__config_dict['account'].sign_request(prepared_req,
                                                   self.__config_dict['endpoint'])
        res = session.send(prepared_req, stream=stream)
        LOG.debug('response.status_code %d' % res.status_code)
        LOG.debug('response.headers: \n%s' % res.headers)
        if not stream: LOG.debug('response.content: %s\n' % (res.content))
        # Automatically detect error
        if not self.is_ok(res):
            errors.throw_if_parsable(res)
        return res

    def get(self, stream=False, **kwargs):
        return self.request('get', stream=stream, **kwargs)

    def post(self, data, stream=False, **kwargs):
        return self.request('post', stream=stream, data=data, **kwargs)

    def put(self, data, stream=False, **kwargs):
        return self.request('put', stream=stream, data=data, **kwargs)

    def __getattr__(self, name):
        if name in self.__dict__:
            return super(RestClient, self).__getattr__(name)
        return self[name]

    def __getitem__(self, name):
        self.__cached_tokens.append(name)
        return self

    # Misc helper methods
    def is_ok(self, resp):
        return resp.status_code / 100 == 2

    def chunk_upload(self, fp, **kwds):
        def gen():
            fp.seek(0)
            data = fp.read()
            yield data
        resp = self.put(data=gen(), **kwds)
        if not self.is_ok(resp):
            raise Exception('upload error')
