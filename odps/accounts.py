# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

"""A couple of authentication types in ODPS.
"""

import base64
import cgi
import hmac
import hashlib
import logging
import threading
import time

import requests

from .compat import six
from .compat import urlparse, unquote, parse_qsl
from . import compat, utils, options


LOG = logging.getLogger(__name__)


class BaseAccount(object):
    def _build_canonical_str(self, url_components, req):
        # Build signing string
        lines = [req.method, ]
        headers_to_sign = dict()

        canonical_resource = url_components.path
        params = dict()
        if url_components.query:
            params_list = sorted(parse_qsl(url_components.query, True),
                                 key=lambda it: it[0])
            assert len(params_list) == len(set(it[0] for it in params_list))
            params = dict(params_list)
            convert = lambda kv: kv if kv[1] != '' else (kv[0], )
            params_str = '&'.join(['='.join(convert(kv)) for kv in params_list])

            canonical_resource = '%s?%s' % (canonical_resource, params_str)

        headers = req.headers
        LOG.debug('headers before signing: %s' % headers)
        for k, v in six.iteritems(headers):
            k = k.lower()
            if k in ('content-type', 'content-md5') or k.startswith('x-odps'):
                headers_to_sign[k] = v
        for k in ('content-type', 'content-md5'):
            if k not in headers_to_sign:
                headers_to_sign[k] = ''
        date_str = headers.get('Date')
        if not date_str:
            req_date = utils.formatdate(usegmt=True)
            headers['Date'] = req_date
            date_str = req_date
        headers_to_sign['date'] = date_str
        for param_key, param_value in six.iteritems(params):
            if param_key.startswith('x-odps-'):
                headers_to_sign[param_key] = param_value

        headers_to_sign = compat.OrderedDict([(k, headers_to_sign[k])
                                              for k in sorted(headers_to_sign)])
        LOG.debug('headers to sign: %s' % headers_to_sign)
        for k, v in six.iteritems(headers_to_sign):
            if k.startswith('x-odps-'):
                lines.append('%s:%s' % (k, v))
            else:
                lines.append(v)

        lines.append(canonical_resource)
        return '\n'.join(lines)

    def sign_request(self, req, endpoint):
        raise NotImplementedError


class AliyunAccount(BaseAccount):
    """
    Account of aliyun.com
    """
    def __init__(self, access_id, secret_access_key):
        self.access_id = access_id
        self.secret_access_key = secret_access_key

    def sign_request(self, req, endpoint):
        url = req.url[len(endpoint):]
        url_components = urlparse(unquote(url))

        canonical_str = self._build_canonical_str(url_components, req)
        LOG.debug('canonical string: ' + canonical_str)

        signature = base64.b64encode(hmac.new(
            utils.to_binary(self.secret_access_key), utils.to_binary(canonical_str),
            hashlib.sha1).digest())
        auth_str = 'ODPS %s:%s' % (self.access_id, utils.to_text(signature))
        req.headers['Authorization'] = auth_str
        LOG.debug('headers after signing: ' + repr(req.headers))


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
                self.send_response(500)
                self.end_headers()

        def _do_POST(self):
            ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
            if ctype == 'multipart/form-data':
                postvars = cgi.parse_multipart(self.rfile, pdict)
            elif ctype == 'application/x-www-form-urlencoded':
                length = int(self.headers.get('content-length'))
                postvars = six.moves.urllib.parse.parse_qs(self.rfile.read(length), keep_blank_values=1)
            else:
                self.send_response(400)
                self.end_headers()
                return

            if self.server._token is not None:
                auth = self.headers.get('Authorization')
                if not auth:
                    self.send_response(401)
                    self.end_headers()
                    return
                method, content = auth.split(' ', 1)
                method = method.lower()
                if method == 'token':
                    if content != self.server._token:
                        self.send_response(401)
                        self.end_headers()
                        return
                else:
                    self.send_response(401)
                    self.end_headers()
                    return

            assert len(postvars[b'access_id']) == 1 and len(postvars[b'canonical']) == 1
            access_id = utils.to_str(postvars[b'access_id'][0])
            canonical = utils.to_str(postvars[b'canonical'][0])
            secret_access_key = self.server._accounts[access_id]

            signature = base64.b64encode(hmac.new(
                utils.to_binary(secret_access_key), utils.to_binary(canonical),
                hashlib.sha1).digest())
            auth_str = 'ODPS %s:%s' % (access_id, utils.to_text(signature))

            self.send_response(200)
            self.send_header("Content-Type", "text/json")
            self.end_headers()
            self.wfile.write(utils.to_binary(auth_str))

        def log_message(self, *args):
            return

    class SignServerCore(six.moves.socketserver.ThreadingMixIn, six.moves.BaseHTTPServer.HTTPServer):
        def __init__(self, *args, **kwargs):
            self._accounts = kwargs.pop('accounts', {})
            self._token = kwargs.pop('token', None)
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
            self._server = self.SignServerCore(endpoint, self.SignServerHandler,
                                               accounts=self.accounts, token=self.token)
            self._server.serve_forever()

        thread = threading.Thread(target=starter)
        thread.daemon = True
        thread.start()
        while self._server is None or not self._server._ready:
            time.sleep(0.05)

    def stop(self):
        self._server.stop()


class SignServerError(Exception):
    def __init__(self, msg, code):
        super(SignServerError, self).__init__(msg)
        self.code = code


class SignServerAccount(BaseAccount):
    _session_local = threading.local()

    def __init__(self, access_id, sign_endpoint=None, server=None, port=None, token=None):
        self.access_id = access_id
        self.sign_endpoint = sign_endpoint or (server, port)
        self.token = token

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

    def sign_request(self, req, endpoint):
        url = req.url[len(endpoint):]
        url_components = urlparse(unquote(url))

        canonical_str = self._build_canonical_str(url_components, req)
        LOG.debug('canonical string: ' + canonical_str)

        headers = dict()
        if self.token:
            headers['Authorization'] = 'token ' + self.token

        resp = self.session.request('post', 'http://%s:%s' % self.sign_endpoint, headers=headers,
                                    data=dict(access_id=self.access_id, canonical=canonical_str))
        if resp.status_code < 400:
            req.headers['Authorization'] = resp.text
            LOG.debug('headers after signing: ' + repr(req.headers))
        else:
            raise SignServerError('Sign server returned error code: %d' % resp.status_code, resp.status_code)
