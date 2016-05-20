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

"""A couple of authentication types in ODPS.
"""

import base64
import hmac
import hashlib
import logging

from odps.compat import six
from odps.compat import urlparse, unquote, parse_qsl

from . import compat, utils


LOG = logging.getLogger(__name__)


class AliyunAccount(object):
    """Account of aliyun.com
    """
    
    def __init__(self, access_id, secret_access_key):
         self.access_id = access_id
         self.secret_access_key = secret_access_key

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
