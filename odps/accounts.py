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
import urllib
import urlparse
import logging
import email.utils

from . import compat


LOG = logging.getLogger(__name__)


def parse_qsl(query):
    params = []
    tokens = query.split('&')
    for token in tokens:
        kv = token.split('=', 1)
        if len(kv) == 2 and not kv[1]:
            kv = [kv[0],]
        params.append(kv)
    return params


class AliyunAccount(object):
    """Account of aliyun.com
    """
    
    def __init__(self, access_id, secret_access_key):
         self.access_id = access_id
         self.secret_access_key = secret_access_key

    def sign_request(self, req, endpoint):
        url = req.url[len(endpoint):]
        url_compo = urlparse.urlparse(urllib.unquote(url))
        CanonicalizedResource = url_compo.path
        param_dict = {}
        if url_compo.query:
            params = parse_qsl(url_compo.query)
            params.sort(key=lambda a:a[0])
            for kv in params:
                assert kv[0] not in param_dict
                if len(kv) == 1:
                    # Parameters with only key lacking of value, such as 'key0' in
                    # 'key0=&k2=v2'
                    param_dict[kv[0]] = ''
                else:
                    param_dict[kv[0]] = kv[1]
            paramstr = '&'.join(['='.join(p) for p in params])
            CanonicalizedResource += '?%s' % paramstr
        # Build signing string
        lines = []
        headers_to_sign = {}
        method = req.method
        lines.append(method)
        
        headers = req.headers
        LOG.debug('headers before signing: %s' % headers)
        for k, v in headers.iteritems():
            k = k.lower()
            if k in ('content-type', 'content-md5') or k.startswith('x-odps'):
               headers_to_sign[k] = v
        for k in ('content-type', 'content-md5'):
            if k not in headers_to_sign:
                headers_to_sign[k] = ''
        date_str = headers.get('Date')
        if not date_str:
            req_date = email.utils.formatdate(usegmt=True)
            headers['Date'] = req_date
            date_str = req_date
        headers_to_sign['date'] = date_str
        for param_key, param_value in param_dict.iteritems():
            if param_key.startswith('x-odps-'):
                headers_to_sign[param_key] = param_value
        headers_to_sign = compat.OrderedDict([(k, headers_to_sign[k])
                                              for k in sorted(headers_to_sign)])
        LOG.debug('headers to sign: %s' % headers_to_sign)
        for k, v in headers_to_sign.iteritems():
            if k.startswith('x-odps-'):
                lines.append('%s:%s' % (k, v))
            else:
                lines.append(v)
        
        lines.append(CanonicalizedResource)
        canonicalString = '\n'.join(lines)
        LOG.debug('canonical string: ' + canonicalString)
        signature = base64.b64encode(hmac.new(self.secret_access_key, canonicalString, hashlib.sha1).digest())
        auth_str = 'ODPS %s:%s' % (self.access_id, signature)
        headers['Authorization'] = auth_str
        LOG.debug('headers after signing: ' + repr(headers))
