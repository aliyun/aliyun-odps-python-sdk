# -*- coding: utf-8 -*-
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


import json
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import logging

from . import utils


LOG = logging.getLogger(__name__)


class DependencyNotInstalledError(Exception):
    pass


class InteractiveError(Exception):
    pass


def parse_response(resp):
    """Parses the content of response and returns an exception object.
    """
    host_id, msg, code = None, None, None
    try:
        content = resp.content
        root = ET.fromstring(content)
        code = root.find('./Code').text
        msg = root.find('./Message').text
        request_id = root.find('./RequestId').text
        host_id = root.find('./HostId').text
    except ET.ParseError:
        request_id = resp.headers.get('x-odps-request-id', None)
        if len(resp.content) > 0:
            obj = json.loads(resp.text)
            msg = obj['Message']
            code = obj['Code']
        else:
            return
    clz = globals().get(code, ODPSError)
    return clz(msg, request_id, code, host_id)


def throw_if_parsable(resp):
    """Try to parse the content of the response and raise an exception
    if neccessary.
    """
    e = None
    try:
        e = parse_response(resp)
    except:
        # Error occurred during parsing the response. We ignore it and delegate
        # the situation to caller to handle.
        LOG.debug(utils.stringify_expt())

    if e is not None:
        raise e

    if resp.status_code == 404:
        raise NoSuchObject('No such object.')
    else:
        raise ODPSError(str(resp.status_code))


class ODPSError(RuntimeError):
    """
    """

    def __init__(self, msg, request_id=None, code=None, host_id=None):
        super(ODPSError, self).__init__(msg)
        self.request_id = request_id
        self.code = code
        self.host_id = host_id

    def __str__(self):
        if hasattr(self, 'message'):
            message = self.message
        else:
            message = self.args[0]  # py3
        if self.code:
            return '%s: %s' % (self.code, message)
        return message
    
    @classmethod
    def parse(cls, resp):
        content = resp.content
        try:
            error = parse_response(resp)
        except:
            try:
                root = json.loads(content)
                code = root.get('Code')
                msg = root.get('Message')
                request_id = root.get('RequestId')
                host_id = root.get('HostId')
                error = ODPSError(msg, request_id, code, host_id)
            except:
                # XXX: Can this happen?
                error = ODPSError(content, None)
        return error


class ServerDefinedException(ODPSError):

    def __str__(self):
        return super(ODPSError, self).__str__()


# A long list of server defined exceptions

class MethodNotAllowed(ServerDefinedException):
    pass


class NoSuchObject(ServerDefinedException):
    pass


class InvalidArgument(ServerDefinedException):
    pass


class Unauthorized(ServerDefinedException):
    pass


class SchemaParseError(ServerDefinedException):
    pass


class InvalidStateSetting(ServerDefinedException):
    pass
