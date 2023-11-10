# -*- coding: utf-8 -*-
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

import calendar
import json
import logging
import operator
from datetime import datetime

from . import utils
from .compat import (
    six,
    reduce,
    ElementTree as ET,
    ElementTreeParseError as ETParseError,
    TimeoutError
)
from .lib.requests import ConnectTimeout as RequestsConnectTimeout

logger = logging.getLogger(__name__)


class DatetimeOverflowError(OverflowError):
    pass


class DependencyNotInstalledError(Exception):
    pass


class InteractiveError(Exception):
    pass


def parse_response(resp, endpoint=None, tag=None):
    """Parses the content of response and returns an exception object.
    """
    try:
        try:
            content = resp.content
            root = ET.fromstring(content)
            code = root.find('./Code').text
            msg = root.find('./Message').text
            request_id = root.find('./RequestId').text
            host_id = root.find('./HostId').text
        except ETParseError:
            request_id = resp.headers.get('x-odps-request-id', None)
            if len(resp.content) > 0:
                obj = json.loads(resp.text)
                msg = obj['Message']
                code = obj.get('Code')
                host_id = obj.get('HostId')
                if request_id is None:
                    request_id = obj.get('RequestId')
            else:
                raise
        clz = globals().get(code, ODPSError)
        return clz(
            msg, request_id=request_id, code=code, host_id=host_id, endpoint=endpoint, tag=tag
        )
    except:
        # Error occurred during parsing the response. We ignore it and delegate
        # the situation to caller to handle.
       logger.debug(utils.stringify_expt())

    if resp.status_code == 404:
        return NoSuchObject('No such object.', endpoint=endpoint, tag=tag)
    else:
        text = resp.content.decode() if six.PY3 else resp.content
        if text:
            if resp.status_code == 502 and _nginx_bad_gateway_message in text:
                return BadGatewayError(
                    text, code=str(resp.status_code), endpoint=endpoint, tag=tag
                )
            else:
                return ODPSError(text, code=str(resp.status_code), endpoint=endpoint, tag=tag)
        else:
            return ODPSError(str(resp.status_code), endpoint=endpoint, tag=tag)


def throw_if_parsable(resp, endpoint=None, tag=None):
    """Try to parse the content of the response and raise an exception
    if necessary.
    """
    raise parse_response(resp, endpoint, tag)


_CODE_MAPPING = {
    "ODPS-0010000": "InternalServerError",
    "ODPS-0110141": "DataVersionError",
    "ODPS-0123055": "ScriptError",
    "ODPS-0130131": "NoSuchTable",
    "ODPS-0130013": "NoPermission",
    "ODPS-0430055": "InternalConnectionError",
}

_SQA_CODE_MAPPING = {
    'ODPS-180': 'SQAGenericError',
    'ODPS-181': 'SQARetryError',
    'ODPS-182': 'SQAAccessDenied',
    'ODPS-183': 'SQAResourceNotEnough',
    'ODPS-184': 'SQAServiceUnavailable',
    'ODPS-185': 'SQAUnsupportedFeature',
    'ODPS-186': 'SQAQueryTimedout',
}

_nginx_bad_gateway_message = "the page you are looking for is currently unavailable"


def parse_instance_error(msg):
    msg = utils.to_str(msg)
    msg_parts = reduce(operator.add, (pt.split(':') for pt in msg.split(' - ')))
    msg_parts = [pt.strip() for pt in msg_parts]
    try:
        msg_code = next(p for p in msg_parts if p.startswith('ODPS-'))
        if msg_code in _CODE_MAPPING:
            cls = globals().get(_CODE_MAPPING[msg_code], ODPSError)
        elif len(msg_code) > 8 and msg_code[:8] in _SQA_CODE_MAPPING:
            # sometimes SQA will report nested odps errors.
            # return the outer error type instead of the inner one.
            cls = globals().get(_SQA_CODE_MAPPING[msg_code[:8]], ODPSError)
            return cls(msg, code=msg_code)
        else:
            cls = ODPSError
    except StopIteration:
        cls = ODPSError
        msg_code = None

    return cls(msg, code=msg_code)


class ODPSError(RuntimeError):
    """Base class of ODPS error"""
    def __init__(
        self, msg, request_id=None, code=None, host_id=None, instance_id=None, endpoint=None, tag=None
    ):
        super(ODPSError, self).__init__(msg)
        self.request_id = request_id
        self.instance_id = instance_id
        self.code = code
        self.host_id = host_id
        self.endpoint = endpoint
        self.tag = tag

    def __str__(self):
        message = self.args[0]

        head_parts = []
        if self.code:
            head_parts.append("%s:" % self.code)
        if self.request_id:
            head_parts.append("RequestId: %s" % self.request_id)
        if self.instance_id:
            head_parts.append("InstanceId: %s" % self.instance_id)
        if self.tag:
            head_parts.append("Tag: %s" % self.tag)
        if self.endpoint:
            head_parts.append("Endpoint: %s" % self.endpoint)

        if head_parts:
            return '%s\n%s' % (" ".join(head_parts), message)
        return message

    @classmethod
    def parse(cls, resp):
        return parse_response(resp)


class ODPSClientError(ODPSError):
    pass


class ConnectTimeout(ODPSError, TimeoutError, RequestsConnectTimeout):
    pass


class DataHealthManagerError(ODPSError):
    pass


class ServerDefinedException(ODPSError):
    pass


# A long list of server defined exceptions

class MethodNotAllowed(ServerDefinedException):
    pass


class NoSuchObject(ServerDefinedException):
    pass


class NoSuchPartition(NoSuchObject):
    pass


class NoSuchPath(NoSuchObject):
    pass


class NoSuchTable(NoSuchObject):
    pass


class InvalidArgument(ServerDefinedException):
    pass


class AuthorizationRequired(ServerDefinedException):
    pass


class Unauthorized(AuthorizationRequired):
    pass


class SchemaParseError(ServerDefinedException):
    pass


class InvalidStateSetting(ServerDefinedException):
    pass


class InvalidProjectTable(ServerDefinedException):
    pass


class NoPermission(ServerDefinedException):
    pass


class InternalServerError(ServerDefinedException):
    pass


class ReadMetaError(InternalServerError):
    pass


class ServiceUnavailable(InternalServerError):
    pass


class ScriptError(ServerDefinedException):
    pass


class ParseError(ServerDefinedException):
    pass


class DataVersionError(InternalServerError):
    pass


class BadGatewayError(InternalServerError):
    pass


class InstanceTypeNotSupported(ServerDefinedException):
    pass


class InvalidParameter(ServerDefinedException):
    pass


class StreamSessionNotFound(ServerDefinedException):
    pass


class UpsertSessionNotFound(ServerDefinedException):
    pass


class OverwriteModeNotAllowed(ServerDefinedException):
    pass


class TableModified(ServerDefinedException):
    pass


class RequestTimeTooSkewed(ServerDefinedException):
    def __init__(self, msg, *args, **kwargs):
        super(RequestTimeTooSkewed, self).__init__(msg, *args, **kwargs)
        try:
            parts = msg.split(',')
            kv_dict = dict(tuple(s.strip() for s in p.split(':', 1)) for p in parts)
            self.max_interval_date = int(kv_dict['max_interval_date'])
            self.expire_date = self._parse_error_date(kv_dict['expire_date'])
            self.now_date = self._parse_error_date(kv_dict['now_date'])
        except:
            self.max_interval_date = None
            self.expire_date = None
            self.now_date = None

    @staticmethod
    def _parse_error_date(date_str):
        date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        micros = date_obj.microsecond
        return datetime.fromtimestamp(calendar.timegm(date_obj.timetuple())).replace(microsecond=micros)

# Handling error code typo in ODPS error message
RequestTimeTooSkewd = RequestTimeTooSkewed


class NotSupportedError(ODPSError):
    pass


class WaitTimeoutError(ODPSError, TimeoutError):
    pass


class SecurityQueryError(ODPSError):
    pass


class NoSuchProject(ODPSError):
    pass


class OSSSignUrlError(ODPSError):
    def __init__(self, err):
        if isinstance(err, six.string_types):
            super(OSSSignUrlError, self).__init__(err)
            self.oss_exception = None
        else:
            super(OSSSignUrlError, self).__init__(str(err))
            self.oss_exception = err


class SQAError(ODPSError):
    pass


class SQAGenericError(SQAError):
    pass


# if this error is thrown, you may retry your request.
class SQARetryError(SQAError):
    pass


class SQAAccessDenied(SQAError):
    pass


class SQAResourceNotEnough(SQAError):
    pass


class SQAServiceUnavailable(SQAError):
    pass


class SQAUnsupportedFeature(SQAError):
    pass


class SQAQueryTimedout(SQAError):
    pass
