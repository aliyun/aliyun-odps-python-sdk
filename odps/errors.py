# -*- coding: utf-8 -*-
# Copyright 1999-2026 Alibaba Group Holding Ltd.
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
from functools import reduce

from requests import ConnectTimeout as RequestsConnectTimeout

from . import utils
from .compat import ElementTree as ET
from .compat import ElementTreeParseError as ETParseError

logger = logging.getLogger(__name__)


class DatetimeOverflowError(OverflowError):
    pass


class DependencyNotInstalledError(Exception):
    pass


class InteractiveError(Exception):
    pass


def parse_response(resp, endpoint=None, tag=None):
    """Parses the content of response and returns an exception object."""
    try:
        try:
            content = resp.content
            root = ET.fromstring(content)
            code = root.find("./Code").text
            msg = root.find("./Message").text
            request_id = root.find("./RequestId").text
            host_id = root.find("./HostId").text
        except ETParseError:
            request_id = resp.headers.get("x-odps-request-id", None)
            if len(resp.content) > 0:
                obj = json.loads(resp.text)
                if tag == "Catalog":
                    msg = obj["message"]
                    reason = obj.get("reason")
                    code = _CATALOG_ERROR_MAPPING.get(reason, reason)
                    host_id = None
                else:
                    msg = obj["Message"]
                    code = obj.get("Code")
                    host_id = obj.get("HostId")
                    if request_id is None:
                        request_id = obj.get("RequestId")
            else:
                raise
        clz, msg = _resolve_error_class_and_message(code, msg)
        return clz(
            msg,
            request_id=request_id,
            code=code,
            host_id=host_id,
            endpoint=endpoint,
            tag=tag,
            status_code=resp.status_code,
            response_headers=resp.headers,
        )
    except Exception:
        # Error occurred during parsing the response. We ignore it and delegate
        # the situation to caller to handle.
        logger.debug(utils.stringify_expt())

    if resp.status_code == 404:
        msg = "Not found error reported by server."
        if endpoint:
            msg += f" Endpoint {endpoint} might be malfunctioning."
        return NoSuchObject(
            msg, endpoint=endpoint, tag=tag, status_code=resp.status_code
        )
    elif resp.status_code == 401:
        return Unauthorized(
            "Unauthorized.", endpoint=endpoint, tag=tag, status_code=resp.status_code
        )
    else:
        text = resp.content.decode()
        if text:
            if resp.status_code == 502 and _nginx_bad_gateway_message in text:
                return BadGatewayError(
                    text,
                    code=str(resp.status_code),
                    endpoint=endpoint,
                    tag=tag,
                    status_code=resp.status_code,
                )
            else:
                return ODPSError(
                    text,
                    code=str(resp.status_code),
                    endpoint=endpoint,
                    tag=tag,
                    status_code=resp.status_code,
                )
        else:
            return ODPSError(
                str(resp.status_code),
                endpoint=endpoint,
                tag=tag,
                status_code=resp.status_code,
            )


def throw_if_parsable(resp, endpoint=None, tag=None):
    """Try to parse the content of the response and raise an exception
    if necessary.
    """
    raise parse_response(resp, endpoint, tag)


_CODE_MAPPING = {
    "ODPS-0010000": "InternalServerError",
    "ODPS-0110141": "DataVersionError",
    "ODPS-0123055": "ScriptError",
    "ODPS-0130013": "NoPermission",
    "ODPS-0130131": "NoSuchTable",
    "ODPS-0130161": "ParseError",
    "ODPS-0420153": "InternalServerError",
    "ODPS-0420411": "InvalidArgument",
    "ODPS-0430055": "InternalConnectionError",
}

_SQA_CODE_MAPPING = {
    "ODPS-180": "SQAGenericError",
    "ODPS-181": "SQARetryError",
    "ODPS-182": "SQAAccessDenied",
    "ODPS-183": "SQAResourceNotEnough",
    "ODPS-184": "SQAServiceUnavailable",
    "ODPS-185": "SQAUnsupportedFeature",
    "ODPS-186": "SQAQueryTimedout",
}

_CATALOG_ERROR_MAPPING = {
    "NotFound": "NoSuchObject",
}


def _resolve_error_class_and_message(code, msg):
    """Resolve the error class and combined message from an ODPS error code.

    Server responses may return the code as a full string like
    ``"ODPS-0420411: Invalid argument - "`` rather than a bare class name.
    This function extracts the ODPS code prefix by splitting on the first
    colon and looks it up in ``globals()`` and ``_CODE_MAPPING``.
    Falls back to a direct ``globals()`` lookup for backward compatibility
    with codes that are already class names.

    When the code is a composite string (not a bare class name), it already
    contains a human-readable description, so the code and message are
    combined into a single message string.

    Returns a tuple ``(clz, msg)``.
    """
    msg = msg or ""

    # Direct class-name lookup (e.g. "InvalidArgument", "NoSuchObject")
    clz = globals().get(code)
    if clz is not None:
        return clz, msg

    # Extract prefix by splitting on first colon
    prefix = code.split(":", 1)[0] if code else None
    prefix = prefix.strip()

    if prefix:
        # Try globals lookup on prefix (prefix itself may be a class name)
        clz = globals().get(prefix)
        # Try _CODE_MAPPING lookup on prefix
        if clz is None:
            class_name = _CODE_MAPPING.get(prefix)
            clz = globals().get(class_name) if class_name else None

    if isinstance(code, str) and not msg.startswith(code):
        msg = code.rstrip() + " " + msg.lstrip()

    return (clz or ODPSError), msg


_nginx_bad_gateway_message = "the page you are looking for is currently unavailable"


def parse_instance_error(msg):
    raw_msg = msg
    try:
        root = ET.fromstring(msg)
        code = root.find("./Code").text
        msg = root.find("./Message").text
        request_id_node = root.find("./RequestId")
        request_id = request_id_node.text if request_id_node else None
        host_id_node = root.find("./HostId")
        host_id = host_id_node.text if host_id_node else None

        clz, msg = _resolve_error_class_and_message(code, msg)
        return clz(msg, request_id=request_id, code=code, host_id=host_id)
    except Exception:
        pass

    msg = utils.to_str(raw_msg)
    msg_parts = reduce(operator.add, (pt.split(":") for pt in msg.split(" - ")))
    msg_parts = [pt.strip() for pt in msg_parts]
    try:
        msg_code = next(p for p in msg_parts if p.startswith("ODPS-"))
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


class BaseODPSError(Exception):
    """Base class of ODPS error"""

    def __init__(
        self,
        msg,
        request_id=None,
        code=None,
        host_id=None,
        instance_id=None,
        endpoint=None,
        tag=None,
        response_headers=None,
        status_code=None,
    ):
        super(BaseODPSError, self).__init__(msg)
        self.request_id = request_id
        self.instance_id = instance_id
        self.code = code
        self.host_id = host_id
        self.endpoint = endpoint
        self.tag = tag
        self.status_code = status_code

    def __str__(self):
        message = self.args[0]

        head_parts = []
        if self.code:
            head_parts.append(f"{self.code}:")
        if self.request_id:
            head_parts.append(f"RequestId: {self.request_id}")
        if self.instance_id:
            head_parts.append(f"InstanceId: {self.instance_id}")
        if self.tag:
            head_parts.append(f"Tag: {self.tag}")
        if self.endpoint:
            head_parts.append(f"Endpoint: {self.endpoint}")

        if head_parts:
            return f"{' '.join(head_parts)}\n{message}"
        return message

    @classmethod
    def parse(cls, resp):
        return parse_response(resp)


class ODPSError(BaseODPSError, RuntimeError):
    pass


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


class NoSuchProject(NoSuchObject):
    pass


class NoSuchPartition(NoSuchObject):
    pass


class NoSuchPath(NoSuchObject):
    pass


class NoSuchTable(NoSuchObject):
    pass


class NoSuchVolume(NoSuchObject):
    pass


class InvalidArgument(ServerDefinedException):
    pass


class AuthenticationRequestExpired(ServerDefinedException):
    pass


class AuthorizationRequired(ServerDefinedException):
    pass


class Unauthorized(AuthorizationRequired):
    pass


class SignatureNotMatch(ServerDefinedException):
    pass


class SchemaParseError(ServerDefinedException):
    pass


class InvalidStateSetting(ServerDefinedException):
    pass


class InstanceNotTerminate(ServerDefinedException):
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
    def __init__(self, *args, **kw):
        super(ParseError, self).__init__(*args, **kw)
        self.statement = None

    def __str__(self):
        message = super(ParseError, self).__str__()
        if self.statement is None:
            return message
        first_row, rests = message.split("\n", 1)
        statement_row = "SQL Statement: " + self.statement
        return "\n".join([first_row, statement_row, rests])


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


class SchemaModified(ServerDefinedException):
    def __init__(self, *args, **kw):
        super(SchemaModified, self).__init__(*args, **kw)
        response_headers = kw.get("response_headers") or dict()
        self.latest_schema_version = response_headers.get(
            "odps-tunnel-latest-schema-version"
        )


class NoSuchSchema(ServerDefinedException):
    pass


class RequestTimeTooSkewed(ServerDefinedException):
    def __init__(self, msg, *args, **kwargs):
        super(RequestTimeTooSkewed, self).__init__(msg, *args, **kwargs)
        try:
            parts = msg.split(",")
            kv_dict = dict(tuple(s.strip() for s in p.split(":", 1)) for p in parts)
            self.max_interval_date = int(kv_dict["max_interval_date"])
            self.expire_date = self._parse_error_date(kv_dict["expire_date"])
            self.now_date = self._parse_error_date(kv_dict["now_date"])
        except Exception:
            self.max_interval_date = None
            self.expire_date = None
            self.now_date = None

    @staticmethod
    def _parse_error_date(date_str):
        date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        micros = date_obj.microsecond
        return datetime.fromtimestamp(calendar.timegm(date_obj.timetuple())).replace(
            microsecond=micros
        )


# Handling error code typo in ODPS error message
RequestTimeTooSkewd = RequestTimeTooSkewed


class RequestQuotaExceeded(ServerDefinedException):
    pass


class SlotExceeded(RequestQuotaExceeded):
    pass


class QPSExceeded(RequestQuotaExceeded):
    pass


class FlowExceeded(RequestQuotaExceeded):
    pass


class NotSupportedError(ODPSError):
    pass


class WaitTimeoutError(ODPSError, TimeoutError):
    pass


class SecurityQueryError(ODPSError):
    pass


class OSSSignUrlError(ODPSError):
    def __init__(self, err):
        if isinstance(err, str):
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


class EmptyTaskInfoError(ODPSError):
    pass


class ChecksumError(ODPSError, IOError):
    pass


class StreamTruncatedException(ODPSError, IOError):
    """
    Raised when a tunnel download stream ends without receiving the footer
    tag (TUNNEL_META_COUNT), indicating the data stream may be truncated.

    Unlike regular I/O errors, stream truncation means data was lost and
    retrying will not recover it, so callers should not retry on this
    exception.
    """

    def __init__(self, message, records_read=0):
        super(StreamTruncatedException, self).__init__(message)
        self.records_read = records_read
