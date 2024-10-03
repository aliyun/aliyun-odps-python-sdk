#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from collections import namedtuple

from ..errors import (
    BadGatewayError,
    InternalServerError,
    ODPSError,
    RequestTimeTooSkewed,
    ScriptError,
    parse_instance_error,
    parse_response,
)

_PseudoResponse = namedtuple("PseudoResponse", "content text headers status_code")


def test_xml_parse_response():
    xml_response = """
    <Error>
        <Code>InternalServerError</Code>
        <Message>System internal error</Message>
        <RequestId>REQ_ID</RequestId>
        <HostId>host</HostId>
    </Error>
    """
    exc = parse_response(_PseudoResponse(xml_response, None, {}, 500))
    assert isinstance(exc, InternalServerError)
    assert exc.code == "InternalServerError"
    assert exc.args[0] == "System internal error"
    assert exc.request_id == "REQ_ID"
    assert exc.host_id == "host"


def test_json_parse_response():
    json_response = """
    {
        "Code": "InternalServerError",
        "Message": "System internal error",
        "HostId": "host"
    }
    """
    exc = parse_response(
        _PseudoResponse(
            json_response.encode(), json_response, {"x-odps-request-id": "REQ_ID"}, 500
        )
    )
    assert isinstance(exc, InternalServerError)
    assert exc.code == "InternalServerError"
    assert exc.args[0] == "System internal error"
    assert exc.request_id == "REQ_ID"
    assert exc.host_id == "host"


def test_nginx_gateway_error():
    nginx_unavailable_response = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>Error</title>
    <style>
        body {
            width: 35em;
            margin: 0 auto;
            font-family: Tahoma, Verdana, Arial, sans-serif;
        }
    </style>
    </head>
    <body>
    <h1>An error occurred.</h1>
    <p>Sorry, the page you are looking for is currently unavailable.<br/>
    Please try again later.</p>
    <p>If you are the system administrator of this resource then you should check
    the <a href="http://nginx.org/r/error_log">error log</a> for details.</p>
    <p><em>Faithfully yours, nginx.</em></p>
    </body>
    </html>
    """.strip()

    exc = parse_response(
        _PseudoResponse(
            nginx_unavailable_response.encode(), nginx_unavailable_response, {}, 502
        ),
        endpoint="http://mock_endpoint",
        tag="mock",
    )
    assert isinstance(exc, BadGatewayError)
    assert "mock" in str(exc)


def test_instance_error():
    err_msg = """
    <Error>
        <Code>InternalServerError</Code>
        <Message>System internal error</Message>
        <HostId>host</HostId>
    </Error>
    """
    exc = parse_instance_error(err_msg)
    assert isinstance(exc, InternalServerError)

    err_msg = "ODPS-0123055:Script exception - ValueError: unmarshallable object"
    exc = parse_instance_error(err_msg)
    assert isinstance(exc, ScriptError)

    err_msg = "502 Update replicas failed"
    exc = parse_instance_error(err_msg)
    assert isinstance(exc, ODPSError)


def test_parse_request_time_skew():
    import time
    from datetime import datetime

    from ..compat import utc

    def get_timestamp(dt):
        if dt.tzinfo:
            delta = dt.astimezone(utc) - datetime(1970, 1, 1, tzinfo=utc)
            return (
                delta.microseconds
                + 0.0
                + (delta.seconds + delta.days * 24 * 3600) * 10**6
            ) / 10**6
        else:
            return time.mktime(dt.timetuple()) + dt.microsecond / 1000000.0

    xml_response = """
    <Error>
        <Code>RequestTimeTooSkewd</Code>
        <RequestId>REQ_ID</RequestId>
        <HostId>host</HostId>
        <Message>ODPS-0410031:Authentication request expired - the expire time interval  exceeds the max limitation: 900000, max_interval_date:900000,expire_date:2018-01-20T16:20:17.012Z,now_date:2018-01-20T19:20:09.034Z</Message>
    </Error>
    """
    exc = parse_response(_PseudoResponse(xml_response, None, {}, 500))
    assert isinstance(exc, RequestTimeTooSkewed)
    assert exc.max_interval_date == 900000
    assert get_timestamp(exc.expire_date) == get_timestamp(
        datetime(2018, 1, 20, 16, 20, 17, 12000, tzinfo=utc)
    )
    assert get_timestamp(exc.now_date) == get_timestamp(
        datetime(2018, 1, 20, 19, 20, 9, 34000, tzinfo=utc)
    )

    xml_response = """
    <Error>
        <Code>RequestTimeTooSkewd</Code>
        <RequestId>REQ_ID</RequestId>
        <HostId>host</HostId>
        <Message>ODPS-0410031:Authentication request expired - the expire time interval  exceeds the max limitation: 900000</Message>
    </Error>
    """
    exc = parse_response(_PseudoResponse(xml_response, None, {}, 500))
    assert isinstance(exc, RequestTimeTooSkewed)
    assert exc.max_interval_date == None
    assert exc.now_date is None
    assert exc.expire_date is None
