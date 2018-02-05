#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from collections import namedtuple

from odps.tests.core import TestBase
from odps.errors import ODPSError, InternalServerError, ScriptError, RequestTimeTooSkewed, \
    parse_instance_error

_PseudoResponse = namedtuple('PseudoResponse', 'content text headers status_code')


class Test(TestBase):
    def testParseResponse(self):
        xml_response = """
        <Error>
            <Code>InternalServerError</Code>
            <Message>System internal error</Message>
            <RequestId>REQ_ID</RequestId>
            <HostId>host</HostId>
        </Error>
        """
        exc = ODPSError.parse(_PseudoResponse(xml_response, None, {}, 500))
        self.assertIsInstance(exc, InternalServerError)
        self.assertEqual(exc.code, 'InternalServerError')
        self.assertEqual(exc.args[0], 'System internal error')
        self.assertEqual(exc.request_id, 'REQ_ID')
        self.assertEqual(exc.host_id, 'host')

        json_response = """
        {
            "Code": "InternalServerError",
            "Message": "System internal error",
            "HostId": "host"
        }
        """
        exc = ODPSError.parse(_PseudoResponse(json_response, json_response,
                                              {'x-odps-request-id': 'REQ_ID'}, 500))
        self.assertIsInstance(exc, InternalServerError)
        self.assertEqual(exc.code, 'InternalServerError')
        self.assertEqual(exc.args[0], 'System internal error')
        self.assertEqual(exc.request_id, 'REQ_ID')
        self.assertEqual(exc.host_id, 'host')

    def testInstanceError(self):
        err_msg = "ODPS-0123055:Script exception - ValueError: unmarshallable object"
        exc = parse_instance_error(err_msg)
        self.assertIsInstance(exc, ScriptError)

        err_msg = "502 Update replicas failed"
        exc = parse_instance_error(err_msg)
        self.assertIsInstance(exc, ODPSError)

    def testParseRequestTimeSkew(self):
        import time
        from datetime import datetime
        from odps.compat import utc

        def get_timestamp(dt):
            if dt.tzinfo:
                delta = dt.astimezone(utc) - datetime(1970, 1, 1, tzinfo=utc)
                return (delta.microseconds + 0.0 +
                        (delta.seconds + delta.days * 24 * 3600) * 10 ** 6) / 10 ** 6
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
        exc = ODPSError.parse(_PseudoResponse(xml_response, None, {}, 500))
        self.assertIsInstance(exc, RequestTimeTooSkewed)
        self.assertEqual(exc.max_interval_date, 900000)
        self.assertEqual(get_timestamp(exc.expire_date),
                         get_timestamp(datetime(2018, 1, 20, 16, 20, 17, 12000, tzinfo=utc)))
        self.assertEqual(get_timestamp(exc.now_date),
                         get_timestamp(datetime(2018, 1, 20, 19, 20, 9, 34000, tzinfo=utc)))

        xml_response = """
        <Error>
            <Code>RequestTimeTooSkewd</Code>
            <RequestId>REQ_ID</RequestId>
            <HostId>host</HostId>
            <Message>ODPS-0410031:Authentication request expired - the expire time interval  exceeds the max limitation: 900000</Message>
        </Error>
        """
        exc = ODPSError.parse(_PseudoResponse(xml_response, None, {}, 500))
        self.assertIsInstance(exc, RequestTimeTooSkewed)
        self.assertEqual(exc.max_interval_date, None)
        self.assertIsNone(exc.now_date)
        self.assertIsNone(exc.expire_date)
