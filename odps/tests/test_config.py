#!/usr/bin/env python
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


from odps.config import options, option_context, is_integer, is_null, any_validator
from odps.tests.core import TestBase
from odps.compat import unittest


class Test(TestBase):

    def testOptions(self):
        with option_context() as local_options:
            self.assertIsNone(local_options.access_id)
            self.assertIsNone(local_options.access_key)
            self.assertIsNone(local_options.end_point)
            self.assertIsNone(local_options.default_project)
            self.assertIsNone(local_options.log_view_host)
            self.assertIsNone(local_options.tunnel_endpoint)
            self.assertGreater(local_options.chunk_size, 0)
            self.assertGreater(local_options.connect_timeout, 0)
            self.assertGreater(local_options.read_timeout, 0)
            self.assertIsNone(local_options.console.max_lines)
            self.assertIsNone(local_options.console.max_width)

            local_options.access_id = 'test'
            self.assertEqual(local_options.access_id, 'test')

            local_options.register_option('nest.inner.value', 50,
                                          validator=any_validator(is_null, is_integer))
            self.assertEqual(local_options.nest.inner.value, 50)
            def set(val):
                local_options.nest.inner.value = val
            self.assertRaises(ValueError, lambda: set('test'))
            set(None)
            self.assertIsNone(local_options.nest.inner.value)
            set(30)
            self.assertEqual(local_options.nest.inner.value, 30)

            local_options.console.max_width = 40
            self.assertEqual(local_options.console.max_width, 40)
            local_options.console.max_lines = 30
            self.assertEqual(local_options.console.max_lines, 30)

        self.assertIsNone(options.access_id)
        self.assertIsNone(options.access_key)
        self.assertIsNone(options.end_point)
        self.assertIsNone(options.default_project)
        self.assertIsNone(options.log_view_host)
        self.assertIsNone(options.tunnel_endpoint)
        self.assertGreater(options.chunk_size, 0)
        self.assertGreater(options.connect_timeout, 0)
        self.assertGreater(options.read_timeout, 0)
        self.assertIsNone(options.console.max_lines)
        self.assertIsNone(options.console.max_width)
        self.assertRaises(AttributeError, lambda: options.nest.inner.value)
        self.assertFalse(options.interactive)


if __name__ == '__main__':
    unittest.main()
