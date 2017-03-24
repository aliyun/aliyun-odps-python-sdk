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

from copy import deepcopy

from odps.accounts import AliyunAccount
from odps.config import Config, options, option_context, is_integer, is_null, any_validator, OptionError
from odps.tests.core import TestBase, pandas_case
from odps.compat import unittest


class Test(TestBase):
    def testOptions(self):
        old_config = Config(deepcopy(options._config))

        with option_context() as local_options:
            if options.account is None:
                self.assertEqual(options.account, old_config.account)
            else:
                self.assertEqual(options.account.access_id, old_config.account.access_id)
                self.assertEqual(options.account.secret_access_key, old_config.account.secret_access_key)
            self.assertEqual(options.end_point, old_config.end_point)
            self.assertEqual(options.default_project, old_config.default_project)
            self.assertIsNotNone(local_options.log_view_host)
            self.assertIsNone(local_options.tunnel_endpoint)
            self.assertGreater(local_options.chunk_size, 0)
            self.assertGreater(local_options.connect_timeout, 0)
            self.assertGreater(local_options.read_timeout, 0)
            self.assertIsNone(local_options.console.max_lines)
            self.assertIsNone(local_options.console.max_width)

            local_options.account = AliyunAccount('test', '')
            self.assertEqual(local_options.account.access_id, 'test')

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

        if options.account is None:
            self.assertEqual(options.account, old_config.account)
        else:
            self.assertEqual(options.account.access_id, old_config.account.access_id)
            self.assertEqual(options.account.secret_access_key, old_config.account.secret_access_key)
        self.assertEqual(options.end_point, old_config.end_point)
        self.assertEqual(options.default_project, old_config.default_project)
        self.assertIsNotNone(options.log_view_host)
        self.assertIsNone(options.tunnel_endpoint)
        self.assertGreater(options.chunk_size, 0)
        self.assertGreater(options.connect_timeout, 0)
        self.assertGreater(options.read_timeout, 0)
        self.assertIsNone(options.console.max_lines)
        self.assertIsNone(options.console.max_width)
        self.assertRaises(AttributeError, lambda: options.nest.inner.value)
        self.assertFalse(options.interactive)

        def set_notexist():
            options.display.val = 3
        self.assertRaises(OptionError, set_notexist)

    def testRedirection(self):
        local_config = Config()

        local_config.register_option('test.redirect_src', 10)
        local_config.redirect_option('test.redirect_redir', 'test.redirect_src')

        self.assertIn('test', dir(local_config))
        self.assertIn('redirect_redir', dir(local_config.test))

        local_config.test.redirect_redir = 20
        self.assertEqual(local_config.test.redirect_src, 20)
        local_config.test.redirect_src = 10
        self.assertEqual(local_config.test.redirect_redir, 10)

        local_config.unregister_option('test.redirect_redir')
        local_config.unregister_option('test.redirect_src')
        self.assertRaises(AttributeError, lambda: local_config.test.redirect_redir)
        self.assertRaises(AttributeError, lambda: local_config.test.redirect_src)

    def testSetDisplayOption(self):
        options.display.max_rows = 10
        options.display.unicode.ambiguous_as_wide = True
        self.assertEqual(options.display.max_rows, 10)
        self.assertTrue(options.display.unicode.ambiguous_as_wide)
        options.register_pandas('display.non_exist', True)
        self.assertEqual(options.display.non_exist, True)

        try:
            import pandas as pd
            self.assertEqual(pd.options.display.max_rows, 10)
            self.assertTrue(pd.options.display.unicode.ambiguous_as_wide)
        except ImportError:
            pass

if __name__ == '__main__':
    unittest.main()
