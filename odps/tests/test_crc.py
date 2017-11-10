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

from odps.tests.core import TestBase
from odps.compat import unittest, reload_module
from odps import crc as _crc
from odps.config import option_context


def bothPyAndC(func):
    def inner(self, *args, **kwargs):
        try:
            import cython
            ts = 'py', 'c'
        except ImportError:
            ts = 'py',
            import warnings
            warnings.warn('No c code tests for crc32c')
        for t in ts:
            with option_context() as options:
                setattr(options, 'force_{0}'.format(t), True)

                reload_module(_crc)

                if t == 'py':
                    self.assertEquals(_crc.Crc32c._method, t)
                else:
                    self.assertFalse(hasattr(_crc.Crc32c, '_method'))
                func(self, *args, **kwargs)
    return inner


class Test(TestBase):
    @bothPyAndC
    def testCrc32c(self):
        crc = _crc.Crc32c()
        self.assertEquals(0, crc.getvalue())
        buf = bytearray(b'abc')
        crc.update(buf)
        self.assertEquals(910901175, crc.getvalue())
        buf = bytearray(b'1111111111111111111')
        crc.update(buf)
        self.assertEquals(2917307201, crc.getvalue())


if __name__ == '__main__':
    unittest.main()
