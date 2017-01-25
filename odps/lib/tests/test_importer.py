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

from __future__ import absolute_import
import zipfile
import tarfile
import sys
import os

from odps.compat import BytesIO as StringIO, six
from odps.tests.core import TestBase
from odps.compat import unittest
from odps.lib.importer import CompressImporter
from odps.utils import to_binary


class Test(TestBase):
    def testImport(self):
        raw_meta_path = sys.meta_path

        zip_io = StringIO()
        zip_f = zipfile.ZipFile(zip_io, 'w')
        zip_f.writestr('testa/a.py', 'a = 1')
        zip_f.close()

        tar_io = StringIO()
        tar_f = tarfile.TarFile(fileobj=tar_io, mode='w')
        tar_f.addfile(tarfile.TarInfo(name='testb/__init__.py'), fileobj=StringIO())
        info = tarfile.TarInfo(name='testb/b.py')
        c = to_binary('from a import a; b = a + 1')
        s = StringIO(c)
        info.size = len(c)
        tar_f.addfile(info, fileobj=s)
        tar_f.close()

        zip_io.seek(0)
        tar_io.seek(0)

        zip_f = zipfile.ZipFile(zip_io)
        tar_f = tarfile.TarFile(fileobj=tar_io)

        sys.meta_path.append(CompressImporter(zip_f, tar_f))

        try:
            from testb.b import b
            self.assertEqual(b, 2)
        finally:
            sys.meta_path = raw_meta_path

    def testRealImport(self):
        raw_meta_path = sys.meta_path

        six_path = os.path.join(os.path.dirname(os.path.abspath(six.__file__)), 'six.py')
        zip_io = StringIO()
        zip_f = zipfile.ZipFile(zip_io, 'w')
        zip_f.write(six_path, arcname='mylib/five.py')
        zip_f.close()
        zip_io.seek(0)

        zip_f = zipfile.ZipFile(zip_io)

        sys.meta_path.append(CompressImporter(zip_f))

        try:
            import five
            self.assertEqual(list(to_binary('abc')), list(five.binary_type(to_binary('abc'))))
        finally:
            sys.meta_path = raw_meta_path



if __name__ == '__main__':
    unittest.main()