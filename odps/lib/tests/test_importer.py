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
import shutil
import tempfile

from odps.compat import BytesIO as StringIO, six
from odps.tests.core import TestBase
from odps.compat import unittest
from odps.lib import importer
from odps.lib.importer import CompressImporter
from odps.utils import to_binary


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        importer.ALLOW_BINARY = True
        self.sys_path = [p for p in sys.path]
        self.meta_path = [mp for mp in sys.meta_path]

    def tearDown(self):
        super(Test, self).tearDown()
        importer.ALLOW_BINARY = True
        sys.path = self.sys_path
        sys.meta_path = self.meta_path

    def testImport(self):
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

        dict_io_init = dict()
        dict_io_init['/opt/test_pyodps_dev/testc/__init__.py'] = StringIO()
        dict_io_init['/opt/test_pyodps_dev/testc/c.py'] = StringIO(to_binary('from a import a; c = a + 2'))

        dict_io_file = dict()
        dict_io_file['/opt/test_pyodps_dev/testd/d.py'] = StringIO(to_binary('from a import a; d = a + 3'))

        zip_io.seek(0)
        tar_io.seek(0)

        zip_f = zipfile.ZipFile(zip_io)
        tar_f = tarfile.TarFile(fileobj=tar_io)

        importer.ALLOW_BINARY = False
        imp = CompressImporter(zip_f, tar_f, dict_io_init, dict_io_file)
        self.assertEqual(len(imp._files), 4)
        sys.meta_path.append(imp)

        from testb.b import b
        self.assertEqual(b, 2)
        from testc.c import c
        self.assertEqual(c, 3)
        from d import d
        self.assertEqual(d, 4)

    def testRealImport(self):
        six_path = os.path.join(os.path.dirname(os.path.abspath(six.__file__)), 'six.py')
        zip_io = StringIO()
        zip_f = zipfile.ZipFile(zip_io, 'w')
        zip_f.write(six_path, arcname='mylib/five.py')
        zip_f.close()
        zip_io.seek(0)

        zip_f = zipfile.ZipFile(zip_io)

        sys.meta_path.append(CompressImporter(zip_f))

        import five
        self.assertEqual(list(to_binary('abc')), list(five.binary_type(to_binary('abc'))))

    def testBinaryImport(self):
        zip_io = StringIO()
        zip_f = zipfile.ZipFile(zip_io, 'w')
        zip_f.writestr('testa/a.so', '')
        zip_f.close()
        self.assertRaises(SystemError, CompressImporter, zip_f)

        temp_path = tempfile.mkdtemp(prefix='tmp_pyodps')
        lib_path = os.path.join(temp_path, 'mylib')
        os.makedirs(lib_path)

        lib_dict = dict()
        try:
            six_path = os.path.join(os.path.dirname(os.path.abspath(six.__file__)), 'six.py')
            shutil.copy(six_path, os.path.join(lib_path, 'five.py'))
            dummy_bin = open(os.path.join(lib_path, 'dummy.so'), 'w')
            dummy_bin.close()

            lib_files = ['five.py', 'dummy.so']
            lib_dict = dict((os.path.join(lib_path, fn), open(os.path.join(lib_path, fn), 'r'))
                            for fn in lib_files)

            importer.ALLOW_BINARY = False
            self.assertRaises(SystemError, CompressImporter, lib_dict)

            importer.ALLOW_BINARY = True
            sys.meta_path.append(CompressImporter(lib_dict))
            import five
            self.assertEqual(list(to_binary('abc')), list(five.binary_type(to_binary('abc'))))
        finally:
            [f.close() for f in six.itervalues(lib_dict)]
            shutil.rmtree(temp_path)


if __name__ == '__main__':
    unittest.main()