#!/usr/bin/env python
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
        self.sys_modules = sys.modules
        self.sys_path = [p for p in sys.path]
        self.meta_path = [mp for mp in sys.meta_path]

    def tearDown(self):
        super(Test, self).tearDown()
        importer.ALLOW_BINARY = True
        for mod_name in sys.modules:
            if mod_name not in self.sys_modules:
                del sys.modules[mod_name]
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
        self.assertRaises(ImportError, __import__, 'c', fromlist=[])
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
        zip_f.writestr('test_a/a.so', '')
        zip_f.writestr('test_a/__init__.py', '')
        zip_f.writestr('testdir/test_b/b.so', '')
        zip_f.writestr('testdir/test_b/__init__.py', '')
        zip_f.writestr('test_direct.py', '')
        zip_f.close()

        zip_io.seek(0)
        self.assertRaises(SystemError, CompressImporter, zipfile.ZipFile(zip_io, 'r'))

        try:
            zip_io.seek(0)
            CompressImporter(zipfile.ZipFile(zip_io, 'r'), extract=True)
            self.assertTrue(os.path.exists(CompressImporter._extract_path))
            import test_direct, test_a, test_b
            del test_direct, test_a, test_b
            del sys.modules['test_direct'], sys.modules['test_a'], sys.modules['test_b']
        finally:
            shutil.rmtree(CompressImporter._extract_path)
            CompressImporter._extract_path = None

        six_path = os.path.join(os.path.dirname(os.path.abspath(six.__file__)), 'six.py')

        temp_path = tempfile.mkdtemp(prefix='tmp-pyodps-')
        lib_path = os.path.join(temp_path, 'mylib')
        lib_path2 = os.path.join(temp_path, 'mylib2')
        os.makedirs(lib_path)
        os.makedirs(lib_path2)

        lib_dict = dict()
        try:
            with open(os.path.join(lib_path, '__init__.py'), 'w'):
                pass
            shutil.copy(six_path, os.path.join(lib_path, 'fake_six.py'))
            dummy_bin = open(os.path.join(lib_path, 'dummy.so'), 'w')
            dummy_bin.close()

            sub_lib_path = os.path.join(lib_path, 'sub_path')
            os.makedirs(sub_lib_path)

            with open(os.path.join(sub_lib_path, '__init__.py'), 'w'):
                pass
            shutil.copy(six_path, os.path.join(sub_lib_path, 'fake_six.py'))

            lib_files = ['__init__.py', 'fake_six.py', 'dummy.so',
                         os.path.join('sub_path', '__init__.py'),
                         os.path.join('sub_path', 'fake_six.py')]
            lib_dict = dict((os.path.join(lib_path, fn), open(os.path.join(lib_path, fn), 'r'))
                            for fn in lib_files)

            importer.ALLOW_BINARY = False
            self.assertRaises(SystemError, CompressImporter, lib_dict)

            importer.ALLOW_BINARY = True
            importer_obj = CompressImporter(lib_dict)
            sys.meta_path.append(importer_obj)
            from mylib import fake_six
            self.assertEqual(list(to_binary('abc')), list(fake_six.binary_type(to_binary('abc'))))
            self.assertRaises(ImportError, __import__, 'sub_path', fromlist=[])

            with open(os.path.join(lib_path2, '__init__.py'), 'w'):
                pass
            shutil.copy(six_path, os.path.join(lib_path2, 'fake_six.py'))
            dummy_bin = open(os.path.join(lib_path2, 'dummy.so'), 'w')
            dummy_bin.close()

            sub_lib_path = os.path.join(lib_path2, 'sub_path2')
            os.makedirs(sub_lib_path)

            with open(os.path.join(sub_lib_path, '__init__.py'), 'w'):
                pass
            shutil.copy(six_path, os.path.join(sub_lib_path, 'fake_six.py'))

            lib_files = ['__init__.py', 'fake_six.py', 'dummy.so',
                         os.path.join('sub_path2', '__init__.py'),
                         os.path.join('sub_path2', 'fake_six.py')]

            importer.ALLOW_BINARY = True
            importer_obj = CompressImporter(lib_files)
            sys.meta_path.append(importer_obj)
            from mylib2 import fake_six
            self.assertEqual(list(to_binary('abc')), list(fake_six.binary_type(to_binary('abc'))))
            self.assertRaises(ImportError, __import__, 'sub_path2', fromlist=[])
        finally:
            [f.close() for f in six.itervalues(lib_dict)]
            shutil.rmtree(temp_path)


if __name__ == '__main__':
    unittest.main()
