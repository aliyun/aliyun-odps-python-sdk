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
import os
import shutil
import sys
import tarfile
import tempfile
import textwrap
import zipfile

from odps.compat import BytesIO, six
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

    @staticmethod
    def _add_tar_directory(tar_file, path):
        info = tarfile.TarInfo(name=path)
        info.type = tarfile.DIRTYPE
        tar_file.addfile(info, fileobj=BytesIO())

    def testImport(self):
        zip_io = BytesIO()
        zip_f = zipfile.ZipFile(zip_io, 'w')
        zip_f.writestr('testa/a.py', 'a = 1')
        zip_f.close()

        tar_io = BytesIO()
        tar_f = tarfile.TarFile(fileobj=tar_io, mode='w')
        tar_f.addfile(tarfile.TarInfo(name='testb/__init__.py'), fileobj=BytesIO())
        info = tarfile.TarInfo(name='testb/b.py')
        c = b'from a import a; b = a + 1'
        s = BytesIO(c)
        info.size = len(c)
        tar_f.addfile(info, fileobj=s)
        tar_f.close()

        dict_io_init = dict()
        dict_io_init['/opt/test_pyodps_dev/testc/__init__.py'] = BytesIO()
        dict_io_init['/opt/test_pyodps_dev/testc/c.py'] = BytesIO(b'from a import a; c = a + 2')
        dict_io_init['/opt/test_pyodps_dev/testc/sub/__init__.py'] = BytesIO(b'from . import mod')
        dict_io_init['/opt/test_pyodps_dev/testc/sub/mod.py'] = BytesIO(b'from ..c import c')

        dict_io_file = dict()
        dict_io_file['/opt/test_pyodps_dev/testd/d.py'] = BytesIO(b'from a import a; d = a + 3')

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
        self.assertRaises(ImportError, __import__, 'sub', fromlist=[])
        from d import d
        self.assertEqual(d, 4)

    def testRootedArchiveImport(self):
        tar_io = BytesIO()
        tar_f = tarfile.TarFile(fileobj=tar_io, mode='w')
        self._add_tar_directory(tar_f, 'root')
        self._add_tar_directory(tar_f, 'root/testb.1.0')
        self._add_tar_directory(tar_f, 'root/testb.1.0/testb.info')
        tar_f.addfile(tarfile.TarInfo(name='root/testb.1.0/testb.info/INFO.txt'), fileobj=BytesIO())
        self._add_tar_directory(tar_f, 'root/testb.1.0/testb')
        tar_f.addfile(tarfile.TarInfo(name='root/testb.1.0/testb/__init__.py'), fileobj=BytesIO())
        info = tarfile.TarInfo(name='root/testb.1.0/testb/b.py')
        c = b'b = 2'
        s = BytesIO(c)
        info.size = len(c)
        tar_f.addfile(info, fileobj=s)
        tar_f.close()

        tar_io.seek(0)

        tar_f = tarfile.TarFile(fileobj=tar_io)
        importer.ALLOW_BINARY = False
        imp = CompressImporter(tar_f)
        sys.meta_path.append(imp)

        from testb.b import b
        self.assertEqual(b, 2)

    def testRealImport(self):
        six_path = os.path.join(os.path.dirname(os.path.abspath(six.__file__)), 'six.py')
        zip_io = BytesIO()
        zip_f = zipfile.ZipFile(zip_io, 'w')
        zip_f.write(six_path, arcname='mylib/five.py')
        zip_f.close()
        zip_io.seek(0)

        zip_f = zipfile.ZipFile(zip_io)

        sys.meta_path.append(CompressImporter(zip_f))

        import five
        self.assertEqual(list(to_binary('abc')), list(five.binary_type(to_binary('abc'))))

    def testImportWithPackageResource(self):
        test_src = textwrap.dedent("""
        import os
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src_file.txt"), "r") as sf:
            pass
        """)

        zip_io = BytesIO()
        zip_f = zipfile.ZipFile(zip_io, 'w')
        zip_f.writestr('test_all_imp/src_file.txt', '')
        zip_f.writestr('test_all_imp/__init__.py', test_src)
        zip_f.close()

        zip_io.seek(0)
        old_meta_path = [mp for mp in sys.meta_path]

        with self.assertRaises(IOError):
            sys.meta_path.append(CompressImporter(zipfile.ZipFile(zip_io, 'r'), extract_binary=True))
            __import__("test_all_imp")

        sys.meta_path = old_meta_path
        sys.modules.pop("test_all_imp", None)

        try:
            CompressImporter(zipfile.ZipFile(zip_io, 'r'), extract_all=True)
            __import__("test_all_imp")
        finally:
            shutil.rmtree(CompressImporter._extract_path)
            sys.meta_path = old_meta_path
            sys.modules.pop("test_all_imp", None)

    def testBinaryImport(self):
        zip_io = BytesIO()
        zip_f = zipfile.ZipFile(zip_io, 'w')
        zip_f.writestr('test_a/a.so', '')
        zip_f.writestr('test_a/__init__.py', '')
        zip_f.writestr('test_direct.py', '')
        zip_f.close()

        zip_io.seek(0)
        self.assertRaises(SystemError, CompressImporter, zipfile.ZipFile(zip_io, 'r'))

        try:
            zip_io.seek(0)
            CompressImporter(zipfile.ZipFile(zip_io, 'r'), extract_binary=True)
            self.assertTrue(os.path.exists(CompressImporter._extract_path))
            import test_direct, test_a
            del test_direct, test_a
            del sys.modules['test_direct'], sys.modules['test_a']
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

    def testRepeatImport(self):
        dict_io_init = dict()
        dict_io_init['/opt/test_pyodps_dev/testc/__init__.py'] = BytesIO()
        dict_io_init['/opt/test_pyodps_dev/testc/c.py'] = BytesIO(b'from a import a; c = a + 2')
        dict_io_init['/opt/test_pyodps_dev/testc/sub/__init__.py'] = BytesIO(b'from . import mod')
        dict_io_init['/opt/test_pyodps_dev/testc/sub/mod.py'] = BytesIO(b'from ..c import c')

        sys.meta_path.append(CompressImporter(dict_io_init))
        sys.meta_path.append(CompressImporter(dict_io_init))
        self.assertIn('/opt/test_pyodps_dev', sys.path)
        self.assertNotIn('/opt/test_pyodps_dev/testc', sys.path)
        self.assertNotIn('/opt/test_pyodps_dev/testc/sub', sys.path)


if __name__ == '__main__':
    unittest.main()
