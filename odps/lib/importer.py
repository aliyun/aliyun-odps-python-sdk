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

import zipfile
import tarfile
from collections import defaultdict
import os
import sys
import types


_SEARCH_ORDER = [
    ('.py', False),
    ('/__init__.py', True),
]


class CompressImportError(ImportError):
    """Exception raised by CompressImporter objects."""
    pass


class CompressImporter(object):
    """
    A PEP-302-style importer that can import from a zipfile.

    Just insert or append this class (not an instance) to sys.path_hooks
    and you're in business.  Instances satisfy both the 'importer' and
    'loader' APIs specified in PEP 302.
    """

    def __init__(self, *compressed_files):
        """
        Constructor.

        Args:
          compressed_files zipfile.ZipFile or tarfile.TarFile
        """
        self._files = compressed_files
        self._prefixes = defaultdict(lambda: set(['']))

        for f in self._files:
            if isinstance(f, zipfile.ZipFile):
                for name in f.namelist():
                    name = name if name.endswith(os.sep) else name.rsplit(os.sep, 1)[0]
                    if name in self._prefixes[f]:
                        continue
                    try:
                        f.getinfo(name + '__init__.py')
                    except KeyError:
                        self._prefixes[f].add(name)
            elif isinstance(f, tarfile.TarFile):
                for member in f.getmembers():
                    name = member.name if member.isdir() else member.name.rsplit(os.sep, 1)[0]
                    if name in self._prefixes[f]:
                        continue
                    try:
                        f.getmember(name + '/__init__.py')
                    except KeyError:
                        self._prefixes[f].add(name + '/')
            else:
                raise TypeError('Compressed file can be zipfile.ZipFile or tarfile.TarFile')

    def _get_info(self, fullmodname):
        """
        Internal helper for find_module() and load_module().

        Args:
          fullmodname: The dot-separated full module name, e.g. 'django.core.mail'.

        Returns:
          A tuple (submodname, is_package, relpath, fileobj) where:
            submodname: The final component of the module name, e.g. 'mail'.
            is_package: A bool indicating whether this is a package.
            relpath: The path to the module's source code within to the zipfile or tarfile.
            fileobj: The file object

        Raises:
          ImportError if the module is not found in the archive.
        """
        parts = fullmodname.split('.')
        submodname = parts[-1]
        for f in self._files:
            for prefix in self._prefixes[f]:
                for suffix, is_package in _SEARCH_ORDER:
                    l = [prefix] + parts[:-1] + [submodname + suffix.replace('/', os.sep)]
                    relpath = os.path.join(*l)
                    try:
                        relpath = relpath.replace(os.sep, '/')
                        if isinstance(f, zipfile.ZipFile):
                            f.getinfo(relpath)
                        else:
                            f.getmember(relpath)
                    except KeyError:
                        pass
                    else:
                        return submodname, is_package, relpath, f

        msg = 'Can\'t find module %s' % fullmodname
        raise CompressImportError(msg)

    def _get_source(self, fullmodname):
        """
        Internal helper for load_module().

        Args:
          fullmodname: The dot-separated full module name, e.g. 'django.core.mail'.

        Returns:
          A tuple (submodname, is_package, fullpath, source) where:
            submodname: The final component of the module name, e.g. 'mail'.
            is_package: A bool indicating whether this is a package.
            fullpath: The path to the module's source code including the
              zipfile's or tarfile's filename.
            source: The module's source code.

        Raises:
          ImportError if the module is not found in the archive.
        """

        submodname, is_package, relpath, fileobj = self._get_info(fullmodname)
        fullpath = '%s%s%s' % (fileobj, os.sep, relpath)
        if isinstance(fileobj, zipfile.ZipFile):
            source = fileobj.read(relpath.replace(os.sep, '/'))
        else:
            source = fileobj.extractfile(relpath.replace(os.sep, '/')).read()
        source = source.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        return submodname, is_package, fullpath, source

    def find_module(self, fullmodname, path=None):
        """
        PEP-302-compliant find_module() method.

        Args:
          fullmodname: The dot-separated full module name, e.g. 'django.core.mail'.
          path: Optional and ignored; present for API compatibility only.

        Returns:
          None if the module isn't found in the archive; self if it is found.
        """
        try:
            self._get_info(fullmodname)
        except ImportError:
            return None
        else:
            return self

    def load_module(self, fullmodname):
        """
        PEP-302-compliant load_module() method.

        Args:
          fullmodname: The dot-separated full module name, e.g. 'django.core.mail'.

        Returns:
          The module object constructed from the source code.

        Raises:
          SyntaxError if the module's source code is syntactically incorrect.
          ImportError if there was a problem accessing the source code.
          Whatever else can be raised by executing the module's source code.
        """
        submodname, is_package, fullpath, source = self._get_source(fullmodname)
        code = compile(source, fullpath, 'exec')
        mod = sys.modules.get(fullmodname)
        try:
            if mod is None:
                mod = sys.modules[fullmodname] = types.ModuleType(fullmodname)
            mod.__loader__ = self
            mod.__file__ = fullpath
            mod.__name__ = fullmodname
            if is_package:
                mod.__path__ = [os.path.dirname(mod.__file__)]
            exec(code, mod.__dict__)
        except:
            if fullmodname in sys.modules:
                del sys.modules[fullmodname]
            raise
        return mod
