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
_EXTRA_PACKAGE_PATH = 'work/extra_packages'


if hasattr(dict, 'itervalues'):
    iterkeys = lambda d: d.iterkeys()
    itervalues = lambda d: d.itervalues()
else:
    iterkeys = lambda d: d.keys()
    itervalues = lambda d: d.values()


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
        self._files = []
        self._prefixes = defaultdict(lambda: set(['']))
        self._bin_path = None

        for f in compressed_files:
            if isinstance(f, zipfile.ZipFile):
                bin_package = any(n.endswith('.so') or n.endswith('.pxd') or n.endswith('.dylib')
                                  for n in f.namelist())
                need_extract = True
            elif isinstance(f, tarfile.TarFile):
                bin_package = any(m.name.endswith('.so') or m.name.endswith('.pxd') or m.name.endswith('.dylib')
                                  for m in f.getmembers())
                need_extract = True
            elif isinstance(f, dict):
                bin_package = any(name.endswith('.so') or name.endswith('.pxd') or name.endswith('.dylib')
                                  for name in iterkeys(f))
                need_extract = False
            else:
                raise TypeError('Compressed file can only be zipfile.ZipFile or tarfile.TarFile')

            if bin_package:
                if not ALLOW_BINARY:
                    raise SystemError('Cannot load binary package. It is quite possible that you are using an old '
                                      'MaxCompute service which does not support binary packages. If this is '
                                      'not true, please set `odps.isolation.session.enable` to True or ask your '
                                      'project owner to change project-level configuration.')
                if need_extract:
                    raise SystemError('We do not allow file-type resource for binary packages. Please upload an '
                                      'archive-typed resource instead.')

            prefixes = set([''])
            dir_prefixes = set()
            if isinstance(f, zipfile.ZipFile):
                for name in f.namelist():
                    name = name if name.endswith('/') else (name.rsplit('/', 1)[0] + '/')
                    if name in prefixes:
                        continue
                    try:
                        f.getinfo(name + '__init__.py')
                    except KeyError:
                        prefixes.add(name)
                        if self._bin_path:
                            dir_prefixes.add(os.path.join(self._bin_path, name))
            elif isinstance(f, tarfile.TarFile):
                for member in f.getmembers():
                    name = member.name if member.isdir() else member.name.rsplit('/', 1)[0]
                    if name in prefixes:
                        continue
                    try:
                        f.getmember(name + '/__init__.py')
                    except KeyError:
                        prefixes.add(name + '/')
                        if self._bin_path:
                            dir_prefixes.add(os.path.join(self._bin_path, name + '/'))
            elif isinstance(f, dict):
                # Force ArchiveResource to run under binary mode to resolve manually
                # opening __file__ paths in pure-python code.
                if ALLOW_BINARY:
                    bin_package = True

                for name in iterkeys(f):
                    name = name.replace(os.sep, '/')
                    name = name if name.endswith('/') else (name.rsplit('/', 1)[0] + '/')
                    if name in prefixes or '/tests/' in name:
                        continue
                    if name + '__init__.py' not in f:
                        prefixes.add(name)
                        dir_prefixes.add(name)
                    else:
                        if '/' in name.rstrip('/'):
                            ppath = name.rstrip('/').rsplit('/', 1)[0]
                        else:
                            ppath = ''
                        prefixes.add(ppath)
                        dir_prefixes.add(ppath)

            if bin_package:
                for p in sorted(dir_prefixes):
                    if p not in sys.path:
                        sys.path.append(p)
            else:
                self._files.append(f)
                if prefixes:
                    self._prefixes[id(f)] = sorted(prefixes)

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
            for prefix in self._prefixes[id(f)]:
                for suffix, is_package in _SEARCH_ORDER:
                    l = [prefix] + parts[:-1] + [submodname + suffix.replace('/', os.sep)]
                    relpath = os.path.join(*l)
                    try:
                        relpath = relpath.replace(os.sep, '/')
                        if isinstance(f, zipfile.ZipFile):
                            f.getinfo(relpath)
                        elif isinstance(f, tarfile.TarFile):
                            f.getmember(relpath)
                        else:
                            if relpath not in f:
                                raise KeyError
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
        elif isinstance(fileobj, tarfile.TarFile):
            source = fileobj.extractfile(relpath.replace(os.sep, '/')).read()
        else:
            source = fileobj[relpath.replace(os.sep, '/')].read()
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


try:
    os.path.exists('/tmp')
    ALLOW_BINARY = True
except:
    ALLOW_BINARY = False
