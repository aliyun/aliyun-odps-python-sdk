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

import zipfile
import tarfile
import os
import random
import sys
import types
from collections import defaultdict


_SEARCH_ORDER = [
    ('.py', False),
    ('/__init__.py', True),
]


try:
    os.path.exists('/tmp')
    ALLOW_BINARY = True
except:
    ALLOW_BINARY = False


if sys.version_info[0] <= 2:
    iterkeys = lambda d: d.iterkeys()
    itervalues = lambda d: d.itervalues()
    string_types = (basestring, unicode)
    BASE = object
else:
    import importlib.abc

    iterkeys = lambda d: d.keys()
    itervalues = lambda d: d.values()
    string_types = (str, bytes)
    BASE = importlib.abc.MetaPathFinder


def _clean_extract():
    if CompressImporter._extract_path:
        import shutil
        shutil.rmtree(CompressImporter._extract_path, ignore_errors=True)


class CompressImportError(ImportError):
    """Exception raised by CompressImporter objects."""
    pass


class CompressImporter(BASE):
    """
    A PEP-302-style importer that can import from a zipfile.

    Just insert or append this class (not an instance) to sys.path_hooks
    and you're in business.  Instances satisfy both the 'importer' and
    'loader' APIs specified in PEP 302.
    """
    _extract_path = None

    def __init__(self, *compressed_files, **kwargs):
        """
        Constructor.

        Args:
          compressed_files zipfile.ZipFile or tarfile.TarFile
        """
        self._files = []
        self._prefixes = defaultdict(lambda: set(['']))
        self._extract_binary = kwargs.get('extract_binary', kwargs.get('extract', False))
        self._extract_all = kwargs.get('extract_all', False)
        self._supersede = kwargs.get('supersede', True)

        for f in compressed_files:
            if isinstance(f, zipfile.ZipFile):
                bin_package = any(
                    n.endswith('.so') or n.endswith('.pxd') or n.endswith('.dylib')
                    for n in f.namelist()
                )
                need_extract = True
            elif isinstance(f, tarfile.TarFile):
                bin_package = any(
                    m.name.endswith('.so') or m.name.endswith('.pxd') or m.name.endswith('.dylib')
                    for m in f.getmembers()
                )
                need_extract = True
            elif isinstance(f, dict):
                bin_package = any(
                    name.endswith('.so') or name.endswith('.pxd') or name.endswith('.dylib')
                    for name in iterkeys(f)
                )
                need_extract = False
            elif isinstance(f, list):
                bin_package = any(
                    name.endswith('.so') or name.endswith('.pxd') or name.endswith('.dylib')
                    for name in f
                )
                need_extract = False
            else:
                raise TypeError('Compressed file can only be zipfile.ZipFile or tarfile.TarFile')

            if bin_package:
                # binary packages need to be extracted before use
                if not ALLOW_BINARY:
                    raise SystemError(
                        'Cannot load binary package. It is quite possible that you are using an old '
                        'MaxCompute service which does not support binary packages. If this is '
                        'not true, please set `odps.isolation.session.enable` to True or ask your '
                        'project owner to change project-level configuration.'
                    )
                if need_extract:
                    f = self._extract_archive(f)
            elif need_extract and self._extract_all:
                # when it is forced to extract even if it is a text package, also extract
                f = self._extract_archive(f)

            prefixes = set([''])
            dir_prefixes = set()  # only for lists or dicts
            if isinstance(f, zipfile.ZipFile):
                for name in f.namelist():
                    name = name if name.endswith('/') else (name.rsplit('/', 1)[0] + '/')
                    if name in prefixes:
                        continue
                    try:
                        f.getinfo(name + '__init__.py')
                    except KeyError:
                        prefixes.add(name)
            elif isinstance(f, tarfile.TarFile):
                for member in f.getmembers():
                    name = member.name if member.isdir() else member.name.rsplit('/', 1)[0]
                    if name in prefixes:
                        continue
                    try:
                        f.getmember(name + '/__init__.py')
                    except KeyError:
                        prefixes.add(name + '/')
            elif isinstance(f, (list, dict)):
                # Force ArchiveResource to run under binary mode to resolve manually
                # opening __file__ paths in pure-python code.
                if ALLOW_BINARY:
                    bin_package = True

                rendered_names = set()
                for name in f:
                    name = name.replace(os.sep, '/')
                    rendered_names.add(name)

                for name in rendered_names:
                    name = name if name.endswith('/') else (name.rsplit('/', 1)[0] + '/')
                    if name in prefixes or '/tests/' in name or '/__pycache__/' in name:
                        continue
                    if name + '__init__.py' not in rendered_names:
                        prefixes.add(name)
                        dir_prefixes.add(name)
                    else:
                        if '/' in name.rstrip('/'):
                            ppath = name.rstrip('/').rsplit('/', 1)[0]
                        else:
                            ppath = ''
                        prefixes.add(ppath)
                        dir_prefixes.add(ppath)

            # make sure only root packages are included, otherwise relative imports might be broken
            # NOTE that it is needed to check sys.path duplication after all pruning done,
            #  otherwise path might be error once CompressImporter is called twice.
            path_patch = []
            for p in sorted(dir_prefixes):
                parent_exist = False
                for pp in path_patch:
                    if p[:len(pp)] == pp:
                        parent_exist = True
                        break
                if parent_exist:
                    continue
                path_patch.append(p)

            if bin_package:
                path_patch = [p for p in path_patch if p not in sys.path]
                if self._supersede:
                    sys.path = path_patch + sys.path
                else:
                    sys.path = sys.path + path_patch
            else:
                self._files.append(f)
                if path_patch:
                    path_patch = [p for p in path_patch if p not in sys.path]
                    self._prefixes[id(f)] = sorted([''] + path_patch)
                elif prefixes:
                    self._prefixes[id(f)] = sorted(prefixes)

    def _extract_archive(self, archive):
        if not self._extract_binary and not self._extract_all:
            raise SystemError(
                'We do not allow file-type resource for binary packages. Please upload an '
                'archive-typed resource instead.'
            )

        cls = type(self)
        if not cls._extract_path:
            import tempfile
            import atexit
            cls._extract_path = tempfile.mkdtemp(prefix='tmp-pyodps-')
            atexit.register(_clean_extract)

        extract_dir = os.path.join(cls._extract_path,
                                   'archive-' + str(random.randint(100000, 999999)))
        os.makedirs(extract_dir)
        if isinstance(archive, zipfile.ZipFile):
            archive.extractall(extract_dir)
        elif isinstance(archive, tarfile.TarFile):
            archive.extractall(extract_dir)

        mock_archive = list()
        for root, dirs, files in os.walk(extract_dir):
            for name in files:
                full_name = os.path.join(root, name)
                mock_archive.append(full_name)
        return mock_archive

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
        elif isinstance(fileobj, dict):
            source = fileobj[relpath.replace(os.sep, '/')].read()
        else:
            source = open(fileobj[relpath.replace(os.sep, '/')], 'rb').read()
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
