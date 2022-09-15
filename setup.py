#!/usr/bin/env python
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

# Parts of this file were taken from the pandas project
# (https://github.com/pandas-dev/pandas), which is permitted for use under
# the BSD 3-Clause License

from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from distutils.cmd import Command
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion

import sys
import os
import platform
import shutil


# From https://github.com/pandas-dev/pandas/pull/24274:
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if sys.platform == 'darwin':
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(
            get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

repo_root = os.path.dirname(os.path.abspath(__file__))

try:
    execfile
except NameError:
    def execfile(fname, globs, locs=None):
        locs = locs or globs
        exec(compile(open(fname).read(), fname, "exec"), globs, locs)

version_ns = {}
execfile(os.path.join(repo_root, 'odps', '_version.py'), version_ns)

extra_install_cmds = []


def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


# http://stackoverflow.com/questions/12683834/how-to-copy-directory-recursively-in-python-and-overwrite-all
def recursive_overwrite(src, dest, filter_func=None):
    destinations = []
    filter_func = filter_func or (lambda s: True)
    if os.path.isdir(src):
        if not os.path.isdir(dest):
            os.makedirs(dest)
        files = os.listdir(src)
        for f in files:
            if not filter_func(f):
                continue
            destinations.extend(
                recursive_overwrite(os.path.join(src, f), os.path.join(dest, f))
            )
    else:
        shutil.copyfile(src, dest)
        destinations.append(dest)
    return destinations


class CustomInstall(install):
    def run(self):
        global extra_install_cmds
        install.run(self)
        [self.run_command(cmd) for cmd in extra_install_cmds]

version = sys.version_info
PY2 = version[0] == 2
PY3 = version[0] == 3
PY26 = PY2 and version[1] == 6
PYPY = platform.python_implementation().lower() == 'pypy'

if PY2 and version[:2] < (2, 6):
    raise Exception('PyODPS supports Python 2.6+ (including Python 3+).')

try:
    import distribute
    raise Exception("PyODPS cannot be installed when 'distribute' is installed. "
                    "Please uninstall it before installing PyODPS.")
except ImportError:
    pass

try:
    import pip
    for pk in pip.get_installed_distributions():
        if pk.key == 'odps':
            raise Exception('Package `odps` collides with PyODPS. Please uninstall it before installing PyODPS.')
except (ImportError, AttributeError):
    pass

try:
    from jupyter_core.paths import jupyter_data_dir
    has_jupyter = True
except ImportError:
    has_jupyter = False
try:
    from jupyterlab import __version__
    has_jupyterlab = True
except ImportError:
    has_jupyterlab = False

if len(sys.argv) > 1 and sys.argv[1] == 'clean':
    build_cmd = sys.argv[1]
else:
    build_cmd = None

requirements = []
with open('requirements.txt') as f:
    requirements.extend(f.read().splitlines())

if PY26:
    requirements.append('ordereddict>=1.1')
    requirements.append('simplejson>=2.1.0')
    requirements.append('importlib>=1.0')

full_requirements = [
    'jupyter>=1.0.0',
    'ipython>=4.0.0',
    'numpy>=1.6.0',
    'pandas>=0.17.0',
    'matplotlib>=1.4',
    'graphviz>=0.4',
    'greenlet>=0.4.10',
]
mars_requirements = [
    'pymars>=0.5.4',
    'protobuf>=3.6,<4.0',
]
if sys.version_info[0] == 2:
    full_requirements.append('ipython<6.0.0')
if sys.platform != 'win32':
    full_requirements.append('cython>=0.20')

long_description = None
if os.path.exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()

setup_options = dict(
    name='pyodps',
    version=version_ns['__version__'],
    description='ODPS Python SDK and data analysis framework',
    long_description=long_description,
    author='Wu Wei',
    author_email='weiwu@cacheme.net',
    maintainer='Qin Xuye',
    maintainer_email='qin@qinxuye.me',
    url='http://github.com/aliyun/aliyun-odps-python-sdk',
    license='Apache License 2.0',
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development :: Libraries',
    ],
    cmdclass={'install': CustomInstall},
    packages=find_packages(exclude=('*.tests.*', '*.tests')),
    include_package_data=True,
    scripts=['scripts/pyou', ],
    install_requires=requirements,
    include_dirs=[],
    extras_require={'full': full_requirements, 'mars': mars_requirements},
    entry_points={
        'sqlalchemy.dialects': ['odps = odps.sqlalchemy_odps:ODPSDialect']
    },
)

if build_cmd != 'clean' and not PYPY:  # skip cython in pypy
    try:
        from Cython.Build import cythonize
        from Cython.Distutils import build_ext
        import cython

        # detect if cython works
        if sys.platform == 'win32':
            cython.inline('return a + b', a=1, b=1)

        extension_kw = dict(language='c++', include_dirs=[])
        if 'MSC' in sys.version:
            extension_kw['extra_compile_args'] = ['/Ot', '/I' + os.path.join(repo_root, 'misc')]
        else:
            extension_kw['extra_compile_args'] = ['-O3']

        extensions = [
            Extension('odps.src.types_c', ['odps/src/types_c.pyx'], **extension_kw),
            Extension('odps.src.crc32c_c', ['odps/src/crc32c_c.pyx'], **extension_kw),
            Extension('odps.src.utils_c', ['odps/src/utils_c.pyx'], **extension_kw),
            Extension('odps.tunnel.pb.encoder_c', ['odps/tunnel/pb/encoder_c.pyx'], **extension_kw),
            Extension('odps.tunnel.pb.decoder_c', ['odps/tunnel/pb/decoder_c.pyx'], **extension_kw),
            Extension('odps.tunnel.io.writer_c', ['odps/tunnel/io/writer_c.pyx'], **extension_kw),
            Extension('odps.tunnel.io.reader_c', ['odps/tunnel/io/reader_c.pyx'], **extension_kw),
            Extension('odps.tunnel.checksum_c', ['odps/tunnel/checksum_c.pyx'], **extension_kw),
        ]
        try:
            import numpy as np
            np_extension_kw = extension_kw.copy()
            np_extension_kw['include_dirs'].append(np.get_include())
            extensions.extend([
                Extension('odps.tunnel.pdio.pdreader_c', ['odps/tunnel/pdio/pdreader_c.pyx'], **np_extension_kw),
                Extension('odps.tunnel.pdio.pdwriter_c', ['odps/tunnel/pdio/pdwriter_c.pyx'], **np_extension_kw),
                Extension('odps.tunnel.pdio.block_decoder_c', ['odps/tunnel/pdio/block_decoder_c.pyx'], **np_extension_kw),
                Extension('odps.tunnel.pdio.block_encoder_c', ['odps/tunnel/pdio/block_encoder_c.pyx'], **np_extension_kw),
            ])
            setup_options['include_dirs'].append(np.get_include())
        except ImportError:
            pass

        setup_options['cmdclass'].update({'build_ext': build_ext})
        force_recompile = bool(int(os.getenv("CYTHON_FORCE_RECOMPILE", "0")))
        setup_options['ext_modules'] = cythonize(extensions, force=force_recompile)
    except:
        pass

if build_cmd != 'clean' and has_jupyter:
    class InstallJS(Command):
        description = "install JavaScript extensions"
        user_options = []

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            src_dir = os.path.join(repo_root, 'odps', 'static', 'ui', 'target')
            dest_dir = os.path.join(jupyter_data_dir(), 'nbextensions', 'pyodps')
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            recursive_overwrite(src_dir, dest_dir)

            try:
                from notebook.nbextensions import enable_nbextension
            except ImportError:
                return
            enable_nbextension('notebook', 'pyodps/main')


    class BuildJS(Command):
        description = "build JavaScript files"
        user_options = [
            ('registry=', None, 'npm registry')
        ]

        def initialize_options(self):
            self.registry = None

        def finalize_options(self):
            pass

        def run(self):
            if not which('npm'):
                raise Exception('You need to install npm before building the scripts.')

            cwd = os.getcwd()

            os.chdir(os.path.join(os.path.abspath(os.getcwd()), 'odps', 'static', 'ui'))
            cmd = 'npm install'
            if getattr(self, 'registry', None):
                cmd += ' --registry=' + self.registry
            print('executing ' + cmd)
            ret = os.system(cmd)
            ret >>= 8
            if ret != 0:
                print(cmd + ' exited with error: %d' % ret)

            print('executing grunt')
            ret = os.system('npm run grunt')
            ret >>= 8
            if ret != 0:
                print('grunt exited with error: %d' % ret)

            os.chdir(cwd)


    setup_options['cmdclass'].update({'install_js': InstallJS, 'build_js': BuildJS})
    extra_install_cmds.append('install_js')

if build_cmd != 'clean' and has_jupyterlab:
    class InstallJupyterLabExtension(Command):
        description = "install Jupyterlab Extension"
        user_options = [
            ('registry=', 'r', 'npm registry')
        ]

        def initialize_options(self):
            self.registry = 'https://registry.npm.taobao.org'

        def finalize_options(self):
            pass

        def run(self):
            print(self.registry)
            os.chdir(os.path.join(os.path.abspath(os.getcwd()), 'odps', 'lab_extension'))
            print("\033[1;34m" + "Install pyodps-lab-extension" + "\033[0;0m")
            os.system('npm install --registry=' + self.registry)
            os.system('pip install .')
            print("\033[0;32m" + "pyodps-lab-extension install success" + "\033[0;0m")


    setup_options['cmdclass'].update({'install_jlab': InstallJupyterLabExtension})
    extra_install_cmds.append('install_jlab')

setup(**setup_options)

if build_cmd == 'clean':
    for root, dirs, files in os.walk(os.path.normpath('odps/')):
        pyx_files = set()
        c_file_pairs = []
        if '__pycache__' in dirs:
            full_path = os.path.join(root, '__pycache__')
            print("removing '%s'" % full_path)
            shutil.rmtree(full_path)
        for f in files:
            fn, ext = os.path.splitext(f)
            # delete compiled binaries
            if ext.lower() in ('.pyd', '.so', '.pyc'):
                full_path = os.path.join(root, f)
                print("removing '%s'" % full_path)
                os.unlink(full_path)
            elif ext.lower() == '.pyx':
                pyx_files.add(fn)
            elif ext.lower() in ('.c', '.cpp', '.cc'):
                c_file_pairs.append((fn, f))

        # remove cython-generated files
        for cfn, cf in c_file_pairs:
            if cfn in pyx_files:
                full_path = os.path.join(root, cf)
                print("removing '%s'" % full_path)
                os.unlink(full_path)
