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

import atexit
import copy
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import stat
import threading
import time
import uuid

from .accounts import AliyunAccount
from .compat import PY26, pickle, six, builtins, futures
from .config import options
from .errors import NoSuchObject
from . import utils

TEMP_ROOT = utils.build_pyodps_dir('tempobjs')
SESSION_KEY = '%d_%s' % (int(time.time()), uuid.uuid4())
CLEANER_THREADS = 100
USER_FILE_RIGHTS = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR

CLEANUP_SCRIPT_TMPL = u"""
#-*- coding:utf-8 -*-
import os
import sys
import json

try:
    os.unlink(os.path.realpath(__file__))
except Exception:
    pass

temp_codes = json.loads({odps_info!r})
import_paths = json.loads({import_paths!r})
biz_ids = json.loads({biz_ids!r})

if sys.version_info[0] < 3:
    if sys.platform == 'win32':
        import_paths = [p.encode('mbcs') for p in import_paths]
    else:
        import_paths = [p.encode() for p in import_paths]
        
normed_paths = set(os.path.normcase(os.path.normpath(p)) for p in sys.path)
import_paths = [p for p in import_paths
                if os.path.normcase(os.path.normpath(p)) not in normed_paths]

sys.path.extend(import_paths)
from odps import ODPS, tempobj

if os.environ.get('WAIT_CLEANUP') == '1':
    tempobj.cleanup_timeout = None
else:
    tempobj.cleanup_timeout = 5
tempobj.cleanup_mode = True
tempobj.host_pid = {host_pid}
tempobj.ObjectRepositoryLib.biz_ids = set(biz_ids)

for o_desc in temp_codes:
    ODPS(**tempobj.compat_kwargs(o_desc))
os._exit(0)
""".lstrip()


cleanup_mode = False
cleanup_timeout = 0
host_pid = os.getpid()
if six.PY3:  # make flake8 happy
    unicode = str


class ExecutionEnv(object):
    def __init__(self, **kwargs):
        self.cleaned = False
        self.os = os
        self.sys = sys
        self._g_env = copy.copy(globals())
        self.is_windows = 'windows' in platform.platform().lower()
        self.pid = os.getpid()
        self.os_sep = os.sep
        self.executable = sys.executable
        self.six = six

        import_paths = copy.deepcopy(sys.path)
        package_root = os.path.dirname(__file__)
        if package_root not in import_paths:
            import_paths.append(package_root)
        self.import_path_json = utils.to_text(json.dumps(import_paths, ensure_ascii=False))

        self.builtins = builtins
        self.io = __import__('io', fromlist=[''])
        if six.PY3:
            self.conv_bytes = (lambda s: s.encode() if isinstance(s, str) else s)
            self.conv_unicode = (lambda s: s if isinstance(s, str) else s.decode())
        else:
            self.conv_bytes = (lambda s: s.encode() if isinstance(s, unicode) else s)
            self.conv_unicode = (lambda s: s if isinstance(s, unicode) else s.decode())
        self.subprocess = subprocess
        self.temp_dir = tempfile.gettempdir()
        self.template = CLEANUP_SCRIPT_TMPL
        self.file_right = USER_FILE_RIGHTS
        self.is_main_process = utils.is_main_process()
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)


class TempObject(object):
    __slots__ = []
    _type = ''
    _priority = 0

    def __init__(self, *args, **kwargs):
        for k, v in zip(self.__slots__, args):
            setattr(self, k, v)
        for k in self.__slots__:
            if hasattr(self, k):
                continue
            setattr(self, k, kwargs.get(k))

    def __hash__(self):
        if self.__slots__:
            return hash(tuple(getattr(self, k) for k in self.__slots__))
        return super(TempObject, self).__hash__()

    def __eq__(self, other):
        if not isinstance(other, TempObject):
            return False
        if self._type != other._type:
            return False
        return all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        return dict((slot, getattr(self, slot)) for slot in self.__slots__ if hasattr(self, slot))

    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)


class TempTable(TempObject):
    __slots__ = 'table', 'project'
    _type = 'Table'

    def drop(self, odps):
        odps.run_sql('drop table if exists %s' % self.table, project=self.project)


class TempModel(TempObject):
    __slots__ = 'model', 'project'
    _type = 'OfflineModel'

    def drop(self, odps):
        try:
            odps.delete_offline_model(self.model, self.project)
        except NoSuchObject:
            pass


class TempFunction(TempObject):
    __slots__ = 'function', 'project'
    _type = 'Function'
    _priority = 1

    def drop(self, odps):
        try:
            odps.delete_function(self.function, self.project)
        except NoSuchObject:
            pass


class TempResource(TempObject):
    __slots__ = 'resource', 'project'
    _type = 'Resource'

    def drop(self, odps):
        try:
            odps.delete_resource(self.resource, self.project)
        except NoSuchObject:
            pass


class TempVolumePartition(TempObject):
    __slots__ = 'volume', 'partition', 'project'
    _type = 'VolumePartition'

    def drop(self, odps):
        try:
            odps.delete_volume_partition(self.volume, self.partition, self.project)
        except NoSuchObject:
            pass


class ObjectRepository(object):
    def __init__(self, file_name):
        self._container = set()
        self._file_name = file_name
        if file_name and os.path.exists(file_name):
            self.load()

    def put(self, obj, dump=True):
        self._container.add(obj)
        if dump:
            self.dump()

    def cleanup(self, odps, use_threads=True):
        cleaned = []

        def _cleaner(obj):
            try:
                obj.drop(odps)
                cleaned.append(obj)
            except:
                pass

        if self._container:
            if use_threads:
                pool = futures.ThreadPoolExecutor(CLEANER_THREADS)
                list(pool.map(_cleaner, reversed(list(self._container))))
            else:
                for o in sorted(list(self._container), key=lambda ro: type(ro)._priority, reverse=True):
                    _cleaner(o)
        for obj in cleaned:
            if obj in self._container:
                self._container.remove(obj)
        if not self._container and self._file_name:
            try:
                os.unlink(self._file_name)
            except OSError:
                pass
        else:
            self.dump()

    def dump(self):
        if self._file_name is None:
            return
        try:
            with open(self._file_name, 'wb') as outf:
                pickle.dump(list(self._container), outf, protocol=0)
                outf.close()
        except OSError:
            return
        os.chmod(self._file_name, USER_FILE_RIGHTS)

    def load(self):
        try:
            with open(self._file_name, 'rb') as inpf:
                contents = pickle.load(inpf)
            self._container.update(contents)
        except (EOFError, OSError):
            pass


class ObjectRepositoryLib(dict):
    biz_ids = set([options.biz_id, ]) if options.biz_id else set(['default', ])
    odps_info = dict()

    biz_ids_json = json.dumps(list(biz_ids))
    odps_info_json = json.dumps([v for v in six.itervalues(odps_info)])

    def __init__(self, *args, **kwargs):
        super(ObjectRepositoryLib, self).__init__(*args, **kwargs)
        self._env = ExecutionEnv()

    def __del__(self):
        self._exec_cleanup_script()

    @classmethod
    def add_biz_id(cls, biz_id):
        cls.biz_ids.add(biz_id)
        cls.biz_ids_json = json.dumps(list(cls.biz_ids))

    @classmethod
    def add_odps_info(cls, odps):
        odps_key = _gen_repository_key(odps)
        cls.odps_info[odps_key] = dict(
            access_id=odps.account.access_id, secret_access_key=odps.account.secret_access_key,
            project=odps.project, endpoint=odps.endpoint
        )
        cls.odps_info_json = json.dumps([v for v in six.itervalues(cls.odps_info)])

    def _exec_cleanup_script(self):
        global cleanup_mode

        if not self:
            return

        env = self._env
        if cleanup_mode or not env.is_main_process or env.cleaned:
            return
        env.cleaned = True

        script = env.template.format(import_paths=env.import_path_json, odps_info=self.odps_info_json,
                                     host_pid=env.pid, biz_ids=self.biz_ids_json)

        script_name = env.temp_dir + env.os_sep + 'tmp_' + str(env.pid) + '_cleanup_script.py'
        script_file = env.io.FileIO(script_name, 'w')
        script_file.write(env.conv_bytes(script))
        script_file.close()
        try:
            if env.is_windows:
                env.os.chmod(script_name, env.file_right)
            else:
                env.subprocess.call(['chmod', oct(env.file_right).replace('o', ''), script_name])
        except:
            pass

        kwargs = dict(close_fds=True)
        if env.is_windows:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs['startupinfo'] = si
        env.subprocess.call([env.executable, script_name], **kwargs)


_cleaned_keys = set()
_obj_repos = ObjectRepositoryLib()  # this line should be put last due to initialization dependency
atexit.register(_obj_repos._exec_cleanup_script)


def _is_pid_running(pid):
    if 'windows' in platform.platform().lower():
        task_lines = os.popen('TASKLIST /FI "PID eq {0}" /NH'.format(pid)).read().strip().splitlines()
        if not task_lines:
            return False
        return str(pid) in set(task_lines[0].split())
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def clean_objects(odps, biz_ids=None):
    odps_key = _gen_repository_key(odps)
    files = []
    biz_ids = biz_ids or _obj_repos.biz_ids
    for biz_id in biz_ids:
        files.extend(glob.glob(os.path.join(TEMP_ROOT, biz_id, odps_key, '*.his')))

    for fn in files:
        repo = ObjectRepository(fn)
        repo.cleanup(odps, use_threads=False)


def clean_stored_objects(odps):
    global cleanup_timeout, host_pid

    if not utils.is_main_process():
        return

    odps_key = _gen_repository_key(odps)
    if odps_key in _cleaned_keys:
        return
    _cleaned_keys.add(odps_key)

    files = []
    for biz_id in _obj_repos.biz_ids:
        files.extend(glob.glob(os.path.join(TEMP_ROOT, biz_id, odps_key, '*.his')))

    def clean_thread():
        for fn in files:
            writer_pid = int(fn.rsplit('__', 1)[-1].split('.', 1)[0])

            # we do not clean running process, unless its pid equals host_pid
            if writer_pid != host_pid and _is_pid_running(writer_pid):
                continue

            repo = ObjectRepository(fn)
            repo.cleanup(odps)

    thread_obj = threading.Thread(target=clean_thread)
    thread_obj.start()
    if cleanup_timeout == 0:
        return
    else:
        if cleanup_timeout is not None and cleanup_timeout < 0:
            cleanup_timeout = None
        thread_obj.join(cleanup_timeout)


def _gen_repository_key(odps):
    if hasattr(odps.account, 'access_id'):
        keys = [odps.account.access_id, odps.endpoint, str(odps.project)]
    elif hasattr(odps.account, 'token'):
        keys = [utils.to_str(odps.account.token), odps.endpoint, str(odps.project)]
    return hashlib.md5(utils.to_binary('####'.join(keys))).hexdigest()


def _put_objects(odps, objs):
    odps_key = _gen_repository_key(odps)

    biz_id = options.biz_id if options.biz_id else 'default'
    ObjectRepositoryLib.add_biz_id(biz_id)
    if odps_key not in _obj_repos:
        if isinstance(odps.account, AliyunAccount):
            ObjectRepositoryLib.add_odps_info(odps)
        file_dir = os.path.join(TEMP_ROOT, biz_id, odps_key)
        try:
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
        except OSError:
            pass
        file_name = os.path.join(file_dir, 'temp_objs_{0}__{1}.his'.format(SESSION_KEY, os.getpid()))
        _obj_repos[odps_key] = ObjectRepository(file_name)
    [_obj_repos[odps_key].put(o, False) for o in objs]
    _obj_repos[odps_key].dump()


def register_temp_table(odps, table, project=None):
    if isinstance(table, six.string_types):
        table = [table, ]
    _put_objects(odps, [TempTable(t, project if project else odps.project) for t in table])


def register_temp_model(odps, model, project=None):
    if isinstance(model, six.string_types):
        model = [model, ]
    _put_objects(odps, [TempModel(m, project if project else odps.project) for m in model])


def register_temp_resource(odps, resource, project=None):
    if isinstance(resource, six.string_types):
        resource = [resource, ]
    _put_objects(odps, [TempResource(r, project if project else odps.project) for r in resource])


def register_temp_function(odps, func, project=None):
    if isinstance(func, six.string_types):
        func = [func, ]
    _put_objects(odps, [TempFunction(f, project if project else odps.project) for f in func])


def register_temp_volume_partition(odps, volume_partition_tuple, project=None):
    if isinstance(volume_partition_tuple, tuple):
        volume_partition_tuple = [volume_partition_tuple, ]
    _put_objects(odps, [TempVolumePartition(v, p, project if project else odps.project)
                        for v, p in volume_partition_tuple])


def compat_kwargs(kwargs):
    if PY26:
        new_desc = dict()
        for k, v in six.iteritems(kwargs):
            new_desc[k.encode('utf-8') if isinstance(k, unicode) else k] = v.encode('utf-8')
        return new_desc
    else:
        return kwargs
