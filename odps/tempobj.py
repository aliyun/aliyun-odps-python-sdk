#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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

from .compat import PY26, pickle, six, builtins
from .config import options
from .errors import NoSuchObject
from .utils import is_main_process, build_pyodps_dir

if PY26:
    # Compatible ThreadPool to avoid Issue 10015 of Python 2.6
    import threadpool

    class ThreadPool(threadpool.ThreadPool):
        def map(self, func, iterable):
            [self.putRequest(threadpool.WorkRequest(func, (v, ))) for v in iterable]

        def close(self):
            pass

        def join(self):
            self.wait()
else:
    from multiprocessing.pool import ThreadPool

TEMP_ROOT = build_pyodps_dir('tempobjs')
SESSION_KEY = '%d_%s' % (int(time.time()), uuid.uuid4())
CLEANER_THREADS = 100
USER_FILE_RIGHTS = stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC

CLEANUP_SCRIPT_TMPL = """
import os
import sys
import json

try:
    os.unlink(os.path.realpath(__file__))
except Exception:
    pass

temp_codes = json.loads(r\"\"\"
{odps_info}
\"\"\".strip())
import_paths = json.loads(r\"\"\"
{import_paths}
\"\"\".strip())
biz_ids = json.loads(r\"\"\"
{biz_ids}
\"\"\".strip())

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
"""


cleanup_mode = False
cleanup_timeout = 0
host_pid = os.getpid()


class ExecutionEnv(object):
    def __init__(self, **kwargs):
        self.os = os
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
        self.import_path_json = json.dumps(import_paths)

        self.builtins = builtins
        self.io = __import__('io', fromlist=[''])
        self.conv_bytes = (lambda s: s.encode()) if six.PY3 else (lambda s: s)
        self.subprocess = subprocess
        self.temp_dir = tempfile.gettempdir()
        self.template = CLEANUP_SCRIPT_TMPL
        self.file_right = USER_FILE_RIGHTS
        self.is_main_process = is_main_process()
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)


class TempObject(object):
    __slots__ = []
    _type = ''

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
        if os.path.exists(file_name):
            self.load()

    def put(self, obj, dump=True):
        self._container.add(obj)
        if dump:
            self.dump()

    def cleanup(self, odps):
        cleaned = []

        def cleaner_thread(obj):
            try:
                obj.drop(odps)
                cleaned.append(obj)
            except:
                pass

        pool = ThreadPool(CLEANER_THREADS)
        if self._container:
            pool.map(cleaner_thread, self._container)
            pool.close()
            pool.join()
        for obj in cleaned:
            if obj in self._container:
                self._container.remove(obj)
        if not self._container:
            try:
                os.unlink(self._file_name)
            except OSError:
                pass
        else:
            self.dump()

    def dump(self):
        with open(self._file_name, 'wb') as outf:
            pickle.dump(list(self._container), outf, protocol=0)
            outf.close()
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
        global cleanup_mode
        if cleanup_mode or not self._env.is_main_process:
            return
        self._exec_cleanup_script()

    @classmethod
    def add_biz_id(cls, biz_id):
        cls.biz_ids.add(biz_id)
        cls.biz_ids_json = json.dumps(list(cls.biz_ids))

    @classmethod
    def add_odps_info(cls, odps):
        odps_key = _gen_repository_key(odps)
        cls.odps_info[odps_key] = dict(access_id=odps.account.access_id, secret_access_key=odps.account.secret_access_key,
                                       project=odps.project, endpoint=odps.endpoint)
        cls.odps_info_json = json.dumps([v for v in six.itervalues(cls.odps_info)])

    def _exec_cleanup_script(self):
        if not self:
            return

        env = self._env
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
                env.subprocess.call(['chmod', str(env.file_right), script_name])
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


def clean_objects(odps):
    global cleanup_timeout, host_pid

    if not is_main_process():
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
    return hashlib.md5('####'.join([odps.account.access_id, odps.account.secret_access_key, odps.endpoint,
                                    odps.project]).encode('utf-8')).hexdigest()


def _put_objects(odps, objs):
    odps_key = _gen_repository_key(odps)

    biz_id = options.biz_id if options.biz_id else 'default'
    ObjectRepositoryLib.add_biz_id(biz_id)
    if odps_key not in _obj_repos:
        ObjectRepositoryLib.add_odps_info(odps)
        file_dir = os.path.join(TEMP_ROOT, biz_id, odps_key)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
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
