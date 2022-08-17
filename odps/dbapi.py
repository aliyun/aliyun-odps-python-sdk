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

import warnings

from .compat import enum, six
from .core import ODPS
from .errors import NotSupportedError, InstanceTypeNotSupported, ODPSError
from .utils import to_str
from .models.session import PUBLIC_SESSION_NAME


# PEP 249 module globals
apilevel = '2.0'
threadsafety = 2  # Threads may share the module and connections.
paramstyle = 'named'  # Python extended format codes, e.g. ...WHERE name=%(name)s


class Error(Exception):
    pass


class State(enum.Enum):
    NONE = 0
    RUNNING = 1
    FINISHED = 2


def connect(*args, **kwargs):
    """Constructor for creating a connection to the database. See class :py:class:`Connection` for
    arguments.
    :returns: a :py:class:`Connection` object.
    """
    return Connection(*args, **kwargs)


FALLBACK_POLICIES = {
    "unsupported": ["ODPS-185"],
    "upgrading": ["ODPS-182", "ODPS-184"],
    "noresource": ["ODPS-183"],
    "timeout": ["ODPS-186"],
    "generic": ["ODPS-180"]
}

FALLBACK_POLICY_ALIASES = {
    "default": ["unsupported", "upgrading", "noresource", "timeout"],
    "all": ["unsupported", "upgrading", "noresource", "timeout", "generic"]
}


class Connection(object):
    def __init__(self, access_id=None, secret_access_key=None, project=None,
                 endpoint=None, session_name=None, odps=None, **kw):
        if isinstance(access_id, ODPS):
            access_id, odps = None, access_id

        if odps is None:
            # pop unsupported
            kw.pop("use_sqa", None)
            kw.pop("fallback_policy", None)
            self._odps = ODPS(access_id=access_id, secret_access_key=secret_access_key,
                              project=project, endpoint=endpoint, **kw)
        else:
            if access_id is not None:
                raise ValueError('Either access_id or odps can be specified')
            self._odps = odps
        self._session_name = PUBLIC_SESSION_NAME
        if session_name is not None:
            self._session_name = session_name
        self._use_sqa = (kw.pop('use_sqa', False) != False)
        self._fallback_policy = kw.pop('fallback_policy', '')

    @property
    def odps(self):
        return self._odps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def cursor(self, *args, **kwargs):
        """Return a new :py:class:`Cursor` object using the connection."""
        return Cursor(self, *args, use_sqa=self._use_sqa, fallback_policy=self._fallback_policy, **kwargs)

    def close(self):
        # there is no long polling for ODPS
        # do nothing for close
        pass

    def commit(self):
        # ODPS does not support transactions
        # do nothing for commit
        pass

    def rollback(self):
        raise NotSupportedError("ODPS does not have transactions")


default_arraysize = 1000


class Cursor(object):
    def __init__(self, connection, arraysize=default_arraysize,
            use_sqa=False, fallback_policy='', **kwargs):
        self._connection = connection
        self._arraysize = arraysize
        self._reset_state()
        self.lastrowid = None
        self._use_sqa = use_sqa
        self._fallback_policy = []
        fallback_policies = map(lambda x: x.strip(), fallback_policy.split(','))
        for policy in fallback_policies:
            if policy in FALLBACK_POLICY_ALIASES:
                self._fallback_policy.extend(FALLBACK_POLICY_ALIASES[policy])
            else:
                self._fallback_policy.append(policy)

    def _reset_state(self):
        self._state = State.NONE
        self._description = None
        # odps instance and download session
        self._instance = None
        self._download_session = None

    @property
    def arraysize(self):
        return self._arraysize

    @arraysize.setter
    def arraysize(self, value):
        try:
            self._arraysize = max(int(value), default_arraysize)
        except TypeError:
            warnings.warn('arraysize has to be a integer, got {}, '
                          'will set default value 1000'.format(value))
            self._arraysize = default_arraysize

    @property
    def description(self):
        """This read-only attribute is a sequence of 7-item sequences.

        Each of these sequences contains information describing one result column:

        - name
        - type_code
        - display_size (None in current implementation)
        - internal_size (None in current implementation)
        - precision (None in current implementation)
        - scale (None in current implementation)
        - null_ok (always True in current implementation)

        This attribute will be ``None`` for operations that do not return rows or if the cursor has
        not had an operation invoked via the :py:meth:`execute` method yet.

        The ``type_code`` can be interpreted by comparing it to the Type Objects specified in the
        section below.
        """
        if self._instance is None:
            return
        if self._description is None:
            self._check_download_session()
            self._description = []
            if self._download_session is not None:
                for col in self._download_session.schema.columns:
                    self._description.append((
                        col.name, col.type.name,
                        None, None, None, None, True
                    ))
            else:
                self._description.append((
                    '_c0', 'string', None, None,
                    None, None, True
                ))
        return self._description

    @staticmethod
    def escape_string(item):
        item = to_str(item)
        return "'{}'".format(
            item
                .replace('\\', '\\\\')
                .replace("'", "\\'")
                .replace('\r', '\\r')
                .replace('\n', '\\n')
                .replace('\t', '\\t')
        )

    def execute(self, operation, parameters=None, **kwargs):
        """Prepare and execute a database operation (query or command).

        Parameters may be provided as sequence or mapping and will be bound to variables in the
        operation. Variables are specified in a database-specific notation (see the module's
        ``paramstyle`` attribute for details).

        Return values are not defined.
        """
        for k in ['async', 'async_']:
            if k in kwargs:
                async_ = kwargs[k]
                break
        else:
            async_ = False

        # prepare statement
        if parameters is None:
            sql = operation
        else:
            sql = operation
            for origin, replacement in parameters.items():
                if isinstance(replacement, six.string_types):
                    replacement = self.escape_string(replacement)
                sql = to_str(sql).replace(':' + to_str(origin), to_str(replacement))

        self._reset_state()

        odps = self._connection.odps
        run_sql = odps.execute_sql
        if self._use_sqa:
            run_sql = self._run_sqa_with_fallback
        if async_:
            run_sql = odps.run_sql
        self._instance = run_sql(sql)

    def executemany(self, operation, seq_of_parameters):
        for parameter in seq_of_parameters:
            self.execute(operation, parameter)

    def _sqa_error_should_fallback(self, err_str):
        if 'ODPS-18' not in err_str:
            return False
        for fallback_case in self._fallback_policy:
            fallback_error = FALLBACK_POLICIES.get(fallback_case, None)
            if fallback_error is None:
                continue
            for error_code in fallback_error:
                if error_code in err_str:
                    return True
        return False

    def _run_sqa_with_fallback(self, sql, **kw):
        odps = self._connection.odps
        session_name = self._connection._session_name
        inst = None
        while True:
            try:
                if inst is None:
                    inst = odps.run_sql_interactive(sql, service_name = session_name)
                else:
                    inst.wait_for_success(interval=0.5)
                rd = inst.open_reader(tunnel=True, limit=False)
                if not rd:
                    raise ODPSError('failed to create direct download')
                rd.schema  # will check if task is ok
                self._download_session = rd
                return inst
            except ODPSError as e:
                if self._sqa_error_should_fallback(str(e)):
                    return odps.execute_sql(sql)
                elif "OdpsTaskTimeout" in str(e):
                    # tunnel failed to wait data cache result. fallback to normal wait.
                    pass
                else:
                    raise e

    def cancel(self):
        if self._instance is not None:
            self._instance.stop()

    def close(self):
        self._reset_state()

    def _check_download_session(self):
        if not self._download_session and self._instance:
            try:
                self._download_session = self._instance.open_reader(
                    tunnel=True, limit=False)
            except InstanceTypeNotSupported:
                # not select, cannot create session
                self._download_session = None

    def _fetch_non_select(self):
        # not select
        # just return reader.raw
        with self._instance.open_reader() as reader:
            return [(reader.raw,)]

    def _fetch(self, size):
        self._check_download_session()

        if self._download_session is None:
            return self._fetch_non_select()

        results = []
        i = 0
        while size == -1 or i < size:
            try:
                results.append(next(self._download_session).values)
            except StopIteration:
                break
            i += 1
        return results

    def __iter__(self):
        while True:
            res = self.fetchone()
            if res is not None:
                yield res
            else:
                break

    def next(self):
        res = self.fetchone()
        if res is not None:
            yield res
        else:
            raise StopIteration

    def fetchone(self):
        self._check_download_session()
        results = self._fetch(1)
        if len(results) == 1:
            return results[0]

    def fetchmany(self, size=None):
        if size is None:
            size = self._arraysize
        return self._fetch(size)

    def fetchall(self):
        return self._fetch(-1)
