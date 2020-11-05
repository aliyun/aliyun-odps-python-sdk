#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from .errors import NotSupportedError, InstanceTypeNotSupported
from .utils import to_str


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


class Connection(object):
    def __init__(self, access_id=None, secret_access_key=None, project=None,
                 endpoint=None, odps=None, **kw):
        if odps is None:
            self._odps = ODPS(access_id=access_id, secret_access_key=secret_access_key,
                              project=project, endpoint=endpoint, **kw)
        else:
            if access_id is not None:
                raise ValueError('Either access_id or odps can be specified')
            self._odps = odps

    @property
    def odps(self):
        return self._odps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def cursor(self, *args, **kwargs):
        """Return a new :py:class:`Cursor` object using the connection."""
        return Cursor(self, *args, **kwargs)

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
    def __init__(self, connection, arraysize=default_arraysize):
        self._connection = connection
        self._arraysize = arraysize
        self._reset_state()
        self.lastrowid = None

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
        run_sql = odps.run_sql if async_ else odps.execute_sql
        self._instance = run_sql(sql)

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
