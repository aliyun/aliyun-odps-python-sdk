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

import time

from odps.inter import enter, setup, teardown, list_rooms
from odps.compat import StringIO
from odps import types as odps_types
from odps import options, ODPS
from odps.utils import init_progress_bar, replace_sql_parameters
from odps.df import DataFrame, Scalar
from odps.df.backends.frame import ResultFrame
from odps.ui.common import html_notify

from IPython.core.magic import Magics, magics_class, line_cell_magic, line_magic
from IPython.core.display import display_javascript

try:
    import numpy as np

    np_int_types = map(np.dtype, [np.int_, np.int8, np.int16, np.int32, np.int64])
    np_float_types = map(np.dtype, [np.float, np.float16, np.float32, np.float64])
    np_to_odps_types = dict([(t, odps_types.bigint) for t in np_int_types] +
                            [(t, odps_types.double) for t in np_float_types])
except ImportError:
    pass


@magics_class
class ODPSSql(Magics):

    _odps = None

    def _set_odps(self):
        if self._odps is not None:
            return

        if options.access_id is not None and \
                    options.access_key is not None and \
                    options.default_project is not None:
            self._odps = ODPS(
                options.access_id, options.access_key, options.default_project,
                endpoint=options.end_point, tunnel_endpoint=options.tunnel_endpoint
            )
        else:
            self._odps = enter().odps

    @line_magic('enter')
    def enter(self, line):
        room = line.strip()
        if room:
            r = enter(room)
            self._odps = r.odps
        else:
            r = enter()
            self._odps = r.odps

        if 'o' not in self.shell.user_ns:
            self.shell.user_ns['o'] = self._odps

        return r

    @line_magic('setup')
    def setup(self, line):
        args = line.strip().split()
        name, args = args[0], args[1:]
        setup(*args, room=name)
        html_notify('setup succeeded')

    @line_magic('teardown')
    def teardown(self, line):
        name = line.strip()
        teardown(name)
        html_notify('teardown succeeded')

    @line_magic('list_rooms')
    def list_rooms(self, line):
        return list_rooms()

    @line_magic('stores')
    def list_stores(self, line):
        line = line.strip()

        if line:
            room = enter(line)
        else:
            room = enter()

        return room.display()

    def _get_task_percent(self, instance, task_name):
        progress = instance.get_task_progress(task_name)

        if len(progress.stages) > 0:
            all_percent = sum((float(stage.terminated_workers) / stage.total_workers)
                              for stage in progress.stages if stage.total_workers > 0)
            return all_percent / len(progress.stages)
        else:
            return 0

    def _to_stdout(cls, msg):
        print(msg)

    @line_magic('set')
    def set_hint(self, line):
        if '=' not in line:
            raise ValueError('Hint for sql is not allowed')

        key, val = line.strip().strip(';').split('=', 1)
        key, val = key.strip(), val.strip()

        settings = options.sql.settings
        if settings is None:
            options.sql.settings = {key: val}
        else:
            options.sql.settings[key] = val

    @line_cell_magic('sql')
    def execute(self, line, cell=''):
        self._set_odps()

        content = line + '\n' + cell
        content = content.strip()

        sql = None
        hints = dict()

        splits = content.split(';')
        for s in splits:
            stripped = s.strip()
            if stripped.lower().startswith('set '):
                hint = stripped.split(' ', 1)[1]
                k, v = hint.split('=', 1)
                k, v = k.strip(), v.strip()
                hints[k] = v
            elif len(stripped) == 0:
                continue
            else:
                if sql is None:
                    sql = s
                else:
                    sql = '%s;%s' % (sql, s)

        # replace user defined parameters
        sql = replace_sql_parameters(sql, self.shell.user_ns)

        if sql:
            bar = init_progress_bar()

            instance = self._odps.run_sql(sql, hints=hints)
            if options.verbose:
                stdout = options.verbose_log or self._to_stdout
                stdout('Instance ID: ' + instance.id)
                stdout('  Log view: ' + instance.get_logview_address())

            percent = 0
            while not instance.is_terminated():
                task_names = instance.get_task_names()
                last_percent = percent
                if len(task_names) > 0:
                    percent = sum(self._get_task_percent(instance, name)
                                  for name in task_names) / len(task_names)
                else:
                    percent = 0
                percent = min(1, max(percent, last_percent))
                bar.update(percent)

                time.sleep(1)

            instance.wait_for_success()
            bar.update(1)

            try:
                with instance.open_reader() as reader:
                    try:
                        import pandas as pd
                        from pandas.parser import CParserError

                        try:
                            res = pd.read_csv(StringIO(reader.raw))
                        except (ValueError, CParserError):
                            res = reader.raw
                    except ImportError:
                        try:
                            res = ResultFrame(list(reader), columns=reader._columns)
                        except TypeError:
                            res = reader.raw

                html_notify('SQL execution succeeded')
                return res
            finally:
                bar.close()

    @line_magic('persist')
    def persist(self, line):
        try:
            import pandas as pd
            has_pandas = True
        except ImportError:
            has_pandas = False

        self._set_odps()

        line = line.strip().strip(';')

        frame_name, table_name = line.split(None, 1)

        if '.' in table_name:
            project_name, table_name = tuple(table_name.split('.', 1))
        else:
            project_name = None

        frame = self.shell.user_ns[frame_name]
        if self._odps.exist_table(table_name, project=project_name):
            raise TypeError('%s already exists' % table_name)

        if isinstance(frame, DataFrame):
            frame.persist(name=table_name, project=project_name, notify=False)
        elif has_pandas and isinstance(frame, pd.DataFrame):
            frame = DataFrame(frame)
            frame.persist(name=table_name, project=project_name, notify=False)
        html_notify('Persist succeeded')


def load_ipython_extension(ipython):
    ipython.register_magics(ODPSSql)
    js = "IPython.CodeCell.config_defaults.highlight_modes['magic_sql'] = {'reg':[/^%%sql/]};"
    display_javascript(js, raw=True)

    # Do global import when load extension
    ipython.user_ns['DataFrame'] = DataFrame
    ipython.user_ns['Scalar'] = Scalar
    ipython.user_ns['options'] = options
