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

import itertools
import time

from odps.inter import enter, setup, teardown, list_rooms
from odps.compat import StringIO
from odps import types as odps_types
from odps.models import Schema
from odps.utils import init_progress_bar
from odps.df.backends.frame import ResultFrame

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

    @line_magic('enter')
    def enter(self, line):
        room = line.strip()
        if room:
            r = enter(room)
            self._odps = r.odps
        else:
            r = enter()
            self._odps = r.odps

        return r

    @line_magic('setup')
    def setup(self, line):
        args = line.strip().split()
        name, args = args[0], args[1:]
        setup(*args, room=name)

    @line_magic('teardown')
    def teardown(self, line):
        name = line.strip()
        teardown(name)

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

    @line_cell_magic('sql')
    def execute(self, line, cell=''):
        if self._odps is None:
            self._odps = enter().odps

        sql = line + '\n' + cell
        sql = sql.strip()

        if sql:
            bar = init_progress_bar()

            instance = self._odps.run_sql(sql)

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

            with instance.open_reader() as reader:
                try:
                    import pandas as pd

                    try:
                        return pd.read_csv(StringIO(reader.raw))
                    except ValueError:
                        return reader.raw
                except ImportError:
                    return ResultFrame(list(reader), columns=reader._columns)

    @line_magic('persist')
    def persist(self, line):
        import pandas as pd

        if self._odps is None:
            self._odps = enter().odps

        line = line.strip().strip(';')

        frame_name, table_name = line.split(None, 1)

        if '.' in table_name:
            project_name, table_name = tuple(table_name.split('.', 1))
        else:
            project_name = None

        frame = self.shell.user_ns[frame_name]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError('%s is not a Pandas DataFrame' % frame_name)

        columns = list(frame.columns)
        types = [np_to_odps_types.get(tp, odps_types.string) for tp in frame.dtypes]

        if self._odps.exist_table(table_name, project=project_name):
            raise TypeError('%s already exists')

        tb = self._odps.create_table(table_name, Schema.from_lists(columns, types))

        def gen(df):
            size = len(df)

            bar = init_progress_bar(size)

            try:
                c = itertools.count()
                for row in df.values:
                    i = next(c)
                    if i % 50 == 0:
                        bar.update(min(i, size))

                    yield tb.new_record(list(row))

                bar.update(size)
            finally:
                bar.close()

        with tb.open_writer() as writer:
            writer.write(gen(frame))


def load_ipython_extension(ipython):
    ipython.register_magics(ODPSSql)
    js = "IPython.CodeCell.config_defaults.highlight_modes['magic_sql'] = {'reg':[/^%%sql/]};"
    display_javascript(js, raw=True)
