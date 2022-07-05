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

import time

from ..inter import enter, setup, teardown, list_rooms
from ..compat import six, StringIO
from .. import types as odps_types
from .. import options, ODPS, log
from ..utils import replace_sql_parameters, init_progress_ui
from ..df import DataFrame, Scalar, NullScalar, Delay
from ..df.backends.frame import ResultFrame
from ..df.backends.odpssql.types import odps_schema_to_df_schema, odps_type_to_df_type
from ..models import Schema
from ..ui.common import html_notify
from ..ui.progress import create_instance_group, reload_instance_status, fetch_instance_group

try:
    import numpy as np

    np_int_types = map(np.dtype, [np.int_, np.int8, np.int16, np.int32, np.int64])
    np_float_types = map(np.dtype, [np.float_, np.float16, np.float32, np.float64])
    np_to_odps_types = dict([(t, odps_types.bigint) for t in np_int_types] +
                            [(t, odps_types.double) for t in np_float_types])
except ImportError:
    pass

try:
    from IPython.core.magic import Magics, magics_class, line_cell_magic, line_magic
except ImportError:
    # skipped for ci
    ODPSSql = None
    pass
else:
    @magics_class
    class ODPSSql(Magics):

        _odps = None

        def _set_odps(self):
            if self._odps is not None:
                return

            if options.account is not None and options.default_project is not None:
                self._odps = ODPS._from_account(
                    options.account, options.default_project,
                    endpoint=options.endpoint, tunnel_endpoint=options.tunnel.endpoint
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
                self.shell.user_ns['odps'] = self._odps

            return r

        @line_magic('setup')
        def setup(self, line):
            args = line.strip().split()
            name, args = args[0], args[1:]
            setup(*args, room=name)
            html_notify('Setup succeeded')

        @line_magic('teardown')
        def teardown(self, line):
            name = line.strip()
            teardown(name)
            html_notify('Teardown succeeded')

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

        @staticmethod
        def _get_task_percent(task_progress):
            if len(task_progress.stages) > 0:
                all_percent = sum((float(stage.terminated_workers) / stage.total_workers)
                                  for stage in task_progress.stages if stage.total_workers > 0)
                return all_percent / len(task_progress.stages)
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
                progress_ui = init_progress_ui()
                group_id = create_instance_group('SQL Query')
                progress_ui.add_keys(group_id)

                instance = self._odps.run_sql(sql, hints=hints)
                if options.verbose:
                    log('Instance ID: ' + instance.id)
                    log('  Log view: ' + instance.get_logview_address())
                reload_instance_status(self._odps, group_id, instance.id)
                progress_ui.status('Executing')

                percent = 0
                while not instance.is_terminated(retry=True):
                    last_percent = percent

                    reload_instance_status(self._odps, group_id, instance.id)
                    inst_progress = fetch_instance_group(group_id).instances.get(instance.id)

                    if inst_progress is not None and len(inst_progress.tasks) > 0:
                        percent = sum(self._get_task_percent(task)
                                      for task in six.itervalues(inst_progress.tasks)) / len(inst_progress.tasks)
                    else:
                        percent = 0

                    percent = min(1, max(percent, last_percent))

                    progress_ui.update(percent)
                    progress_ui.update_group()

                    time.sleep(1)

                instance.wait_for_success()
                progress_ui.update(1)

                try:
                    with instance.open_reader() as reader:
                        try:
                            import pandas as pd
                            try:
                                from pandas.io.parsers import ParserError as CParserError
                            except ImportError:
                                pass
                            try:
                                from pandas.parser import CParserError
                            except ImportError:
                                CParserError = ValueError

                            if not hasattr(reader, 'raw'):
                                res = ResultFrame([rec.values for rec in reader],
                                                  schema=odps_schema_to_df_schema(reader._schema))
                            else:
                                try:
                                    res = pd.read_csv(StringIO(reader.raw))
                                    if len(res.values) > 0:
                                        schema = DataFrame(res).schema
                                    else:
                                        cols = res.columns.tolist()
                                        schema = odps_schema_to_df_schema(
                                            Schema.from_lists(cols, ['string' for _ in cols]))
                                    res = ResultFrame(res.values, schema=schema)
                                except (ValueError, CParserError):
                                    res = reader.raw
                        except ImportError:
                            if not hasattr(reader, 'raw'):
                                res = ResultFrame([rec.values for rec in reader],
                                                  schema=odps_schema_to_df_schema(reader._schema))
                            else:
                                try:
                                    columns = [odps_types.Column(name=col.name,
                                                                 typo=odps_type_to_df_type(col.type))
                                               for col in reader._columns]
                                    res = ResultFrame(list(reader), columns=columns)
                                except TypeError:
                                    res = reader.raw

                    html_notify('SQL execution succeeded')
                    return res
                finally:
                    progress_ui.close()

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

        # Do global import when load extension
        ipython.user_ns['DataFrame'] = DataFrame
        ipython.user_ns['Scalar'] = Scalar
        ipython.user_ns['NullScalar'] = NullScalar
        ipython.user_ns['options'] = options
        ipython.user_ns['Schema'] = Schema
        ipython.user_ns['Delay'] = Delay
