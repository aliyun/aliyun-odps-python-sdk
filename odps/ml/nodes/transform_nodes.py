# encoding: utf-8
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

from bisect import bisect_left

from ...errors import NoPermission
from ...config import options
from ...compat import OrderedDict, six
from ...runner import BaseRunnerNode, PortType, EngineType, RunnerContext
from ..nodes.exporters import get_input_table_name, get_input_partitions, get_output_table_name, \
    get_output_table_partitions, get_input_field_names
from ..utils import is_temp_table


class ReusableMixIn(BaseRunnerNode):
    def __init__(self, *args, **kwargs):
        super(ReusableMixIn, self).__init__(*args, **kwargs)
        self._init_objs()

    def _init_objs(self):
        self._cmd_hash = None
        self._table_names = None
        self._sink = None

    @property
    def table_names(self):
        return self._table_names

    @table_names.setter
    def table_names(self, value):
        self._table_names = value

    def get_computed(self, odps):
        return None

    def set_computed(self, odps, value):
        pass

    def compute_result(self, odps):
        pass

    def after_exec(self, odps, is_success):
        super(ReusableMixIn, self).after_exec(odps, is_success)

        if is_success:
            computed = self.get_computed(odps)
            if not computed:
                self.compute_result(odps)
            RunnerContext.instance()._result_container[self._cmd_hash] = self.get_computed(odps)

    def before_exec(self, odps, conv_params):
        super(ReusableMixIn, self).before_exec(odps, conv_params)
        context = RunnerContext.instance()

        self._cmd_hash = self.calc_node_hash(odps)
        if self._cmd_hash in context._result_container:
            self.executed = True
            self.set_computed(odps, context._result_container[self._cmd_hash])


class TableReusableMixIn(ReusableMixIn):
    def __init__(self, *args, **kwargs):
        super(TableReusableMixIn, self).__init__(*args, **kwargs)

    def get_computed(self, odps):
        tables = dict()
        for oname, output in six.iteritems(self.outputs):
            tables[oname] = output.obj.table
        return tables

    def set_computed(self, odps, value):
        for oname, output in six.iteritems(self.outputs):
            if oname in value:
                output.obj.table = value[oname]


class SummaryNode(TableReusableMixIn, BaseRunnerNode):
    class DataSummary(dict):
        def _repr_html_(self):
            cols, col_set = [], set()
            for r in six.itervalues(self):
                keys = [rn for rn in six.iterkeys(r) if rn not in col_set]
                cols.extend(keys)
                col_set.update(keys)

            html = '<table width="100%">'
            html += '<thead><tr><th></th>' + ''.join('<th>%s</th>' % c for c in cols) + '</tr></thead>'
            html += '<tbody>'
            for row_name, row_dict in sorted(list(six.iteritems(self)), key=lambda p: p[0]):
                html += '<tr>'
                html += '<td>%s</td>' % row_name
                html += ''.join('<td>%s</td>' % row_dict.get(c) for c in cols)
                html += '</tr>'
            html += '</tbody></table>'
            return html

    def __init__(self, columns, force_categorical):
        super(SummaryNode, self).__init__(code_name='summary')
        self.marshal({
            'parameters': {
                'forceCategorical': ','.join(force_categorical)
            },
            'inputs': [(1, 'input', PortType.DATA)],
            'outputs': [(1, 'output', PortType.DATA)]
        })
        self._columns = columns
        self._feature_count = 2048

        def selected_column_names_exporter():
            if self._columns:
                fields = columns
            else:
                fields = get_input_field_names(self, 'selectedColNames', 'input', field_func=lambda f: f.role)
            self._feature_count = len(fields)
            return fields

        self.add_exporter('inputTableName', lambda: get_input_table_name(self, 'input'))
        self.add_exporter('inputTablePartitions', lambda: get_input_partitions(self, 'input'))
        self.add_exporter('selectedColNames', selected_column_names_exporter)
        self.add_exporter('outputTableName', lambda: get_output_table_name(self, 'output'))

    @property
    def sink(self):
        return self.DataSummary(self._sink)

    def after_exec(self, odps, is_success):
        super(SummaryNode, self).after_exec(odps, is_success)

        input_roles = dict((f.name, f.role) for f in self.inputs['input'].obj._fields)
        table_name = get_output_table_name(self, 'output')
        try:
            recs = list(odps.read_table(table_name))
        except NoPermission:
            table_obj = odps.get_table(table_name)
            reader_inst = odps.execute_sql('select * from `%s`' % table_name)
            with reader_inst.open_reader(schema=table_obj.schema) as reader:
                recs = list(reader)

        field_stats = OrderedDict()
        for rec in recs:
            field_name = rec.values[0]
            field_stat = OrderedDict([('field_role', ','.join(r.name for r in input_roles[field_name])), ])
            field_stat.update(OrderedDict([(rec._columns[idx + 1].name, val) for idx, val in enumerate(rec.values[1:])]))
            field_stats[field_name] = field_stat
        self._sink = field_stats


class SQLNode(BaseRunnerNode):
    def __init__(self, stmt=None, code_name='sql'):
        super(SQLNode, self).__init__(code_name, EngineType.SQL)
        self.marshal({
            "parameters": {
                "script": stmt
            },
            "inputs": [(idx, "input%d" % idx, PortType.DATA) for idx in range(1, 5)],
            "outputs": [(idx, "output%d" % idx, PortType.DATA) for idx in range(1, 5)]
        })
        self.reload_on_finish = True

        self.add_exporter("script", self.generate_sql)

    def get_input_tables(self):
        return [pair for pair in (('input%d' % idx, get_input_table_name(self, 'input%d' % idx))
                                          for idx in range(1, 5)) if pair[1] is not None]

    def generate_sql(self):
        core_stmt = self.parameters['script']
        input_tables = self.get_input_tables()
        # replace placeholders in user sql statement
        for placeholder, table in input_tables:
            if table is not None:
                core_stmt = core_stmt.replace('$' + placeholder, table)

        # special: the only input can be denoted as $input
        if len(input_tables) == 1 and input_tables[0][0] == 'input1':
            core_stmt = core_stmt.replace('$input', input_tables[0][1])

        return self.wrap_create_sql(1, core_stmt)

    def wrap_create_sql(self, port_id, core_stmt):
        output_name = 'output%d' % port_id
        temp_lifecycle = options.temp_lifecycle
        global_lifecycle = options.lifecycle
        ds = self.outputs[output_name].obj
        if ds is None:
            raise ValueError('Failed to get data set object.')

        if global_lifecycle is None and not is_temp_table(ds.table):
            lifecycle_stmt = ''
        else:
            lifecycle_stmt = ' lifecycle ' + str(temp_lifecycle if is_temp_table(ds.table) else global_lifecycle)

        if ds.partitions is None:
            return 'create table ' + ds.table + lifecycle_stmt + ' as ' + core_stmt
        else:
            return ['insert overwrite table ' + ds.table + ' partition (' +
                    ','.join(ds.partitions.to_partition_spec(0)) + ') ' + core_stmt,
                    'alter table %s add if not exists partition (%s)' %
                    (ds.table, ds.partitions.to_partition_fields())]

    @staticmethod
    def assert_sql(stmt):
        quote_poses = [pos for pos, ch in enumerate(stmt) if ch == '\'']
        quote_props = dict((qpos, pos_id % 2 != 0) for pos_id, qpos in enumerate(quote_poses))

        def assert_pos_not_quoted(pos, msg):
            idx = bisect_left(quote_poses, pos)
            if idx:
                qpos = quote_poses[idx - 1]
                if not quote_props[qpos]:
                    raise Exception(msg)

        # assure that there is only one sql statement
        for pos, ch in enumerate(stmt):
            if ch != ';':
                continue
            assert_pos_not_quoted(pos, 'Multiple SQL statements not supported.')

        # assure that there are no denied keywords
        start_pos = 0
        find_src = stmt.upper()
        while start_pos >= 0:
            poses = [(kw, find_src.find(kw, start_pos)) for kw in ['CREATE', 'ALTER', 'UPDATE', 'DELETE', 'INSERT']
                     if kw in find_src]
            if len(poses) == 0:
                break
            pos_pair = min(poses, key=lambda p: p[1])
            assert_pos_not_quoted(pos_pair[1], "Denied statement %s detected." % pos_pair[0])
            start_pos += len(pos_pair[0])


class DataCopyNode(SQLNode):
    def __init__(self):
        super(DataCopyNode, self).__init__('SELECT * FROM $input', code_name='data_copy')


class MergeColumnNode(BaseRunnerNode):
    def __init__(self, input_count, auto_rename_col=False, selected_cols=None, excluded_cols=None):
        super(MergeColumnNode, self).__init__('AppendColumns', engine=EngineType.XFLOW)
        self.marshal({
            'parameters': {
                'autoRenameCol': 'True' if auto_rename_col else 'False'
            },
            'inputs': [(seq, 'input%d' % seq, PortType.DATA) for seq in range(1, input_count + 1)],
            'outputs': [(1, 'output', PortType.DATA)]
        })

        if selected_cols is None:
            selected_cols = dict()

        if excluded_cols is None:
            excluded_cols = dict()

        def fetch_adapters():
            return [self.inputs['input%d' % seq].obj for seq in range(1, input_count + 1)]

        def table_names_exporter():
            dses = fetch_adapters()
            return ','.join(ds.table for ds in dses)

        def selected_cols_exporter():
            dses = fetch_adapters()

            def fetch_ds_cols(seq, ds):
                if seq in selected_cols:
                    return selected_cols[seq]
                elif seq in excluded_cols:
                    return [f for f in ds._fields if f.role and f.name not in excluded_cols[seq]]
                else:
                    return [f for f in ds._fields if f.role]

            return ';'.join(','.join(f.name for f in fetch_ds_cols(seq, ds)) for seq, ds in enumerate(dses))

        def input_partitions_exporter():
            dses = fetch_adapters()
            return ';'.join(','.join(ds._partitions) if ds._partitions else '' for ds in dses)

        self.add_exporter("inputTableNames", table_names_exporter)
        self.add_exporter("outputTableName", lambda: get_output_table_name(self, "output"))
        self.add_exporter("selectedColNamesList", selected_cols_exporter)
        self.add_exporter("inputPartitionsInfoList", input_partitions_exporter)
        self.add_exporter("outputPartition", lambda: get_output_table_partitions(self, "output"))
