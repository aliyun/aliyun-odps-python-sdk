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

from __future__ import absolute_import

import json
import logging
from abc import abstractmethod

from ... import options
from ...compat import six
from ...utils import escape_odps_string
from ...df.core import DataFrame
from ...runner import BaseRunnerNode, RunnerContext, RunnerObject, ObjectDescription, PortType, gen_table_name, \
    adapter_from_df, DFAdapter
from ..nodes.io_nodes import TablesModelInputNode, TablesModelOutputNode
from ..utils import import_class_member, TABLE_MODEL_SEPARATOR, TABLE_MODEL_PREFIX

logger = logging.getLogger(__name__)


class MLModel(RunnerObject):
    """
    Base class for models in PyODPS ML.
    """
    def __init__(self, odps, port):
        super(MLModel, self).__init__(odps, port)
        RunnerContext.instance()._obj_container.register(self)
        port.obj_uuid = self._obj_uuid

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def reload(self):
        if not hasattr(self, '_multiplexer') or not self._multiplexer:
            return
        if options.runner.dry_run:
            return
        for ep in (ep for ep in six.itervalues(self._multiplexer.outputs) if ep.obj_uuid is not None):
            obj = ep.obj
            if obj:
                obj.reload()


class MultiplexerNode(BaseRunnerNode):
    def __init__(self, output_names):
        super(MultiplexerNode, self).__init__("Multiplexer")
        self.marshal({
            'inputs': [(1, 'model', PortType.MODEL)],
            'outputs': [(idx + 1, name, PortType.DATA) for idx, name in enumerate(output_names)],
        })
        self.virtual = True


class TablesModel(MLModel):
    """
    Table model reference for PyODPS ML. This type of model is designed for non-pmml models as well as for compatibility
    needs. Table names are similar to 'otm_${table_prefix}__${table_name}'.

    ``predict`` method can be used to predict a data set using the referenced model.
    ``persist`` method can be used to store models with given names.

    :param odps: ODPS object
    :param str prefix: model name prefix
    :param str project: project name, if the model is stored in a different project
    """
    def __new__(cls, odps, name=None, project=None, **kw):
        if name is not None:
            name_parts = name.split('.', 1)
            if len(name_parts) > 1:
                project = project or name_parts[0]
                name = name_parts[1]

            source_node = TablesModelInputNode(name, project=project)

            out_table_dict = dict()
            table_comment = None
            for tb in odps.list_tables(project=project, prefix=TABLE_MODEL_PREFIX + name):
                if TABLE_MODEL_SEPARATOR not in tb.name:
                    continue
                _, sub_table_name = tb.name.rsplit(TABLE_MODEL_SEPARATOR)
                out_table_dict[sub_table_name] = tb.name
                if not table_comment:
                    table_comment = json.loads(tb.comment)

            if 'className' not in table_comment:
                model_class = TablesModel
            else:
                model_class = import_class_member(table_comment['className'])

            obj = MLModel.__new__(model_class, odps, table_prefix=name, project=project, **kw)
            obj.__out_table_dict = out_table_dict
            obj.__table_comment = table_comment
            obj.__port = source_node.outputs.get('model')
            return obj
        else:
            return MLModel.__new__(cls, odps, table_prefix=name, project=project, **kw)

    def __init__(self, odps, prefix=None, project=None, **kw):
        self._dfs = dict()
        self._params = {}

        if prefix is not None:
            super(TablesModel, self).__init__(odps, self.__port)
            self._set_outputs(six.iterkeys(self.__out_table_dict))

            for oname, otable in six.iteritems(self.__out_table_dict):
                self._dfs[oname] = DataFrame(odps.get_table(otable, project=project))
                DFAdapter(odps, self._multiplexer.outputs[oname], df=self._dfs[oname])
            self._params = self.__table_comment
        else:
            super(TablesModel, self).__init__(odps, kw.get('port'))

    def _set_outputs(self, output_names):
        output_names = list(output_names)
        self._multiplexer = MultiplexerNode(output_names)
        self._link_node(self._multiplexer, 'model')

        cls = type(self)
        if not hasattr(cls, '__perinstance'):
            cls = type(cls.__name__, (cls,), {})
            cls.__perinstance = True

        for oname in output_names:
            setattr(cls, oname, self._gen_port_property(oname))
        self.__class__ = cls

    def _gen_port_property(self, port_name):
        def _getter(self):
            if port_name in self._dfs:
                return self._dfs[port_name]
            else:
                raise AttributeError('Port %s not exist.' % port_name)

        return property(_getter)

    def _get_output_port(self, port_name):
        return self._multiplexer.outputs.get(port_name)

    def gen_temp_names(self):
        tables = dict()
        for k in six.iterkeys(self._dfs):
            table_name = gen_table_name(self._bind_node.code_name, node_id=self._bind_node.node_id, suffix=k)
            adapter = adapter_from_df(self._dfs[k])
            if not adapter.table:
                tables[k] = adapter.table = table_name
        return ObjectDescription(tables=tables)

    def describe(self):
        ds_dict = dict((k, adapter_from_df(v).describe()) for k, v in six.iteritems(self._dfs))
        table_dict = dict((k, (v.table, v.partitions) if v.partitions is not None else v.table)
                          for k, v in ((k, adapter_from_df(v_)) for k, v_ in six.iteritems(self._dfs)))
        return ObjectDescription(tables=table_dict, dfs=ds_dict, params=self._params)

    def fill(self, desc):
        for k, v in six.iteritems(desc.dfs):
            self._dfs[k].fill(v)
        self._params = desc.params

    @property
    def predictor(self):
        return self._params.get('predictor')

    @predictor.setter
    def predictor(self, value):
        self._params['predictor'] = value

    def predict(self, *args, **kwargs):
        """
        Predict given data set using the given model. Actual prediction steps will not
        be executed till an operational step is called.

        :param list[DataFrame] args: input data sets to be predicted
        :param kwargs: named input data sets or prediction parameters, details can be found in ''Predictor Parameters'' section of training algorithms.

        A :class:`DataFrame` object will be generated for input data. Output fields may be referenced in the
        documents of training algorithms.
        """
        from ..algorithms.loader import dispatch_args
        _predictor = import_class_member(self.predictor)
        members = set(p for p in dir(_predictor) if not p.startswith('_') and
                      not p.startswith('set_') and not callable(getattr(_predictor, p)))
        ml_args, p_args, ml_kw, p_kw = dispatch_args(members, *args, **kwargs)
        return _predictor(*p_args, **p_kw).transform(self, *ml_args, **ml_kw)

    def persist(self, table_prefix, project=None):
        """
        Store model into ODPS with given table prefix.

        Note that this method will trigger the defined flow to execute.

        :param str table_prefix: name of the prefix of the model. Actual table name will be 'otm_${table_prefix}__${table_name}'.
        :param str project: name of project, if you want to store into a new one.
        """
        self._params['className'] = self._cls_path
        param_json = json.dumps(self._params)

        model_node = TablesModelOutputNode(table_prefix, project=project)
        self._link_node(model_node, 'model')

        RunnerContext.instance()._run(self._bind_node)
        for key, df in six.iteritems(self._dfs):
            adapter = adapter_from_df(df)
            self._odps.run_sql("alter table %s set comment '%s'" % (adapter.table, escape_odps_string(param_json)))


class TablesRecommendModel(TablesModel):
    """
    Table model reference with recommending functionality for PyODPS ML. This type of model is designed for recommendation
    algorithms built on TablesModel.

    Besides methods in :class:`odps.ml.TablesModel`, you can also use `recommend` method to do recommendation.
    """
    @property
    def recommender(self):
        return self._params.get('recommender')

    @recommender.setter
    def recommender(self, value):
        self._params['recommender'] = value

    def recommend(self, *args, **kwargs):
        """
        Recommend given data set using the given recommendation model. Actual recommending steps will not
        be executed till an operational step is called.

        A :class:`DataFrame` object will be generated for input data. Output fields may be referenced in the
        documents of recommendation algorithms.
        """
        from ..algorithms.loader import dispatch_args
        _recommender = import_class_member(self.recommender)
        members = set(p for p in dir(_recommender) if not p.startswith('_') and
                      not p.startswith('set_') and not callable(getattr(_recommender, p)))
        ml_args, p_args, ml_kw, p_kw = dispatch_args(members, *args, **kwargs)
        return _recommender(*p_args, **p_kw).transform(self, *ml_args, **ml_kw)


def list_tables_model(odps, prefix='', project=None):
    """
    List all TablesModel in the given project.

    :param odps: ODPS object
    :param prefix: model prefix
    :param str project: project name, if you want to look up in another project
    :rtype: list[str]
    """
    name_parts = prefix.split('.', 1)
    if len(name_parts) > 1:
        project = project or name_parts[0]
        prefix = name_parts[1]

    tset = set()
    for tb in odps.list_tables(project=project, prefix=TABLE_MODEL_PREFIX + prefix):
        if TABLE_MODEL_SEPARATOR not in tb.name:
            continue
        tname, _ = tb.name.rsplit(TABLE_MODEL_SEPARATOR, 1)
        tset.add(tname.replace(TABLE_MODEL_PREFIX, ''))
    return sorted(tset)
