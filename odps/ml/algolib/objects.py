# encoding: utf-8
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

from itertools import groupby

from ..enums import PortType, PortDirection
from ...compat import six
from ...utils import str_to_bool, camel_to_underline
from ... import types as odps_types, serializers


class ParamDef(serializers.XMLSerializableModel):
    """
    Description for algorithm parameter.

    :param name: parameter name
    :type name: str
    :param default: default value
    :type default: str
    :param type: parameter type
    :type type: str
    """
    _root = 'param'

    name = serializers.XMLNodeAttributeField(attr='name')
    required = serializers.XMLNodeAttributeField(attr='required', parse_callback=str_to_bool)
    docs = serializers.XMLNodeField('docs')
    alias = serializers.XMLNodeField('alias')
    _value = serializers.XMLNodeField('value')
    type = serializers.XMLNodeField('type')
    exported = serializers.XMLNodeField('exported', parse_callback=str_to_bool)
    internal = serializers.XMLNodeField('internal', parse_callback=str_to_bool)
    exporter = serializers.XMLNodeField('exporter')
    setter = serializers.XMLNodeField('setter')
    input_name = serializers.XMLNodeField('inputName')
    output_name = serializers.XMLNodeField('outputName')
    sequence = serializers.XMLNodeField('sequence', parse_callback=int)

    def __init__(self, *args, **kwargs):
        if len(args) >= 1 and 'name' not in kwargs:
            kwargs['name'] = args[0]
        if len(args) >= 2 and 'default' not in kwargs:
            kwargs['value'] = args[1]
        super(ParamDef, self).__init__(**kwargs)
        self.exported = True if self.exported is None else self.exported
        if kwargs.get('default') is not None:
            self._value = kwargs.get('default')

    @property
    def friendly_name(self):
        return camel_to_underline(self.name) if not self.alias else camel_to_underline(self.alias)

    @property
    def value(self):
        if self._value is None or self.type is None:
            return self._value
        odps_type = getattr(odps_types, self.type)
        return odps_type.cast_value(self._value, odps_types.infer_primitive_data_type(self._value))

    @value.setter
    def value(self, v):
        self._value = v

    def copy(self):
        cls = self.__class__
        keys = dict((k, getattr(self, k)) for k in six.iterkeys(getattr(cls, '__fields')))
        return cls(**keys)

    @classmethod
    def build_input_table(cls, name='inputTableName', input_name='input'):
        """
        Build an input table parameter

        :param name: parameter name
        :type name: str
        :param input_name: bind input port name
        :param input_name: str
        :return: input description
        :rtype: ParamDef
        """
        obj = cls(name)
        obj.exporter = 'get_input_table_name'
        obj.input_name = input_name
        return obj

    @classmethod
    def build_input_partitions(cls, name='inputTablePartitions', input_name='input'):
        """
        Build an input table partition parameter

        :param name: parameter name
        :type name: str
        :param input_name: bind input port name
        :param input_name: str
        :return: input description
        :rtype: ParamDef
        """
        obj = cls(name)
        obj.exporter = 'get_input_partitions'
        obj.input_name = input_name
        return obj

    @classmethod
    def build_feature_col_names(cls, name='featureColNames', input_name='input'):
        obj = cls(name)
        obj.exporter = 'get_feature_columns'
        obj.input_name = input_name
        return obj

    @classmethod
    def build_label_col_name(cls, name='labelColName', input_name='input'):
        obj = cls(name)
        obj.exporter = 'get_label_column'
        obj.input_name = input_name
        return obj

    @classmethod
    def build_output_table(cls, name='inputTableName', output_name='output'):
        """
        Build an output table parameter

        :param name: parameter name
        :type name: str
        :param output_name: bind input port name
        :type output_name: str
        :return: output description
        :rtype: ParamDef
        """
        obj = cls(name)
        obj.exporter = 'get_output_table_name'
        obj.output_name = output_name
        return obj

    @classmethod
    def build_output_partitions(cls, name='inputTablePartitions', output_name='output'):
        """
        Build an output table partition parameter

        :param name: parameter name
        :type name: str
        :param output_name: bind input port name
        :type output_name: str
        :return: output description
        :rtype: ParamDef
        """
        obj = cls(name)
        obj.exporter = 'get_output_table_partition'
        obj.output_name = output_name
        return obj

    @classmethod
    def build_model_name(cls, name='modelName', output_name='output'):
        """
        Build an output model name parameter.

        :param name: model name
        :type name: str
        :param output_name: bind output port name
        :type output_name: str
        :return: output description
        :rtype: ParamDef
        """
        obj = cls(name)
        obj.exporter = 'generate_model_name'
        obj.output_name = output_name
        return obj


class SchemaDef(serializers.XMLSerializableModel):
    name = serializers.XMLNodeAttributeField(attr='name')
    dynamic = serializers.XMLNodeField('dynamic', type='bool')
    copy_input = serializers.XMLNodeField('copyInput')
    schema = serializers.XMLNodeField('schema')
    direct_copy = serializers.XMLNodeField('directCopy')
    mixin = serializers.XMLNodeField('mixin')

    def __init__(self, **kwargs):
        super(SchemaDef, self).__init__(**kwargs)
        self.dynamic = False if self.dynamic is None else self.dynamic

    @property
    def programmatic(self):
        return self.schema and '.' in self.schema


class ModelDef(serializers.XMLSerializableModel):
    type = serializers.XMLNodeField('type', default='PmmlModel')
    schemas = serializers.XMLNodesReferencesField(SchemaDef, 'schemas', 'schema')
    copy_params = serializers.XMLNodeField('copyParams')
    mixin = serializers.XMLNodeField('mixin')


class PortDef(serializers.XMLSerializableModel):
    """
    Description for algorithm port.

    :param name: port name
    :type name: str
    :param direction: port direction (input / output)
    :type direction: PortDirection
    :param type: port type (model / data)
    :type type: PortType
    :param copy_input: input name where the schema is copied from.
    :type copy_input: str
    :param schema: k1:v1,k2:v2 string describing the schema to be appended
    :type schema: str
    """
    name = serializers.XMLNodeAttributeField(attr='name')
    docs = serializers.XMLNodeField('docs')
    type = serializers.XMLNodeField('type', parse_callback=lambda v: PortType(v.upper()),
                                    serialize_callback=lambda v: v.value)
    io_type = serializers.XMLNodeField('ioType', parse_callback=lambda v: PortDirection(v.upper()),
                                       serialize_callback=lambda v: v.value)
    sequence = serializers.XMLNodeField('sequence', parse_callback=int)
    required = serializers.XMLNodeField('required', parse_callback=str_to_bool, default=True)
    schema = serializers.XMLNodeReferenceField(SchemaDef, 'schema')
    model = serializers.XMLNodeReferenceField(ModelDef, 'model')
    exporter = serializers.XMLNodeField('exporter')

    def __init__(self, *args, **kwargs):
        if len(args) >= 1 and 'name' not in kwargs:
            kwargs['name'] = args[0]
        if len(args) >= 2 and 'io_type' not in kwargs:
            kwargs['io_type'] = args[1]
        super(PortDef, self).__init__(**kwargs)

    @classmethod
    def build_data_input(cls, name='input'):
        """
        Build a data input port.

        :param name: port name
        :type name: str
        :return: port object
        :rtype: PortDef
        """
        return cls(name, PortDirection.INPUT, type=PortType.DATA)

    @classmethod
    def build_data_output(cls, name='output', copy_input=None, schema=None):
        """
        Build a data output port.

        :param name: port name
        :type name: str
        :return: port object
        :param copy_input: input name where the schema is copied from.
        :type copy_input: str
        :param schema: k1:v1,k2:v2 string describing the schema to be appended
        :type schema: str
        :rtype: PortDef
        """
        return cls(name, PortDirection.OUTPUT, type=PortType.DATA, copy_input=copy_input, schema=schema)

    @classmethod
    def build_model_input(cls, name='input'):
        """
        Build a model input port.

        :param name: port name
        :type name: str
        :return: port object
        :rtype: PortDef
        """
        return cls(name, PortDirection.INPUT, type=PortType.MODEL)

    @classmethod
    def build_model_output(cls, name='output'):
        """
        Build a model output port.

        :param name: port name
        :type name: str
        :return: port object
        :rtype: PortDef
        """
        return cls(name, PortDirection.OUTPUT, type=PortType.MODEL)


class MetaDef(serializers.XMLSerializableModel):
    name = serializers.XMLNodeAttributeField(attr='name')
    _param_value = serializers.XMLNodeAttributeField(attr='value')
    _text_value = serializers.XMLNodeField('.')

    @property
    def value(self):
        return self._text_value or self._param_value

    @value.setter
    def value(self, v):
        self._text_value = v
        self._param_value = None

    def __init__(self, name=None, value=None, **kwargs):
        if name:
            kwargs['name'] = name
        if value:
            kwargs['_text_value'] = value
        super(MetaDef, self).__init__(**kwargs)


class AlgorithmDef(serializers.XMLSerializableModel):
    """
    Description for an algorithm. This class can help users publish their own algorithms.

    :param code_name: Code name of the algorithm

    :type params: list[ParamDef]
    :type ports: list[PortDef]
    :type metas: list[MetaDef]
    :type code_name: str
    """
    _root = 'algorithm'
    code_name = serializers.XMLNodeAttributeField(attr='codeName')
    docs = serializers.XMLNodeField('docs')
    entry_docs = serializers.XMLNodeField('entryDocs')
    export_function = serializers.XMLNodeField('exportFunction', parse_callback=str_to_bool)
    enabled = serializers.XMLNodeField('enabled', parse_callback=str_to_bool)
    reload_fields = serializers.XMLNodeField('reloadFields', parse_callback=str_to_bool)
    base_class = serializers.XMLNodeField('baseClass')
    params = serializers.XMLNodesReferencesField(ParamDef, 'params', 'param')
    ports = serializers.XMLNodesReferencesField(PortDef, 'ports', 'port')
    metas = serializers.XMLNodesReferencesField(MetaDef, 'metas', 'meta')
    public = serializers.XMLNodeField('public', parse_callback=str_to_bool, default=True)
    port_seqs = dict()

    def __init__(self, code_name=None, **kwargs):
        super(AlgorithmDef, self).__init__(**kwargs)
        self.code_name = code_name
        self.params = []
        self.ports = []
        self.metas = []

    def add_param(self, param):
        """
        Add a parameter object to the definition

        :param param: parameter
        :type param: ParamDef
        """
        self.params.append(param)
        return self

    def add_port(self, port):
        """
        Add a port object to the definition

        :param port: port definition
        :type port: PortDef
        """
        self.ports.append(port)
        if port.io_type not in self.port_seqs:
            self.port_seqs[port.io_type] = 0
        self.port_seqs[port.io_type] += 1
        port.sequence = self.port_seqs[port.io_type]
        return self

    def add_meta(self, name, value):
        """
        Add a pair of meta data to the definition

        :param name: name of the meta
        :type name: str
        :param value: value of the meta
        :type value: str
        """
        for mt in self.metas:
            if mt.name == name:
                mt.value = value
                return self
        self.metas.append(MetaDef(name, value))
        return self

    def get_input_ports(self):
        return sorted([p for p in self.ports if p.io_type == PortDirection.INPUT], key=lambda p: p.sequence)

    def get_output_ports(self):
        return sorted([p for p in self.ports if p.io_type == PortDirection.OUTPUT], key=lambda p: p.sequence)

    @property
    def meta_dict(self):
        return dict((m.name, m.value) for m in self.metas)

    def serialize(self):
        """
        Serialize the algorithm definition
        """
        # fill sequences
        for keys, groups in groupby(self.ports, lambda x: x.io_type):
            for seq, port in enumerate(groups):
                port.sequence = seq
        return super(AlgorithmDef, self).serialize()


class XflowAlgorithmDef(AlgorithmDef):
    """
    Description for Xflow algorithm.
    """

    def __init__(self, code_name=None, xflow_name=None, project='algo_public', **kwargs):
        super(XflowAlgorithmDef, self).__init__(code_name, **kwargs)
        if xflow_name is None:
            xflow_name = code_name
        if xflow_name is not None:
            self.add_meta('xflowName', xflow_name)
        self.add_meta('xflowProjectName', project)


class AlgorithmsDef(serializers.XMLSerializableModel):
    algorithms = serializers.XMLNodesReferencesField(AlgorithmDef, 'algorithm')
    base_class = serializers.XMLNodeAttributeField(attr='baseClass')

    @classmethod
    def parse(cls, response, obj=None, **kw):
        if six.PY2:
            response = response.encode('utf-8')
        return super(AlgorithmsDef, cls).parse(response, obj=obj, **kw)
