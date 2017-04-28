# -*- coding: utf-8 -*-
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

import json
import time
from decimal import Decimal

from ..core import LazyLoad
from ...compat import six, OrderedDict
from ...rest import RestClient
from ...config import options
from ... import serializers, errors, compat, utils, types as odps_types

PREDICT_TYPE_CODES = dict(bool=1, int=10, long=20, float=40, str=50)
if six.PY3:
    PREDICT_TYPE_CODES['int'] = 20
PREDICT_ODPS_TYPE_CODES = dict()
PREDICT_ODPS_TYPE_CODES[type(odps_types.boolean).__name__] = 1
PREDICT_ODPS_TYPE_CODES[type(odps_types.bigint).__name__] = 20
PREDICT_ODPS_TYPE_CODES[type(odps_types.double).__name__] = 40
PREDICT_ODPS_TYPE_CODES[type(odps_types.string).__name__] = 50


class JSONXMLSerializableModelMetaClass(serializers.SerializableModelMetaClass):
    def __new__(mcs, name, bases, kv):
        cls = serializers.SerializableModelMetaClass.__new__(mcs, name, bases, kv)
        if 'JSON' in kv:
            kv['JSON']._xml_class = cls
        return cls


class JSONInputXMLSerializableModel(six.with_metaclass(JSONXMLSerializableModelMetaClass,
                                                       serializers.XMLSerializableModel)):
    @classmethod
    def deserial(cls, content, obj=None, **kw):
        if not hasattr(cls, 'JSON') and 'json_cls' not in kw:
            return super(JSONInputXMLSerializableModel, cls).deserial(content, obj=obj, **kw)
        json_cls = kw.pop('json_cls', None) or cls.JSON
        if isinstance(content, dict):
            json_dict = content
        else:
            try:
                if isinstance(content, six.string_types):
                    json_dict = json.loads(content)
                else:
                    json_dict = json.loads(content.text)
            except ValueError:
                return super(JSONInputXMLSerializableModel, cls).deserial(content, obj=obj, **kw)

        json_obj = json_cls.deserial(json_dict, obj=obj, **kw)
        xml_cls = getattr(json_cls, '_xml_class', None) or cls
        obj = obj or xml_cls(**kw)
        obj._copy_from(json_obj)
        return obj

    def _copy_from(self, src):
        def _copy_model_field(v, model_cls):
            if v is None:
                return None
            override_cls = getattr(type(v), '_xml_class', None)
            new_obj = (override_cls or model_cls)(_parent=self)
            new_obj._copy_from(v)
            return new_obj

        for nm, f in six.iteritems(getattr(type(self), '__fields')):
            if isinstance(f, serializers.HasSubModelField):
                model_cls = f._model
                json_val = getattr(src, nm)
                if isinstance(f, serializers.XMLNodesReferencesField):
                    if json_val is not None:
                        ret_list = [_copy_model_field(v, model_cls) for v in json_val]
                    else:
                        ret_list = None
                    setattr(self, nm, ret_list)
                elif isinstance(f, serializers.XMLNodeReferenceField):
                    setattr(self, nm, _copy_model_field(json_val, model_cls))
            else:
                setattr(self, nm, getattr(src, nm))


class ModelAbTestItem(JSONInputXMLSerializableModel):
    _root = 'Item'

    class JSON(serializers.JSONSerializableModel):
        project = serializers.JSONNodeField('project')
        target_model = serializers.JSONNodeField('targetModel')
        pct = serializers.JSONNodeField('pct')

    project = serializers.XMLNodeField
    target_model = serializers.XMLNodeField
    pct = serializers.XMLNodeField


class ModelAbTest(JSONInputXMLSerializableModel):
    _root = 'ABTest'

    class JSON(serializers.JSONSerializableModel):
        _items = serializers.JSONNodesReferencesField(ModelAbTestItem.JSON, 'items')

    _items = serializers.XMLNodesReferencesField(ModelAbTestItem, 'Item')

    def __repr__(self):
        return repr(self._items)

    @classmethod
    def _wrap_list_methods(cls):
        def wrapper_gen(mt_name):
            def _wrapper(self, *args, **kwargs):
                if self._items is None:
                    self._items = []
                mt = getattr(self._items, mt_name)
                return mt(*args, **kwargs)
            _wrapper.__name__ = mt_name
            return _wrapper

        for mt_name in dir(list):
            attr = getattr(list, mt_name)
            if not hasattr(JSONInputXMLSerializableModel, mt_name) and callable(attr):
                setattr(cls, mt_name, wrapper_gen(mt_name))
ModelAbTest._wrap_list_methods()
delattr(ModelAbTest, '_wrap_list_methods')


class ModelAbTestRequest(serializers.XMLSerializableModel):
    _root = 'Onlinemodel'

    project = serializers.XMLNodeField
    name = serializers.XMLNodeField('Name')
    items = serializers.XMLNodesReferencesField(ModelAbTestItem, 'ABTest', 'Item')


class BaseProcessor(JSONInputXMLSerializableModel):
    _processors = []

    class JSON(serializers.JSONSerializableModel):
        @classmethod
        def deserial(cls, content, obj=None, **kw):
            if cls is not BaseProcessor.JSON:
                return super(BaseProcessor.JSON, cls).deserial(content, obj=obj, **kw)

            BaseProcessor._load_processors()
            if isinstance(content, six.string_types):
                content = json.loads(content)
            json_cls = None
            for processor in BaseProcessor._processors:
                if processor._accepts_json(content):
                    json_cls = processor
                    break
            json_cls = json_cls or CustomProcessor
            parsed_obj = json_cls.JSON.deserial(content, obj=obj, **kw)
            return parsed_obj

    @classmethod
    def deserial(cls, content, obj=None, **kw):
        cls._load_processors()
        json_cls = None
        for processor in BaseProcessor._processors:
            if processor._root == content.tag:
                json_cls = processor
                break
        json_cls = json_cls or CustomProcessor
        return super(BaseProcessor, json_cls).deserial(content, obj=obj, **kw)

    @classmethod
    def _load_processors(cls):
        if not cls._processors:
            cls._processors = [
                c for c in six.itervalues(globals())
                if isinstance(c, type) and issubclass(c, BaseProcessor) and c is not BaseProcessor
            ]

    @classmethod
    def _accepts_json(cls, json_obj):
        return False

    @staticmethod
    def _parse_resources(s):
        return [p.strip() for p in s.split(';')]

    @staticmethod
    def _serialize_resources(a):
        return ';'.join(a) if isinstance(a, list) else a


class BuiltinProcessor(BaseProcessor):
    _root = 'BuiltinProcessor'

    class JSON(BaseProcessor.JSON):
        offline_model_project = serializers.JSONNodeField('offlinemodelProject')
        offline_model_name = serializers.JSONNodeField('offlinemodelName')

    offline_model_project = serializers.XMLNodeField('OfflinemodelProject')
    offline_model_name = serializers.XMLNodeField('OfflinemodelName')

    @classmethod
    def _accepts_json(cls, json_obj):
        return bool(json_obj.get('offlinemodelName'))


class PmmlRunMode(compat.Enum):
    Evaluator = 'Evaluator'
    Converter = 'Converter'


class PmmlProcessor(BaseProcessor):
    _root = 'PmmlProcessor'

    class JSON(BaseProcessor.JSON):
        pmml = serializers.JSONNodeField('pmml')
        resources = serializers.JSONNodeField('refResource', parse_callback=BaseProcessor._parse_resources,
                                              serialize_callback=BaseProcessor._serialize_resources)
        run_mode = serializers.JSONNodeField('runMode', parse_callback=lambda s: PmmlRunMode(s) if s else None)

    pmml = serializers.XMLNodeField('Pmml')
    resources = serializers.XMLNodeField('RefResource', parse_callback=BaseProcessor._parse_resources,
                                         serialize_callback=BaseProcessor._serialize_resources)
    run_mode = serializers.XMLNodeField('RunMode', parse_callback=lambda s: PmmlRunMode(s.lower().capitalize()),
                                        serialize_callback=lambda v: v if isinstance(v, six.string_types) else v.value)

    @classmethod
    def _accepts_json(cls, json_obj):
        return bool(json_obj.get('pmml'))


class CustomProcessor(BaseProcessor):
    _root = 'Processor'

    class JSON(BaseProcessor.JSON):
        class_name = serializers.JSONNodeField('id')
        lib = serializers.JSONNodeField('libName')
        resources = serializers.JSONNodeField('refResource', parse_callback=BaseProcessor._parse_resources,
                                              serialize_callback=BaseProcessor._serialize_resources)
        config = serializers.JSONNodeField('configuration')

    class_name = serializers.XMLNodeField('Id')
    lib = serializers.XMLNodeField('LibName')
    resources = serializers.XMLNodeField('RefResource', parse_callback=BaseProcessor._parse_resources,
                                         serialize_callback=BaseProcessor._serialize_resources)
    config = serializers.XMLNodeField('Configuration')


class ModelPredictor(JSONInputXMLSerializableModel):
    __slots__ = 'runtime', 'instance_num'
    _root = 'PredictDesc'

    class JSON(serializers.JSONSerializableModel):
        pipeline = serializers.JSONNodesReferencesField(BaseProcessor.JSON, 'pipeline', 'processor')
        target_name = serializers.JSONNodeField('target', 'name')

    pipeline = serializers.XMLNodesReferencesField(BaseProcessor, 'Pipeline', '*')
    target_name = serializers.XMLNodeField('Target', 'Name')

    def __init__(self, **kw):
        if 'runtime' in kw:
            self.runtime = kw.pop('runtime')
        if 'instance_num' in kw:
            self.instance_num = kw.pop('instance_num')
        super(ModelPredictor, self).__init__(**kw)
        if not self.pipeline:
            self.pipeline = []


class ModelResource(serializers.XMLSerializableModel):
    cpu = serializers.XMLNodeField
    memory = serializers.XMLNodeField


class UsedResource(serializers.JSONSerializableModel):
    cpu = serializers.JSONNodeField('cpu')
    memory = serializers.JSONNodeField('mem')


class ModelPredictResult(serializers.JSONSerializableModel):
    label = serializers.JSONNodeField('outputLabel')
    scores = serializers.JSONNodeField('outputMulti')
    value = serializers.JSONNodeField('outputValue', 'dataValue')


class ModelPredictResults(serializers.JSONSerializableModel):
    outputs = serializers.JSONNodesReferencesField(ModelPredictResult, 'outputs')


class OnlineModel(LazyLoad):
    """
    Representing an ODPS online model.
    """
    __slots__ = '_predict_rest', '_endpoint'
    _root = 'Onlinemodel'

    class Status(compat.Enum):
        DEPLOYING = 'DEPLOYING'
        DEPLOY_FAILED = 'DEPLOYFAILED'
        SERVING = 'SERVING'
        UPDATING = 'UPDATING'
        DELETING = 'DELETING'
        DELETE_FAILED = 'DELETEFAILED'

    _project = serializers.XMLNodeField('Project')
    name = serializers.XMLNodeField
    version = serializers.XMLNodeField
    owner = serializers.XMLNodeField
    create_time = serializers.XMLNodeField(parse_callback=utils.parse_rfc822)
    offline_model_project = serializers.XMLNodeField('OfflinemodelProject')
    offline_model_name = serializers.XMLNodeField('OfflinemodelName')
    offline_model_id = serializers.XMLNodeField('OfflinemodelId')
    used_resource = serializers.XMLNodeReferenceField(UsedResource, 'UsedRes')
    _model_resource = serializers.XMLNodeReferenceField(ModelResource, 'Resource')
    qos = serializers.XMLNodeField('QOS')
    instance_num = serializers.XMLNodeField
    _status = serializers.XMLNodeField(parse_callback=lambda v: OnlineModel.Status(v.upper()),
                                       serialize_callback=lambda v: v.value)
    service_tag = serializers.XMLNodeField
    service_name = serializers.XMLNodeField
    last_fail_msg = serializers.XMLNodeField
    predictor = serializers.XMLNodeReferenceField(ModelPredictor, 'PredictDesc')
    runtime = serializers.XMLNodeField
    ab_test = serializers.XMLNodeReferenceField(ModelAbTest, 'ABTest')

    def __init__(self, **kwargs):
        super(OnlineModel, self).__init__(**kwargs)
        self._predict_rest = None
        self._endpoint = options.predict_endpoint

        predictor = kwargs.get('predictor')
        if hasattr(predictor, 'runtime') and 'runtime' not in kwargs:
            self.runtime = self.predictor.runtime
        if hasattr(predictor, 'instance_num') and 'instance_num' not in kwargs:
            self.instance_num = self.predictor.instance_num

        if kwargs.get('_model_resource') is None:
            self._model_resource = ModelResource()
        if kwargs.get('cpu') is not None:
            self._model_resource.cpu = kwargs.get('cpu')
        if kwargs.get('memory') is not None:
            self._model_resource.memory = kwargs.get('memory')
        if self._model_resource.cpu is None or self._model_resource.memory is None:
            self._model_resource = None

    @property
    def cpu(self):
        return self._model_resource.cpu

    @cpu.setter
    def cpu(self, value):
        self._model_resource.cpu = value

    @property
    def memory(self):
        return self._model_resource.memory

    @memory.setter
    def memory(self, value):
        self._model_resource.memory = value

    @property
    def project(self):
        return self.parent.parent

    @property
    def status(self):
        self.reload()
        return self._status

    def reload(self):
        resp = self._client.get(self.resource())
        self.parse(self._client, resp, obj=self)

    def serial(self):
        self._project = self.project.name
        return super(OnlineModel, self).serial()

    def update(self, async=False, **kw):
        """
        Update online model parameters to server.
        """
        headers = {'Content-Type': 'application/xml'}

        new_kw = dict()
        if self.offline_model_name:
            upload_keys = ('_parent', 'name', 'offline_model_name', 'offline_model_project', 'qos', 'instance_num')
        else:
            upload_keys = ('_parent', 'name', 'qos', '_model_resource', 'instance_num', 'predictor', 'runtime')

        for k in upload_keys:
            new_kw[k] = getattr(self, k)
        new_kw.update(kw)

        obj = type(self)(version='0', **new_kw)

        data = obj.serialize()
        self._client.put(self.resource(), data, headers=headers)
        self.reload()

        if not async:
            self.wait_for_service()

    def drop(self, async=False):
        self.parent.delete(self, async=async)

    def wait_for_service(self, interval=1):
        """
        Wait for the online model to be ready for service.

        :param interval: check interval
        """
        while self.status in (OnlineModel.Status.DEPLOYING, OnlineModel.Status.UPDATING):
            time.sleep(interval)
        if self.status == OnlineModel.Status.DEPLOY_FAILED:
            raise OnlineModelError(self.last_fail_msg, self)
        elif self.status != OnlineModel.Status.SERVING:
            raise OnlineModelError('Unexpected status occurs: %s' % self.status.value, self)

    def wait_for_deletion(self, interval=1):
        """
        Wait for the online model to be deleted.

        :param interval: check interval
        """
        deleted = False
        while True:
            try:
                if self.status != OnlineModel.Status.DELETING:
                    break
            except errors.NoSuchObject:
                deleted = True
                break
            time.sleep(interval)

        if not deleted:
            if self.status == OnlineModel.Status.DELETE_FAILED:
                raise OnlineModelError(self.last_fail_msg, self)
            else:
                raise OnlineModelError('Unexpected status occurs: %s' % self.status.value, self)

    def config_ab_test(self, *args):
        """
        Config AB-Test percentages of the online model.

        This method should be called like

        >>> result = model.config_ab_test(model1, percentage1, model2, percentage2, ...)

        where `modelx` can be model names or :class:`odps.models.ml.OnlineModel` instances, while `percentagex` should
        be percentage of `modelx` in AB-Test, ranging from 0 to 100.
        """
        obj = ModelAbTestRequest(project=self.project.name, name=self.name, items=[])
        for model, pct in zip(args[::2], args[1::2]):
            if isinstance(model, six.string_types):
                if '.' in model:
                    model_proj, model_name = model.split('.', 1)
                else:
                    model_proj, model_name = self.project.name, model
            elif isinstance(model, OnlineModel):
                model_proj, model_name = model.project.name, model.name
            else:
                raise ValueError('Unsupported model specification.')
            obj.items.append(ModelAbTestItem(project=model_proj, target_model=model_name, pct=pct))

        data = obj.serialize()

        headers = {'Content-Type': 'application/xml'}
        self._client.put(self.resource(), data, headers=headers)
        self.reload()

    @staticmethod
    def _build_predict_request(data, schema=None):
        from .. import Schema

        def _conv_row(row):
            new_row = OrderedDict()
            if not row:
                raise ValueError('Row should not be empty.')
            if isinstance(row, list):
                if isinstance(row[0], tuple):
                    row = OrderedDict(row)
                elif isinstance(schema, Schema):
                    row = odps_types.Record(schema=schema, values=row)
                elif isinstance(schema, list) and schema:
                    if isinstance(schema[0], odps_types.Column):
                        row = odps_types.Record(columns=schema, values=row)
                    elif isinstance(schema[0], six.string_types):
                        row = OrderedDict(zip(schema, row))
                    else:
                        raise ValueError('Unsupported row type.')
                else:
                    raise ValueError('Unsupported schema type.')

            if isinstance(row, odps_types.Record):
                for col, val in zip(row._columns, row._values):
                    if col.type == odps_types.decimal:
                        type_code = PREDICT_TYPE_CODES['float']
                        val = float(val)
                    elif type(col.type).__name__ in PREDICT_ODPS_TYPE_CODES:
                        type_code = PREDICT_ODPS_TYPE_CODES[type(col.type).__name__]
                    else:
                        raise ValueError('Unsupported prediction value type.')
                    new_row[col.name] = dict(dataType=type_code, dataValue=val)
            elif isinstance(row, dict):
                for col, val in six.iteritems(row):
                    if isinstance(val, Decimal):
                        type_code = PREDICT_TYPE_CODES['float']
                        val = float(val)
                    elif type(val).__name__ in PREDICT_TYPE_CODES:
                        type_code = PREDICT_TYPE_CODES[type(val).__name__]
                    else:
                        raise ValueError('Unsupported prediction value type.')
                    new_row[col] = dict(dataType=type_code, dataValue=val)
            else:
                raise ValueError('Unsupported row type.')
            return new_row

        if not data:
            raise ValueError('Data should not be empty.')

        if isinstance(data, odps_types.Record):
            data = [data, ]
        elif isinstance(data, dict):
            data = [data, ]
        elif isinstance(data, (list, tuple)) and not isinstance(data[0], list):
            data = [data, ]

        return dict(inputs=[_conv_row(d) for d in data])

    def predict(self, data, schema=None, endpoint=None):
        """
        Predict data labels with current online model.

        :param data: data to be predicted
        :param schema: schema of input data
        :param endpoint: endpoint of predict service
        :return: prediction result
        """
        from .. import Projects

        if endpoint is not None:
            self._endpoint = endpoint
        if self._predict_rest is None:
            # do not add project option
            self._predict_rest = RestClient(self._client.account, self._endpoint, proxy=options.data_proxy)

        json_data = json.dumps(self._build_predict_request(data, schema))
        headers = {'Content-Type': 'application/json'}

        predict_model = Projects(client=self._predict_rest)[self.project.name].online_models[self.name]
        resp = self._predict_rest.post(predict_model.resource(), json_data, headers=headers)

        if not self._client.is_ok(resp):
            e = errors.ODPSError.parse(resp)
            raise e
        return ModelPredictResults.parse(resp).outputs


class OnlineModelError(errors.ODPSError):
    def __init__(self, msg, model=None, request_id=None, code=None, host_id=None):
        self.model = model
        super(OnlineModelError, self).__init__(msg, request_id=request_id, code=code, host_id=host_id)
