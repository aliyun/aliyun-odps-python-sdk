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

import re
import json
from xml.dom import minidom
import inspect

import requests

from . import compat, utils
from .compat import BytesIO, ElementTree, PY26, six
from .utils import to_text


def _route_xml_path(root, *keys, **kw):
    create_if_not_exists = kw.get('create_if_not_exists', False)

    if isinstance(root, six.string_types):
        root = ElementTree.fromstring(root)

    for key in keys:
        if key == '.':
            return root
        prev = root
        root = root.find(key)
        if root is None:
            if not create_if_not_exists:
                return
            root = ElementTree.Element(key)
            prev.append(root)

    return root


def _route_json_path(root, *keys, **kw):
    create_if_not_exists = kw.get('create_if_not_exists', False)

    if isinstance(root, six.string_types):
        root = json.loads(root)

    for key in keys:
        prev = root
        root = root.get(key)
        if root is None:
            if not create_if_not_exists:
                return
            root = prev[key] = compat.OrderedDict()

    return root


def parse_ndarray(array):
    try:
        import numpy as np
        return np.asarray(array)
    except ImportError:
        return array


def serialize_ndarray(array):
    try:
        return array.tolist()
    except AttributeError:
        return array


_serialize_types = dict()
_serialize_types['bool'] = (utils.str_to_bool, utils.bool_to_str)
_serialize_types['json'] = (
    lambda s: json.loads(s) if s is not None else None,
    lambda s: json.dumps(s) if s is not None else None,
)
_serialize_types['rfc822'] = (utils.parse_rfc822, utils.gen_rfc822)
_serialize_types['rfc822l'] = (utils.parse_rfc822, lambda s: utils.gen_rfc822(s, localtime=True))
_serialize_types['ndarray'] = (parse_ndarray, serialize_ndarray)


class SerializeField(object):
    def __init__(self, *keys, **kwargs):
        self._path_keys = keys

        self._required = kwargs.get('required', False)  # used when serialized
        self._blank_if_null = kwargs.get('blank_if_null',
                                         True if self._required else False)
        self._default = kwargs.get('default')
        if 'type' in kwargs:
            self._parse_callback, self._serialize_callback = _serialize_types[kwargs.pop('type')]
        else:
            self._parse_callback = kwargs.get('parse_callback')
            self._serialize_callback = kwargs.get('serialize_callback')

        self.set_to_parent = kwargs.get('set_to_parent', False)

    def _to_str(self, val):
        if isinstance(val, six.string_types):
            return utils.to_str(val)
        return val

    def _set_default_keys(self, *keys):
        if not self._path_keys:
            self._path_keys = keys

    def parse(self, root, **kwargs):
        raise NotImplementedError

    def serialize(self, root, value):
        raise NotImplementedError


class HasSubModelField(SerializeField):
    def __init__(self, model, *args, **kwargs):
        if isinstance(model, six.string_types):
            self._model_cls = None
            self._model_str = model
        else:
            self._model_cls = model
            self._model_str = None

        super(HasSubModelField, self).__init__(*args, **kwargs)

    @property
    def _model(self):
        if self._model_cls is not None:
            return self._model_cls

        models = self._model_str.split('.')
        model_name = models[0]

        module = None
        for stack in inspect.stack():
            globs = stack[0].f_globals
            if model_name in globs:
                possible_module = globs[model_name]
                if inspect.isclass(possible_module) and \
                        issubclass(possible_module, SerializableModel):
                    module = possible_module
                    break

        if module is None:
            raise ValueError('Unknown model name: %s' % self._model_str)

        res = None
        for model in models[1:]:
            if res is None:
                res = getattr(module, model)
            else:
                res = getattr(res, model)

        self._model_cls = res
        return res


_default_name_maker = dict(capitalized=utils.underline_to_capitalized, raw=lambda v: v, camel=utils.underline_to_camel)


class SerializableModelMetaClass(type):
    def __new__(mcs, name, bases, kv):
        slots = []
        fields = dict()
        for base in bases:
            base_slots = list(getattr(base, '__slots__', []))
            if '__weakref__' in base_slots:
                base_slots.remove('__weakref__')
            slots.extend(base_slots)
            fields.update(getattr(base, '__fields', dict()))
        slots.extend(kv.get('__slots__', []))
        fields.update(kv.get('__fields', dict()))

        attrs = []
        parent_attrs = []
        def_name = kv.pop('_' + name + '__default_name', 'capitalized')
        for attr, field in (pair for pair in six.iteritems(kv) if not pair[0].startswith('__')):
            if inspect.isclass(field) and issubclass(field, SerializeField):
                field = field()
            if isinstance(field, SerializeField):
                field._set_default_keys(_default_name_maker[def_name](attr))
                if not field.set_to_parent:
                    slots.append(attr)
                    attrs.append(attr)
                if field.set_to_parent:
                    parent_attrs.append(attr)
                fields[attr] = field
        kv['_parent_attrs'] = set(parent_attrs)

        slots = tuple(compat.OrderedDict.fromkeys(slots))

        slots_pos = dict([(v, k) for k, v in enumerate(slots)])
        fields = compat.OrderedDict(
            sorted(six.iteritems(fields), key=lambda s: slots_pos.get(s[0], float('inf'))))

        for attr in attrs:
            if attr in kv:
                del kv[attr]

        slots = tuple(slot for slot in slots if slot not in kv)
        if len(slots) > 0:
            kv['__slots__'] = slots
        if len(fields) > 0:
            kv['__fields'] = fields

        return type.__new__(mcs, name, bases, kv)


class SerializableModel(six.with_metaclass(SerializableModelMetaClass)):
    __slots__ = '_parent', '__weakref__'

    def __init__(self, **kwargs):
        slots = getattr(self, '__slots__', [])

        for k, v in six.iteritems(kwargs):
            if k in slots:
                setattr(self, k, v)
        for attr in slots:
            try:
                super(SerializableModel, self).__getattribute__(attr)
            except AttributeError:
                setattr(self, attr, None)

    @property
    def parent(self):
        return self._parent

    @classmethod
    def _is_null(cls, v):
        if v is None:
            return True
        if isinstance(v, (list, dict)) and len(v) == 0:
            return True
        return False

    @classmethod
    def _setattr(cls, obj, k, v, skip_null=True):
        if cls._is_null(v) and object.__getattribute__(obj, k) is not None:
            if not skip_null:
                setattr(obj, k, v)
            return

        fields = getattr(type(obj), '__fields')
        if not isinstance(fields[k], HasSubModelField):
            setattr(obj, k, v)
        elif isinstance(v, list):
            setattr(obj, k, v)
        else:
            sub_obj = object.__getattribute__(obj, k)
            new_obj = v
            if sub_obj is None:
                setattr(obj, k, v)
                return
            sub_fields = getattr(new_obj, '__fields', {})
            for k in six.iterkeys(sub_fields):
                if sub_fields[k].set_to_parent is True:
                    continue
                cls._setattr(sub_obj, k, object.__getattribute__(new_obj, k),
                             skip_null=skip_null)

    @classmethod
    def _init_obj(cls, content, obj=None, **kw):
        fields = dict(getattr(cls, '__fields'))

        _type = getattr(cls, '_type_indicator', None)
        _name = 'name' if 'name' in fields else None
        if obj is None and (_type is not None or 'name' in fields):
            kwargs = dict(kw)

            for field in (_name, _type):
                if field is None:
                    continue
                typo = fields[field].parse(content, **kw)
                kwargs[field] = typo

            obj = cls(**kwargs)

        return obj or cls(**kw)

    @classmethod
    def deserial(cls, content, obj=None, **kw):
        obj = cls._init_obj(content, obj=obj, **kw)
        obj_type = type(obj)

        fields = dict(getattr(obj_type, '__fields'))

        if isinstance(content, six.string_types):
            if issubclass(obj_type, XMLSerializableModel):
                content = ElementTree.fromstring(content)
            else:
                content = json.loads(content)

        parent_kw = dict()
        self_kw = dict()

        for attr, prop in six.iteritems(fields):
            if isinstance(prop, SerializeField):
                kwargs = dict(kw)
                if isinstance(prop, HasSubModelField):
                    kwargs['_parent'] = obj
                if not prop.set_to_parent:
                    self_kw[attr] = prop.parse(content, **kwargs)
                else:
                    parent_kw[attr] = prop.parse(content, **kwargs)

        for k, v in six.iteritems(self_kw):
            obj_type._setattr(obj, k, v, skip_null=getattr(obj_type, 'skip_null', True))

        if obj.parent is not None:
            for k, v in six.iteritems(parent_kw):
                # remember that do not use `hasattr` here
                try:
                    old_v = object.__getattribute__(obj.parent, k)
                except AttributeError:
                    continue
                if v is not None and old_v != v:
                    setattr(obj.parent, k, v)

        return obj

    def serial(self):
        if isinstance(self, XMLSerializableModel):
            assert self._root is not None
            root = ElementTree.Element(self._root)
        else:
            root = compat.OrderedDict()

        for attr, prop in six.iteritems(getattr(self, '__fields')):
            if isinstance(prop, SerializeField):
                try:
                    prop.serialize(root, object.__getattribute__(self, attr))
                except NotImplementedError:
                    continue

        return root

    def extract(self, **base_kw):
        kwargs = base_kw.copy()
        for attr in self.__slots__:
            try:
                kwargs[attr] = object.__getattribute__(self, attr)
            except AttributeError:
                pass
        return kwargs


class XMLSerializableModel(SerializableModel):
    __slots__ = '_root',

    @classmethod
    def parse(cls, response, obj=None, **kw):
        if 'parent' in kw:
            kw['_parent'] = kw.pop('parent')
        if isinstance(response, requests.Response):
            # PY2 prefer bytes, while PY3 prefer str
            response = response.content.decode() if six.PY3 else response.content
        return cls.deserial(response, obj=obj, **kw)

    def serialize(self):
        root = self.serial()

        sio = BytesIO()
        if PY26:
            ElementTree.ElementTree(root).write(sio, encoding="utf-8")
        else:
            ElementTree.ElementTree(root).write(sio, encoding="utf-8", xml_declaration=True)
        xml_content = sio.getvalue()

        prettified_xml = minidom.parseString(xml_content).toprettyxml(indent=' '*2, encoding='utf-8')
        prettified_xml = to_text(prettified_xml, encoding='utf-8')

        cdata_re = re.compile(r'&lt;!\[CDATA\[.*\]\]&gt;', (re.M | re.S))
        for src_cdata in cdata_re.finditer(prettified_xml):
            src_cdata = src_cdata.group(0)
            dest_cdata = src_cdata.replace('&amp;', '&').replace('&lt;', '<'). \
                replace('&quot;', '"').replace('&gt;', '>')
            prettified_xml = prettified_xml.replace(src_cdata, dest_cdata)

        return prettified_xml.replace('&quot;', '"')


class JSONSerializableModel(SerializableModel):
    @classmethod
    def parse(cls, response, obj=None, **kw):
        if 'parent' in kw:
            kw['_parent'] = kw.pop('parent')
        if isinstance(response, requests.Response):
            # PY2 prefer bytes, while PY3 prefer str
            response = response.content.decode() if six.PY3 else response.content
        return cls.deserial(response, obj=obj, **kw)

    def serialize(self, **kwargs):
        root = self.serial()
        return json.dumps(root, **kwargs)


class XMLTagField(SerializeField):
    def parse(self, root, **kwargs):
        node = _route_xml_path(root, *self._path_keys)

        val = self._default
        if node is not None:
            val = node.tag

        if val is None:
            return

        if self._parse_callback:
            return self._parse_callback(val)
        return val

    def _set_default_keys(self, *keys):
        super(XMLTagField, self)._set_default_keys('.')


class XMLNodeField(SerializeField):
    def parse(self, root, **kwargs):
        node = _route_xml_path(root, *self._path_keys)

        val = self._default
        if node is not None:
            val = node.text or self._default

        if val is None:
            return

        if self._parse_callback:
            return self._parse_callback(val)
        return val

    def serialize(self, root, value):
        value = value if value is not None else self._default
        if value is None and self._blank_if_null:
            value = ''

        if not self._required and value is None:
            return

        node = _route_xml_path(root, create_if_not_exists=True, *self._path_keys)
        if self._serialize_callback:
            node.text = utils.to_text(self._serialize_callback(value))
        else:
            node.text = utils.to_text(value)


class XMLNodeAttributeField(SerializeField):
    def __init__(self, *keys, **kwargs):
        self._attr = kwargs.pop('attr', None)

        super(XMLNodeAttributeField, self).__init__(*keys, **kwargs)

    def parse(self, root, **kwargs):
        assert self._attr is not None

        node = _route_xml_path(root, *self._path_keys)

        val = self._default
        if node is not None:
            val = node.get(self._attr)

        if val is None:
            return

        if self._parse_callback:
            return self._parse_callback(val)
        return node.get(self._attr)

    def serialize(self, root, value):
        assert self._attr is not None

        value = value if value is not None else self._default
        if value is None:
            if self._default is not None:
                value = self._default
            elif self._blank_if_null:
                value = ''

        if not self._required and value is None:
            return

        node = _route_xml_path(root, create_if_not_exists=True, *self._path_keys)
        if self._serialize_callback:
            node.set(self._attr, self._serialize_callback(value))
        else:
            node.set(self._attr, value)

    def _set_default_keys(self, *keys):
        if self._attr is None:
            self._attr = keys[0]


class XMLNodesField(SerializeField):
    def parse(self, root, **kwargs):
        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            root = _route_xml_path(root, *prev_path_keys)

        values = self._default
        if root is not None:
            values = [node.text for node in root.findall(self._path_keys[-1])]

        if values is None:
            return

        if self._parse_callback:
            return self._parse_callback(values)
        return values

    def serialize(self, root, value):
        value = value if value is not None else self._default
        if value is None and self._blank_if_null:
            value = []

        if not self._required and value is None:
            return

        if self._serialize_callback:
            value = self._serialize_callback(value)

        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            root = _route_xml_path(root, create_if_not_exists=True, *prev_path_keys)

        for val in value:
            element = ElementTree.Element(self._path_keys[-1])
            element.text = utils.to_text(val)
            root.append(element)


class XMLNodeReferenceField(HasSubModelField):
    def parse(self, root, **kwargs):
        node = _route_xml_path(root, *self._path_keys)

        instance = self._default
        if node is not None:
            if issubclass(self._model, JSONSerializableModel):
                node = node.text

            instance = self._model.deserial(node, **kwargs)
            if isinstance(instance, XMLSerializableModel) and \
                            instance._root is None:
                instance._root = node.tag

        if instance is None:
            return

        if self._parse_callback:
            return self._parse_callback(instance)
        return instance

    def serialize(self, root, value):
        value = value if value is not None else self._default

        if not self._required and value is None:
            return

        if self._serialize_callback:
            value = self._serialize_callback(value)

        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            root = _route_xml_path(root, create_if_not_exists=True, *prev_path_keys)

        if isinstance(value, XMLSerializableModel) and getattr(value, '_root') is None:
            setattr(value, '_root', self._path_keys[-1])

        val = value.serial()
        if isinstance(value, JSONSerializableModel):  # JSON mixed in XML
            element = ElementTree.Element(self._path_keys[-1])
            val = json.dumps(val)
            element.text = val
            root.append(element)
        else:
            root.append(val)


class XMLNodesReferencesField(HasSubModelField):
    def parse(self, root, **kwargs):
        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            root = _route_xml_path(root, *prev_path_keys)

        instances = self._default

        if root is not None:
            instances = []

            tag = self._path_keys[-1]
            if tag == '*':
                nodes = list(root)
            else:
                nodes = root.findall(self._path_keys[-1])

            for node in nodes:
                instance = self._model.deserial(node, **kwargs)
                if isinstance(instance, XMLSerializableModel) and instance._root is None:
                    instance._root = node.tag
                instances.append(instance)

        if instances is None:
            return

        if self._parse_callback:
            return self._parse_callback(instances)
        return instances

    def serialize(self, root, value):
        value = value if value is not None else self._default
        if value is None and self._blank_if_null:
            value = []

        if not self._required and value is None:
            return

        if self._serialize_callback:
            value = self._serialize_callback(value)

        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            root = _route_xml_path(root, create_if_not_exists=True, *prev_path_keys)

        for it in value:
            if isinstance(it, XMLSerializableModel) and \
                            getattr(it, '_root') is None:
                setattr(it, '_root', self._path_keys[-1])

            val = it.serial()
            if isinstance(it, JSONSerializableModel):
                element = ElementTree.Element(self._path_keys[-1])
                val = json.dumps(val)
                element.text = val
                root.append(element)
            else:
                root.append(val)


class XMLNodePropertiesField(SerializeField):
    def __init__(self, *keys, **kwargs):
        super(XMLNodePropertiesField, self).__init__(*keys, **kwargs)
        self._key_tag = kwargs['key_tag']
        self._value_tag = kwargs['value_tag']

    def parse(self, root, **kwargs):
        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            root = _route_xml_path(root, *prev_path_keys)

        results = self._default

        if root is not None:
            results = compat.OrderedDict()

            for node in root.findall(self._path_keys[-1]):
                key_node = node.find(self._key_tag)
                value_node = node.find(self._value_tag)
                if key_node is not None and value_node is not None:
                    results[key_node.text] = value_node.text

        if results is None:
            return

        if self._parse_callback:
            return self._parse_callback(results)
        return results

    def serialize(self, root, value):
        value = value if value is not None else self._default
        if value is None and self._blank_if_null:
            value = compat.OrderedDict()

        if not self._required and value is None:
            return

        if self._serialize_callback:
            value = self._serialize_callback(value)

        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            root = _route_xml_path(root, create_if_not_exists=True, *prev_path_keys)

        for k, v in six.iteritems(value):
            element = ElementTree.Element(self._path_keys[-1])
            key_node = ElementTree.Element(self._key_tag)
            key_node.text = utils.to_text(k)
            element.append(key_node)
            value_node = ElementTree.Element(self._value_tag)
            value_node.text = utils.to_text(v)
            element.append(value_node)

            root.append(element)


class JSONNodeField(SerializeField):
    def parse(self, root, **kwargs):
        val = self._default
        if root is not None:
            val = _route_json_path(root, *self._path_keys)

        if val is None:
            return

        val = self._to_str(val)
        if self._parse_callback:
            return self._parse_callback(val)
        return val

    def serialize(self, root, value):
        value = value if value is not None else self._default
        if value is None and self._blank_if_null:
            value = ''

        if not self._required and value is None:
            return

        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            node = _route_json_path(root, create_if_not_exists=True, *prev_path_keys)
        else:
            node = root

        if self._serialize_callback:
            node[self._path_keys[-1]] = self._serialize_callback(value)
        else:
            node[self._path_keys[-1]] = value


class JSONNodesField(SerializeField):
    def parse(self, root, **kwargs):
        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            root = _route_json_path(root, *prev_path_keys)

        values = self._default
        if root is not None:
            values = [self._to_str(node[self._path_keys[-1]])
                      for node in root]

        if values is None:
            return

        if self._parse_callback:
            return self._parse_callback(values)
        return values

    def serialize(self, root, value):
        value = value if value is not None else self._default
        if value is None and self._blank_if_null:
            value = []

        if not self._required and value is None:
            return

        if self._serialize_callback:
            value = self._serialize_callback(value)

        assert len(self._path_keys) >= 2

        prev_path_keys = self._path_keys[:-2]
        if prev_path_keys:
            root = _route_json_path(root, create_if_not_exists=True, *prev_path_keys)

        node = self._path_keys[-2]
        if node not in root:
            root[node] = []
        root = root[node]

        for val in value:
            root.append({self._path_keys[-1]: val})


class JSONNodeReferenceField(HasSubModelField):
    def __init__(self, model, *keys, **kwargs):
        super(JSONNodeReferenceField, self).__init__(model, *keys, **kwargs)

        self._check_before = kwargs.get('check_before')

    def parse(self, root, **kwargs):
        instance = self._default

        if root is not None:
            node = _route_json_path(root, *self._path_keys)

            if self._check_before is not None:
                _check = _route_json_path(root, *self._check_before)
                if not _check:
                    return

            instance = self._model.deserial(node, **kwargs)

        if instance is None:
            return

        if self._parse_callback:
            return self._parse_callback(instance)
        return instance

    def serialize(self, root, value):
        value = value if value is not None else self._default

        if not self._required and value is None:
            return

        if self._serialize_callback:
            value = self._serialize_callback(value)

        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            root = _route_json_path(root, create_if_not_exists=True, *prev_path_keys)

        root[self._path_keys[-1]] = value.serial()


class JSONNodesReferencesField(HasSubModelField):
    def parse(self, root, **kwargs):
        instances = self._default

        if isinstance(root, list):
            instances = [self._model.deserial(node, **kwargs)
                         for node in root]
        elif root is not None:
            prev_path_keys = self._path_keys[:-1]
            if prev_path_keys:
                root = _route_json_path(root, *prev_path_keys)

            if root is not None:
                root = root.get(self._path_keys[-1])
                if root is not None:
                    instances = [self._model.deserial(node, **kwargs)
                                 for node in root]

        if instances is None:
            return

        if self._parse_callback:
            return self._parse_callback(instances)
        return instances

    def serialize(self, root, value):
        value = value if value is not None else self._default
        if value is None and self._blank_if_null:
            value = []

        if not self._required and value is None:
            return

        if self._serialize_callback:
            value = self._serialize_callback(value)

        prev_path_keys = self._path_keys[:-1]
        if prev_path_keys:
            root = _route_json_path(root, create_if_not_exists=True, *prev_path_keys)

        key = self._path_keys[-1]
        if key not in root:
            root[key] = []
        [root[key].append(it.serial()) for it in value]


class JSONRawField(JSONNodeField):
    def _set_default_keys(self, *keys):
        return
