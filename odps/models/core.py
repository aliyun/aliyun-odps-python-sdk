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

try:
    import xml.etree.cElementTree as ElementTree
except ImportError:
    import xml.etree.ElementTree as ElementTree

from .. import serializers
from .cache import cache, del_cache
from ..config import options
from ..compat import six, quote_plus


class XMLRemoteModel(serializers.XMLSerializableModel):
    __slots__ = '_parent', '_client'

    def __init__(self, **kwargs):
        if 'parent' in kwargs:
            kwargs['_parent'] = kwargs.pop('parent')
        if 'client' in kwargs:
            kwargs['_client'] = kwargs.pop('client')

        assert frozenset(kwargs).issubset(self.__slots__)
        super(XMLRemoteModel, self).__init__(**kwargs)

    @classmethod
    def parse(cls, client, response, obj=None, **kw):
        kw['_client'] = client
        return super(XMLRemoteModel, cls).parse(response, obj=obj, **kw)


class AbstractXMLRemoteModel(XMLRemoteModel):
    __slots__ = '_type_indicator',


class JSONRemoteModel(serializers.JSONSerializableModel):
    __slots__ = '_parent', '_client'

    def __init__(self, **kwargs):
        if 'parent' in kwargs:
            kwargs['_parent'] = kwargs.pop('parent')
        if 'client' in kwargs:
            kwargs['_client'] = kwargs.pop('client')
        assert frozenset(kwargs).issubset(self.__slots__)
        super(JSONRemoteModel, self).__init__(**kwargs)

    @classmethod
    def parse(cls, client, response, obj=None, **kw):
        kw['_client'] = client
        return super(JSONRemoteModel, cls).parse(response, obj=obj, **kw)


class RestModel(XMLRemoteModel):
    def _name(self):
        return type(self).__name__.lower()

    @classmethod
    def _encode(cls, name):
        name = quote_plus(name).replace('+', '%20')
        return name

    def resource(self):
        parent = self._parent
        if parent is None:
            parent_res = self._client.endpoint
        else:
            parent_res = parent.resource()
        name = self._name()
        if name is None:
            return parent_res
        return '/'.join([parent_res, self._encode(name)])

    @classmethod
    def _to_stdout(cls, msg):
        print(msg)

    def log(self, msg):
        if options.verbose:
            (options.verbose_log or self._to_stdout)(msg)

    def __eq__(self, other):
        if other is None:
            return False

        if not isinstance(other, type(self)):
            return False

        return self._name() == other._name() and \
            self.parent == other.parent

    def __hash__(self):
        return hash(type(self)) * hash(self._parent) * hash(self._name())


class LazyLoad(RestModel):
    __slots__ = '_loaded',

    @cache
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, **kwargs):
        self._loaded = False
        kwargs.pop('no_cache', None)
        super(LazyLoad, self).__init__(**kwargs)

    def _name(self):
        return self._getattr('name')

    def _getattr(self, attr):
        return object.__getattribute__(self, attr)

    def __getattribute__(self, attr):
        val = object.__getattribute__(self, attr)
        if val is None and not self._loaded:
            fields = getattr(type(self), '__fields')
            if attr in fields:
                self.reload()
                val = self._getattr(attr)
        return val

    def reload(self):
        raise NotImplementedError

    def reset(self):
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def __repr__(self):
        if hasattr(self, '_repr'):
            return self._repr()
        return super(LazyLoad, self).__repr__()

    def __hash__(self):
        return hash((self.name, self.parent))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return self.name == other.name and self.parent == other.parent

    def __getstate__(self):
        return self.name, self._parent, self._client

    def __setstate__(self, state):
        name, parent, client = state
        self.__init__(name=name, _parent=parent, _client=client)


class Container(RestModel):
    skip_null = False

    @cache
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def _get(self, item):
        raise NotImplementedError

    def __getitem__(self, item):
        if isinstance(item, six.string_types):
            return self._get(item)
        raise ValueError('Unsupported getitem value: %s' % item)

    @del_cache
    def __delitem__(self, key):
        pass

    def __contains__(self, item):
        raise NotImplementedError

    def __getstate__(self):
        return self._parent, self._client

    def __setstate__(self, state):
        parent, client = state
        self.__init__(_parent=parent, _client=client)


class Iterable(Container):
    __slots__ = '_iter',

    def __init__(self, **kwargs):
        super(Iterable, self).__init__(**kwargs)
        self._iter = iter(self)

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        return next(self._iter)

    next = __next__
