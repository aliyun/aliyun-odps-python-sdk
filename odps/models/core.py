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

from .. import serializers
from .cache import cache, del_cache
from ..compat import six, quote_plus


class XMLRemoteModel(serializers.XMLSerializableModel):
    __slots__ = '_parent', '_client'

    def __init__(self, **kwargs):
        if 'parent' in kwargs:
            kwargs['_parent'] = kwargs.pop('parent')
        if 'client' in kwargs:
            kwargs['_client'] = kwargs.pop('client')

        if not frozenset(kwargs).issubset(self.__slots__):
            unexpected = sorted(set(kwargs) - set(self.__slots__))
            raise TypeError("%s() meet illegal arguments (%s)" % (
                type(self).__name__, ', '.join(unexpected)))
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
        if not frozenset(kwargs).issubset(self.__slots__):
            unexpected = sorted(set(kwargs) - set(self.__slots__))
            raise TypeError("%s() meet illegal arguments (%s)" % (
                type(self).__name__, ', '.join(unexpected)))
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

    def resource(self, client=None):
        parent = self._parent
        if parent is None:
            parent_res = (client or self._client).endpoint
        else:
            parent_res = parent.resource(client=client)
        name = self._name()
        if name is None:
            return parent_res
        return '/'.join([parent_res, self._encode(name)])

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
        try:
            r = self._repr()
        except:
            r = None
        if r:
            return r
        else:
            return super(LazyLoad, self).__repr__()

    def _repr(self):
        name = self._name()
        if name:
            return '<%s %s>' % (type(self).__name__, name)
        else:
            raise ValueError

    def __hash__(self):
        return hash((self._name(), self.parent))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return self._name() == other._name() and self.parent == other.parent

    def __getstate__(self):
        return self._name(), self._parent, self._client

    def __setstate__(self, state):
        name, parent, client = state
        self._set_state(name, parent, client)

    def _set_state(self, name, parent, client):
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
