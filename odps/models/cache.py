#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import weakref
import inspect

from ..compat import six


class ObjectCache(object):
    def __init__(self):
        self._caches = weakref.WeakValueDictionary()

    @staticmethod
    def _get_cache_class(cls):
        if hasattr(cls, '_cache_class'):
            cls._cache_class = ObjectCache._get_cache_class(cls._cache_class)
            return cls._cache_class
        return cls

    def _fetch(self, cache_key):
        client, parent, _, _ = cache_key
        if parent is None:
            return self._caches.get(cache_key)

        ancestor = getattr(parent, '_parent')
        parent_cls = self._get_cache_class(type(parent))
        if parent_cls is None:
            return None
        name = getattr(parent, 'name', parent_cls.__name__.lower())

        parent_cache_key = client, ancestor, parent_cls, name

        if self._fetch(parent_cache_key):
            return self._caches.get(cache_key)

    def _get_cache(self, cls, **kw):
        kwargs = dict(kw)
        parent = kwargs.pop('parent', None) or kwargs.pop('_parent', None)
        name = kwargs.pop(getattr(cls, '_cache_name_arg', 'name'), None)
        client = kwargs.pop('client', None) or kwargs.pop('_client', None)
        cache_cls = self._get_cache_class(cls)

        cache_key = client, parent, cache_cls, name
        obj = None
        if name is not None:
            obj = self._fetch(cache_key)
            if obj is not None:
                if not frozenset(kwargs).issubset(obj.__slots__):
                    obj = None
                else:
                    for k, v in six.iteritems(kwargs):
                        setattr(obj, k, v)
            if obj is not None:
                return cache_key, obj

        return cache_key, obj

    def cache_lazyload(self, func, cls, **kwargs):
        cache_key, obj = self._get_cache(cls, **kwargs)

        if obj is not None:
            return obj

        obj = func(cls, **kwargs)

        if not hasattr(cls, '_filter_cache'):
            self._caches[cache_key] = obj
        elif cls._filter_cache(func, **obj.extract(**kwargs)):
            self._caches[cache_key] = obj
        return obj

    def cache_container(self, func, cls, **kwargs):
        parent = kwargs.get('parent') or kwargs.get('_parent')
        client = kwargs.get('client') or kwargs.get('_client')
        name = cls.__name__.lower()

        cache_key = client, parent, cls, name
        if name is not None:
            obj = self._fetch(cache_key)
            if obj is not None:
                return obj

        obj = func(cls, **kwargs)
        self._caches[cache_key] = obj
        return obj

    def del_item_cache(self, obj, item):
        item = obj[item]

        client = getattr(item, '_client')
        parent = getattr(item, '_parent')
        name = item._name()

        if name is not None:
            clz = self._get_cache_class(type(item))
            cache_key = client, parent, clz, name
            if cache_key in self._caches:
                del self._caches[cache_key]


_object_cache = ObjectCache()


def cache(func):
    def inner(cls, **kwargs):
        bases = [base.__name__ for base in inspect.getmro(cls)]
        if 'LazyLoad' in bases:
            return _object_cache.cache_lazyload(func, cls, **kwargs)
        elif 'Container' in bases:
            return _object_cache.cache_container(func, cls, **kwargs)

        return func(cls, **kwargs)

    inner.__name__ = func.__name__
    inner.__doc__ = func.__doc__
    return inner


def del_cache(func):
    def inner(obj, item):
        if func.__name__ == '__delitem__':
            _object_cache.del_item_cache(obj, item)
        return func(obj, item)

    inner.__name__ = func.__name__
    inner.__doc__ = func.__doc__
    inner._cache_maker = True
    return inner


def cache_parent(cls):
    cls._cache_class = cls.__bases__[0]
    return cls
