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

from collections import namedtuple, OrderedDict

try:
    string_types = (unicode, basestring)
    iteritems = lambda d: d.iteritems()
except:
    string_types = (bytes, str)
    iteritems = lambda d: d.items()


class NamedTupleMixin(object):
    def __getattr__(self, item):
        if item in self._name_map:
            return self[self._name_map[item]]
        elif item == "get":
            return self._NamedTupleMixin_get
        elif item == "items":
            return self._NamedTupleMixin_items
        elif item == "keys":
            return self._NamedTupleMixin_keys
        elif item == "values":
            return self._NamedTupleMixin_values
        else:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, str(item))
            )

    def __getitem__(self, item):
        try:
            return self._base.__getitem__(self, item)
        except TypeError:
            pass

        return self[self._name_map[item]]

    def __eq__(self, other):
        if isinstance(other, tuple):
            return tuple(self) == other
        else:
            return self.asdict() == other

    def _NamedTupleMixin_get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default

    def _NamedTupleMixin_items(self):
        for k in self._names:
            yield k, getattr(self, k)

    def _NamedTupleMixin_keys(self):
        return self._names

    def _NamedTupleMixin_values(self):
        for k in self._names:
            yield getattr(self, k)

    def asdict(self):
        return OrderedDict(zip(self._names, self))

    def replace(self, **kwds):
        new_kw = {self._fields[self._name_map[k]]: v for k, v in iteritems(kwds)}
        return self._replace(**new_kw)


def xnamedtuple(typename, field_names):
    if isinstance(field_names, string_types):
        field_names = field_names.replace(",", " ").split()
    base_nt = namedtuple(typename + "_base", field_names, rename=True)
    nt = type(typename, (NamedTupleMixin, base_nt), {})
    nt._base = base_nt
    nt._name_map = {v: k for k, v in enumerate(field_names)}
    nt._names = field_names
    return nt
