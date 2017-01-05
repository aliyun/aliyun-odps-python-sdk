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
    from collections import OrderedDict
    from collections import namedtuple
except ImportError:
    from ordereddict import OrderedDict
    from collections import namedtuple as _namedtuple

    def namedtuple(typename, field_names, verbose=False, rename=False):
        return _namedtuple(typename, field_names, verbose=verbose)

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
        else:
            raise AttributeError

    def asdict(self):
        return OrderedDict(zip(self._names, self))

    def replace(self, **kwds):
        new_kw = dict((self._fields[self._name_map[k]], v) for k, v in iteritems(kwds))
        return self._replace(**new_kw)


def xnamedtuple(typename, field_names, verbose=False):
    if isinstance(field_names, string_types):
        field_names = field_names.replace(',', ' ').split()
    base_nt = namedtuple(typename + '_base', field_names, verbose=verbose, rename=True)
    nt = type(typename, (base_nt, NamedTupleMixin), {})
    nt._name_map = dict((v, k) for k, v in enumerate(field_names))
    nt._names = field_names
    return nt
