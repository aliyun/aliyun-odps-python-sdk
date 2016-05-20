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

import inspect


class FunctionWrapper(object):
    def __init__(self, func):
        self._func = func
        self.output_names = None
        self.output_types = None

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def output_names(*names):
    if len(names) == 1 and isinstance(names[0], (tuple, list)):
        names = tuple(names[0])

    def inner(func):
        if isinstance(func, FunctionWrapper):
            wrapper = func
        else:
            wrapper = FunctionWrapper(func)
        wrapper.output_names = names
        return wrapper
    return inner


def output_types(*types):
    if len(types) == 1 and isinstance(types[0], (tuple, list)):
        types = tuple(types[0])

    def inner(func):
        if isinstance(func, FunctionWrapper):
            wrapper = func
        else:
            wrapper = FunctionWrapper(func)
        wrapper.output_types = types
        return wrapper
    return inner


def output(names, types):
    if isinstance(names, tuple):
        names = list(names)
    if not isinstance(names, list):
        names = [names, ]

    if isinstance(types, tuple):
        types = list(types)
    if not isinstance(types, list):
        types = [types, ]

    def inner(func):
        if isinstance(func, FunctionWrapper):
            wrapper = func
        else:
            wrapper = FunctionWrapper(func)
        wrapper.output_names = names
        wrapper.output_types = types
        return wrapper
    return inner


def make_copy(f):
    if inspect.isfunction(f):
        return lambda *args, **kwargs: f(*args, **kwargs)
    elif inspect.isclass(f):
        class NewCls(f):
            pass
        return NewCls
    else:
        return f