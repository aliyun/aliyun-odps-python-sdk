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

from copy import deepcopy
import contextlib

import six

DEFAULT_CHUNK_SIZE = 1496
DEFAULT_CONNECT_RETRY_TIMES = 4
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_READ_TIMEOUT = 120


class AttributeDict(dict):
    def __getattr__(self, item):
        if item in self:
            obj = self[item]
            if isinstance(obj, dict):
                return AttributeDict(obj)
            return obj
        return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        self[key] = value


class Config(object):
    def __init__(self, config=None):
        self._config = config or AttributeDict()

    def __getattr__(self, item):
        if item == '_config':
            return object.__getattribute__(self, '_config')
        return getattr(self._config, item)

    def __setattr__(self, key, value):
        if key == '_config':
            object.__setattr__(self, key, value)
        setattr(self._config, key, value)

    def register_option(self, option, value):
        splits = option.split('.')
        conf = self._config
        for name in splits[:-1]:
            config = conf.get(name)
            if config is None:
                conf[name] = dict()
                conf = conf[name]
            elif not isinstance(config, dict):
                raise AttributeError(
                    'Fail to set option: %s, conflict has encountered' % option)

        key = splits[-1]
        if conf.get(key) is not None:
            raise AttributeError(
                'Fail to set option: %s, option has been set' % option)

        conf[key] = value


@contextlib.contextmanager
def option_context(config=None):
    global options
    global_options = options

    try:
        config = config or dict()
        local_options = Config(deepcopy(global_options._config))
        for option, value in six.iteritems(config):
            local_options.register_option(option, value)
        options = local_options
        yield options
    finally:
        options = global_options


options = Config()
options.register_option('access_id', None)
options.register_option('access_key', None)
options.register_option('end_point', None)
options.register_option('default_project', None)
options.register_option('log_view_host', None)
options.register_option('tunnel_endpoint', None)

# network connections
options.register_option('chunk_size', DEFAULT_CHUNK_SIZE)
options.register_option('retry_times', DEFAULT_CONNECT_RETRY_TIMES)
options.register_option('connect_timeout', DEFAULT_CONNECT_TIMEOUT)
options.register_option('read_timeout', DEFAULT_READ_TIMEOUT)



