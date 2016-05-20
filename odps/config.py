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

from .compat import six


DEFAULT_CHUNK_SIZE = 1496
DEFAULT_CONNECT_RETRY_TIMES = 4
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_READ_TIMEOUT = 120


class OptionError(Exception):
    pass


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        self._inited = False
        super(AttributeDict, self).__init__(*args, **kwargs)
        self._inited = True

    def __getattr__(self, item):
        if item in self:
            val = self[item]
            if isinstance(val, AttributeDict):
                return val
            else:
                return val[0]
        return object.__getattribute__(self, item)

    def register(self, key, value, validator=None):
        self[key] = value, validator

    def _setattr(self, key, value, silent=False):
        if not silent and key not in self:
            raise OptionError('You can only set the value of existing options')

        if not isinstance(value, AttributeDict):
            validate = None
            if key in self:
                validate = self[key][1]
                if validate is not None:
                    if not validate(value):
                        raise ValueError('Cannot set value %s' % value)
            self[key] = value, validate
        else:
            self[key] = value

    def __setattr__(self, key, value):
        if key == '_inited':
            super(AttributeDict, self).__setattr__(key, value)
            return
        try:
            object.__getattribute__(self, key)
            super(AttributeDict, self).__setattr__(key, value)
            return
        except AttributeError:
            pass

        if not self._inited:
            super(AttributeDict, self).__setattr__(key, value)
        else:
            self._setattr(key, value)


class DisplayAttributeDict(AttributeDict):
    def __init__(self, *args, **kwargs):
        self._inited = False
        self._prefix = None
        super(DisplayAttributeDict, self).__init__(*args, **kwargs)

    def register(self, key, value, validator=None):
        self[key] = value, validator

    def _setattr(self, key, value, silent=False):
        if not silent and key not in self:
            raise OptionError('You can only set the value of existing options')

        assert not isinstance(value, AttributeDict)

        validate = None
        if key in self:
            validate = self[key][1]
            if validate is not None:
                if not validate(value):
                    raise ValueError('Cannot set value %s' % value)
        self[key] = value, validate

        try:
            import pandas as pd

            try:
                pd.set_option('%s.%s' % (self._prefix, key), value)
            except:
                pass
        except ImportError:
            pass


class Config(object):
    def __init__(self, config=None):
        self._config = config or AttributeDict()

    def __getattr__(self, item):
        return getattr(self._config, item)

    def __setattr__(self, key, value):
        if key == '_config':
            object.__setattr__(self, key, value)
            return
        setattr(self._config, key, value)

    def register_option(self, option, value, validator=None):
        splits = option.split('.')
        conf = self._config

        if splits[0] == 'display':
            dict_cls = DisplayAttributeDict
        else:
            dict_cls = AttributeDict

        for i, name in enumerate(splits[:-1]):
            config = conf.get(name)
            if config is None:
                val = dict_cls()
                conf[name] = val
                if isinstance(val, DisplayAttributeDict):
                    # set the prefix used in the pandas option
                    if i == 0:
                        val._prefix = 'display'
                    else:
                        val._prefix = '%s.%s' % (conf._prefix, name)
                conf = val
            elif not isinstance(config, dict):
                raise AttributeError(
                    'Fail to set option: %s, conflict has encountered' % option)
            else:
                conf = config

        key = splits[-1]
        if conf.get(key) is not None:
            raise AttributeError(
                'Fail to set option: %s, option has been set' % option)

        conf.register(key, value, validator)


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


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


# validators
def any_validator(*validators):
    def validate(x):
        return any(validator(x) for validator in validators)
    return validate


def all_validator(*validators):
    def validate(x):
        return all(validator(x) for validator in validators)
    return validate


is_null = lambda x: x is None
is_bool = lambda x: isinstance(x, bool)
is_integer = lambda x: isinstance(x, six.integer_types)
is_string = lambda x: isinstance(x, six.string_types)
is_dict = lambda x: isinstance(x, dict)
def is_in(vals):
    def validate(x):
        return x in vals
    return validate


options = Config()
options.register_option('access_id', None)
options.register_option('access_key', None)
options.register_option('end_point', None)
options.register_option('default_project', None)
options.register_option('log_view_host', None)
options.register_option('log_view_hours', 24, validator=is_integer)
options.register_option('tunnel_endpoint', None)
options.register_option('biz_id', None)
options.register_option('temp_lifecycle', 1, validator=is_integer)
options.register_option('lifecycle', None, validator=any_validator(is_null, is_integer))

# c or python mode, use for UT, in other cases, please do not modify the value
options.register_option('force_c', False, validator=is_integer)
options.register_option('force_py', False, validator=is_integer)

# network connections
options.register_option('chunk_size', DEFAULT_CHUNK_SIZE, validator=is_integer)
options.register_option('retry_times', DEFAULT_CONNECT_RETRY_TIMES, validator=is_integer)
options.register_option('connect_timeout', DEFAULT_CONNECT_TIMEOUT, validator=is_integer)
options.register_option('read_timeout', DEFAULT_READ_TIMEOUT, validator=is_integer)

# terminal
options.register_option('console.max_lines', None)
options.register_option('console.max_width', None)
options.register_option('console.use_color', False, validator=is_bool)

# SQL
options.register_option('sql.settings', None, validator=any_validator(is_null, is_dict))

# DataFrame
options.register_option('interactive', is_interactive(), validator=is_bool)
options.register_option('verbose', False, validator=is_bool)
options.register_option('verbose_log', None)
options.register_option('df.optimize', True, validator=is_bool)
options.register_option('df.analyze', True, validator=is_bool)
options.register_option('df.use_cache', True, validator=is_bool)
options.register_option('df.quote', True, validator=is_bool)

# PAI
options.register_option('pai.xflow_project', 'algo_public', validator=is_string)
options.register_option('pai.parallel_num', 5, validator=is_integer)
options.register_option('pai.dry_run', False, validator=is_bool)
options.register_option('pai.retry_times', 3, validator=is_integer)

# display
from .console import detect_console_encoding

options.register_option('display.encoding', detect_console_encoding(), validator=is_string)
options.register_option('display.max_rows', 60, validator=any_validator(is_null, is_integer))
options.register_option('display.max_columns', 20, validator=any_validator(is_null, is_integer))
options.register_option('display.large_repr', 'truncate', validator=is_in(['truncate', 'info']))
options.register_option('display.notebook_repr_html', True, validator=is_bool)
options.register_option('display.precision', 6, validator=is_integer)
options.register_option('display.float_format', None)
options.register_option('display.chop_threshold', None)
options.register_option('display.column_space', 12, validator=is_integer)
options.register_option('display.pprint_nest_depth', 3, validator=is_integer)
options.register_option('display.max_seq_items', 100, validator=is_integer)
options.register_option('display.max_colwidth', 50, validator=is_integer)
options.register_option('display.multi_sparse', True, validator=is_bool)
options.register_option('display.colheader_justify', 'right', validator=is_string)
options.register_option('display.unicode.ambiguous_as_wide', False, validator=is_bool)
options.register_option('display.unicode.east_asian_width', False, validator=is_bool)
options.register_option('display.height', 60, validator=any_validator(is_null, is_integer))
options.register_option('display.width', 80, validator=any_validator(is_null, is_integer))
options.register_option('display.expand_frame_repr', True)
options.register_option('display.show_dimensions', 'truncate', validator=is_in([True, False, 'truncate']))




