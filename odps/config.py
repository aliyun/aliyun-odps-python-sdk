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

from copy import deepcopy
import collections
import contextlib
import traceback
import warnings

from .compat import six


DEFAULT_CHUNK_SIZE = 1496
DEFAULT_CONNECT_RETRY_TIMES = 4
DEFAULT_CONNECT_TIMEOUT = 120
DEFAULT_READ_TIMEOUT = 120
DEFAULT_POOL_CONNECTIONS = 10
DEFAULT_POOL_MAXSIZE = 10
_DEFAULT_REDIRECT_WARN = 'Option {source} has been replaced by {target} and might be removed in a future release.'


class OptionError(Exception):
    pass


class Redirection(object):
    def __init__(self, item, warn=None):
        self._items = item.split('.')
        self._warn = warn
        self._warned = True
        self._parent = None

    def bind(self, attr_dict):
        self._parent = attr_dict
        self.getvalue()
        self._warned = False

    def getvalue(self, silent=False):
        if not silent and self._warn and not self._warned:
            in_completer = any(1 for st in traceback.extract_stack()
                               if 'completer' in st[0].lower())
            if not in_completer:
                self._warned = True
                warnings.warn(self._warn)
        conf = self._parent.root
        for it in self._items:
            conf = getattr(conf, it)
        return conf

    def setvalue(self, value, silent=False):
        if not silent and self._warn and not self._warned:
            self._warned = True
            warnings.warn(self._warn)
        conf = self._parent.root
        for it in self._items[:-1]:
            conf = getattr(conf, it)
        setattr(conf, self._items[-1], value)


class PandasRedirection(Redirection):
    def __init__(self, *args, **kw):
        super(PandasRedirection, self).__init__(*args, **kw)
        self._val = None
        try:
            import pandas  # noqa: F401

            self._use_pd = True
        except ImportError:
            self._use_pd = False

    def getvalue(self, silent=False):
        if self._use_pd:
            import pandas as pd
            try:
                return pd.get_option('.'.join(self._items))
            except (KeyError, LookupError, AttributeError):
                self._use_pd = False
        else:
            return self._val

    def setvalue(self, value, silent=False):
        if self._use_pd:
            import pandas as pd
            key = '.'.join(self._items)
            if value != pd.get_option(key):
                pd.set_option(key, value)
        else:
            self._val = value


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        self._inited = False
        self._parent = kwargs.pop('_parent', None)
        self._root = None
        super(AttributeDict, self).__init__(*args, **kwargs)
        self._inited = True

    @property
    def root(self):
        if self._root is not None:
            return self._root
        if self._parent is None:
            self._root = self
        else:
            self._root = self._parent.root
        return self._root

    def __getattr__(self, item):
        if item in self:
            val = self[item]
            if isinstance(val, AttributeDict):
                return val
            elif isinstance(val[0], Redirection):
                return val[0].getvalue()
            else:
                return val[0]
        return object.__getattribute__(self, item)

    def __dir__(self):
        return list(six.iterkeys(self))

    def register(self, key, value, validator=None):
        self[key] = value, validator
        if isinstance(value, Redirection):
            value.bind(self)

    def unregister(self, key):
        del self[key]

    def _setattr(self, key, value, silent=False):
        if not silent and key not in self:
            raise OptionError("Cannot identify configuration name '%s'." % str(key))

        if not isinstance(value, AttributeDict):
            validate = None
            if key in self:
                val = self[key]
                validate = self[key][1]
                if validate is not None:
                    if not validate(value):
                        raise ValueError('Cannot set value %s' % value)
                if isinstance(val[0], Redirection):
                    val[0].setvalue(value)
                else:
                    self[key] = value, validate
            else:
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

    def loads(self, d):
        dispatches = collections.defaultdict(dict)
        for k, v in six.iteritems(d):
            if '.' in k:
                sk, rk = k.split('.', 1)
                dispatches[sk][rk] = v
            elif isinstance(self[k][0], Redirection):
                self[k][0].setvalue(v, silent=True)
            else:
                setattr(self, k, v)
        for k, v in six.iteritems(dispatches):
            self[k].loads(v)

    def dumps(self):
        from .accounts import BaseAccount

        result_dict = dict()
        for k, v in six.iteritems(self):
            if isinstance(v, AttributeDict):
                result_dict.update((k + '.' + sk, sv) for sk, sv in six.iteritems(v.dumps()))
            elif isinstance(v[0], BaseAccount) or callable(v[0]):
                # ignore accounts in config dumps
                result_dict[k] = None
            elif isinstance(v[0], Redirection):
                result_dict[k] = v[0].getvalue(silent=True)
            else:
                result_dict[k] = v[0]
        return result_dict


class Config(object):
    def __init__(self, config=None):
        self._config = config or AttributeDict()

    def __dir__(self):
        return list(six.iterkeys(self._config))

    def __getattr__(self, item):
        return getattr(self._config, item)

    def __setattr__(self, key, value):
        if key == '_config':
            object.__setattr__(self, key, value)
            return
        setattr(self._config, key, value)

    def register_option(self, option, value, validator=None):
        assert validator is None or callable(validator)
        splits = option.split('.')
        conf = self._config

        for i, name in enumerate(splits[:-1]):
            config = conf.get(name)
            if config is None:
                val = AttributeDict(_parent=conf)
                conf[name] = val
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

    def register_pandas(self, option, value=None, validator=None):
        redir = PandasRedirection(option)
        self.register_option(option, redir, validator=validator)
        if value is not None:
            redir.setvalue(value)

    def redirect_option(self, option, target, warn=_DEFAULT_REDIRECT_WARN):
        redir = Redirection(target, warn=warn.format(source=option, target=target))
        self.register_option(option, redir)

    def unregister_option(self, option):
        splits = option.split('.')
        conf = self._config
        for name in splits[:-1]:
            config = conf.get(name)
            if not isinstance(config, dict):
                raise AttributeError(
                    'Fail to unregister option: %s, conflict has encountered' % option)
            else:
                conf = config

        key = splits[-1]
        if key not in conf:
            raise AttributeError('Option %s not configured, thus failed to unregister.' % option)
        conf.unregister(key)

    def loads(self, d):
        return self._config.loads(d)

    def dumps(self):
        return self._config.dumps()


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
options.register_option('is_global_account_overwritable', True, validator=is_bool)
options.register_option('account', None)
options.register_option('endpoint', None)
options.redirect_option('end_point', 'endpoint')
options.register_option('default_project', None)
options.register_option('app_account', None)
options.register_option('local_timezone', None)
options.register_option('allow_antique_date', False)
options.register_option('user_agent_pattern', '$pyodps_version $python_version $os_version')
options.register_option('logview_host', None)
options.register_option('logview_hours', 24 * 30, validator=is_integer)
options.redirect_option('log_view_host', 'logview_host')
options.redirect_option('log_view_hours', 'logview_hours')
options.register_option('api_proxy', None)
options.register_option('data_proxy', None)
options.redirect_option('tunnel_proxy', 'data_proxy')
options.register_option('seahawks_url', None)
options.register_option('biz_id', None)
options.register_option('priority', None, validator=any_validator(is_null, is_integer))
options.register_option('get_priority', None)
options.register_option('temp_lifecycle', 1, validator=is_integer)
options.register_option('lifecycle', None, validator=any_validator(is_null, is_integer))
options.register_option('table_read_limit', None, validator=any_validator(is_null, is_integer))
options.register_option('completion_size', 10, validator=is_integer)
options.register_option('default_task_settings', None, validator=any_validator(is_null, is_dict))

# c or python mode, use for UT, in other cases, please do not modify the value
options.register_option('force_c', False, validator=is_integer)
options.register_option('force_py', False, validator=is_integer)

# callbacks for wrappers
options.register_option('instance_create_callback', None)
options.register_option("tunnel_session_create_callback", None)
options.register_option("result_reader_create_callback", None)
options.register_option('tunnel_read_timeout_callback', None)

# network connections
options.register_option('chunk_size', DEFAULT_CHUNK_SIZE, validator=is_integer)
options.register_option('retry_times', DEFAULT_CONNECT_RETRY_TIMES, validator=is_integer)
options.register_option('connect_timeout', DEFAULT_CONNECT_TIMEOUT, validator=is_integer)
options.register_option('read_timeout', DEFAULT_READ_TIMEOUT, validator=is_integer)
options.register_option('pool_connections', DEFAULT_POOL_CONNECTIONS, validator=is_integer)
options.register_option('pool_maxsize', DEFAULT_POOL_MAXSIZE, validator=is_integer)

# Tunnel
options.register_option('tunnel.endpoint', None)
options.register_option('tunnel.string_as_binary', False, validator=is_bool)
options.register_option('tunnel.use_instance_tunnel', True, validator=is_bool)
options.register_option('tunnel.limit_instance_tunnel', None, validator=any_validator(is_null, is_bool))
options.register_option('tunnel.pd_mem_cache_size', 1024 * 4, validator=is_integer)
options.register_option('tunnel.pd_row_cache_size', 1024 * 16, validator=is_integer)
options.redirect_option('tunnel_endpoint', 'tunnel.endpoint')
options.redirect_option('use_instance_tunnel', 'tunnel.use_instance_tunnel')
options.redirect_option('limited_instance_tunnel', 'tunnel.limit_instance_tunnel')
options.redirect_option('tunnel.limited_instance_tunnel', 'tunnel.limit_instance_tunnel')

# terminal
options.register_option('console.max_lines', None)
options.register_option('console.max_width', None)
options.register_option('console.use_color', False, validator=is_bool)

# SQL
options.register_option('sql.settings', None, validator=any_validator(is_null, is_dict))
options.register_option('sql.use_odps2_extension', None, validator=any_validator(is_null, is_bool))

# DataFrame
options.register_option('interactive', is_interactive(), validator=is_bool)
options.register_option('verbose', False, validator=is_bool)
options.register_option('verbose_log', None)
options.register_option('df.optimize', True, validator=is_bool)
options.register_option('df.optimizes.cp', True, validator=is_bool)
options.register_option('df.optimizes.pp', True, validator=is_bool)
options.register_option('df.optimizes.tunnel', True, validator=is_bool)
options.register_option('df.analyze', True, validator=is_bool)
options.register_option('df.use_cache', True, validator=is_bool)
options.register_option('df.quote', True, validator=is_bool)
options.register_option('df.dump_udf', False, validator=is_bool)
options.register_option('df.supersede_libraries', True, validator=is_bool)
options.register_option('df.libraries', None)
options.register_option('df.odps.sort.limit', 10000)
options.register_option('df.sqlalchemy.execution_options', None, validator=any_validator(is_null, is_dict))
options.register_option('df.seahawks.max_size', 10 * 1024 * 1024 * 1024)  # 10G

# PyODPS ML
options.register_option('ml.xflow_project', 'algo_public', validator=is_string)
options.register_option('ml.xflow_settings', None, validator=any_validator(is_null, is_dict))
options.register_option('ml.dry_run', False, validator=is_bool)
options.register_option('ml.use_model_transfer', False, validator=is_bool)
options.register_option('ml.use_old_metrics', True, validator=is_bool)
options.register_option('ml.model_volume', 'pyodps_volume', validator=is_string)

# Runner
options.redirect_option('runner.dry_run', 'ml.dry_run')

# display
from .console import detect_console_encoding

options.register_pandas('display.encoding', detect_console_encoding(), validator=is_string)
options.register_pandas('display.max_rows', 60, validator=any_validator(is_null, is_integer))
options.register_pandas('display.max_columns', 20, validator=any_validator(is_null, is_integer))
options.register_pandas('display.large_repr', 'truncate', validator=is_in(['truncate', 'info']))
options.register_pandas('display.notebook_repr_html', True, validator=is_bool)
options.register_pandas('display.precision', 6, validator=is_integer)
options.register_pandas('display.float_format', None)
options.register_pandas('display.chop_threshold', None)
options.register_pandas('display.column_space', 12, validator=is_integer)
options.register_pandas('display.pprint_nest_depth', 3, validator=is_integer)
options.register_pandas('display.max_seq_items', 100, validator=is_integer)
options.register_pandas('display.max_colwidth', 50, validator=is_integer)
options.register_pandas('display.multi_sparse', True, validator=is_bool)
options.register_pandas('display.colheader_justify', 'right', validator=is_string)
options.register_pandas('display.unicode.ambiguous_as_wide', False, validator=is_bool)
options.register_pandas('display.unicode.east_asian_width', False, validator=is_bool)
options.redirect_option('display.height', 'display.max_rows')
options.register_pandas('display.width', 80, validator=any_validator(is_null, is_integer))
options.register_pandas('display.expand_frame_repr', True)
options.register_pandas('display.show_dimensions', 'truncate', validator=is_in([True, False, 'truncate']))

options.register_option('display.notebook_widget', True, validator=is_bool)
options.redirect_option('display.notebook_repr_widget', 'display.notebook_widget')

# Mars
options.register_option('mars.use_common_proxy', True, validator=is_bool)
options.register_option('mars.launch_notebook', False, validator=is_bool)
options.register_option('mars.to_dataframe_memory_scale', None, validator=any_validator(is_null, is_integer))
options.register_option('mars.container_status_timeout', 120, validator=is_integer)
