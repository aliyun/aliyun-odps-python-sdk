# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

import gc
import os
import sys
import tempfile
import time
import warnings

from .. import compat, utils
from .. import ODPS
from ..errors import InternalServerError, NoSuchObject
from ..tunnel import TableTunnel
from ..compat import six, ConfigParser

LOCK_FILE_NAME = os.path.join(tempfile.gettempdir(), 'pyodps_test_lock_')

LOGGING_CONFIG = {
    'version': 1,
    "filters": {
        "odps": {
            "name": "odps"
        },
    },
    "formatters": {
        "msgonly": {
            "format": "%(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": 'INFO',
            "formatter": "msgonly",
            "filters": ["odps",],
        },
    },
    "root": {
            "level": "NOTSET",
            "handlers": ["console"]
        },
    "disable_existing_loggers": False
}


class Config(object):
    config = None
    odps = None
    tunnel = None
    admin = None


def get_config():
    global LOGGING_CONFIG

    if not Config.config:
        config = ConfigParser.ConfigParser()
        Config.config = config
        config_path = os.path.join(os.path.dirname(__file__), 'test.conf')
        if not os.path.exists(config_path):
            raise Exception('Please configure test.conf (you can rename'
                            ' test.conf.template)')
        config.read(config_path)
        access_id = config.get("odps", "access_id")
        secret_access_key = config.get("odps", "secret_access_key")
        project = config.get("odps", "project")
        endpoint = config.get("odps", "endpoint")
        try:
            seahawks_url = config.get("odps", "seahawks_url")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            seahawks_url = None
        try:
            tunnel_endpoint = config.get("odps", "tunnel_endpoint")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            tunnel_endpoint = None
        try:
            predict_endpoint = config.get("odps", "predict_endpoint")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            predict_endpoint = None

        config.odps = ODPS(access_id, secret_access_key, project, endpoint,
                           tunnel_endpoint=tunnel_endpoint, predict_endpoint=predict_endpoint,
                           seahawks_url=seahawks_url)
        config.tunnel = TableTunnel(config.odps, endpoint=tunnel_endpoint)
        logging_level = config.get('test', 'logging_level')
        LOGGING_CONFIG['handlers']['console']['level'] = logging_level
    else:
        config = Config.config

    compat.dictconfig(LOGGING_CONFIG)
    return config


def to_str(s):
    if isinstance(s, six.binary_type):
        s = s.decode('utf-8')
    return s


def tn(s, limit=128):
    if os.environ.get('TEST_NAME_SUFFIX') is not None:
        suffix = '_' + os.environ.get('TEST_NAME_SUFFIX').lower()
        if len(s) + len(suffix) > limit:
            s = s[:limit - len(suffix)]
        return s + suffix
    else:
        if len(s) > limit:
            s = s[:limit]
        return s


def in_coverage_mode():
    return 'COVERAGE_FILE' in os.environ or 'unittest' in sys.argv[0]


def start_coverage():
    if not in_coverage_mode():
        return
    os.environ['COVERAGE_PROCESS_START'] = ''
    try:
        import coverage
        coverage.process_startup()
    except ImportError:
        pass


def ignore_case(case, reason):
    from odps.compat import unittest
    decorator = unittest.skip(reason)
    return decorator(case)


def ci_skip_case(obj):
    if 'CI_MODE' in os.environ:
        return ignore_case(obj, 'Intentionally skipped in CI mode.')
    else:
        return obj


def module_depend_case(mod_names):
    if isinstance(mod_names, six.string_types):
        mod_names = [mod_names, ]

    def _decorator(obj):
        for mod_name in mod_names:
            try:
                __import__(mod_name, fromlist=[''])
            except ImportError:
                return ignore_case(obj, 'Skipped due to absence of %s.' % mod_name)
        return obj
    return _decorator


numpy_case = module_depend_case('numpy')
pandas_case = module_depend_case('pandas')
snappy_case = module_depend_case('snappy')
sqlalchemy_case = module_depend_case('sqlalchemy')


def odps2_typed_case(func):
    def _wrapped(*args, **kwargs):
        from odps import options
        options.sql.use_odps2_extension = True

        old_settings = options.sql.settings
        options.sql.settings = old_settings or {}
        options.sql.settings.update({'odps.sql.hive.compatible': True})
        try:
            func(*args, **kwargs)
        finally:
            options.sql.use_odps2_extension = False
            options.sql.settings = old_settings

    _wrapped.__name__ = func.__name__
    return _wrapped


def global_locked(lock_key):

    def _decorator(func):
        if callable(lock_key):
            file_name = LOCK_FILE_NAME + '_' + func.__module__.replace('.', '__') + '__' + func.__name__ + '.lck'
        else:
            file_name = LOCK_FILE_NAME + '_' + lock_key + '.lck'

        def _decorated(*args, **kwargs):
            while os.path.exists(file_name):
                time.sleep(0.5)
            open(file_name, 'w').close()
            try:
                return func(*args, **kwargs)
            finally:
                os.unlink(file_name)

        _decorated.__name__ = func.__name__
        return _decorated

    if callable(lock_key):
        return _decorator(lock_key)
    else:
        return _decorator


class TestMeta(type):
    def __init__(cls, what, bases=None, d=None):
        # switch cases given switches defined in tests/__init__.py
        cls_path = cls.__module__
        if '.tests' in cls_path:
            main_pack_name, _ = cls_path.split('.tests', 1)
            test_pack_name = main_pack_name + '.tests'
            test_pack = __import__(test_pack_name, fromlist=[''])

            check_symbol = lambda s: hasattr(test_pack, s) and getattr(test_pack, s)

            # disable cases in CI_MODE when tests.SKIP_IN_CI = True
            if 'CI_MODE' in os.environ and check_symbol('SKIP_IN_CI'):
                for k, v in six.iteritems(d):
                    if k.startswith('test') and hasattr(v, '__call__'):
                        d[k] = None

        for k, v in six.iteritems(d):
            if k.startswith('test') and v is None:
                delattr(cls, k)

        super(TestMeta, cls).__init__(what, bases, d)


try:
    from flaky import flaky
    import requests

    def is_internal_error(err, fun_name, *_):
        ex = err[0]
        if fun_name in ('testInstanceQueueingInfo', 'testBatchStop', 'testAsyncExecute') \
                and issubclass(ex, AssertionError):
            return True
        if issubclass(ex, (InternalServerError, NoSuchObject)):
            return True
        if issubclass(ex, (requests.ConnectionError, requests.Timeout)):
            return True
        if 'Faithfully yours, tengine.' in str(ex):
            return True
        if 'status=CANCELLED' in str(ex):
            return True
        try:
            import psycopg2
            if issubclass(ex, psycopg2.ProgrammingError) and 'does not exist' in str(ex):
                return True
            if issubclass(ex, psycopg2.DatabaseError) and 'unknown error' in str(ex):
                return True
        except ImportError:
            pass
        return False

    @flaky(max_runs=3, rerun_filter=is_internal_error)
    class _TestBase(six.with_metaclass(TestMeta, compat.unittest.TestCase)):
        pass

except ImportError:
    class _TestBase(six.with_metaclass(TestMeta, compat.unittest.TestCase)):
        pass


class TestBase(_TestBase):

    def setUp(self):
        gc.collect()

        self.config = get_config()
        self.odps = self.config.odps
        self.tunnel = self.config.tunnel
        self.setup()

    def tearDown(self):
        self.teardown()

    def setup(self):
        pass

    def teardown(self):
        pass

    def _get_result(self, res):
        from odps.df.backends.frame import ResultFrame
        if isinstance(res, ResultFrame):
            res = res.values
        try:
            import pandas
            import numpy

            def conv(t):
                try:
                    if numpy.isnan(t):
                        return None
                except (TypeError, ValueError):
                    pass
                if isinstance(t, pandas.Timestamp):
                    t = t.to_pydatetime()
                elif not isinstance(t, (list, dict, tuple)) and pandas.isnull(t):
                    t = None
                return t

            if isinstance(res, pandas.DataFrame):
                return [list(conv(i) for i in it) for it in res.values]
            else:
                return res
        except ImportError:
            return res

    def assertWarns(self, func, warn_type=Warning):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            ret_val = func()
            # Verify some things
            assert len(w) == 1
            assert issubclass(w[-1].category, warn_type)

            return ret_val

    def assertNoWarns(self, func):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            ret_val = func()
            # Verify some things
            assert len(w) == 0

            return ret_val

    def assertListAlmostEqual(self, first, second, **kw):
        self.assertEqual(len(first), len(second))
        only_float = kw.pop('only_float', True)
        for f, s in zip(first, second):
            if only_float:
                self.assertAlmostEqual(f, s, **kw)
            else:
                if isinstance(f, float) and isinstance(s, float):
                    self.assertAlmostEqual(f, s, **kw)
                elif isinstance(f, list) and isinstance(s, list):
                    self.assertListAlmostEqual(f, s, only_float=False, **kw)
                else:
                    self.assertEqual(f, s)

    @staticmethod
    def waitContainerFilled(container_fun, countdown=10):
        while len(container_fun()) == 0:
            time.sleep(1)
            countdown -= 1
            if countdown <= 0:
                raise SystemError('Waiting for container content time out.')
