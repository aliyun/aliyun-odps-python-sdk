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

import importlib
import math
import os
import random
import sys
import tempfile
import time
import types

import pytest
try:
    from flaky import flaky as _raw_flaky
except ImportError:
    _raw_flaky = None

from .. import compat, errors, options
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
    oss = None
    tunnel = None
    admin = None


def _load_config_odps(config, section_name, overwrite_global=True):
    from ..core import ODPS

    try:
        config.options(section_name)
    except ConfigParser.NoSectionError:
        return

    access_id = config.get(section_name, "access_id")
    secret_access_key = config.get(section_name, "secret_access_key")
    project = config.get(section_name, "project")
    endpoint = config.get(section_name, "endpoint")
    try:
        seahawks_url = config.get(section_name, "seahawks_url")
    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
        seahawks_url = None
    try:
        schema = config.get(section_name, "schema")
    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
        schema = None
    try:
        tunnel_endpoint = config.get(section_name, "tunnel_endpoint")
    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
        tunnel_endpoint = None

    try:
        attr_name = config.get(section_name, "attr")
    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
        attr_name = section_name

    odps_entry = ODPS(
        access_id, secret_access_key, project, endpoint,
        schema=schema, tunnel_endpoint=tunnel_endpoint,
        seahawks_url=seahawks_url,
        overwrite_global=overwrite_global,
    )
    setattr(config, attr_name, odps_entry)


def get_config():
    global LOGGING_CONFIG

    from ..tunnel.tabletunnel import TableTunnel

    if not Config.config:
        config = ConfigParser.ConfigParser()
        Config.config = config
        config_path = os.path.join(os.path.dirname(__file__), 'test.conf')
        if not os.path.exists(config_path):
            raise OSError(
                'Please configure test.conf (you can rename test.conf.template)'
            )
        config.read(config_path)

        _load_config_odps(config, "odps_daily", overwrite_global=False)
        _load_config_odps(config, "odps_with_storage_tier", overwrite_global=False)
        _load_config_odps(config, "odps_with_schema", overwrite_global=False)
        _load_config_odps(config, "odps_with_schema_tenant", overwrite_global=False)
        _load_config_odps(config, "odps_with_tunnel_quota", overwrite_global=False)
        # make sure main config overrides other configs
        _load_config_odps(config, "odps")
        config.tunnel = TableTunnel(config.odps, endpoint=config.odps._tunnel_endpoint)

        try:
            from cupid import options as cupid_options

            cupid_options.cupid.proxy_endpoint = config.get("cupid", "proxy_endpoint")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError, ImportError):
            pass

        try:
            oss_access_id = config.get("oss", "access_id")
            oss_secret_access_key = config.get("oss", "secret_access_key")
            oss_bucket_name = config.get("oss", "bucket_name")
            oss_endpoint = config.get("oss", "endpoint")

            config.oss_config = (
                oss_access_id, oss_secret_access_key, oss_bucket_name, oss_endpoint
            )

            import oss2

            auth = oss2.Auth(oss_access_id, oss_secret_access_key)
            config.oss_bucket = oss2.Bucket(auth, oss_endpoint, oss_bucket_name)
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError, ImportError):
            pass

        logging_level = config.get('test', 'logging_level')
        LOGGING_CONFIG['handlers']['console']['level'] = logging_level
    else:
        config = Config.config

    compat.dictconfig(LOGGING_CONFIG)
    return config


_test_tables_to_drop = set()


def tn(s, limit=128):
    if os.environ.get('TEST_NAME_SUFFIX') is not None:
        suffix = '_' + os.environ.get('TEST_NAME_SUFFIX').lower()
        if len(s) + len(suffix) > limit:
            s = s[:limit - len(suffix)]
        table_name = s + suffix
        _test_tables_to_drop.add(table_name)
        return table_name
    else:
        if len(s) > limit:
            s = s[:limit]
        return s


def drop_test_tables(odps):
    global _test_tables_to_drop
    for table_name in _test_tables_to_drop:
        odps.delete_table(table_name, if_exists=True, async_=True)
    _test_tables_to_drop = set()


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


def flaky(o=None, *args, **kwargs):
    platform = kwargs.pop("platform", "")
    if _raw_flaky is None or not sys.platform.startswith(platform):
        if o is not None:
            return o

        def ident(x):
            return x

        return ident
    elif o is not None:
        return _raw_flaky(o, *args, **kwargs)
    else:
        return _raw_flaky(*args, **kwargs)


def ignore_case(case, reason):
    if isinstance(case, types.FunctionType) and not case.__name__.startswith("test"):
        @six.wraps(case)
        def wrapped(*args, **kwargs):
            pytest.skip(reason)
            return case(*args, **kwargs)

        return wrapped

    decorator = pytest.mark.skip(reason)
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
            # avoid reimporting modules
            if sys.version_info[0] == 2 and mod_name in sys.modules:
                continue
            try:
                __import__(mod_name, fromlist=[''])
            except ImportError:
                return ignore_case(obj, 'Skipped due to absence of %s.' % mod_name)
        return obj
    return _decorator


numpy_case = module_depend_case('numpy')
pandas_case = module_depend_case('pandas')
pyarrow_case = module_depend_case('pyarrow')
sqlalchemy_case = module_depend_case('sqlalchemy')


def odps2_typed_case(func):
    @six.wraps(func)
    def _wrapped(*args, **kwargs):
        from odps import options
        options.sql.use_odps2_extension = True

        old_settings = options.sql.settings
        options.sql.settings = old_settings or {}
        options.sql.settings.update({'odps.sql.hive.compatible': True})
        options.sql.settings.update({'odps.sql.decimal.odps2': True})
        try:
            func(*args, **kwargs)
        finally:
            options.sql.use_odps2_extension = None
            options.sql.settings = old_settings

    return _wrapped


def global_locked(lock_key):

    def _decorator(func):
        if callable(lock_key):
            file_name = LOCK_FILE_NAME + '_' + func.__module__.replace('.', '__') + '__' + func.__name__ + '.lck'
        else:
            file_name = LOCK_FILE_NAME + '_' + lock_key + '.lck'

        @six.wraps(func)
        def _decorated(*args, **kwargs):
            while os.path.exists(file_name):
                time.sleep(0.5)
            open(file_name, 'w').close()
            try:
                return func(*args, **kwargs)
            finally:
                os.unlink(file_name)

        return _decorated

    if callable(lock_key):
        return _decorator(lock_key)
    else:
        return _decorator


def approx_list(val, **kw):
    res = [None] * len(val)
    for idx, x in enumerate(val):
        if isinstance(x, float):
            res[idx] = pytest.approx(x, **kw)
        elif isinstance(x, list):
            res[idx] = approx_list(x)
        else:
            res[idx] = x
    return res


def wait_filled(container_fun, countdown=10):
    while len(container_fun()) == 0:
        time.sleep(1)
        countdown -= 1
        if countdown <= 0:
            raise SystemError('Waiting for container content time out.')


def run_sub_tests_in_parallel(n_parallel, sub_tests):
    test_pool = compat.futures.ThreadPoolExecutor(n_parallel)
    futures = [
        test_pool.submit(sub_test) for idx, sub_test in enumerate(sub_tests)
    ]
    try:
        first_exc = None
        for fut in futures:
            try:
                fut.result()
            except:
                if first_exc is None:
                    first_exc = sys.exc_info()
        if first_exc is not None:
            six.reraise(*first_exc)
    finally:
        test_pool.shutdown(wait=True)


def force_drop_schema(schema):
    insts = []
    if schema.name not in schema.project.schemas:
        return
    try:
        for tb in schema.tables:
            insts.append(tb.drop(async_=True))
        for res in schema.resources:
            res.drop()
        for func in schema.functions:
            func.drop()
        for inst in insts:
            inst.wait_for_completion()
        schema.drop()
    except errors.InternalServerError as ex:
        if "Database not found" not in str(ex):
            raise
    except errors.NoSuchObject:
        pass


def get_result(res):
    from odps.df.backends.frame import ResultFrame
    if isinstance(res, ResultFrame):
        res = res.values
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        np = pd = None

    def conv(t):
        try:
            if np is not None and np.isnan(t):
                return None
            if math.isnan(t):
                return None
        except (TypeError, ValueError):
            pass

        if pd is not None:
            if isinstance(t, pd.Timestamp):
                t = t.to_pydatetime()
            elif not isinstance(t, (list, dict, tuple)) and pd.isnull(t):
                t = None
        return t

    if pd is not None and isinstance(res, pd.DataFrame):
        return [list(conv(i) for i in it) for it in res.values]
    elif res and isinstance(res, list) and isinstance(res[0], list):
        return [list(conv(i) for i in it) for it in res]
    else:
        return res


def get_code_mode():
    from odps import crc as _crc
    if hasattr(_crc.Crc32c, "_method"):
        return _crc.Crc32c._method
    else:
        return "c"


def py_and_c(modules=None, reloader=None):
    fixture_name = "mod_reloader_%s" % random.randint(0, 99999)
    if isinstance(modules, six.string_types):
        modules = [modules]
    if "odps.crc" not in modules:
        modules.append("odps.crc")

    try:
        import cython
        has_cython = True
    except ImportError:
        has_cython = False

    def mod_reloader(request):
        impl = request.param
        if impl == "c" and not has_cython:
            pytest.skip("Must install cython to run this test.")

        old_config = getattr(options, 'force_{0}'.format(impl))
        setattr(options, 'force_{0}'.format(impl), True)

        for mod_name in (modules or []):
            mod = importlib.import_module(mod_name)
            compat.reload_module(mod)

        if callable(reloader):
            reloader()

        assert get_code_mode() == impl

        try:
            yield
        finally:
            setattr(options, 'force_{0}'.format(impl), old_config)

    mod_reloader.__name__ = fixture_name

    def wrap_fun(fun):
        func_mod = __import__(fun.__module__, fromlist=[''])
        if not hasattr(func_mod, fixture_name):
            setattr(func_mod, fixture_name, mod_reloader)

            mod_reloader.__module__ = fun.__module__
            try:
                mod_fixture = pytest.fixture(mod_reloader)
            except ValueError:
                mod_fixture = mod_reloader
            setattr(func_mod, fixture_name, mod_fixture)

        fixture_deco = pytest.mark.usefixtures(fixture_name)
        param_deco = pytest.mark.parametrize(fixture_name, ["py", "c"], indirect=True)
        return param_deco(fixture_deco(fun))

    return wrap_fun
