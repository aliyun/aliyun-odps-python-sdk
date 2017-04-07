# encoding: utf-8
# copyright 1999-2017 alibaba group holding ltd.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#      http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import types


def get_field_name_by_role(df, role):
    fields = [f for f in df._ml_fields if role in f.role]
    if not fields:
        raise ValueError('Input df does not contain a field with role %s.' % role.name)
    return fields[0].name


class FuncArgs(Exception):
    def __init__(self, *args, **kwargs):
        self._args = kwargs.pop('args', ())
        self._kwargs = kwargs.pop('kwargs', {})
        super(FuncArgs, self).__init__(*args, **kwargs)


def replace_function_globals(fun, repl_dict):
    glob = fun.__globals__.copy()
    glob.update(repl_dict)
    new_fun = types.FunctionType(fun.__code__, glob)
    new_fun.__name__ = fun.__name__
    new_fun.__defaults__ = fun.__defaults__
    if hasattr(fun, '__kwdefaults__'):
        new_fun.__kwdefaults__ = fun.__kwdefaults__
    return new_fun


def fetch_call_args(fun_name, callee, *args, **kwargs):
    def _raiser(*args, **kwargs):
        raise FuncArgs(args=args, kwargs=kwargs)

    new_callee = replace_function_globals(callee, {fun_name: _raiser})
    try:
        new_callee(*args, **kwargs)
    except FuncArgs as ex:
        return ex._args, ex._kwargs
    return (), {}


def metrics_result(runner_fun, fun_name=None):
    fun_name = fun_name or runner_fun.__name__

    def _decorator(fun):
        def _decorated(*args, **kwargs):
            execute_now = kwargs.pop('execute_now', True)
            if execute_now:
                return fun(*args, **kwargs)
            else:
                def result_callback(ret):
                    new_fun = replace_function_globals(fun, {fun_name: lambda *args, **kw: ret})
                    return new_fun(*args, **kwargs)

                node_args, node_kw = fetch_call_args(fun_name, fun, *args, **kwargs)
                node_kw.update(dict(execute_now=False, result_callback=result_callback))
                return runner_fun(*node_args, **node_kw)

        _decorated.__name__ = fun.__name__
        _decorated.__doc__ = fun.__doc__
        return _decorated
    return _decorator

_metrics_fallback_detected = False


def detect_metrics_fallback(df):
    global _metrics_fallback_detected
    from ... import ODPS, errors, options

    if _metrics_fallback_detected or options.ml.use_old_metrics:
        return

    _metrics_fallback_detected = True
    odps = ODPS.from_global()
    for src in df.data_source():
        if hasattr(src, 'odps'):
            odps = src.odps

    try:
        odps.execute_xflow('MultiClassEvaluation', xflow_project=options.ml.xflow_project)
    except errors.ODPSError as ex:
        if 'xflow not found' in str(ex):
            options.ml.use_old_metrics = True
