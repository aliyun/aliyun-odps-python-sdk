#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import collections
import functools
import sys
import threading

from ..compat import izip, six, futures
from .backends.core import Engine


class Delay(object):
    def __init__(self):
        self._idx = 0
        self._lock = threading.RLock()
        self._calls = dict()
        self._futures = dict()
        self._running = False

    def execute(self, ui=None, async=False, n_parallel=1, timeout=None,
                close_and_notify=True):

        if self._running:
            raise RuntimeError('Cannot execute on an executing delay object.')

        with self._lock:
            _idx_calls = sorted(six.iteritems(self._calls), key=lambda p: p[0])
            _indices = [p[0] for p in _idx_calls]
            _calls = [p[1] for p in _idx_calls]
            _futures = [p[1] for p in sorted(six.iteritems(self._futures), key=lambda p: p[0])]

        from .backends.engine import get_default_engine
        engine = get_default_engine(*[call[1] for call in _calls])

        ui = ui or Engine._create_ui(async=async, n_parallel=n_parallel)

        batch_kw = dict(ui=ui, async=True, n_parallel=n_parallel, timeout=timeout,
                        close_and_notify=close_and_notify)
        fs = engine.batch(*_calls, **batch_kw)

        if not isinstance(fs, collections.Iterable):
            fs = [fs]
        if len(fs) == 0:
            return

        self._running = True

        def relay_future(src, index=None, dest=None):
            if dest.done():
                return
            with self._lock:
                try:
                    dest.set_result(src.result())
                except:
                    e, tb = sys.exc_info()[1:]
                    if six.PY2:
                        dest.set_exception_info(e, tb)
                    else:
                        dest.set_exception(e)
                finally:
                    del self._calls[index]
                    del self._futures[index]

        for idx, uf, bf in izip(_indices, _futures, fs):
            uf.set_running_or_notify_cancel()
            bf.add_done_callback(functools.partial(relay_future, index=idx, dest=uf))
            if bf.done():
                relay_future(bf, idx, uf)

        if not async:
            try:
                futures.wait(_futures, timeout)
                for uf in _futures:
                    uf.result()
                ui.notify('DataFrame execution succeeded')
            except:
                ui.notify('DataFrame execution failed')
                raise
            finally:
                self._running = False
        else:
            delay_future = futures.Future()
            delay_future.set_running_or_notify_cancel()
            result_store, exc_store = [None], [None]

            def _check_results(src):
                if delay_future.done():
                    return
                try:
                    result_store[0] = src.result()
                except:
                    exc_store[0] = sys.exc_info()[1:]

                with self._lock:
                    if all(f.done() for f in fs):
                        self._running = False
                        if exc_store[0] is not None:
                            e, tb = exc_store[0]
                            if six.PY2:
                                delay_future.set_exception_info(e, tb)
                            else:
                                delay_future.set_exception(e)
                        else:
                            delay_future.set_result(result_store[0])
            for f in fs:
                f.add_done_callback(_check_results)
                if f.done():
                    _check_results(f)

            return delay_future

    def register_item(self, action, expr, *args, **kwargs):
        with self._lock:
            call_idx = self._idx
            self._idx += 1
            user_future = futures.Future()
            self._calls[call_idx] = (action, expr, args, kwargs)
            self._futures[call_idx] = user_future

        return user_future
