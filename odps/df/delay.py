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


class Delay(object):
    def __init__(self):
        self._calls = []
        self._futures = []

    def execute(self, ui=None, async=False, n_parallel=1, timeout=None,
                close_and_notify=True):
        from .backends.engine import get_default_engine
        future_lock = threading.RLock()
        engine = get_default_engine(*[call[1] for call in self._calls])

        batch_kw = dict(ui=ui, async=True, n_parallel=n_parallel, timeout=timeout,
                        close_and_notify=close_and_notify)
        fs = engine.batch(*self._calls, **batch_kw)

        if not isinstance(fs, collections.Iterable):
            fs = [fs]
        if len(fs) == 0:
            return

        def relay_future(src, dest=None):
            if dest.done():
                return
            with future_lock:
                try:
                    dest.set_result(src.result())
                except:
                    e, tb = sys.exc_info()[1:]
                    if six.PY2:
                        dest.set_exception_info(e, tb)
                    else:
                        dest.set_exception(e)

        for uf, bf in izip(self._futures, fs):
            uf.set_running_or_notify_cancel()
            bf.add_done_callback(functools.partial(relay_future, dest=uf))
            if bf.done():
                relay_future(bf, uf)

        if not async:
            futures.wait(self._futures, timeout)
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

                with future_lock:
                    if all(f.done() for f in fs):
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
        user_future = futures.Future()
        self._calls.append((action, expr, args, kwargs))
        self._futures.append(user_future)
        return user_future
