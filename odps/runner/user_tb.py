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

import os
import sys
import inspect
import traceback

_old_extract_tb = traceback.extract_tb
_old_getinnerframes = inspect.getinnerframes


def _new_extract_tb(tb, limit=None):
    if not isinstance(tb, list):
        return _old_extract_tb(tb, limit)
    stack = list(reversed(tb))
    stack_start = 0
    while stack_start < len(stack) and stack[stack_start][1].startswith(sys.prefix):
        stack_start += 1
    stack_end = len(stack)
    if limit is not None:
        stack_end = min(stack_end, stack_start + limit)
    return [(s[1], s[2], s[3], ','.join(s[4]).strip()) for s in stack[stack_start:stack_end]]

_new_extract_tb._pyodps_packed = True


def _new_getinnerframes(tb, context=1):
    if not isinstance(tb, list):
        return _old_getinnerframes(tb, context)
    stack = list(reversed(tb))
    stack_start = 0
    while stack_start < len(stack) and stack[stack_start][1].startswith(sys.prefix):
        stack_start += 1
    stack_end = len(stack)
    return stack[stack_start:stack_end]

_new_getinnerframes._pyodps_packed = True


def get_user_stack():
    from .. import core
    odps_dir = os.path.dirname(core.__file__)
    stack = inspect.stack()
    stack_start = 0
    while stack_start < len(stack) and stack[stack_start][1].startswith(odps_dir):
        stack_start += 1
    return stack[stack_start:]


def tb_patched(func):
    def _wrapper(*args, **kwargs):
        global _old_excepthook
        try:
            if not hasattr(inspect.getinnerframes, '_pyodps_packed'):
                inspect.getinnerframes = _new_getinnerframes
            if not hasattr(traceback.extract_tb, '_pyodps_packed'):
                traceback.extract_tb = _new_extract_tb
            func(*args, **kwargs)
        finally:
            inspect.getinnerframes = _old_getinnerframes
            traceback.extract_tb = _old_extract_tb

    _wrapper.__name__ = func.__name__
    _wrapper.__doc__ = func.__doc__
    return _wrapper
