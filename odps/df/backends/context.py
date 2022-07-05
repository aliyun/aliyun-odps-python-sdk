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


class ExecuteContext(object):
    def __init__(self):
        self._expr_id_cached_data = dict()

    def cache(self, expr, data):
        if data is not None:
            self._expr_id_cached_data[expr._id] = data

    def is_cached(self, expr):
        return expr._id in self._expr_id_cached_data

    def get_cached(self, expr):
        return self._expr_id_cached_data[expr._id]

    def uncache(self, expr):
        if expr._id in self._expr_id_cached_data:
            del self._expr_id_cached_data[expr._id]

context = ExecuteContext()
