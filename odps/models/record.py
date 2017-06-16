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

from .. import types as _types
from ..config import options
try:
    if not options.force_py:
        from ..src.types_c import BaseRecord

        Record = _types.RecordMeta('Record', (_types.RecordReprMixin, BaseRecord),
                                   {'__doc__': _types.Record.__doc__})
    else:
        Record = _types.Record
except (ImportError, AttributeError):
    if options.force_c:
        raise
    Record = _types.Record

