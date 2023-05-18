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

import random
import time
from datetime import datetime
from decimal import Decimal
try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters

from .... import types
from ....tests.core import tn, pandas_case
from ....utils import to_text
from ..frame import ResultFrame


class NumGenerators(object):
    @staticmethod
    def gen_random_bigint(value_range=None):
        return random.randint(*(value_range or types.bigint._bounds))

    @staticmethod
    def gen_random_string(max_length=15):
        gen_letter = lambda: letters[random.randint(0, 51)]
        return to_text(''.join([gen_letter() for _ in range(random.randint(1, max_length))]))

    @staticmethod
    def gen_random_double():
        return random.uniform(-2**32, 2**32)

    @staticmethod
    def gen_random_datetime():
        dt = datetime.fromtimestamp(random.randint(0, int(time.time())))
        if dt.year >= 1986 or dt.year <= 1992:  # ignore years when daylight saving time is used
            return dt.replace(year=1996)
        else:
            return dt

    @staticmethod
    def gen_random_boolean():
        return random.uniform(-1, 1) > 0

    @classmethod
    def gen_random_decimal(cls):
        return Decimal(str(cls.gen_random_double()))


__all__ = ['NumGenerators', 'tn', 'pandas_case']
