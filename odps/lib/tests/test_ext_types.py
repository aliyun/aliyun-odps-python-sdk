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

import pickle
import sys
from datetime import datetime

from odps.lib.ext_types import Monthdelta
from odps.tests.core import TestBase


class Test(TestBase):
    def testMonthdelta(self):
        ym = lambda md: (md.years, md.months)

        self.assertEqual(Monthdelta(years=2, months=5).total_months(), 29)
        self.assertEqual(int(Monthdelta(years=2, months=5)), 29)
        if sys.version_info[0] <= 2:
            self.assertEqual(long(Monthdelta(years=2, months=5)), 29)

        self.assertEqual(ym(Monthdelta(12)), (1, 0))
        self.assertEqual(ym(Monthdelta(26)), (2, 2))
        self.assertEqual(ym(Monthdelta(-11)), (-1, 1))

        self.assertEqual(ym(Monthdelta('29')), (2, 5))
        self.assertEqual(ym(Monthdelta('-11')), (-1, 1))
        self.assertEqual(ym(Monthdelta('-1year1month')), (-1, 1))
        self.assertEqual(ym(Monthdelta('1 year 11 month')), (1, 11))
        self.assertEqual(ym(Monthdelta('12 years')), (12, 0))
        self.assertEqual(ym(Monthdelta('11 months')), (0, 11))
        self.assertEqual(ym(Monthdelta('-11 months')), (-1, 1))
        self.assertRaises(ValueError, Monthdelta, 't months')

        self.assertEqual(pickle.loads(pickle.dumps(Monthdelta(30, 10))), Monthdelta(30, 10))

        self.assertEqual(str(Monthdelta(0)), '0')
        self.assertEqual(str(Monthdelta(1, 0)), '1 year')
        self.assertEqual(str(Monthdelta(years=-5, months=0)), '-5 years')
        self.assertEqual(str(Monthdelta(years=-1, months=1)), '-1 year 1 month')
        self.assertEqual(str(Monthdelta(months=1)), '1 month')
        self.assertEqual(str(Monthdelta(10)), '10 months')
        self.assertEqual(str(Monthdelta(years=20, months=3)), '20 years 3 months')
        self.assertEqual(repr(Monthdelta(years=20, months=3)), "Monthdelta('20 years 3 months')")

        self.assertEqual(Monthdelta(20, 11), Monthdelta(19, 23))
        self.assertNotEqual(Monthdelta(20, 12), Monthdelta(19, 23))
        self.assertNotEqual(Monthdelta(20, 12), 10)
        self.assertGreater(Monthdelta(20, 12), Monthdelta(19, 23))
        self.assertGreaterEqual(Monthdelta(20, 11), Monthdelta(19, 23))
        self.assertGreaterEqual(Monthdelta(20, 12), Monthdelta(19, 23))
        self.assertLess(Monthdelta(19, 23), Monthdelta(20, 12))
        self.assertLessEqual(Monthdelta(19, 23), Monthdelta(20, 11))
        self.assertLessEqual(Monthdelta(19, 23), Monthdelta(20, 12))

        self.assertRaises(TypeError, lambda: Monthdelta(20, 12) > 'abcd')
        self.assertRaises(TypeError, lambda: Monthdelta(20, 12) >= 'abcd')
        self.assertRaises(TypeError, lambda: Monthdelta(20, 12) < 'abcd')
        self.assertRaises(TypeError, lambda: Monthdelta(20, 12) <= 'abcd')

        self.assertEqual(Monthdelta(-2, 7), -Monthdelta(1, 5))
        self.assertEqual(abs(Monthdelta(-2, 7)), Monthdelta(1, 5))
        self.assertEqual(Monthdelta(10, 2) + Monthdelta(5, 11), Monthdelta(16, 1))
        self.assertEqual(Monthdelta(10, 2) - Monthdelta(5, 11), Monthdelta(4, 3))
        self.assertEqual(datetime(2012, 3, 5) + Monthdelta(4, 7),
                         datetime(2016, 10, 5))
        self.assertEqual(Monthdelta(4, 7) + datetime(2012, 3, 5),
                         datetime(2016, 10, 5))
        self.assertEqual(Monthdelta(-2, 7) + datetime(2012, 3, 5),
                         datetime(2010, 10, 5))
        self.assertEqual(datetime(2012, 7, 12) - Monthdelta(4, 2),
                         datetime(2008, 5, 12))

        self.assertRaises(TypeError, lambda: Monthdelta(20, 12) + 'abcd')
        self.assertRaises(TypeError, lambda: 'abcd' + Monthdelta(20, 12))
        self.assertRaises(TypeError, lambda: Monthdelta(20, 12) - 'abcd')
        self.assertRaises(TypeError, lambda: 'abcd' - Monthdelta(20, 12))

        try:
            import pandas as pd
            self.assertEqual(pd.Timestamp(2012, 3, 5) + Monthdelta(4, 7),
                             pd.Timestamp(2016, 10, 5))
            self.assertEqual(Monthdelta(4, 7) + pd.Timestamp(2012, 3, 5),
                             pd.Timestamp(2016, 10, 5))
            self.assertEqual(Monthdelta(-2, 7) + pd.Timestamp(2012, 3, 5),
                             pd.Timestamp(2010, 10, 5))
            self.assertEqual(pd.Timestamp(2012, 7, 12) - Monthdelta(4, 2),
                             pd.Timestamp(2008, 5, 12))
        except ImportError:
            pass
