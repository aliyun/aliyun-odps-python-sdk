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

import pytest

from ..ext_types import Monthdelta


def test_monthdelta():
    ym = lambda md: (md.years, md.months)

    assert Monthdelta(years=2, months=5).total_months() == 29
    assert int(Monthdelta(years=2, months=5)) == 29
    if sys.version_info[0] <= 2:
        assert long(Monthdelta(years=2, months=5)) == 29

    assert ym(Monthdelta(12)) == (1, 0)
    assert ym(Monthdelta(26)) == (2, 2)
    assert ym(Monthdelta(-11)) == (-1, 1)

    assert ym(Monthdelta('29')) == (2, 5)
    assert ym(Monthdelta('-11')) == (-1, 1)
    assert ym(Monthdelta('-1year1month')) == (-1, 1)
    assert ym(Monthdelta('1 year 11 month')) == (1, 11)
    assert ym(Monthdelta('12 years')) == (12, 0)
    assert ym(Monthdelta('11 months')) == (0, 11)
    assert ym(Monthdelta('-11 months')) == (-1, 1)
    pytest.raises(ValueError, Monthdelta, 't months')

    assert pickle.loads(pickle.dumps(Monthdelta(30, 10))) == Monthdelta(30, 10)

    assert str(Monthdelta(0)) == '0'
    assert str(Monthdelta(1, 0)) == '1 year'
    assert str(Monthdelta(years=-5, months=0)) == '-5 years'
    assert str(Monthdelta(years=-1, months=1)) == '-1 year 1 month'
    assert str(Monthdelta(months=1)) == '1 month'
    assert str(Monthdelta(10)) == '10 months'
    assert str(Monthdelta(years=20, months=3)) == '20 years 3 months'
    assert repr(Monthdelta(years=20, months=3)) == "Monthdelta('20 years 3 months')"

    assert Monthdelta(20, 11) == Monthdelta(19, 23)
    assert Monthdelta(20, 12) != Monthdelta(19, 23)
    assert Monthdelta(20, 12) != 10
    assert Monthdelta(20, 12) > Monthdelta(19, 23)
    assert Monthdelta(20, 11) >= Monthdelta(19, 23)
    assert Monthdelta(20, 12) >= Monthdelta(19, 23)
    assert Monthdelta(19, 23) < Monthdelta(20, 12)
    assert Monthdelta(19, 23) <= Monthdelta(20, 11)
    assert Monthdelta(19, 23) <= Monthdelta(20, 12)

    pytest.raises(TypeError, lambda: Monthdelta(20, 12) > 'abcd')
    pytest.raises(TypeError, lambda: Monthdelta(20, 12) >= 'abcd')
    pytest.raises(TypeError, lambda: Monthdelta(20, 12) < 'abcd')
    pytest.raises(TypeError, lambda: Monthdelta(20, 12) <= 'abcd')

    assert Monthdelta(-2, 7) == -Monthdelta(1, 5)
    assert abs(Monthdelta(-2, 7)) == Monthdelta(1, 5)
    assert Monthdelta(10, 2) + Monthdelta(5, 11) == Monthdelta(16, 1)
    assert Monthdelta(10, 2) - Monthdelta(5, 11) == Monthdelta(4, 3)
    assert datetime(2012, 3, 5) + Monthdelta(4, 7) == datetime(2016, 10, 5)
    assert Monthdelta(4, 7) + datetime(2012, 3, 5) == datetime(2016, 10, 5)
    assert Monthdelta(-2, 7) + datetime(2012, 3, 5) == datetime(2010, 10, 5)
    assert datetime(2012, 7, 12) - Monthdelta(4, 2) == datetime(2008, 5, 12)

    pytest.raises(TypeError, lambda: Monthdelta(20, 12) + 'abcd')
    pytest.raises(TypeError, lambda: 'abcd' + Monthdelta(20, 12))
    pytest.raises(TypeError, lambda: Monthdelta(20, 12) - 'abcd')
    pytest.raises(TypeError, lambda: 'abcd' - Monthdelta(20, 12))

    try:
        import pandas as pd
        assert pd.Timestamp(2012, 3, 5) + Monthdelta(4, 7) == pd.Timestamp(2016, 10, 5)
        assert Monthdelta(4, 7) + pd.Timestamp(2012, 3, 5) == pd.Timestamp(2016, 10, 5)
        assert Monthdelta(-2, 7) + pd.Timestamp(2012, 3, 5) == pd.Timestamp(2010, 10, 5)
        assert pd.Timestamp(2012, 7, 12) - Monthdelta(4, 2) == pd.Timestamp(2008, 5, 12)
    except ImportError:
        pass
