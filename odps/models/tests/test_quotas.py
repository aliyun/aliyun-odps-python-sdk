# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import pytest


def test_quotas(odps_daily):
    odps = odps_daily

    with pytest.raises(TypeError):
        odps.get_quota()

    first_quota_nick = next(odps.list_quotas())
    assert first_quota_nick.nickname is not None

    assert not odps.exist_quota("non_exist_quota")
    assert odps.exist_quota(first_quota_nick)
    assert odps.exist_quota(first_quota_nick.nickname)

    quota_obj = odps.get_quota(first_quota_nick.nickname)
    quota_obj.reload()
    assert quota_obj.nickname == first_quota_nick.nickname
