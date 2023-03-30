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

from odps.models.tenant import Tenant
from odps.tests.core import TestBase


class Test(TestBase):
    def testTenantProps(self):
        tenant = Tenant(client=self.odps_with_schema.rest)
        assert not tenant._getattr("_loaded")
        assert tenant.name is not None
        assert tenant._getattr("_loaded")
        assert tenant.get_parameter("prop-key") is None
