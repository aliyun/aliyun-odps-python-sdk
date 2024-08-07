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

from ....models import Table


class MockProject(object):
    def __init__(self):
        self.name = 'mocked_project'


class MockTable(Table):
    __slots__ = "_mock_project",

    def __init__(self, **kwargs):
        from ....core import ODPS

        super(MockTable, self).__init__(**kwargs)

        self._loaded = True
        self._mock_project = MockProject()
        if getattr(self, "_client", None) is not None:
            client = self._client
            odps_entry = ODPS(
                account=client._account, project=client.project,
                endpoint=client.endpoint, schema=client.schema,
            )
            self._mock_project.odps = odps_entry

    @property
    def project(self):
        return self._mock_project

