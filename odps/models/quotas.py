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

from .. import errors, serializers
from ..compat import six
from .core import Iterable
from .quota import Quota


class Quotas(Iterable):
    marker = serializers.XMLNodeField("Marker")
    max_items = serializers.XMLNodeField("MaxItems")
    quotas = serializers.XMLNodesReferencesField(Quota, "Quota")

    def _name(self):
        return "quotas"

    def _get(self, nickname, tenant_id=None):
        return Quota(
            client=self._client, parent=self, nickname=nickname, tenant_id=tenant_id
        )

    def get(self, nickname, tenant_id=None):
        return self._get(nickname, tenant_id=tenant_id)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            quota = self._get(item)
        elif isinstance(item, Quota):
            quota = item
        else:
            return False

        try:
            quota.reload()
            return True
        except errors.NoSuchObject:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self, region_id=None, **kw):
        params = kw.copy()
        params.update(
            {
                "expectmarker": "true",
                "version": Quota.VERSION,
                "project": self._client.project,
            }
        )
        if region_id is not None:
            params["region"] = region_id

        def _it():
            last_marker = params.get("marker")
            if "marker" in params and (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource()
            resp = self._client.get(url, params=params)

            f = Quotas.parse(self._client, resp, obj=self)
            params["marker"] = f.marker

            return f.quotas

        while True:
            quotas = _it()
            if quotas is None:
                break
            for quota in quotas:
                yield quota
