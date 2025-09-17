# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from ... import errors, serializers
from ...compat import six
from ..core import JSONContainer
from .model import Model


class ModelVersions(JSONContainer):
    __slots__ = ("_model_name",)

    models = serializers.JSONNodesReferencesField(Model, "models")
    next_page_token = serializers.JSONNodeField("nextPageToken")

    def _get(self, item):
        if isinstance(item, Model):
            return item
        return Model(
            client=self._client, parent=self, name=self._model_name, version_name=item
        )

    def resource(self, client=None, endpoint=None, with_schema=False, url_prefix=None):
        return self._parent.resource(
            client=client,
            endpoint=endpoint,
            with_schema=with_schema,
            url_prefix=url_prefix,
        )

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            model = self._get(item)
        elif isinstance(item, Model):
            model = item
        else:
            return False

        try:
            model.reload()
            return True
        except errors.NoSuchObject:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self):
        params = {}

        def _it():
            last_marker = params.get("pageToken")
            if "pageToken" in params and (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource(with_schema=True) + ":listVersions"
            resp = self._client.get(url, params=params)

            t = ModelVersions.parse(
                self._client, resp, obj=self, parent=self.parent.parent
            )
            params["pageToken"] = t.next_page_token

            return t.models

        while True:
            models = _it()
            if models is None:
                break
            for model in models:
                yield model
