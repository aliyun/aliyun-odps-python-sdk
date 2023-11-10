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

import warnings
from datetime import datetime

from .. import serializers
from ..compat import enum, TimeoutError
from ..errors import ODPSError, InternalServerError
from .core import JSONRemoteModel


class Tenant(JSONRemoteModel):
    __slots__ = "_loaded",

    class State(enum.Enum):
        NORMAL = "NORMAL"
        SUSPEND = "SUSPEND"
        DELETING = "DELETING"
        DELETED = "DELETED"

    class Meta(JSONRemoteModel):
        name = serializers.JSONNodeField("Name", set_to_parent=True)
        owner_id = serializers.JSONNodeField("OwnerId", set_to_parent=True)
        tenant_id = serializers.JSONNodeField("TenantId", set_to_parent=True)
        tenant_state = serializers.JSONNodeField(
            "TenantState", parse_callback=lambda x: Tenant.State(x.upper()), set_to_parent=True
        )
        creation_time = serializers.JSONNodeField(
            "CreateTime", parse_callback=datetime.fromtimestamp, set_to_parent=True
        )
        last_modified_time = serializers.JSONNodeField(
            "UpdateTime", parse_callback=datetime.fromtimestamp, set_to_parent=True
        )
        tenant_properties = serializers.JSONNodeField("TenantMeta", set_to_parent=True)
        parameters = serializers.JSONNodeField("Parameters", set_to_parent=True)

        def __init__(self, **kwargs):
            kwargs.pop("name", None)
            super(Tenant.Meta, self).__init__(**kwargs)

    _meta = serializers.JSONNodeReferenceField(Meta, "Tenant")
    name = serializers.JSONNodeField("Name")
    owner_id = serializers.JSONNodeField("OwnerId")
    tenant_id = serializers.JSONNodeField("TenantId")
    tenant_state = serializers.JSONNodeField(
        "TenantState", parse_callback=lambda x: Tenant.State(x.upper())
    )
    creation_time = serializers.JSONNodeField(
        "CreateTime", parse_callback=datetime.fromtimestamp
    )
    last_modified_time = serializers.JSONNodeField(
        "UpdateTime", parse_callback=datetime.fromtimestamp
    )
    tenant_properties = serializers.JSONNodeField("TenantMeta")
    parameters = serializers.JSONNodeField("Parameters")

    def __init__(self, **kwargs):
        super(Tenant, self).__init__(**kwargs)
        self._loaded = False

    def _getattr(self, attr):
        return object.__getattribute__(self, attr)

    def __getattribute__(self, attr):
        val = object.__getattribute__(self, attr)
        if val is None and not self._getattr("_loaded"):
            fields = getattr(type(self), '__fields')
            if attr in fields:
                self.reload()
        return object.__getattribute__(self, attr)

    @property
    def create_time(self):
        warnings.warn(
            'Tenant.create_time is deprecated and will be replaced '
            'by Tenant.creation_time.',
            DeprecationWarning,
            stacklevel=3,
        )
        return self.creation_time

    def resource(self, client=None, endpoint=None):
        endpoint = endpoint if endpoint is not None else (client or self._client).endpoint
        return endpoint + "/tenants"

    def reload(self):
        try:
            resp = self._client.get(self.resource())
            self.parse(self._client, resp, obj=self)
        except ODPSError as ex:
            if isinstance(ex, (InternalServerError, TimeoutError)):
                raise
        self._loaded = True

    def get_parameter(self, key, default=None):
        try:
            return (self.parameters or {}).get(key, default)
        except ODPSError as ex:
            if isinstance(ex, (InternalServerError, TimeoutError)):
                raise
            return default
