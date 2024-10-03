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

from ... import compat, serializers
from .core import Task


class CopyDataSource(serializers.XMLSerializableModel):
    class CopyDirection(compat.Enum):
        IMPORT = "IMPORT"
        EXPORT = "EXPORT"

    copy_type = serializers.XMLNodeField("Type")
    project = serializers.XMLNodeField("Project")
    table = serializers.XMLNodeField("Table")
    partition = serializers.XMLNodeField("Partition")

    def __init__(self, direction=None, **kw):
        kw["type"] = (
            "Destination" if direction == self.CopyDirection.IMPORT else "Source"
        )
        super(CopyDataSource, self).__init__(**kw)


class LocalCopyDataSource(CopyDataSource):
    _root = "Local"


class TunnelCopyDataSource(CopyDataSource):
    version = serializers.XMLNodeField("Version", default="1")
    endpoint = serializers.XMLNodeField("EndPoint")
    odps_endpoint = serializers.XMLNodeField("OdpsEndPoint")
    signature = serializers.XMLNodeField("Signature")
    application_signature = serializers.XMLNodeField("ApplicationSignature")
    signature_type = serializers.XMLNodeField("SignatureType")


class MappingItem(serializers.XMLSerializableModel):
    _root = "MappingItem"

    src = serializers.XMLNodeField("SourceColumn")
    dest = serializers.XMLNodeField("DestColumn")


class CopyTask(Task):
    _root = "COPY"

    local = serializers.XMLNodeReferenceField(LocalCopyDataSource, "Local")
    tunnel = serializers.XMLNodeReferenceField(TunnelCopyDataSource, "Tunnel")
    _mapping_items = serializers.XMLNodesReferencesField(MappingItem, "MappingItems")
    mode = serializers.XMLNodeField("Mode")
    job_instance_number = serializers.XMLNodeField("JobInstanceNumber")

    @property
    def mapping_items(self):
        return self._mapping_items or []

    @mapping_items.setter
    def mapping_items(self, value):
        self._mapping_items = value
