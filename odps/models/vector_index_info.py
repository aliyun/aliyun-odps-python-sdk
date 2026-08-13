# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from .. import serializers


class VectorIndexInfo(serializers.JSONSerializableModel):
    id = serializers.JSONNodeField("Id")
    name = serializers.JSONNodeField("Name")
    type = serializers.JSONNodeField("Type")
    properties = serializers.JSONNodeField("Properties")

    def __repr__(self):
        return (
            f"VectorIndexInfo(id={self.id!r}, name={self.name!r}, type={self.type!r})"
        )


def parse_vector_indexes(reserved):
    """Parse VectorIndexes from the Reserved JSON dict.

    Returns a list of VectorIndexInfo objects, or None if VectorIndexes is absent.
    Uses attributes dict for forward compatibility: new or missing fields
    will not affect parsing.
    """
    vector_indexes = reserved.get("VectorIndexes")
    if not vector_indexes or not isinstance(vector_indexes, list):
        return None
    result = []
    for item in vector_indexes:
        if not isinstance(item, dict):
            continue
        index = VectorIndexInfo.deserial(item)
        result.append(index)
    return result if result else None
