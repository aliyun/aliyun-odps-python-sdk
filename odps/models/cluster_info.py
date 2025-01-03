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

from .. import serializers
from ..compat import Enum, six


class ClusterType(Enum):
    HASH = "hash"
    RANGE = "range"


class ClusterSortOrder(Enum):
    ASC = "ASC"
    DESC = "DESC"


class ClusterSortCol(serializers.JSONSerializableModel):
    name = serializers.JSONNodeField("col")
    order = serializers.JSONNodeField(
        "order", parse_callback=lambda x: ClusterSortOrder(x.upper()) if x else None
    )


class ClusterInfo(serializers.JSONSerializableModel):
    cluster_type = serializers.JSONNodeField(
        "ClusterType", parse_callback=lambda x: ClusterType(x.lower()) if x else None
    )
    bucket_num = serializers.JSONNodeField("BucketNum")
    cluster_cols = serializers.JSONNodeField("ClusterCols")
    sort_cols = serializers.JSONNodesReferencesField(ClusterSortCol, "SortCols")

    @classmethod
    def deserial(cls, content, obj=None, **kw):
        res = super(ClusterInfo, cls).deserial(content, obj=obj, **kw)
        return res if res.cluster_type is not None else None

    def to_sql_clause(self):
        sio = six.StringIO()
        if self.cluster_type == ClusterType.RANGE:
            cluster_type_str = u"RANGE "
        else:
            cluster_type_str = u""
        cluster_cols = u", ".join(u"`%s`" % col for col in self.cluster_cols)
        sio.write("%sCLUSTERED BY (%s)" % (cluster_type_str, cluster_cols))
        if self.sort_cols:
            sort_cols = u", ".join(
                u"`%s` %s" % (c.name, c.order.value) for c in self.sort_cols
            )
            sio.write(u" SORTED BY (%s)" % sort_cols)
        if self.bucket_num:
            sio.write(" INTO %s BUCKETS" % self.bucket_num)
        return sio.getvalue()
