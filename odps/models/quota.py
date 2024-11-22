#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from .. import serializers
from ..compat import Enum
from .core import LazyLoad

_MCQA_VERSION = "mcqaVersion"


class Quota(LazyLoad):
    """
    Quota provides information about computational resources.
    """

    VERSION = "wlm"

    __slots__ = ("_mcqa_conn_header",)
    _root = "Quota"

    class Strategy(Enum):
        NoPreempt = "NoPreempt"
        Preempt = "Preempt"

    class SchedulerType(Enum):
        Fifo = "Fifo"
        Fair = "Fair"

    class Status(Enum):
        ON = "ON"
        OFF = "OFF"
        INITIALIZING = "INITIALIZING"
        ABNORMAL = "ABNORMAL"

    class ResourceSystemType(Enum):
        FUXI_OFFLINE = "FUXI_OFFLINE"
        FUXI_ONLINE = "FUXI_ONLINE"
        FUXI_STANDALONE = "FUXI_STANDALONE"
        FUXI_VW = "FUXI_VW"

        UNKNOWN = "UNKNOWN"

        @classmethod
        def _missing_(cls, _value):
            return cls.UNKNOWN

    class BillingPolicy(serializers.JSONSerializableModel):
        class BillingMethod(Enum):
            payasyougo = "payasyougo"
            subscription = "subscription"

        billing_method = serializers.JSONNodeField(
            "billingMethod", parse_callback=serializers.none_or(BillingMethod)
        )
        specification = serializers.JSONNodeField("OdpsSpecCode")
        order_id = serializers.JSONNodeField("orderId")

    cluster = serializers.XMLNodeField("Cluster")
    name = serializers.XMLNodeField("Name")
    id = serializers.XMLNodeField("ID")
    is_enabled = serializers.XMLNodeField("IsEnabled", type="bool")
    resource_system_type = serializers.XMLNodeField(
        "ResourceSystemType", parse_callback=serializers.none_or(ResourceSystemType)
    )
    session_service_name = serializers.XMLNodeField("SessionServiceName")
    creation_time = serializers.XMLNodeField("CreateTimeMs", type="timestamp_ms")
    last_modified_time = serializers.XMLNodeField(
        "LastModifiedTimeMs", type="timestamp_ms"
    )

    cpu = serializers.XMLNodeField("CPU", type="int")
    min_cpu = serializers.XMLNodeField("MinCPU", type="int")
    elastic_cpu_max = serializers.XMLNodeField("ElasticCPUMax", type="int")
    elastic_cpu_min = serializers.XMLNodeField("ElasticCPUMin", type="int")
    adhoc_cpu = serializers.XMLNodeField("AdhocCPU", type="int")
    cpu_usage = serializers.XMLNodeField("CPUUsage", type="float")
    adhoc_cpu_usage = serializers.XMLNodeField("AdhocCPUUsage", type="float")
    cpu_ready_ratio = serializers.XMLNodeField("CPUReadyRatio", type="float")

    memory = serializers.XMLNodeField("Memory", type="int")
    min_memory = serializers.XMLNodeField("MinMemory", type="int")
    elastic_memory_max = serializers.XMLNodeField("ElasticMemoryMax", type="int")
    elastic_memory_min = serializers.XMLNodeField("ElasticMemoryMin", type="int")
    adhoc_memory = serializers.XMLNodeField("AdhocMemory", type="int")
    memory_usage = serializers.XMLNodeField("MemoryUsage", type="float")
    adhoc_memory_usage = serializers.XMLNodeField("AdhocMemoryUsage", type="float")
    memory_ready_ratio = serializers.XMLNodeField("MemoryReadyRatio", type="float")

    gpu = serializers.XMLNodeField("GPU", type="int")
    min_gpu = serializers.XMLNodeField("MinGPU", type="int")
    elastic_gpu_max = serializers.XMLNodeField("ElasticGPUMax", type="int")
    elastic_gpu_min = serializers.XMLNodeField("ElasticGPUMin", type="int")
    adhoc_gpu = serializers.XMLNodeField("AdhocGPU", type="int")

    strategy = serializers.XMLNodeField(
        "Strategy", parse_callback=serializers.none_or(Strategy)
    )
    scheduler_type = serializers.XMLNodeField("SchedulerType")
    is_parent_group = serializers.XMLNodeField("IsParGroup", type="bool")
    parent_id = serializers.XMLNodeField("ParGroupId")
    parent_name = serializers.XMLNodeField("ParentName")
    user_defined_tags = serializers.XMLNodePropertiesField(
        "UserDefinedTag", "entry", key_attr="key", value_tag="value"
    )
    virtual_cluster_config = serializers.XMLNodeField(
        "VirtualClusterConfig", type="json"
    )
    tenant_id = serializers.XMLNodeField("TenantId")
    status = serializers.XMLNodeField(
        "Status", parse_callback=serializers.none_or(Status)
    )
    nickname = serializers.XMLNodeField("Nickname")
    parent_nickname = serializers.XMLNodeField("ParentNickname")
    creator_id = serializers.XMLNodeField("CreatorId")
    region_id = serializers.XMLNodeField("Region")
    billing_policy = serializers.XMLNodeReferenceField(BillingPolicy, "BillingPolicy")
    need_auth = serializers.XMLNodeField("NeedAuth", type="bool")
    is_pure_link = serializers.XMLNodeField("IsPureLink", type="bool")
    quota_version = serializers.XMLNodeField("QuotaVersion")
    is_meta_only = serializers.XMLNodeField("IsMetaOnly", type="bool")
    properties = serializers.XMLNodeField("Properties", type="json")

    def __init__(self, *args, **kwds):
        super(Quota, self).__init__(*args, **kwds)
        self._mcqa_conn_header = None

    def _name(self):
        return self._getattr("nickname")

    @property
    def mcqa_conn_header(self):
        if not self._loaded:
            self.reload()
        return self._mcqa_conn_header

    def reload(self):
        params = {
            "project": self._client.project,
            "version": self.VERSION,
        }
        try:
            if self._getattr("region_id"):
                params["region"] = self.region_id
        except AttributeError:
            pass
        try:
            if self._getattr("tenant_id"):
                params["tenant"] = self.tenant_id
        except AttributeError:
            pass
        resp = self._client.get(self.resource(), params=params)
        self.parse(self._client, resp, obj=self)
        self._mcqa_conn_header = resp.headers.get("x-odps-mcqa-conn")
        self._loaded = True

    def is_interactive_quota(self):
        if self.resource_system_type != Quota.ResourceSystemType.FUXI_VW:
            return False
        return (
            self.user_defined_tags is None
            or _MCQA_VERSION not in self.user_defined_tags
        )
