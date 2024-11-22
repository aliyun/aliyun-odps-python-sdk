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

import re

from ...rest import RestClient

_inst_url_regex = re.compile("/instances(?:$|/)")


class McqaV2RestClient(RestClient):
    def __init__(self, *args, **kwargs):
        self._conn_header = kwargs.pop("conn_header", None)
        self._query_cookie = None

        super(McqaV2RestClient, self).__init__(*args, **kwargs)

    def _request(self, url, method, stream=False, **kwargs):
        headers = kwargs.pop("headers", None) or dict()
        if self._conn_header:
            headers["x-odps-mcqa-conn"] = self._conn_header
        if self._query_cookie is not None:
            headers["x-odps-mcqa-query-cookie"] = self._query_cookie
        kwargs["headers"] = headers
        if _inst_url_regex.search(url):
            url = self.endpoint + "/mcqa" + url[len(self.endpoint) :]
        return super(McqaV2RestClient, self)._request(url, method, stream, **kwargs)

    def is_ok(self, resp):
        if "x-odps-mcqa-query-cookie" in resp.headers:
            self._query_cookie = resp.headers["x-odps-mcqa-query-cookie"]
        return super(McqaV2RestClient, self).is_ok(resp)


class McqaV2Methods(object):
    @classmethod
    def _patch_odps(cls, odps):
        odps._quota_to_mcqa_odps = getattr(odps, "_quota_to_mcqa_odps", {})

    @classmethod
    def _load_mcqa_conn(cls, odps, quota_name):
        from ...core import ODPS

        cls._patch_odps(odps)
        if quota_name in odps._quota_to_mcqa_odps:
            return odps._quota_to_mcqa_odps[quota_name]

        conn_header = odps.get_quota(
            quota_name, tenant_id=odps.default_tenant.tenant_id
        ).mcqa_conn_header

        odps._quota_to_mcqa_odps[quota_name] = ODPS(
            account=odps.account,
            project=odps.project,
            endpoint=odps.endpoint,
            rest_client_cls=McqaV2RestClient,
            rest_client_kwargs={"conn_header": conn_header},
        )
        return odps._quota_to_mcqa_odps[quota_name]

    @classmethod
    def run_sql_interactive(cls, odps, sql, hints=None, quota_name=None, **kwargs):
        cls._patch_odps(odps)
        mcqa_odps = cls._load_mcqa_conn(odps, quota_name)
        return mcqa_odps.run_sql(sql, hints=hints, **kwargs)

    @classmethod
    def execute_sql_interactive(cls, odps, sql, hints=None, quota_name=None, **kwargs):
        inst = cls.run_sql_interactive(
            odps, sql, hints=hints, quota_name=quota_name, **kwargs
        )
        inst.wait_for_success()
        return inst
