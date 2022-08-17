#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import base64
import json
import logging
import time
import requests
import warnings

from mars import new_session

from ....utils import write_log
from ....models import Instance
from ....config import options
from .... import errors
from ...utils import get_default_resource_files, build_mars_image_name

NOTEBOOK_NAME = "MarsNotebook"
CUPID_APP_NAME = "MarsWeb"
GS_COORDINATOR_NAME = "GSCoordinator"
DEFAULT_RESOURCES = ["pymars-0.9.0", "pyodps-0.11.2", "pyarrow-4.0.0"]

logger = logging.getLogger(__name__)


class MarsCupidClient:
    def __init__(self, odps, inst=None, project=None):
        from cupid import CupidSession

        self._odps = odps
        self._cupid_session = CupidSession(odps, project=project)
        self._kube_instance = inst
        self._kube_url = None
        self._kube_client = None
        self._kube_namespace = None

        self._supervisor_key = None
        self._supervisor_config = None
        self._worker_config = None
        self._endpoint = None
        self._with_notebook = False
        self._notebook_endpoint = None
        self._with_graphscope = False
        self._graphscope_endpoint = None

        self._mars_session = None
        self._req_session = None

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def notebook_endpoint(self):
        return self._notebook_endpoint

    @property
    def session(self):
        return self._mars_session

    @property
    def instance_id(self):
        return self._kube_instance.id

    def rewrite_mars_request(self, request):
        from ....rest import default_user_agent

        request.headers["User-Agent"] = request.user_agent = default_user_agent()
        if "Content-Type" not in request.headers:
            request.headers["Content-Type"] = "application/octet-stream"

        if options.mars.use_common_proxy:
            if self._odps.project is not None:
                project_query = "curr_project=" + self._odps.project
                if "?" in request.url:
                    request.url += "&" + project_query
                else:
                    request.url += "?" + project_query
            account = self._req_session.account
            account.sign_request(request, self._endpoint)
        return request

    def submit(
        self,
        image=None,
        supervisor_num=1,
        supervisor_cpu=4,
        supervisor_mem=16 * 1024**3,
        worker_num=1,
        worker_cpu=8,
        worker_mem=32 * 1024**3,
        worker_cache_mem=None,
        min_worker_num=None,
        worker_disk_num=1,
        worker_disk_size=100 * 1024**3,
        with_notebook=False,
        notebook_cpu=1,
        notebook_mem=2 * 1024**3,
        with_graphscope=False,
        coordinator_cpu=1,
        coordinator_mem=2 * 1024**3,
        timeout=None,
        extra_env=None,
        extra_modules=None,
        resources=None,
        create_session=True,
        priority=None,
        running_cluster=None,
        task_name=None,
        **kw
    ):
        try:
            async_ = kw.pop("async_", None)

            # compatible with early version
            mars_image = kw.pop("mars_image", None)
            default_resources = kw.pop(
                "default_resources", None
            ) or get_default_resource_files(DEFAULT_RESOURCES)
            instance_idle_timeout = kw.pop("instance_idle_timeout", None)
            node_blacklist = kw.pop("node_blacklist", None)
            if with_notebook is not None:
                self._with_notebook = bool(with_notebook)
            else:
                self._with_notebook = options.mars.launch_notebook
            if with_graphscope is not None:
                self._with_graphscope = bool(with_graphscope)
            if self._kube_instance is None:
                image = image or build_mars_image_name(mars_image)

                extra_modules = extra_modules or []
                if isinstance(extra_modules, (tuple, list)):
                    extra_modules = list(extra_modules) + ["odps.mars_extension"]
                else:
                    extra_modules = [extra_modules, "odps.mars_extension"]

                if resources is not None:
                    if isinstance(resources, (tuple, list)):
                        resources = list(resources)
                        resources.extend(default_resources)
                    else:
                        resources = [resources] + default_resources
                else:
                    resources = default_resources

                if worker_cache_mem is None:
                    worker_cache_mem = int(worker_mem * 0.4)
                else:
                    worker_cache_mem = worker_cache_mem

                cluster_args = dict(
                    image=image,
                    supervisor_num=supervisor_num,
                    supervisor_cpu=supervisor_cpu,
                    supervisor_mem=supervisor_mem,
                    worker_num=worker_num,
                    worker_cpu=worker_cpu,
                    worker_mem=worker_mem,
                    worker_cache_mem=worker_cache_mem,
                    min_worker_num=min_worker_num,
                    worker_disk_num=worker_disk_num,
                    worker_disk_size=worker_disk_size,
                    with_notebook=with_notebook,
                    notebook_cpu=notebook_cpu,
                    notebook_mem=notebook_mem,
                    with_graphscope=with_graphscope,
                    coordinator_cpu=coordinator_cpu,
                    coordinator_mem=coordinator_mem,
                    extra_env=extra_env,
                    extra_modules=extra_modules,
                    node_blacklist=node_blacklist,
                    instance_idle_timeout=instance_idle_timeout,
                    timeout=timeout,
                )

                command = "/srv/entrypoint.sh %s %s" % (
                    __name__.rsplit(".", 1)[0] + ".app",
                    base64.b64encode(json.dumps(cluster_args).encode()).decode(),
                )

                self._kube_instance = self._cupid_session.start_kubernetes(
                    async_=True,
                    running_cluster=running_cluster,
                    priority=priority,
                    app_image=build_mars_image_name(),
                    app_command=command,
                    resources=resources,
                    task_name=task_name,
                    **kw
                )
                write_log(self._kube_instance.get_logview_address())
            if async_:
                return self
            else:
                self.wait_for_success(
                    create_session=create_session,
                    min_worker_num=min_worker_num or worker_num,
                )
                return self

        except KeyboardInterrupt:
            self.stop_server()
            return self

    def check_service_ready(self, timeout=1):
        try:
            resp = self._req_session.get(self._endpoint + "/api", timeout=timeout)
        except (requests.ConnectionError, requests.Timeout, errors.ODPSError):
            return False
        if resp.status_code >= 400:
            return False
        elif b"not found" in resp.content:
            self._endpoint = None
            return False
        return True

    def count_workers(self):
        resp = self._req_session.get(
            self._endpoint + "/api/cluster/nodes?role=1", timeout=1
        )
        json_result = json.loads(resp.text)
        return len(json_result["nodes"])

    def rescale_workers(self, new_scale, min_workers=None, wait=True, timeout=None):
        self._mars_session._sess.rescale_workers(
            new_scale, min_workers=min_workers, wait=wait, timeout=timeout
        )

    def get_logview_address(self):
        return self._kube_instance.get_logview_address()

    def get_mars_endpoint(self):
        return self._cupid_session.get_proxied_url(
            self._kube_instance.id, CUPID_APP_NAME
        )

    def get_notebook_endpoint(self):
        return self._cupid_session.get_proxied_url(
            self._kube_instance.id, NOTEBOOK_NAME
        )

    def get_graphscope_endpoint(self):
        return self._cupid_session.get_proxied_url(
            self._kube_instance.id, GS_COORDINATOR_NAME
        )

    def get_req_session(self):
        from ....rest import RestClient

        if options.mars.use_common_proxy:
            return RestClient(self._odps.account, self._endpoint, self._odps.project)
        else:
            return requests.Session()

    def check_instance_status(self):
        if self._kube_instance.is_terminated():
            for task_name, task in (self._kube_instance.get_task_statuses()).items():
                exc = None
                if task.status == Instance.Task.TaskStatus.FAILED:
                    exc = errors.parse_instance_error(
                        self._kube_instance.get_task_result(task_name)
                    )
                elif task.status != Instance.Task.TaskStatus.SUCCESS:
                    exc = errors.ODPSError(
                        "%s, status=%s" % (task_name, task.status.value)
                    )
                if exc:
                    exc.instance_id = self._kube_instance.id
                    raise exc

    def _post_pyodps_api(self, **data):
        r = self._req_session.post(
            self._endpoint.rstrip("/") + "/api/pyodps", data=data
        )
        try:
            r.raise_for_status()
        except errors.InvalidStateSetting:
            if not self._kube_instance.is_successful():
                raise

    def wait_for_success(self, min_worker_num=0, create_session=True):
        while True:
            self.check_instance_status()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    if self._endpoint is None:
                        self._endpoint = self.get_mars_endpoint()
                        write_log("Mars UI: " + self._endpoint)
                        self._req_session = self.get_req_session()

                        self._post_pyodps_api(
                            action="write_log",
                            content="Mars UI from client: " + self._endpoint,
                        )
                    if self._with_notebook and self._notebook_endpoint is None:
                        self._notebook_endpoint = self.get_notebook_endpoint()
                        write_log("Notebook UI: " + self._notebook_endpoint)

                        self._post_pyodps_api(
                            action="write_log",
                            content="Notebook UI from client: "
                            + self._notebook_endpoint,
                        )
                    if self._with_graphscope and self._graphscope_endpoint is None:
                        self._graphscope_endpoint = self.get_graphscope_endpoint()
                        write_log("Graphscope endpoint: " + self._graphscope_endpoint)

                        self._post_pyodps_api(
                            action="write_log",
                            content="Graphscope endpoint from client: "
                            + self._graphscope_endpoint,
                        )
                except KeyboardInterrupt:
                    raise
                except:
                    time.sleep(1)
                    continue

                try:
                    if not self.check_service_ready():
                        continue
                    if self.count_workers() >= min_worker_num:
                        break
                except:
                    continue
                finally:
                    time.sleep(1)

        if create_session:
            try:
                self._mars_session = new_session(
                    self._endpoint, request_rewriter=self.rewrite_mars_request
                ).as_default()
            except KeyboardInterrupt:
                raise
            except:
                if (
                    self._kube_instance
                    and self._kube_instance.status == self._kube_instance.Status.RUNNING
                ):
                    self._kube_instance.stop()
                raise

    def restart_session(self):
        self._mars_session.close()
        self._mars_session = new_session(
            self._endpoint, request_rewriter=self.rewrite_mars_request
        ).as_default()

    def stop_server(self):
        if not self._kube_instance:
            return

        try:
            self._post_pyodps_api(action="terminate", message="Stopped at client side")
            self._kube_instance.wait_for_completion(
                timeout=options.mars.container_status_timeout
            )
        except BaseException:
            if not self._kube_instance.is_terminated():
                try:
                    self._kube_instance.stop()
                except errors.InvalidStateSetting:
                    pass
        finally:
            self._kube_instance = None
