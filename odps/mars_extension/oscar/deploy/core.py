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

import asyncio
import json
import logging
import multiprocessing
import os
import shutil
import socket
import tempfile
import time
from typing import AsyncGenerator, List, Optional

from mars.deploy.kubernetes.config import MarsSupervisorsConfig
from mars.services.cluster import NodeRole
from mars.services.cluster.backends import (
    AbstractClusterBackend,
    register_cluster_backend,
)

from ..cupid_service import run_cupid_service, CupidServiceClient

logger = logging.getLogger(__name__)


_abs_methods = dict()
for name, method in vars(AbstractClusterBackend).items():
    if getattr(method, "__isabstractmethod__", False):
        _abs_methods[name] = None

_ClusterBackend = type("_ClusterBackend", (AbstractClusterBackend,), _abs_methods)


@register_cluster_backend
class CupidK8SClusterBackend(_ClusterBackend):
    name = "cupid_k8s"
    _pod_to_port = dict()

    def __init__(self, k8s_config=None, k8s_namespace=None):
        from kubernetes import client

        self._k8s_config = k8s_config

        verify_ssl = bool(int(os.environ.get("KUBE_VERIFY_SSL", "1").strip('"')))
        if not verify_ssl:
            c = client.Configuration()
            c.verify_ssl = False
            client.Configuration.set_default(c)

        self._k8s_namespace = (
            k8s_namespace or os.environ.get("MARS_K8S_POD_NAMESPACE") or "default"
        )
        self._full_label_selector = None
        self._client = client.CoreV1Api(client.ApiClient(self._k8s_config))

        self._service_pod_to_ep = dict()

    @classmethod
    async def create(
        cls, node_role: NodeRole, lookup_address: Optional[str], pool_address: str
    ) -> "AbstractClusterBackend":
        from kubernetes import config, client

        if os.environ.get("KUBE_API_ADDRESS"):
            k8s_config = client.Configuration()
            k8s_config.host = os.environ["KUBE_API_ADDRESS"]
            return cls(k8s_config, None)

        if lookup_address is None:
            k8s_namespace = None
            k8s_config = config.load_incluster_config()
        else:
            address_parts = lookup_address.rsplit("?", 1)
            k8s_namespace = None if len(address_parts) == 1 else address_parts[1]

            k8s_config = client.Configuration()
            if "://" in address_parts[0]:
                k8s_config.host = address_parts[0]
            else:
                config.load_kube_config(
                    address_parts[0], client_configuration=k8s_config
                )
        return cls(k8s_config, k8s_namespace)

    def __reduce__(self):
        return type(self), (self._k8s_config, self._k8s_namespace)

    @property
    def cupid_client(self):
        if not hasattr(self, "_cupid_client"):
            self._cupid_client = CupidServiceClient()
        return self._cupid_client

    async def _get_label_selector(self, service_type):
        if self._full_label_selector is not None:
            return self._full_label_selector

        selectors = [f"mars/service-type={service_type}"]
        if "MARS_K8S_GROUP_LABELS" in os.environ:
            group_labels = os.environ["MARS_K8S_GROUP_LABELS"].split(",")
            cur_pod_info = (
                await asyncio.to_thread(
                    self._client.read_namespaced_pod,
                    os.environ["MARS_K8S_POD_NAME"],
                    namespace=self._k8s_namespace,
                )
            ).to_dict()
            for label in group_labels:
                label_val = cur_pod_info["metadata"]["labels"][label]
                selectors.append(f"{label}={label_val}")
        self._full_label_selector = ",".join(selectors)
        logger.debug("Using pod selector %s", self._full_label_selector)
        return self._full_label_selector

    async def _extract_pod_name_ep(self, pod_data):
        pod_name = pod_data["metadata"]["name"]

        if self._pod_to_port.get(pod_name):
            pod_port = self._pod_to_port[pod_name]
        else:
            try:
                pod_kv_data = await asyncio.to_thread(
                    self.cupid_client.get_kv, pod_name
                )
            except OSError:
                logger.warning("Cupid server not ready.")
                return pod_name, None

            if pod_kv_data:
                pod_port = self._pod_to_port[pod_name] = json.loads(pod_kv_data)[
                    "endpoint"
                ].rsplit(":", 1)[-1]
                logger.debug(
                    "Get port from kvstore, name: {}, port: {}".format(
                        pod_name, pod_port
                    )
                )
            else:
                pod_port = None
                if pod_name not in self._pod_to_port:
                    logger.debug(
                        "Cannot get port from kvstore, name: {}".format(pod_name)
                    )
                    self._pod_to_port[pod_name] = None
        pod_endpoint = "%s:%s" % (pod_data["status"]["pod_ip"], pod_port)
        return pod_name, pod_endpoint if pod_port else None

    @staticmethod
    def _extract_pod_ready(obj_data):
        return obj_data["status"]["phase"] == "Running"

    async def _get_pod_to_ep(self, service_type: str, filter_ready: bool = False):
        query = (
            await asyncio.to_thread(
                self._client.list_namespaced_pod,
                namespace=self._k8s_namespace,
                label_selector=await self._get_label_selector(service_type),
                resource_version="0",
            )
        ).to_dict()

        result = dict()
        for el in query["items"]:
            name, pod_ep = await self._extract_pod_name_ep(el)
            if filter_ready and pod_ep is not None and not self._extract_pod_ready(el):
                pod_ep = None
            result[name] = pod_ep
        return result

    async def _get_endpoints_by_service_type(
        self, service_type: str, update: bool = False, filter_ready: bool = True
    ):
        if not self._service_pod_to_ep.get(service_type) or update:
            self._service_pod_to_ep[service_type] = await self._get_pod_to_ep(
                service_type, filter_ready=filter_ready
            )
        return sorted(
            a for a in self._service_pod_to_ep[service_type].values() if a is not None
        )

    async def _watch_service(self, service_type, linger=60):
        from urllib3.exceptions import ReadTimeoutError
        from kubernetes import watch

        cur_pods = set(
            await self._get_endpoints_by_service_type(service_type, update=True)
        )
        w = watch.Watch()

        pod_to_ep = self._service_pod_to_ep[service_type]
        while True:
            # when some pods are not ready, we refresh faster
            linger_seconds = linger() if callable(linger) else linger
            streamer = w.stream(
                self._client.list_namespaced_pod,
                namespace=self._k8s_namespace,
                label_selector=await self._get_label_selector(service_type),
                timeout_seconds=linger_seconds,
                resource_version="0",
            )
            while True:
                try:
                    event = await asyncio.to_thread(next, streamer, StopIteration)
                    if event is StopIteration:
                        # todo change this into a continuous watch
                        #  when watching in a master node is implemented
                        return
                except (ReadTimeoutError, StopIteration):
                    new_pods = set(
                        await self._get_endpoints_by_service_type(
                            service_type, update=True
                        )
                    )
                    if new_pods != cur_pods:
                        cur_pods = new_pods
                        yield await self._get_endpoints_by_service_type(
                            service_type, update=False
                        )
                    break
                except:  # noqa: E722  # pragma: no cover  # pylint: disable=bare-except
                    logger.exception("Unexpected error when watching on kubernetes")
                    break

                obj_dict = event["object"].to_dict()
                pod_name, endpoint = await self._extract_pod_name_ep(obj_dict)
                pod_to_ep[pod_name] = (
                    endpoint if endpoint and self._extract_pod_ready(obj_dict) else None
                )
                yield await self._get_endpoints_by_service_type(
                    service_type, update=False
                )

    def watch_supervisors(self) -> AsyncGenerator[List[str], None]:
        return self._watch_service(MarsSupervisorsConfig.rc_name)

    async def get_supervisors(self, filter_ready: bool = True) -> List[str]:
        return await self._get_endpoints_by_service_type(
            MarsSupervisorsConfig.rc_name,
            update=not filter_ready,
            filter_ready=filter_ready,
        )


class CupidCommandRunnerMixin:
    def fix_hyper_address(self):
        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            os.environ["MARS_CONTAINER_IP"] = socket.gethostbyname(socket.gethostname())

    def fix_protobuf_import(self):
        # The reason is that plasma and tensorflow relies on protobuf 3+,
        # meanwhile, cupid channel relies on protobuf 2.4,
        # however, when cupid channel started below,
        # tensorflow will recognize the old version of protobuf
        # even when we set LD_LIBRARY_PATH,
        # so we import tensorflow in advance to prevent from potential crash.
        from pyarrow import plasma

        try:
            import tensorflow
        except ImportError:
            tensorflow = None
        del plasma, tensorflow

    def start_cupid_service(self):
        self._env_path = tempfile.mkdtemp(prefix="mars-pool-")
        self._channel_file = os.path.join(
            self._env_path, "mars-cupid-channel-%s.json" % os.getpid()
        )
        self._cupid_sock_file = os.environ["CUPID_SERVICE_SOCKET"] = os.path.join(
            self._env_path, "mars-cupid-sock-%s.sock" % os.getpid()
        )

        self._cupid_service_proc = multiprocessing.Process(
            target=run_cupid_service, args=(self._channel_file,)
        )
        self._cupid_service_proc.start()

        from cupid import context

        self._cupid_context = context()

        envs = self._cupid_context.prepare_channel()
        envs_dict = dict((env.name, env.value) for env in envs)
        with open(self._channel_file, "w") as env_file:
            env_file.write(json.dumps(envs_dict))

        while not os.path.exists(self._cupid_sock_file):
            time.sleep(0.1)

    def stop_cupid_service(self):
        shutil.rmtree(self._env_path)
        if self._cupid_context is not None:
            self._cupid_context.channel_client.stop()
        if self._cupid_service_proc is not None:
            self._cupid_service_proc.terminate()

    def _register_application(self, app_name, internal_endpoint, app_endpoint):
        kvstore = self._cupid_context.kv_store()
        kvstore[app_name] = json.dumps(dict(endpoint=internal_endpoint))
        self._cupid_context.register_application(app_name, app_endpoint)

    async def register_application(self, app_name, internal_endpoint, app_endpoint):
        await asyncio.to_thread(
            self._register_application, app_name, internal_endpoint, app_endpoint
        )

    async def write_node_endpoint(self):
        cupid_client = CupidServiceClient(self._cupid_sock_file)
        content = json.dumps(dict(endpoint=self.args.endpoint))
        await asyncio.to_thread(
            cupid_client.put_kv, os.environ["MARS_K8S_POD_NAME"], content
        )

    async def start_readiness_server(self):
        pass

    async def stop_readiness_server(self):
        pass

    def _get_logging_config_paths(self):
        paths = super()._get_logging_config_paths()
        # make config file inside the package as the highest priority
        config_dir = os.path.dirname(self.get_default_config_file())
        return [os.path.join(config_dir, "logging.conf")] + paths
