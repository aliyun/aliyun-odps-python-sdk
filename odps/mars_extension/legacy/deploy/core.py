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

import json
import io
import logging
import os
import tempfile
import time
import uuid
import socket
import sys

from mars.actors.pool.gevent_pool import ActorPool
from mars.deploy.kubernetes.core import K8SPodsIPWatcher

logger = logging.getLogger(__name__)


class CupidK8SPodsIPWatcher(K8SPodsIPWatcher):
    _pod_to_port = dict()

    @property
    def cupid_kv(self):
        if not hasattr(self, "_cupid_kv"):
            from cupid import context

            self._cupid_kv = context().kv_store()
        return self._cupid_kv

    def _extract_pod_name_ep(self, pod_data):
        from cupid.runtime import RuntimeContext

        pod_name = pod_data["metadata"]["name"]
        if not RuntimeContext.is_context_ready():
            logger.debug("Cupid context not ready, pod name: {}".format(pod_name))
            return pod_name, None

        if self._pod_to_port.get(pod_name):
            pod_port = self._pod_to_port[pod_name]
        else:
            pod_kv_data = self.cupid_kv.get(pod_name)
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

    def _get_pod_to_ep(self, service_type=None):
        if service_type is not None:
            query = (
                self._pool.spawn(
                    self._client.list_namespaced_pod,
                    namespace=self._k8s_namespace,
                    label_selector=self._get_label_selector(service_type),
                )
                .result()
                .to_dict()
            )
        else:
            query = (
                self._pool.spawn(
                    self._client.list_namespaced_pod,
                    namespace=self._k8s_namespace,
                    label_selector=self._get_label_selector(),
                )
                .result()
                .to_dict()
            )
        result = dict()
        for el in query["items"]:
            name, pod_ep = self._extract_pod_name_ep(el)
            if pod_ep is not None and not self._extract_pod_ready(el):
                pod_ep = None
            if el["status"]["phase"] == "Running":
                result[name] = pod_ep
        return result

    @staticmethod
    def _extract_pod_ready(obj_data):
        if obj_data["status"]["phase"] != "Running":
            return False
        return True


class CupidActorPool(ActorPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cupid_context = None
        self._env_path = tempfile.mkdtemp(prefix="mars-pool-")
        self._channel_file = dict()

    def _prepare_process_channel(self, idx):
        envs = self._cupid_context.prepare_channel()
        envs_dict = dict((env.name, env.value) for env in envs)
        with open(self._channel_file[idx], "w") as env_file:
            env_file.write(json.dumps(envs_dict))

    def run(self):
        if self.processes:
            return

        super().run()

        from cupid import context

        self._cupid_context = context()

        kvstore = self._cupid_context.kv_store()
        advertise_endpoint = (
            self.advertise_address.split(":")[0] + ":" + self.address.split(":")[-1]
        )
        kvstore[os.environ["MARS_K8S_POD_NAME"]] = json.dumps(
            dict(endpoint=advertise_endpoint)
        )
        logger.debug(
            "Endpoint %s written to %s",
            advertise_endpoint,
            os.environ["MARS_K8S_POD_NAME"],
        )

        for idx in range(len(self.processes)):
            self._prepare_process_channel(idx)

    def stop(self):
        import shutil

        shutil.rmtree(self._env_path)
        if self._cupid_context is not None:
            self._cupid_context.channel_client.stop()
        super().stop()

    def pre_process_start(self, idx):
        self._channel_file[idx] = os.path.join(self._env_path, str(uuid.uuid4()))

    def post_process_start_child(self, idx):
        try:
            # Patch import here.
            # The reason is that tensorflow relies on protobuf 3+,
            # meanwhile, cupid channel relies on protobuf 2.4,
            # however, when cupid channel started below,
            # tensorflow will recognize the old version of protobuf
            # even when we set LD_LIBRARY_PATH,
            # so we import tensorflow in advance to prevent from potential crash.
            import tensorflow  # noqa: F401
        except ImportError:
            pass

        # set STDOUT to unbuffer mode
        sys.stdout = io.TextIOWrapper(
            open(sys.stdout.fileno(), "wb", 0), write_through=True
        )

        while not os.path.exists(self._channel_file[idx]):
            time.sleep(1)
        try:
            with open(self._channel_file[idx], "r") as env_file:
                envs = json.loads(env_file.read())
        except:
            time.sleep(1)
            with open(self._channel_file[idx], "r") as env_file:
                envs = json.loads(env_file.read())

        from cupid import context

        os.environ.update(envs)
        _proc_cupid_context = context()  # noqa: F841
        odps_envs = {
            "ODPS_BEARER_TOKEN": os.environ["BEARER_TOKEN_INITIAL_VALUE"],
            "ODPS_ENDPOINT": os.environ["ODPS_RUNTIME_ENDPOINT"],
        }
        os.environ.update(odps_envs)
        logger.info("Started channel for process index %s.", idx)


class CupidServiceMixin:
    def create_scheduler_discoverer(self):
        try:
            self.scheduler_discoverer = CupidK8SPodsIPWatcher(
                label_selector="name=marsscheduler"
            )
        except TypeError:
            self.scheduler_discoverer = CupidK8SPodsIPWatcher()
        finally:
            os.environ.pop("KUBE_API_ADDRESS", None)

    def create_pool(self, *args, **kwargs):
        kwargs["pool_cls"] = CupidActorPool
        return super().create_pool(*args, **kwargs)

    def parse_args(self, parser, argv, environ=None):
        args = super().parse_args(parser, argv, environ)
        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            args.advertise = socket.gethostbyname(socket.gethostname())
        return args
