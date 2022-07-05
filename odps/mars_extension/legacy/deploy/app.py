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

import argparse
import base64
import logging
import json
import os
import sys

from mars.deploy.kubernetes.client import KubernetesCluster, new_cluster
from mars.deploy.kubernetes.config import (
    VolumeConfig,
    MarsReplicationControllerConfig,
    MarsSchedulersConfig,
    MarsWorkersConfig,
    MarsWebsConfig,
)
from mars.deploy.utils import wait_services_ready
from mars.utils import readable_size

from ...utils import build_mars_image_name

logger = logging.getLogger(__name__)

UI_PORT = 50002
NOTEBOOK_PORT = 50003
GS_COORDINATOR_PORT = 63800
GS_COORDINATOR_GATEWAY_PORT = 64800


class DiskDriverVolumeConfig(VolumeConfig):
    def __init__(self, name, mount_path, device_size, renew_interval=3, timeout=120):
        super().__init__(name, mount_path)
        self._device_size = device_size
        self._timeout = timeout
        self._renew_interval = renew_interval

    def build(self):
        return {
            "name": self.name,
            "diskDriverEphemeralDevice": {
                "deviceSize": self._device_size,
                "timeout": self._timeout,
                "renewInterval": self._renew_interval,
            },
        }


class CupidMarsConfigMixin:
    def __init__(self, *args, **kwargs):
        self._container_port = kwargs.pop("container_port", None)
        super().__init__(*args, **kwargs)

    @property
    def stat_type(self):
        return "cgroup" if os.environ.get("VM_ENGINE_TYPE") != "hyper" else None

    @staticmethod
    def get_local_app_module(mod_name):
        from . import scheduler

        return scheduler.__name__.rsplit(".", 1)[0] + "." + mod_name

    def build_container(self):
        container_config = super().build_container()
        container_config.update({"enableChannel": True, "enableNativeLib": True})
        if self._container_port is not None:
            container_config["ports"] = {
                "containerPort": int(self._container_port),
            }
        return container_config

    def build_template_spec(self):
        spec_config = super().build_template_spec()
        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            spec_config.update({"restartPolicy": "Never"})
        else:
            spec_config.update({"hostNetwork": True, "restartPolicy": "Never"})
        return spec_config

    def config_readiness_probe(self):
        return None

    def build(self):
        result = super().build()
        with open("pod_%s.json" % self.rc_name, "w") as pod_file:
            pod_file.write(json.dumps(result, indent=2, sort_keys=True))
        return result

    def remove_env(self, name):
        try:
            super().remove_env(name)
        except AttributeError:
            self._envs.pop(name, None)


class CupidMarsSchedulersConfig(CupidMarsConfigMixin, MarsSchedulersConfig):
    def add_default_envs(self):
        super().add_default_envs()
        self.remove_env("MARS_USE_CGROUP_STAT")
        self.add_env("MARS_DISABLE_FAILOVER", "1")
        self.add_env("MARS_POD_ROLE", "scheduler")
        if self.stat_type == "cgroup":
            self.add_env("MARS_MEM_USE_CGROUP_STAT", "1")
            self.add_env("MARS_CPU_USE_PROCESS_STAT", "1")


class CupidMarsWorkersConfig(CupidMarsConfigMixin, MarsWorkersConfig):
    def __init__(self, *args, **kwargs):
        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            kwargs["mount_shm"] = kwargs.get("mount_shm", None) or False
        kwargs["min_cache_mem"] = kwargs.get("min_cache_mem", None) or "10%"
        super().__init__(*args, **kwargs)

    def add_default_envs(self):
        super().add_default_envs()
        self.add_env("MARS_LOCK_FREE_FILEIO", "1")
        self.add_env("MARS_DISABLE_PROC_RECOVER", "1")
        self.add_env("MARS_PLASMA_LIMIT", "95%")
        self.add_env("MARS_POD_ROLE", "worker")
        self.remove_env("MARS_USE_CGROUP_STAT")
        if self.stat_type == "cgroup":
            self.add_env("MARS_CPU_USE_PROCESS_STAT", "1")
            self.add_env("MARS_MEM_USE_CGROUP_STAT", "1")


class CupidMarsWebsConfig(CupidMarsConfigMixin, MarsWebsConfig):
    def __init__(self, *args, **kwargs):
        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            default_port = UI_PORT
        else:
            default_port = None
        kwargs["service_port"] = kwargs.get("service_port", None) or default_port
        super().__init__(*args, **kwargs)

    def add_default_envs(self):
        super().add_default_envs()
        self.add_env("MARS_POD_ROLE", "web")
        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            self.add_env("MARS_UI_PORT", str(UI_PORT))


class CupidMarsNotebooksConfig(CupidMarsConfigMixin, MarsReplicationControllerConfig):
    rc_name = "marsnotebook"

    def __init__(self, *args, **kwargs):
        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            default_port = NOTEBOOK_PORT
        else:
            default_port = None
        kwargs["service_port"] = kwargs.get("service_port", None) or default_port
        super().__init__(*args, **kwargs)

    def build_container_command(self):
        return [
            "/srv/entrypoint.sh",
            self.get_local_app_module("notebook"),
        ]

    def add_default_envs(self):
        super().add_default_envs()
        self.add_env("MARS_POD_ROLE", "notebook")
        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            self.add_env("MARS_NOTEBOOK_PORT", self._container_port)


class CupidGSCoordinatorConfig(CupidMarsConfigMixin, MarsReplicationControllerConfig):
    rc_name = "gscoordinator"

    def __init__(self, *args, **kwargs):
        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            default_port = GS_COORDINATOR_GATEWAY_PORT
        else:
            default_port = None
        kwargs["service_port"] = kwargs.get("service_port", None) or default_port
        self._worker_pod_name_list = kwargs.pop("worker_pod_name_list", None)
        self._worker_pod_ip_list = kwargs.pop("worker_pod_ip_list", None)
        self._port = GS_COORDINATOR_PORT
        self._gateway_port = kwargs["service_port"]
        self._worker_num = kwargs.pop("worker_num", 1)
        super().__init__(*args, **kwargs)

    def build_container_command(self):
        # replace command with coordinator
        coordinator_args = dict(
            port=self._port,
            gateway_port=self._gateway_port,
            worker_pod_name_list=self._worker_pod_name_list,
            worker_pod_ip_list=self._worker_pod_ip_list,
            num_workers=self._worker_num,
        )
        cmd = [
            "/srv/entrypoint.sh",
            self.get_local_app_module("gscoordinator"),
            base64.b64encode(json.dumps(coordinator_args).encode()).decode(),
        ]
        return cmd

    def add_default_envs(self):
        super().add_default_envs()
        self.add_env("MARS_POD_ROLE", "gscoordinator")
        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            self.add_env("GS_COORDINATOR_GATEWAY_PORT", self._container_port)
            self.add_env("GS_COORDINATOR_PORT", self._container_port)


class CupidKubernetesCluster(KubernetesCluster):
    _scheduler_config_cls = CupidMarsSchedulersConfig
    _worker_config_cls = CupidMarsWorkersConfig
    _web_config_cls = CupidMarsWebsConfig
    _default_service_port = None

    def __init__(self, *args, **kwargs):
        self._with_notebook = kwargs.pop("with_notebook", False)
        self._notebook_cpu = kwargs.pop("notebook_cpu", None)
        self._notebook_mem = kwargs.pop("notebook_mem", None)

        self._notebook_extra_env = kwargs.get("extra_env", None) or dict()
        self._notebook_extra_env.update(
            kwargs.pop("notebook_extra_env", None) or dict()
        )

        self._with_graphscope = kwargs.pop("with_graphscope", False)
        self._coordinator_extra_env = kwargs.get("extra_env", None) or dict()
        self._coordinator_extra_env.update(
            kwargs.pop("coordinator_extra_env", None) or dict()
        )
        self._coordinator_cpu = kwargs.pop("coordinator_cpu", None)
        self._coordinator_mem = kwargs.pop("coordinator_mem", None)

        self._node_blacklist = set(kwargs.pop("node_blacklist", None) or [])
        self._blacklisted_pods = set()

        kwargs["image"] = build_mars_image_name(kwargs.pop("image", None))
        super().__init__(*args, **kwargs)

    def _create_notebook(self):
        if self._with_notebook:
            notebook_config = CupidMarsNotebooksConfig(
                1,
                image=self._image,
                cpu=self._notebook_cpu,
                memory=self._notebook_mem,
                volumes=self._extra_volumes,
                pre_stop_command=self._pre_stop_command,
            )
            notebook_config.add_simple_envs(self._notebook_extra_env)
            self._core_api.create_namespaced_replication_controller(
                self._namespace, notebook_config.build()
            )

    def _create_graphscope(self):
        if self._with_graphscope:
            # check the worker services are ready
            limits = [self._worker_num]
            selectors = ["mars/service-type=" + CupidMarsWorkersConfig.rc_name]
            wait_services_ready(
                selectors,
                limits,
                lambda sel: self._get_ready_pod_count(sel),
                timeout=self._timeout,
            )
            # get the worker pod name and ip
            worker_pod_name_list, worker_pod_ip_list = self._get_pods_name_and_ip(
                selectors[0]
            )
            coordinator_config = CupidGSCoordinatorConfig(
                1,
                image=self._image,
                cpu=self._coordinator_cpu,
                memory=self._coordinator_mem,
                volumes=self._extra_volumes,
                pre_stop_command=self._pre_stop_command,
                worker_pod_name_list=worker_pod_name_list,
                worker_pod_ip_list=worker_pod_ip_list,
                worker_num=self._worker_num,
            )
            coordinator_config.add_simple_envs(self._coordinator_extra_env)
            self._core_api.create_namespaced_replication_controller(
                self._namespace, coordinator_config.build()
            )

    def _create_services(self):
        super()._create_services()
        self._create_graphscope()
        self._create_notebook()

    def _create_kube_service(self):
        # does not create k8s service as not supported in cupid
        pass

    def _get_ready_pod_count(self, label_selector):
        if self._node_blacklist:
            query = self._core_api.list_namespaced_pod(
                namespace=self._namespace, label_selector=label_selector
            ).to_dict()
            for el in query["items"]:
                node_name = el.get("spec", {}).get("node_name")
                pod_name = el["metadata"]["name"]

                if pod_name in self._blacklisted_pods:
                    continue
                if node_name and node_name in self._node_blacklist:
                    logger.warning(
                        "Found node %s in blacklist, will terminate pod %s",
                        node_name,
                        pod_name,
                    )
                    try:
                        self._core_api.delete_namespaced_pod(
                            name=pod_name, namespace=self._namespace
                        )
                        self._blacklisted_pods.add(pod_name)
                    except:  # noqa: E722
                        pass
        return super()._get_ready_pod_count(label_selector)

    def _get_pods_name_and_ip(self, label_selector):
        pod_name_list = []
        pod_ip_list = []
        query = self._core_api.list_namespaced_pod(
            namespace=self._namespace, label_selector=label_selector
        ).to_dict()
        for el in query["items"]:
            if el["status"]["reason"] == "CupidStarted":
                pod_name_list.append(el["metadata"]["name"])
                pod_ip_list.append(el["status"]["pod_ip"])
        return pod_name_list, pod_ip_list


class MarsCupidServer(object):
    def __init__(self):
        self.args = None
        self._instance_id = None
        self._kube_url = None
        self._kube_client = None

    def __call__(self, argv=None):
        if argv is None:
            argv = sys.argv[1:]
        return self._main(argv)

    def config_logging(self):
        import logging.config
        import mars

        log_conf = self.args.log_conf or "logging.conf"

        conf_file_paths = [
            "",
            os.path.abspath("."),
            os.path.dirname(os.path.dirname(mars.__file__)),
        ]
        log_configured = False
        for path in conf_file_paths:
            conf_path = os.path.join(path, log_conf) if path else log_conf
            if os.path.exists(conf_path):
                logging.config.fileConfig(conf_path, disable_existing_loggers=False)
                log_configured = True
                break

        if not log_configured:
            log_level = self.args.log_level
            level = getattr(logging, log_level.upper()) if log_level else logging.INFO
            logging.getLogger("mars").setLevel(level)
            logging.basicConfig(format=self.args.log_format)

    def _main(self, argv=None):
        parser = argparse.ArgumentParser(description="Mars Cupid Application")
        parser.add_argument("--log-level", help="log level")
        parser.add_argument("--log-format", help="log format")
        parser.add_argument(
            "--log-conf", help="log config file, logging.conf by default"
        )
        parser.add_argument("encoded_args")
        self.args = parser.parse_args(argv)
        self.config_logging()

        self._kube_url = os.environ["KUBE_API_ADDRESS"].strip('"')
        self._instance_id = os.environ["KUBE_NAMESPACE"].strip('"')

        args_dict = json.loads(base64.b64decode(self.args.encoded_args).decode())
        args_dict["namespace"] = self._instance_id

        if args_dict.get("worker_disk_num") is not None:
            disk_num = args_dict.pop("worker_disk_num")
            disk_size = args_dict.pop("worker_disk_size")
            args_dict["worker_spill_paths"] = [
                DiskDriverVolumeConfig(
                    name="diskdriver-volume%d" % i,
                    mount_path="/diskdriver%d" % i,
                    device_size=readable_size(disk_size, trunc=True).lower(),
                )
                for i in range(disk_num)
            ]

        scheduler_extra_env = args_dict.get("scheduler_extra_env") or dict()
        if args_dict.get("instance_idle_timeout") is not None:
            idle_timeout = args_dict.pop("instance_idle_timeout")
            scheduler_extra_env["MARS_INSTANCE_IDLE_TIMEOUT"] = str(idle_timeout)
        args_dict["scheduler_extra_env"] = scheduler_extra_env

        extra_env = args_dict.get("extra_env") or dict()
        extra_env["KUBE_API_ADDRESS"] = self._kube_url
        extra_env["KUBE_NAMESPACE"] = self._instance_id
        args_dict["extra_env"] = extra_env

        new_cluster(
            self.get_instance_kube(), cluster_cls=CupidKubernetesCluster, **args_dict
        )
        logger.info("Cluster creation finished")

    def get_instance_kube(self):
        from kubernetes import client

        config = client.Configuration()
        config.host = self._kube_url
        return client.ApiClient(config)


main = MarsCupidServer()

if __name__ == "__main__":
    main()
