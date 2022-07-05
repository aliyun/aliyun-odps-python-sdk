# !/usr/bin/env python
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

import asyncio
import logging
import os
import socket
import time
from urllib.parse import urlparse

import mars.oscar as mo
from mars.deploy.kubernetes.supervisor import K8SSupervisorCommandRunner
from mars.services.session import SessionAPI
from mars.services.web import OscarWebAPI

from ....config import options
from ..cupid_service import CupidServiceClient
from .core import CupidCommandRunnerMixin

DEFAULT_MARS_IDLE_TIMEOUT = 3 * 3600
CUPID_LAST_IDLE_TIME_KEY = "MarsServiceLastIdleTime"

logger = logging.getLogger(__name__)


class ClusterIdleCheckActor(mo.Actor):
    async def __post_create__(self):
        if "MARS_INSTANCE_IDLE_TIMEOUT" in os.environ:
            self._idle_timeout = int(os.environ["MARS_INSTANCE_IDLE_TIMEOUT"])
        else:
            self._idle_timeout = DEFAULT_MARS_IDLE_TIMEOUT
        self._idle_check_task = asyncio.create_task(self._check_idle())

    async def __pre_destroy__(self):
        self._idle_check_task.cancel()

    async def _check_idle(self):
        try:
            cupid_client = CupidServiceClient()
            session_api = await SessionAPI.create(self.address)

            last_idle_time_str = cupid_client.get_kv(CUPID_LAST_IDLE_TIME_KEY) or 0
            last_idle_time_from_service = await session_api.get_last_idle_time(None)
            if not last_idle_time_str:
                last_idle_time = last_idle_time_from_service or time.time()
            else:
                last_idle_time = float(last_idle_time_str)
        except:
            logger.exception("Failed to check instance idle")
            raise

        has_sessions_ever = False
        while True:
            try:
                await asyncio.sleep(10)
                idle_time_from_service = await session_api.get_last_idle_time(None)
                idle_time_from_service = idle_time_from_service or time.time()
                if idle_time_from_service != last_idle_time_from_service:
                    last_idle_time = idle_time_from_service
                    last_idle_time_from_service = idle_time_from_service

                if not has_sessions_ever:
                    # when no sessions has been created ever, we shall never stop our cluster
                    last_idle_time = time.time()
                    has_sessions_ever = len(await session_api.get_sessions()) > 0

                if time.time() - last_idle_time > self._idle_timeout:
                    logger.warning(
                        "Stopping instance due to idle timed out. %s > %s",
                        time.time() - last_idle_time,
                        self._idle_timeout,
                    )
                    await asyncio.to_thread(self._sync_stop_instance, cupid_client)
                else:
                    # make sure elapsed time is persisted
                    await asyncio.to_thread(
                        cupid_client.put_kv,
                        CUPID_LAST_IDLE_TIME_KEY,
                        str(last_idle_time),
                    )
            except:
                logger.exception("Failed to check instance idle")

    @staticmethod
    def _sync_stop_instance(cupid_client):
        from cupid import ContainerStatus, WorkItemProgress

        cupid_client.report_container_status(
            ContainerStatus.TERMINATED,
            "Instance idle timed out, stopping",
            WorkItemProgress.WIP_TERMINATING,
        )
        time.sleep(options.mars.container_status_timeout)
        # when instance is still not stopped, we kill forcifully
        cupid_client.terminate_instance(os.environ["MARS_K8S_POD_NAMESPACE"])


class CupidSupervisorCommandRunner(CupidCommandRunnerMixin, K8SSupervisorCommandRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_task = None

    def __call__(self, *args, **kwargs):
        try:
            self.fix_protobuf_import()
            self.start_cupid_service()
            super().__call__(*args, **kwargs)
        except:
            import traceback

            traceback.print_exc()
            raise
        finally:
            self.stop_cupid_service()

    def parse_args(self, parser, argv, environ=None):
        self.fix_hyper_address()
        return super().parse_args(parser, argv, environ)

    async def _repeat_register(self, internal_endpoint, app_endpoint):
        from .client import CUPID_APP_NAME

        while True:
            try:
                await self.register_application(
                    CUPID_APP_NAME, internal_endpoint, app_endpoint
                )
                await asyncio.sleep(5)
            except asyncio.asyncio.CancelledError:
                break

    async def _register_web_endpoint(self):
        web_api = await OscarWebAPI.create(self.args.endpoint)
        web_port = urlparse(await web_api.get_web_address()).port

        host_addr = socket.gethostbyname(socket.gethostname())
        internal_endpoint = "http://{0}:{1}".format(host_addr, web_port)

        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            app_endpoint = socket.gethostname() + "-{}".format(web_port)
        else:
            app_endpoint = internal_endpoint

        self._register_task = asyncio.create_task(
            self._repeat_register(internal_endpoint, app_endpoint)
        )

    async def start_services(self):
        await self.write_node_endpoint()
        await super().start_services()
        await mo.create_actor(
            ClusterIdleCheckActor,
            uid=ClusterIdleCheckActor.default_uid(),
            address=self.args.endpoint,
        )
        await self._register_web_endpoint()

    async def stop_services(self):
        if self._register_task is not None:
            self._register_task.cancel()
        return await super().stop_services()


main = CupidSupervisorCommandRunner()

if __name__ == "__main__":
    main()
