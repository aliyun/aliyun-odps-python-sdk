# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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
import os
import socket
import time
from urllib.parse import urlparse

import mars.oscar as mo
from mars.deploy.kubernetes.supervisor import K8SSupervisorCommandRunner
from mars.services.session import SessionAPI
from mars.services.web import OscarWebAPI

from ..cupid_service import CupidServiceClient
from .core import CupidCommandRunnerMixin

DEFAULT_MARS_IDLE_TIMEOUT = 3 * 3600
CUPID_LAST_IDLE_TIME_KEY = 'MarsServiceLastIdleTime'

logger = logging.getLogger(__name__)


class ClusterIdleCheckActor(mo.Actor):
    async def __post_create__(self):
        if 'MARS_INSTANCE_IDLE_TIMEOUT' in os.environ:
            self._idle_timeout = int(os.environ['MARS_INSTANCE_IDLE_TIMEOUT'])
        else:
            self._idle_timeout = DEFAULT_MARS_IDLE_TIMEOUT
        self._idle_check_task = asyncio.create_task(self._check_last_idle())

    async def __pre_destroy__(self):
        self._idle_check_task.cancel()

    async def _check_last_idle(self):
        cupid_client = CupidServiceClient()
        session_api = await SessionAPI.create(self.address)

        last_idle_time_str = cupid_client.get_kv(CUPID_LAST_IDLE_TIME_KEY) or 0
        last_idle_time_from_service = await session_api.get_last_idle_time(None)
        if not last_idle_time_str:
            last_idle_time = last_idle_time_from_service
        else:
            last_idle_time = float(last_idle_time_str)

        while True:
            await asyncio.sleep(10)
            idle_time_from_service = await session_api.get_last_idle_time(None)
            if idle_time_from_service != last_idle_time_from_service:
                last_idle_time = idle_time_from_service
                last_idle_time_from_service = idle_time_from_service

            if time.time() - last_idle_time > self._idle_timeout:
                # timeout: we need to kill the instance
                cupid_client.terminate_instance(os.environ['MARS_K8S_POD_NAMESPACE'])
            else:
                # make sure elapsed time is persisted
                await asyncio.to_thread(
                    cupid_client.put_key,
                    CUPID_LAST_IDLE_TIME_KEY,
                    str(last_idle_time),
                )


class CupidSupervisorCommandRunner(CupidCommandRunnerMixin, K8SSupervisorCommandRunner):
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

    async def _register_web_endpoint(self):
        from .client import CUPID_APP_NAME

        web_api = await OscarWebAPI.create(self.args.endpoint)
        web_port = urlparse(await web_api.get_web_address()).port

        host_addr = socket.gethostbyname(socket.gethostname())
        endpoint = 'http://{0}:{1}'.format(host_addr, web_port)
        kvstore = self._cupid_context.kv_store()
        kvstore[CUPID_APP_NAME] = json.dumps(dict(endpoint=endpoint))

        if os.environ.get('VM_ENGINE_TYPE') == 'hyper':
            endpoint = socket.gethostname() + "-{}".format(web_port)
        await self.register_application(CUPID_APP_NAME, endpoint)

    async def start_services(self):
        await self.write_node_endpoint()
        await super().start_services()
        await mo.create_actor(
            ClusterIdleCheckActor,
            uid=ClusterIdleCheckActor.default_uid(),
            address=self.args.endpoint,
        )
        await self._register_web_endpoint()


main = CupidSupervisorCommandRunner()

if __name__ == '__main__':
    main()
