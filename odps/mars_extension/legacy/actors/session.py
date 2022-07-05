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

import logging
import os
import time

from mars.scheduler.session import SessionManagerActor

from ....config import options

logger = logging.getLogger(__name__)

DEFAULT_MARS_IDLE_TIMEOUT = 3 * 3600
CUPID_LAST_IDLE_TIME_KEY = "MarsServiceLastIdleTime"


class CupidSessionManagerActor(SessionManagerActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "MARS_INSTANCE_IDLE_TIMEOUT" in os.environ:
            self._idle_timeout = int(os.environ["MARS_INSTANCE_IDLE_TIMEOUT"])
        else:
            self._idle_timeout = DEFAULT_MARS_IDLE_TIMEOUT

        self._last_active_time = time.time()
        self._last_active_time_from_service = None
        self._check_started = False

    def create_session(self, *args, **kwargs):
        session_ref = super().create_session(*args, **kwargs)
        self._last_activity_time = time.time()
        if not self._check_started and self._idle_timeout is not None:
            from cupid.runtime import context

            self._check_started = True
            kv_store = context().kv_store()
            last_idle_time_str = kv_store.get(CUPID_LAST_IDLE_TIME_KEY) or 0
            _, self._last_active_time_from_service = self._get_service_activity_info()
            if not last_idle_time_str:
                self._last_active_time = self._last_active_time_from_service
            else:
                self._last_active_time = float(last_idle_time_str)

            self.ref().check_instance_idle(_delay=10, _tell=True, _wait=False)
            logger.info(
                "Instance will go timeout in %s seconds when no active sessions.",
                self._idle_timeout,
            )
        return session_ref

    def _get_service_activity_info(self):
        last_active_time = self._last_active_time
        has_running = False
        for ref in self._session_refs.values():
            for info in ref.get_graph_infos().values():
                if info.get("end_time") is None:
                    has_running = True
                    break
                else:
                    last_active_time = max(info["end_time"], last_active_time)
            if has_running:
                break
        return has_running, last_active_time

    def check_instance_idle(self):
        from cupid.runtime import context

        has_running, active_time_from_service = self._get_service_activity_info()
        if active_time_from_service != self._last_active_time_from_service:
            self._last_active_time = active_time_from_service
            self._last_active_time_from_service = active_time_from_service
        elif has_running:
            self._last_active_time = time.time()

        if self._last_active_time < time.time() - self._idle_timeout:
            # timeout: we need to kill the instance
            logger.warning("Timeout met, killing the instance now.")
            self._stop_instance()
        else:
            kv_store = context().kv_store()
            kv_store[CUPID_LAST_IDLE_TIME_KEY] = str(self._last_active_time)
            self.ref().check_instance_idle(_delay=10, _tell=True, _wait=False)

    @staticmethod
    def _stop_instance():
        from cupid import context, ContainerStatus, WorkItemProgress
        from odps import ODPS
        from odps.accounts import BearerTokenAccount

        cupid_context = context()
        cupid_context.report_container_status(
            ContainerStatus.TERMINATED,
            "Instance idle timed out, stopping",
            WorkItemProgress.WIP_TERMINATING,
        )
        time.sleep(options.mars.container_status_timeout)

        # when instance is still not stopped, we kill forcifully
        bearer_token = cupid_context.get_bearer_token()
        account = BearerTokenAccount(bearer_token)
        project = os.environ["ODPS_PROJECT_NAME"]
        endpoint = os.environ["ODPS_RUNTIME_ENDPOINT"]
        o = ODPS(None, None, account=account, project=project, endpoint=endpoint)

        o.stop_instance(os.environ["MARS_K8S_POD_NAMESPACE"])


try:
    from ...internal.core import DEFAULT_MARS_IDLE_TIMEOUT
except ImportError:
    pass
