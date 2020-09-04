#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

logger = logging.getLogger(__name__)

DEFAULT_MARS_IDLE_TIMEOUT = 3 * 3600


class CupidSessionManagerActor(SessionManagerActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'MARS_INSTANCE_IDLE_TIMEOUT' in os.environ:
            self._idle_timeout = int(os.environ['MARS_INSTANCE_IDLE_TIMEOUT'])
        else:
            self._idle_timeout = DEFAULT_MARS_IDLE_TIMEOUT

        self._last_activity_time = time.time()
        self._check_started = False

    def create_session(self, *args, **kwargs):
        session_ref = super().create_session(*args, **kwargs)
        self._last_activity_time = time.time()
        if not self._check_started and self._idle_timeout is not None:
            self._check_started = True
            self.ref().check_instance_idle(_delay=10, _tell=True, _wait=False)
            logger.info('Instance will go timeout in %s seconds when no active sessions.',
                        self._idle_timeout)
        return session_ref

    def check_instance_idle(self):
        last_active_time = self._last_activity_time
        has_running = False
        for ref in self._session_refs.values():
            for info in ref.get_graph_infos().values():
                if info.get('end_time') is None:
                    has_running = True
                    break
                else:
                    last_active_time = max(info['end_time'], last_active_time)
            if has_running:
                break

        if not has_running and last_active_time < time.time() - self._idle_timeout:
            # timeout: we need to kill the instance
            from odps import ODPS
            from odps.accounts import BearerTokenAccount
            from cupid.runtime import context

            logger.warning('Timeout met, killing the instance now.')

            bearer_token = context().get_bearer_token()
            account = BearerTokenAccount(bearer_token)
            project = os.environ['ODPS_PROJECT_NAME']
            endpoint = os.environ['ODPS_RUNTIME_ENDPOINT']
            o = ODPS(None, None, account=account, project=project, endpoint=endpoint)

            o.stop_instance(os.environ['MARS_K8S_POD_NAMESPACE'])
        else:
            self.ref().check_instance_idle(_delay=10, _tell=True, _wait=False)


try:
    from ...internal.core import DEFAULT_MARS_IDLE_TIMEOUT
except ImportError:
    pass
