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

from mars.services.web.core import MarsRequestHandler
from tornado.web import HTTPError

logger = logging.getLogger(__name__)


class PyODPSHandler(MarsRequestHandler):
    def get(self):
        pass

    def post(self):
        action = self.get_argument("action", "write_log")
        if action == "write_log":
            self._handle_write_log()
        elif action == "terminate":
            self._handle_terminate()
        else:  # pragma: no cover
            raise HTTPError(400, "Invalid action %r" % action)

    def _handle_write_log(self):
        content = self.get_argument("content")
        level = self.get_argument("level", "warning").lower()
        getattr(logger, level)(content)

    def _handle_terminate(self):
        from .cupid_service import CupidServiceClient
        from cupid import ContainerStatus, WorkItemProgress

        message = self.get_argument("message", "")
        CupidServiceClient().report_container_status(
            ContainerStatus.TERMINATED, message, WorkItemProgress.WIP_TERMINATING
        )


web_handlers = {
    "/api/pyodps": PyODPSHandler,
    "/api/pyodps_logger": PyODPSHandler,
}
