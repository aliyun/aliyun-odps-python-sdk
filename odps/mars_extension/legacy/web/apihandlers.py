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

from tornado.web import HTTPError

from mars.web.server import register_web_handler
from mars.web.apihandlers import MarsApiRequestHandler

logger = logging.getLogger("mars.web")


class PyODPSHandler(MarsApiRequestHandler):
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
        from cupid import context, ContainerStatus, WorkItemProgress

        message = self.get_argument("message", "")
        context().report_container_status(
            ContainerStatus.TERMINATED, message, WorkItemProgress.WIP_TERMINATING
        )


register_web_handler("/api/pyodps", PyODPSHandler)
register_web_handler("/api/logger", PyODPSHandler)
