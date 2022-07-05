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

import json
import logging
import os
import threading
import time

from mars.deploy.kubernetes.web import K8SWebApplication

from .client import CUPID_APP_NAME
from .core import CupidServiceMixin

from .. import web as _web_plugin

del _web_plugin

logger = logging.getLogger(__name__)


class CupidWebApplication(CupidServiceMixin, K8SWebApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from cupid import context

        self._cupid_context = context()

    def _app_register_thread(self, internal_endpoint, app_endpoint):
        while True:
            if self._cupid_context is None:
                break
            kvstore = self._cupid_context.kv_store()
            kvstore[CUPID_APP_NAME] = json.dumps(dict(endpoint=internal_endpoint))

            self._cupid_context.register_application(CUPID_APP_NAME, app_endpoint)
            time.sleep(5)

    def start(self):
        import socket

        super().start()

        host_addr = socket.gethostbyname(socket.gethostname())
        internal_endpoint = "http://{0}:{1}".format(host_addr, self.mars_web.port)

        if os.environ.get("VM_ENGINE_TYPE") == "hyper":
            app_endpoint = socket.gethostname() + "-{}".format(self.mars_web.port)
        else:
            app_endpoint = internal_endpoint

        app_register_thread = threading.Thread(
            target=self._app_register_thread,
            args=(internal_endpoint, app_endpoint),
            daemon=True,
        )
        app_register_thread.start()

    def stop(self):
        super().stop()
        if self._cupid_context is not None:
            self._cupid_context.channel_client.stop()
            self._cupid_context = None


main = CupidWebApplication()

if __name__ == "__main__":
    main()
