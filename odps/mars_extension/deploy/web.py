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

import json
import logging
import os

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

    def start(self):
        import socket
        super().start()

        host_addr = socket.gethostbyname(socket.gethostname())
        endpoint = 'http://{0}:{1}'.format(host_addr, self.mars_web.port)
        kvstore = self._cupid_context.kv_store()
        kvstore[CUPID_APP_NAME] = json.dumps(dict(endpoint=endpoint))

        if os.environ.get('VM_ENGINE_TYPE') == 'hyper':
            endpoint = socket.gethostname() + "-{}".format(self.mars_web.port)
        self._cupid_context.register_application(CUPID_APP_NAME, endpoint)

    def stop(self):
        super().stop()
        if self._cupid_context is not None:
            self._cupid_context.channel_client.stop()


main = CupidWebApplication()

if __name__ == '__main__':
    main()
