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

import os
import json
import logging
import socket

from mars.web.__main__ import WebApplication


from .client import CUPID_APP_NAME
from .utils import wait_all_schedulers_ready

logger = logging.getLogger(__name__)


class CupidWebServiceMain(WebApplication):
    def __init__(self):
        super(CupidWebServiceMain, self).__init__()

        from cupid import context
        self.cupid_context = context()

    def config_args(self, parser):
        super(CupidWebServiceMain, self).config_args(parser)
        parser.add_argument('--cupid-scheduler-key', help='scheduler configuration key in cupid')

    def read_cupid_service_info(self, cupid_key):
        kvstore = self.cupid_context.kv_store()
        scheduler_endpoints = wait_all_schedulers_ready(kvstore, cupid_key)
        self.args.schedulers = ','.join(scheduler_endpoints)
        logger.info('Obtained endpoint from K-V store: %s', ','.join(scheduler_endpoints))

    def config_service(self):
        if self.args.cupid_scheduler_key:
            self.read_cupid_service_info(self.args.cupid_scheduler_key)

    def start(self):
        super(CupidWebServiceMain, self).start()

        host_addr = socket.gethostbyname(socket.gethostname())
        endpoint = 'http://{0}:{1}'.format(host_addr, self.mars_web.port)
        kvstore = self.cupid_context.kv_store()
        kvstore[CUPID_APP_NAME] = json.dumps(dict(endpoint=endpoint))

        if os.environ.get('VM_ENGINE_TYPE') == 'hyper':
            endpoint = socket.gethostname() + "-{}".format(self.mars_web.port)
        self.cupid_context.register_application(CUPID_APP_NAME, endpoint)

    def stop(self):
        self.cupid_context.channel_client.stop()
        super(CupidWebServiceMain, self).stop()


main = CupidWebServiceMain()

if __name__ == '__main__':
    main()
