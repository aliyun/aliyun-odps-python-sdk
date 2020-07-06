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

import logging

from mars.worker.__main__ import WorkerApplication
from .utils import wait_all_schedulers_ready

logger = logging.getLogger('mars.worker')


class CupidWorkerServiceMain(WorkerApplication):
    def __init__(self):
        super(CupidWorkerServiceMain, self).__init__()
        self.cupid_context = None

    def config_args(self, parser):
        super(CupidWorkerServiceMain, self).config_args(parser)
        parser.add_argument('--cupid-scheduler-key', help='scheduler configuration key in cupid')

    def read_cupid_service_info(self, cupid_key):
        kvstore = self.cupid_context.kv_store()
        scheduler_endpoints = wait_all_schedulers_ready(kvstore, cupid_key)
        self.args.schedulers = ','.join(scheduler_endpoints)
        logger.info('Obtained endpoint from K-V store: %s', ','.join(scheduler_endpoints))

    def config_service(self):
        if self.args.cupid_scheduler_key:
            self.args.schedulers = ' '

    def start(self):
        from mars.actors import new_client
        from cupid import context

        self.cupid_context = context()
        self.read_cupid_service_info(self.args.cupid_scheduler_key)
        self.create_scheduler_discoverer()

        super(CupidWorkerServiceMain, self).start()

        actor_client = new_client()
        proc_helpers = self._service._process_helper_actors
        for proc_helper_actor in proc_helpers:
            logger.info('Start channel for subprocess %s.', proc_helper_actor.uid)
            envs = self.cupid_context.prepare_channel()
            proc_helper_ref = actor_client.actor_ref(proc_helper_actor)
            new_envs = dict((env.name, env.value) for env in envs)
            proc_helper_ref.start_channel(new_envs)
        logger.info('All channel ready, upload worker status now.')
        self._service._status_ref.enable_status_upload(channel_ready=True, _tell=True)

    def stop(self):
        self.cupid_context.channel_client.stop()
        super(CupidWorkerServiceMain, self).stop()


main = CupidWorkerServiceMain()

if __name__ == '__main__':
    main()
