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

from mars.scheduler.__main__ import SchedulerApplication

from ..actors.prochelper import CupidSchedulerProcessHelperActor

from .utils import wait_all_schedulers_ready

logger = logging.getLogger(__name__)


class CupidSchedulerServiceMain(SchedulerApplication):
    def __init__(self):
        super(CupidSchedulerServiceMain, self).__init__()
        self.cupid_context = None

    def config_args(self, parser):
        super(CupidSchedulerServiceMain, self).config_args(parser)
        parser.add_argument('--cupid-scheduler-key', help='scheduler configuration key in cupid')

    def write_cupid_service_info(self, cupid_key):
        from cupid import context
        self.cupid_context = context()

        kvstore = self.cupid_context.kv_store()
        kvstore[cupid_key] = json.dumps(dict(endpoint=self.endpoint))
        logger.info('Service endpoint %s written in key %s', self.endpoint, cupid_key)

    def wait_for_all_ready(self, scheduler_keys):
        kvstore = self.cupid_context.kv_store()
        scheduler_endpoints = wait_all_schedulers_ready(kvstore, scheduler_keys)
        self._service._cluster_info_ref.set_schedulers(scheduler_endpoints)

    def start(self):
        from mars.actors import new_client

        super(CupidSchedulerServiceMain, self).start()

        # create process helper on every process
        proc_helper_refs = []
        for proc_id in range(self.pool.cluster_info.n_process):
            uid = 's:%d:mars-process-helper' % proc_id
            actor_ref = self.pool.create_actor(CupidSchedulerProcessHelperActor, uid=uid)
            proc_helper_refs.append(actor_ref)

        cupid_scheduler_key, scheduler_keys = self.args.cupid_scheduler_key.split(';')

        if self.args.cupid_scheduler_key:
            self.write_cupid_service_info(cupid_scheduler_key)

        self.wait_for_all_ready(scheduler_keys)
        self.create_scheduler_discoverer()

        actor_client = new_client()
        for proc_helper_actor in proc_helper_refs:
            envs = self.cupid_context.prepare_channel()
            proc_helper_ref = actor_client.actor_ref(proc_helper_actor)
            new_envs = dict((env.name, env.value) for env in envs)
            proc_helper_ref.start_channel(new_envs)

    def stop(self):
        self.cupid_context.channel_client.stop()
        super(CupidSchedulerServiceMain, self).stop()


main = CupidSchedulerServiceMain()

if __name__ == '__main__':
    main()
