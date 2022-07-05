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

import logging

from mars.deploy.kubernetes.scheduler import K8SSchedulerApplication

from .core import CupidServiceMixin, CupidK8SPodsIPWatcher

try:
    from mars.deploy.kubernetes.scheduler import WorkerWatcherActor

    class CupidWorkerWatcherActor(WorkerWatcherActor):
        watcher_cls = CupidK8SPodsIPWatcher

except ImportError:
    WorkerWatcherActor = CupidWorkerWatcherActor = None

logger = logging.getLogger(__name__)


class CupidSchedulerApplication(CupidServiceMixin, K8SSchedulerApplication):
    def __init__(self, *args, **kwargs):
        from mars.actors import register_actor_implementation
        from mars.scheduler import SessionManagerActor
        from odps.mars_extension.legacy.actors import CupidSessionManagerActor

        register_actor_implementation(SessionManagerActor, CupidSessionManagerActor)
        if WorkerWatcherActor is not None:
            register_actor_implementation(WorkerWatcherActor, CupidWorkerWatcherActor)
        super().__init__(*args, **kwargs)


main = CupidSchedulerApplication()

if __name__ == "__main__":
    main()
