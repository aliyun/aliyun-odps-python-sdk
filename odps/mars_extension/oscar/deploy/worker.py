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

from mars.deploy.kubernetes.worker import K8SWorkerCommandRunner

from .core import CupidCommandRunnerMixin

logger = logging.getLogger(__name__)


class CupidWorkerCommandRunner(CupidCommandRunnerMixin, K8SWorkerCommandRunner):
    def __call__(self, *args, **kwargs):
        try:
            self.fix_protobuf_import()
            self.start_cupid_service()
            super().__call__(*args, **kwargs)
        except:
            import traceback

            traceback.print_exc()
            raise
        finally:
            self.stop_cupid_service()

    def parse_args(self, parser, argv, environ=None):
        self.fix_hyper_address()
        return super().parse_args(parser, argv, environ)

    async def start_services(self):
        await self.write_node_endpoint()
        await super().start_services()


main = CupidWorkerCommandRunner()

if __name__ == "__main__":
    main()
