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

from mars.deploy.kubernetes.worker import K8SWorkerApplication

from .core import CupidServiceMixin

logger = logging.getLogger(__name__)


class CupidWorkerApplication(CupidServiceMixin, K8SWorkerApplication):
    pass


main = CupidWorkerApplication()

if __name__ == "__main__":
    main()
