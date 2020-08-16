#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from mars.web.server import register_web_handler
from mars.web.apihandlers import MarsApiRequestHandler

logger = logging.getLogger('mars.web')


class LoggerHandler(MarsApiRequestHandler):
    def get(self):
        pass

    def post(self):
        content = self.get_argument('content')
        level = self.get_argument('level', 'warning').lower()
        getattr(logger, level)(content)


register_web_handler('/api/logger', LoggerHandler)
