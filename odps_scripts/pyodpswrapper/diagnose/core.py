# -*- coding: utf-8 -*-
# Copyright 1999-2026 Alibaba Group Holding Ltd.
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
import textwrap
from collections import OrderedDict

from ..envs import REGION_ENV


class I18NMessage(object):
    _region_maps = {
        "cn-": "cn",
        "d2": "all",
    }
    _lang = None

    def __init__(self, default, **kwargs):
        kwargs.pop("all", None)
        self.messages = OrderedDict([("default", textwrap.dedent(default))])
        self.messages.update(
            OrderedDict((k, self._clean_message(v)) for k, v in kwargs.items())
        )
        # combine all langs for D2
        all_messages = "\n\n".join(s.strip() for s in self.messages.values())
        self.messages["all"] = "\n%s\n" % all_messages

    @classmethod
    def _clean_message(cls, message):
        return textwrap.dedent(message).strip()

    def __str__(self):
        if self._lang is None:
            region = os.environ.get(REGION_ENV) or "d2"
            for region_prefix, lang in self._region_maps.items():
                if region.lower().startswith(region_prefix):
                    self._lang = lang
        if self._lang is None:
            self._lang = "default"
        return self.messages[self._lang]

    def __add__(self, other):
        assert isinstance(other, I18NMessage)
        msg_dict = OrderedDict()
        for k, v in self.messages.items():
            msg_dict[k] = v + (
                other.messages.get(k) or other.messages.get("default") or ""
            )
        for k, v in other.messages.items():
            if k in msg_dict:
                continue
            msg_dict[k] = v
        return I18NMessage(**msg_dict)

    def safe_replace(self, str_to_replace, replacement):
        try:
            return str(self).replace(str_to_replace, replacement)
        except:
            return self.messages["default"].replace(str_to_replace, replacement)
