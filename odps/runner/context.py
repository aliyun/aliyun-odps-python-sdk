# encoding: utf-8
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
import threading

import itertools
from collections import Iterable

from .. import ODPS
from ..compat import six
from ..config import options
from .core import RunnerDAG, ObjectContainer
from .runner import Runner

logger = logging.getLogger(__name__)


class RunnerContext(object):
    def __new__(cls, *args, **kwargs):
        if hasattr(cls, '_instance'):
            return cls._instance
        else:
            return object.__new__(cls)

    def __init__(self, odps=None):
        if odps:
            self._odps = odps
        elif options.account is not None:
            self._odps = ODPS._from_account(
                options.account, options.default_project,
                endpoint=options.end_point, tunnel_endpoint=options.tunnel_endpoint
            )
        else:
            self._odps = None

        self._dag = RunnerDAG()
        self._obj_container = ObjectContainer()
        self._result_container = dict()
        self._node_outputs = dict()

        self._local = threading.local()

        type(self)._instance = self

    @classmethod
    def instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = RunnerContext()
        return cls._instance

    @classmethod
    def reset(cls):
        if hasattr(cls, '_instance'):
            delattr(cls, '_instance')

    def _get_node_odps(self, node):
        if isinstance(node, Iterable):
            node = six.next(node)
        for port in itertools.chain(six.itervalues(node.outputs), six.itervalues(node.inputs)):
            if port.obj_uuid in self._obj_container:
                return self._obj_container[port.obj_uuid]._odps
        return None

    def _run(self, target_node=None, odps=None):
        odps = odps or self._get_node_odps(target_node)
        if hasattr(self._local, 'parallel_nodes'):
            if isinstance(target_node, Iterable):
                self._local.parallel_nodes.extend(target_node)
            else:
                self._local.parallel_nodes.append(target_node)
        if hasattr(self._local, 'actions_before_run'):
            self._run_remaining_actions(odps)
        else:
            runner = Runner(odps)
            runner.run(target_node)

    def _batch_run_actions(self, actions, odps):
        self._local.parallel_nodes = []
        self._local.actions_before_run = iter(actions)

        self._run_remaining_actions(odps)

    def _run_remaining_actions(self, odps):
        while True:
            if not hasattr(self._local, 'actions_before_run'):
                # Execution already done and data is cleaned up.
                break
            try:
                # Fetch next action. If this action calls context._run, an recursion will occur.
                action = next(self._local.actions_before_run)
            except StopIteration:
                # Every action has been enumerated. We need to check if nodes are not executed.
                if hasattr(self._local, 'parallel_nodes'):
                    # Actual execution
                    runner = Runner(odps)
                    runner.run(self._local.parallel_nodes)
                    delattr(self._local, 'parallel_nodes')
                    delattr(self._local, 'actions_before_run')
                break
            action()
