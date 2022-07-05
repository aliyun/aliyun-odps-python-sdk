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
import threading
import warnings

from odps.compat import six

from .config import options
from .errors import CupidMasterTimeoutError
from .rpc import CupidTaskServiceRpcChannel
from .utils import get_environ

try:
    from .proto import cupidtaskparam_pb2 as task_param_pb
except TypeError:
    warnings.warn('Cannot import protos from pycupid: '
        'consider upgrading your protobuf python package.', ImportWarning)
    raise ImportError

logger = logging.getLogger(__name__)

_CUPID_CONF_PREFIXES = 'odps cupid'.split()


class CupidSession(object):
    def __init__(self, odps=None, project=None):
        from .runtime import context
        from odps import ODPS

        self._context = context()

        if odps is not None:
            self.odps = odps
        else:
            self.odps = ODPS.from_global()
        self.project = project if project is not None else self.odps.project

        self.lookup_name = get_environ('META_LOOKUP_NAME', '')
        self.running = False
        self.save_id = None
        self.job_running_event = threading.Event()

        self._kube_url = None

    def check_running(self, timeout=None):
        timeout = timeout or options.cupid.wait_am_start_time
        if not self.running:
            self.job_running_event.wait(timeout)
        return self.running

    @property
    def account(self):
        # todo add support for bearer token
        return self.odps.account

    def job_conf(self, conf=None):
        conf_items = {
            'odps.task.major.version': options.cupid.major_task_version,
            'odps.access.id': getattr(self.odps.account, 'access_id', None),
            'odps.access.key': getattr(self.odps.account, 'secret_access_key', None),
            'odps.end.point': self.odps.endpoint,
            'odps.project.name': self.project,
            'odps.moye.am.cores': '400',
            'odps.cupid.proxy.end.point': options.cupid.proxy_endpoint,
        }

        if conf:
            conf_items.update(conf)
        for k, v in six.iteritems(options.cupid.settings or {}):
            if any(k.startswith(pf) for pf in _CUPID_CONF_PREFIXES):
                conf_items[k] = v
        conf_obj_items = [task_param_pb.JobConfItem(key=k, value=str(v))
                          for k, v in six.iteritems(conf_items) if v is not None]
        return task_param_pb.JobConf(jobconfitem=conf_obj_items)

    def get_proxy_token(self, instance, app_name, expired_in_hours):
        if hasattr(instance, 'id'):
            instance = instance.id

        task_operator = task_param_pb.CupidTaskOperator(moperator='GetCupidProxyToken')
        proxy_token_request = task_param_pb.CupidProxyTokenRequest(
            instanceId=instance,
            appName=app_name,
            expiredInHours=expired_in_hours,
        )
        task_param = task_param_pb.CupidTaskParam(
            mcupidtaskoperator=task_operator,
            cupidProxyTokenRequest=proxy_token_request,
        )
        channel = CupidTaskServiceRpcChannel(self)
        inst = channel.submit_job(task_param, 'eAsyncNotFuxiJob', session=self)
        return channel.wait_cupid_instance(inst)

    def get_proxied_url(self, instance, app_name, expired_in_hours=None):
        expired_in_hours = expired_in_hours or 30 * 24
        return 'http://%s.%s' % (self.get_proxy_token(instance, app_name, expired_in_hours),
                                 options.cupid.proxy_endpoint)

    def start_kubernetes(self, async_=False, priority=None, running_cluster=None,
                         proxy_endpoint=None, major_task_version=None,
                         app_command=None, app_image=None, resources=None, **kw):
        priority = priority or options.priority
        if priority is None and options.get_priority is not None:
            priority = options.get_priority(self.odps)
        menginetype = options.cupid.engine_running_type

        if proxy_endpoint is not None:
            options.cupid.proxy_endpoint = proxy_endpoint
        if major_task_version is not None:
            options.cupid.major_task_version = major_task_version

        async_ = kw.pop('async', async_)
        runtime_endpoint = kw.pop('runtime_endpoint', None)
        task_operator = task_param_pb.CupidTaskOperator(moperator='startam', menginetype=menginetype)
        task_name = kw.pop('task_name', None)

        if len(kw) > 0:
            raise ValueError('Got unexpected arguments: {}'.format(list(kw.keys())[0]))

        kub_conf = {
            'odps.cupid.kube.master.mode': options.cupid.kube.master_mode,
            'odps.cupid.master.type': options.cupid.master_type,
            'odps.cupid.engine.running.type': menginetype,
            'odps.cupid.job.capability.duration.hours': options.cupid.job_duration_hours,
            'odps.cupid.channel.init.timeout.seconds': options.cupid.channel_init_timeout_seconds,
            'odps.moye.runtime.type': options.cupid.application_type,
            'odps.runtime.end.point': runtime_endpoint or options.cupid.runtime.endpoint,
            'odps.cupid.resources': ','.join(resources or []),
            'odps.cupid.kube.appmaster.cmd': app_command,
            'odps.cupid.kube.appmaster.image': app_image
        }
        if running_cluster:
            kub_conf['odps.cupid.task.running.cluster'] = running_cluster
        if options.cupid.container_node_label is not None:
            kub_conf['odps.cupid.container.node.label'] = options.cupid.container_node_label
        if options.cupid.master.virtual_resource is not None:
            kub_conf['odps.cupid.master.virtual.resource'] = options.cupid.master.virtual_resource

        task_param = task_param_pb.CupidTaskParam(
            jobconf=self.job_conf(conf=kub_conf),
            mcupidtaskoperator=task_operator,
        )

        retrial = 0

        while True:
            retrial += 1
            channel = CupidTaskServiceRpcChannel(self)
            inst = channel.submit_job(task_param, 'eHasFuxiJob', with_resource=True, priority=priority,
                                    running_cluster=running_cluster, session=self, task_name=task_name)
            if async_:
                return inst
            else:
                try:
                    return self.get_instance_kube(inst)
                except CupidMasterTimeoutError:
                    if retrial == 3:
                        raise

    def get_instance_kube_api(self, inst, expired_in_hours=None):
        if self._kube_url is None:
            CupidTaskServiceRpcChannel.wait_cupid_instance(inst, 'running')
            self._kube_url = self.get_proxied_url(inst, '', expired_in_hours or 30 * 24)
        return self._kube_url

    def get_instance_kube(self, inst, expired_in_hours=None):
        from kubernetes import client
        config = client.Configuration()
        config.host = self.get_instance_kube_api(inst, expired_in_hours)
        return client.ApiClient(config)


from .io import *

__all__ = ['CupidSession', 'CupidTableUploadSession', 'CupidTableDownloadSession']
