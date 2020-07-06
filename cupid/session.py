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

from odps.compat import six

from .config import options
from .proto import cupidtaskparam_pb2 as task_param_pb
from .rpc import CupidTaskServiceRpcChannel
from .utils import get_environ

logger = logging.getLogger(__name__)

_CUPID_CONF_PREFIXES = 'odps cupid'.split()


def _build_mars_config(app_config):
    options.cupid.application_type = 'mars'

    mars_config = dict()

    resources = app_config['resources']
    module_path = app_config['module_path']
    resources_config = ','.join(resources)
    module_path_config = ','.join(module_path)
    mars_image = app_config.get('mars_image', None)

    mars_config['odps.cupid.resources'] = resources_config
    notebook = app_config.get('notebook', None)
    cmd = '/opt/conda/bin/python /srv/server.py ' \
          '--scheduler-num {} ' \
          '--scheduler-cpu {} ' \
          '--scheduler-mem {} ' \
          '--worker-num {} ' \
          '--worker-cpu {} ' \
          '--worker-mem {} ' \
          '--disk-num {} ' \
          '--cache-mem {} ' \
          '--module-path {} '.format(
        app_config['scheduler_num'],
        app_config['scheduler_cpu'],
        app_config['scheduler_mem'],
        app_config['worker_num'],
        app_config['worker_cpu'],
        app_config['worker_mem'],
        app_config['disk_num'],
        app_config['cache_mem'],
        module_path_config
    )
    if mars_image:
        cmd = cmd + '--mars-image {} '.format(mars_image)
    if notebook:
        cmd = cmd + '--with-nootbook '
    mars_config['odps.cupid.kube.appmaster.cmd'] = cmd
    mars_config['odps.cupid.kube.appmaster.image'] = app_config.get('mars_app_image', None) or \
                                                     options.cupid.mars_image
    return mars_config


def _build_app_config(app_name, app_config):
    if app_name == 'mars':
        return _build_mars_config(app_config)
    else:
        return dict()


class CupidSession(object):
    def __init__(self, odps=None):
        from .runtime import context
        from odps import ODPS

        self._context = context()

        if odps is not None:
            self.odps = odps
        else:
            self.odps = ODPS.from_global()

        self.lookup_name = get_environ('META_LOOKUP_NAME', '')
        self.running = False
        self.save_id = None
        self.job_running_event = threading.Event()

        self._kube_app_name = None
        self._kube_app_config = None
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
            'odps.project.name': self.odps.project,
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

    def start_kubernetes(self, async_=False, priority=None, running_cluster=None, **kw):
        priority = priority or options.priority
        if priority is None and options.get_priority is not None:
            priority = options.get_priority(self.odps)
        menginetype = options.cupid.running_engine_type

        self._kube_app_name = kw.get('app_name', None)
        self._kube_app_config = kw.get('app_config', None)

        proxy_endpoint = self._kube_app_config.pop('proxy_endpoint', None)
        if proxy_endpoint is not None:
            options.cupid.proxy_endpoint = proxy_endpoint

        major_task_version = self._kube_app_config.pop('major_task_version', None)
        if major_task_version is not None:
            options.cupid.major_task_version = major_task_version

        async_ = kw.get('async', async_)
        runtime_endpoint = kw.get('runtime_endpoint', None)
        task_operator = task_param_pb.CupidTaskOperator(moperator='startam', menginetype=menginetype)

        kub_conf = {
            'odps.cupid.kube.master.mode': options.cupid.kube.master_mode,
            'odps.cupid.master.type': options.cupid.master_type,
            'odps.cupid.engine.running.type': menginetype,
            'odps.cupid.job.capability.duration.hours': options.cupid.job_duration_hours,
            'odps.cupid.channel.init.timeout.seconds': options.cupid.channel_init_timeout_seconds,
            'odps.moye.runtime.type': options.cupid.application_type,
            'odps.runtime.end.point': runtime_endpoint or options.cupid.runtime.endpoint
        }
        app_conf = _build_app_config(self._kube_app_name, self._kube_app_config)
        kub_conf.update(app_conf)
        if running_cluster:
            kub_conf['odps.moye.job.runningcluster'] = running_cluster
        task_param = task_param_pb.CupidTaskParam(
            jobconf=self.job_conf(conf=kub_conf),
            mcupidtaskoperator=task_operator,
        )
        channel = CupidTaskServiceRpcChannel(self)
        inst = channel.submit_job(task_param, 'eHasFuxiJob', with_resource=True, priority=priority,
                                  running_cluster=running_cluster, session=self)
        if async_:
            return inst
        else:
            return self.get_instance_kube(inst)

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
