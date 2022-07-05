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

import base64
import json
import uuid
import os
import sys
import logging
import warnings

import struct
import time

from google.protobuf import service as pb_service
from odps import compat
from odps.compat import six
from odps.errors import InvalidStateSetting, NoSuchObject

from . import errors
from .config import options
from .utils import get_property

try:
    from .proto import cupidtaskparam_pb2
except TypeError:
    warnings.warn('Cannot import protos from pycupid: '
        'consider upgrading your protobuf python package.', ImportWarning)
    raise ImportError

logger = logging.getLogger(__name__)
logger_mars = logging.getLogger('odps.mars_extension.cupid')

if sys.version_info[0] >= 3:
    b64decodebytes = base64.decodebytes
else:
    b64decodebytes = base64.decodestring


class CupidRpcController(pb_service.RpcController):
    def __init__(self):
        super(CupidRpcController, self).__init__()
        self._fail_msg = None

    def Reset(self):
        self._fail_msg = None

    def Failed(self):
        return self._fail_msg is not None

    def ErrorText(self):
        return self._fail_msg

    def StartCancel(self):
        pass

    def SetFailed(self, reason):
        self._fail_msg = reason

    def IsCanceled(self):
        return False

    def NotifyOnCancel(self, callback):
        pass


class CupidRpcChannel(pb_service.RpcChannel):
    @staticmethod
    def submit_job(param_pb, running_mode, priority=None, with_resource=False, session=None,
                   running_cluster=None, task_name='cupid_task'):
        from odps import ODPS, compat
        from odps.models import CupidTask

        if logger.getEffectiveLevel() <= logging.DEBUG:
            param_pb_str = str(param_pb)
            if isinstance(param_pb, cupidtaskparam_pb2.CupidTaskParam):
                for conf in param_pb.jobconf.jobconfitem:
                    if conf.key == 'odps.access.id':
                        param_pb_str = param_pb_str.replace(conf.value, '** access-id **')
                    elif conf.key == 'odps.access.key':
                        param_pb_str = param_pb_str.replace(conf.value, '** access-key **')
            logger.debug('Job param proto: %s', param_pb_str)

        odps = session.odps if session else ODPS.from_global()
        plan_string = param_pb.SerializeToString()
        res_kw = dict(fileobj=compat.BytesIO(plan_string))
        if not with_resource:
            res_kw['is_temp_resource'] = True
        res = odps.create_resource('cupid_plan_' + str(uuid.uuid4()), 'file', **res_kw)

        task_info = ','.join([res.name, odps.project, running_mode])

        props = dict()

        if options.cupid.application_type:
            props['odps.cupid.application.type'] = options.cupid.application_type
            props['odps.moye.runtime.type'] = options.cupid.application_type
        if options.biz_id:
            props['biz_id'] = options.biz_id
        if options.cupid.major_task_version:
            props['odps.task.major.version'] = options.cupid.major_task_version
        context_file = get_property('odps.exec.context.file')
        if context_file and os.path.exists(context_file):
            with open(context_file, 'r') as cf:
                file_settings = json.loads(cf.read()).get('settings', {})
            props.update(file_settings)

        task = CupidTask(task_name, task_info, props)
        inst = odps.get_project().instances.create(task=task, priority=priority, running_cluster=running_cluster)
        inst = odps.get_instance(inst.id)
        return inst

    @staticmethod
    def get_cupid_detail(inst):
        inst.reload()

        params = {'instancedetail': '', 'taskname': 'cupid_task'}
        resp = inst._client.get(inst.resource(), params=params)
        if resp.content == b'{}':
            logger.debug('Empty content retrieved. Use get_task_detail2 first.')
            inst.get_task_detail2('cupid_task')
            resp = inst._client.get(inst.resource(), params=params)

        return resp.content

    @classmethod
    def get_cupid_status(cls, inst):
        details = cls.get_cupid_detail(inst)
        if details.startswith(b'Failed') and b'recycled' in details:
            raise errors.InstanceRecycledError(details)
        details_pb = cupidtaskparam_pb2.CupidTaskDetailResultParam()
        details_pb.ParseFromString(b64decodebytes(details))

        status_keys = ('ready', 'waiting', 'running', 'success', 'failed', 'cancelled')
        if not any(details_pb.HasField(sk) for sk in status_keys):
            details_pb.ready.CopyFrom(cupidtaskparam_pb2.Ready())
        return details_pb

    @classmethod
    def wait_cupid_instance(cls, inst, state=None, master_timeout=120):
        state = state or 'success'
        wait_keys = ('running', 'success', 'ready')
        sleep_time = 0

        while True:
            result = cls.get_cupid_status(inst)
            sleep_time = min(5, sleep_time + 1)
            if result.HasField('failed'):
                if result.failed.HasField('cupidTaskFailed'):
                    msg = result.failed.cupidTaskFailed.cupidTaskFailedMsg
                else:
                    msg = result.failed.bizFailed.bizFailedMsg
                if msg.startswith("runTask failed:") or msg.startswith("app run failed!"):
                    raise errors.CupidUserError(msg)
                else:
                    raise errors.CupidError(msg)
            elif result.HasField('cancelled'):
                raise errors.CupidUserError('Instance canceled.')
            elif result.HasField(state):
                return getattr(getattr(result, state), state + 'Msg')
            elif any(result.HasField(wk) for wk in wait_keys):
                time.sleep(sleep_time)
            else:
                msg_pb = str(result)
                if msg_pb != '':
                    logger.warning('Unexpected status: %s' % msg_pb)
                time.sleep(sleep_time)


class CupidTaskServiceRpcChannel(CupidRpcChannel):
    def __init__(self, session):
        super(CupidTaskServiceRpcChannel, self).__init__()
        self.cupid_session = session

    def CallMethod(self, method, rpc_controller,
                   request, response_class, done):
        task_service_req = cupidtaskparam_pb2.TaskServiceRequest(
            methodName=method.name,
            requestInBytes=request.SerializeToString(),
        )
        job_conf = self.cupid_session.job_conf()
        task_operator = cupidtaskparam_pb2.CupidTaskOperator(moperator='TaskServiceRequest')
        task_param = cupidtaskparam_pb2.CupidTaskParam(
            mcupidtaskoperator=task_operator, jobconf=job_conf,
            taskServiceRequest=task_service_req
        )
        inst = self.submit_job(task_param, 'eAsyncNotFuxiJob', session=self.cupid_session)
        logger_mars.debug('Cupid task instance: %s, method: %s' % (inst.id, method.name))

        resp_str = self.wait_cupid_instance(inst)
        if isinstance(resp_str, six.text_type):
            resp_str = resp_str.encode()
        resp = response_class()
        resp.ParseFromString(b64decodebytes(resp_str))
        return resp


class SandboxRpcChannel(pb_service.RpcChannel):
    def CallMethod(self, method, controller, request, response_class, done):
        from .runtime import context

        context = context()

        try:
            sio = compat.BytesIO()
            sio.write(struct.pack('<I', method.index))
            sio.write(request.SerializeToString())

            logger.debug('SandboxRpcChannel sync_call service: %s, method id: %d request: %s',
                         method.containing_service.full_name, method.index, request)
            res = context.channel_client.sync_call(method.containing_service.full_name,
                                                   sio.getvalue())
            resp = response_class()
            resp.ParseFromString(res)
            logger.debug('SandboxRpcChannel sync_call result: %s', resp)
            return resp
        except Exception as exc:
            logger.exception('SandboxRpcChannel CallMethod fail: %s', str(exc))
            controller.SetFailed(str(exc))
