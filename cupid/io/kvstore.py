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

import warnings

from odps.compat import six

from ..rpc import SandboxRpcChannel, CupidRpcController
from ..errors import CupidError

try:
    from ..proto import kv_store_service_pb2 as kv_pb
except TypeError:
    warnings.warn('Cannot import protos from pycupid: '
        'consider upgrading your protobuf python package.', ImportWarning)
    raise ImportError


class CupidKVStore(object):
    def __init__(self):
        self.channel = SandboxRpcChannel()
        self.stub = kv_pb.KVStoreService_Stub(self.channel)

    def __setitem__(self, key, value):
        if isinstance(value, six.text_type):
            value = value.encode()

        controller = CupidRpcController()
        req = kv_pb.PutRequest(key=key, value=value)

        self.stub.Put(controller, req, None)
        if controller.Failed():
            raise CupidError(controller.ErrorText())

    def __getitem__(self, item):
        controller = CupidRpcController()

        req = kv_pb.GetRequest(value=item)
        resp = self.stub.Get(controller, req, None)

        if controller.Failed():
            err_text = controller.ErrorText()
            if 'PANGU_FILE_NOT_FOUND' in err_text:
                raise KeyError(err_text)
            else:
                raise CupidError(err_text)
        return resp.value

    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default
