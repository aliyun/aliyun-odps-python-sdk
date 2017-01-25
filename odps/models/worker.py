#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from ..compat import six, OrderedDict
from .. import serializers

_LOG_TYPES_MAPPING = {
    'hs_err_log': 'hs_err_*.log',
    'coreinfo': 'coreinfo.tmp',
}
_LOG_TYPES_MAPPING.update(dict((k, k) for k in 'stdout stderr waterfall_summary jstack pstack'.split()))


class Worker(serializers.JSONSerializableModel):
    """
    Worker information class for worker information and log retrieval.
    """
    __slots__ = '_client',

    @classmethod
    def extract_from_json(cls, json_obj, client=None, parent=None):
        raise NotImplementedError

    def get_log(self, log_type, size=0):
        """
        Get logs from worker.

        :param log_type: type of logs. Possible log types contains {log_types}
        :param size: length of the log to retrieve
        :return: log content
        """
        params = OrderedDict([('log', ''), ('id', self.log_id)])
        if log_type is not None:
            log_type = log_type.lower()
            if log_type not in _LOG_TYPES_MAPPING:
                raise ValueError('log_type should choose a value in ' +
                                 ' '.join(six.iterkeys(_LOG_TYPES_MAPPING)))
            params['logtype'] = _LOG_TYPES_MAPPING[log_type]
        if size > 0:
            params['size'] = str(size)
        resp = self._client.get(self.parent.resource(), params=params)
        return resp.text

    get_log.__doc__ = get_log.__doc__.format(log_types=', '.join(sorted(six.iterkeys(_LOG_TYPES_MAPPING))))


class WorkerDetail2(Worker):
    id = serializers.JSONNodeField('id')
    log_id = serializers.JSONNodeField('logId')
    start_time = serializers.JSONNodeField('startTime', parse_callback=int)
    end_time = serializers.JSONNodeField('endTime', parse_callback=int)
    status = serializers.JSONNodeField('status')
    gbi_counter = serializers.JSONNodeField('gblCounter')
    input_bytes = serializers.JSONNodeField('input_bytes', parse_callback=int)
    input_records = serializers.JSONNodeField('input_records', parse_callback=int)
    output_bytes = serializers.JSONNodeField('output_bytes', parse_callback=int)
    output_records = serializers.JSONNodeField('output_records', parse_callback=int)

    @classmethod
    def extract_from_json(cls, json_obj, client=None, parent=None):
        workers = []

        def _extract(o):
            if isinstance(o, list):
                for v in o:
                    _extract(v)
            elif isinstance(o, dict):
                if 'instances' in o:
                    for v in o['instances']:
                        w = cls.parse(v, parent=parent)
                        w._client = client
                        workers.append(w)
                    return
                for _, v in six.iteritems(o):
                    _extract(v)

        _extract(json_obj)
        return workers
