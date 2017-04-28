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

from .core import Iterable, XMLRemoteModel
from .xflow import XFlow
from .instances import Instances
from .instance import Instance
from .. import serializers, errors, compat
from ..compat import six


class XFlows(Iterable):

    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems')
    xflows = serializers.XMLNodesReferencesField(XFlow, 'odpsalgo')

    def _get(self, name):
        return XFlow(client=self._client, parent=self, name=name)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            xflow = self._get(item)
        elif isinstance(item, XFlow):
            xflow = item
        else:
            return False

        try:
            xflow.reload()
            return True
        except errors.NoSuchObject:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self, owner=None):
        params = dict()
        if owner is not None:
            params['owner'] = owner

        def _it():
            last_marker = params.get('marker')
            if 'marker' in params and \
                (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource()
            resp = self._client.get(url, params=params)

            inst = XFlows.parse(self._client, resp, obj=self)
            params['marker'] = inst.marker

            return inst.xflows

        while True:
            xflows = _it()
            if xflows is None:
                break
            for xflow in xflows:
                yield xflow

    def create(self, xml_source):
        url = self.resource()
        headers = {'Content-Type': 'application/xml'}
        self._client.post(url, xml_source, headers=headers)

    def delete(self, name):
        if not isinstance(name, XFlow):
            xflow = XFlow(name=name, parent=self)
        else:
            xflow = name
            name = name.name
        del self[name]

        url = xflow.resource()
        self._client.delete(url)

    def update(self, xflow):
        url = xflow.resource()
        headers = {'Content-Type': 'application/xml'}
        self._client.put(url, xflow.xml_source, headers)

        xflow.reload()
        return xflow

    class XFlowInstance(XMLRemoteModel):
        __slots__ = 'xflow_project', 'xflow_name', 'parameters'

        _root = 'XflowInstance'

        xflow_project = serializers.XMLNodeField('Project')
        xflow_name = serializers.XMLNodeField('Xflow')
        parameters = serializers.XMLNodePropertiesField(
            'Parameters', 'Parameter', key_tag='Key', value_tag='Value', required=True)

    class AnonymousSubmitXFlowInstance(XMLRemoteModel):
        _root = 'Instance'

        instance = serializers.XMLNodeReferenceField('XFlows.XFlowInstance', 'XflowInstance')

    def _gen_xflow_instance_xml(self, xflow_instance=None, **kw):
        if xflow_instance is None:
            xflow_instance = XFlows.XFlowInstance(**kw)

        inst = XFlows.AnonymousSubmitXFlowInstance(instance=xflow_instance)
        return inst.serialize()

    def run_xflow(self, xflow_instance=None, project=None, **kw):
        project = project or self.parent
        return project.instances.create(
            xml=self._gen_xflow_instance_xml(xflow_instance=xflow_instance, **kw))

    class XFlowResult(XMLRemoteModel):
        class XFlowAction(XMLRemoteModel):
            node_type = serializers.XMLNodeAttributeField('.', attr='NodeType')
            instance_id = serializers.XMLNodeField('InstanceId')
            name = serializers.XMLNodeField('Name')
            result = serializers.XMLNodeReferenceField(Instance.InstanceResult, 'Result')

        actions = serializers.XMLNodesReferencesField(XFlowAction, 'Actions', 'Action')

    def get_xflow_results(self, instance):
        url = instance.resource()
        params = {'xresult': ''}
        resp = self._client.get(url, params=params)

        xflow_result = XFlows.XFlowResult.parse(self._client, resp)
        return dict((action.name, action) for action in xflow_result.actions)

    def get_xflow_source(self, instance):
        params = {'xsource': ''}
        return self._client.get(instance.resource(), params=params).content

    def get_xflow_instance(self, instance):
        content = self.get_xflow_source()
        try:
            inst = XFlows.AnonymousSubmitXFlowInstance.parse(self._client, content)
            return inst.instance
        except compat.ElementTreeParseError as e:
            raise errors.ODPSError(e)

    def is_xflow_instance(self, instance):
        try:
            self.get_xflow_instance(instance)
            return True
        except errors.ODPSError:
            return False


