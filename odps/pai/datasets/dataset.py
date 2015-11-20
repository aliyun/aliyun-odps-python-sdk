# encoding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import collections
import logging
import uuid
import weakref
from six import string_types

from ..nodes import SplitNode, OdpsTargetNode
from .ops import *

logger = logging.getLogger(__name__)


class DataSet(object):
    def __init__(self, context, endpoint, uplink=None, fields=None):
        self._context = weakref.ref(context)
        self._bind_endpoint = endpoint
        self._bind_node = endpoint.bind_node
        self._bind_output = endpoint.name
        self._data_uuid = uuid.uuid4()
        self._operations = []
        self._uplink = list(uplink) if uplink is not None else []

        self._table = None
        self._partition = None
        self._fields = list(fields) if fields is not None else []

        self._context()._ds_container.register(self)
        if endpoint.data_set_uuid is None:
            endpoint.data_set_uuid = self._data_uuid

    def split(self, ratio):
        split_node = SplitNode(ratio)
        self._link_node(split_node, "input")
        return [
            self._duplicate_data_set(split_node.get_output_endpoint("output1")),
            self._duplicate_data_set(split_node.get_output_endpoint("output2")),
        ]

    def exclude_fields(self, *args):
        if args is None:
            raise ValueError("Field list cannot be None.")
        ds = self._duplicate_data_set(self._bind_endpoint)

        # generate exclusion set from args
        extra_set = set()
        for arg in args:
            if isinstance(arg, collections.Iterable) and not isinstance(arg, string_types):
                extra_set = extra_set.union(arg)
            else:
                extra_set.add(arg)

        ds._perform_operation(ds, ExcludeFeatureDataSetOperation(extra_set))
        return ds

    def select_fields(self, *args):
        if len(args) == 0:
            raise ValueError("Field list cannot be empty.")
        ds = self._duplicate_data_set(self._bind_endpoint)

        # generate selected set from args
        select_set = set()
        for arg in args:
            if isinstance(arg, collections.Iterable) and not isinstance(arg, string_types):
                select_set = select_set.union(arg)
            else:
                select_set.add(arg)

        self._perform_operation(ds, SelectFeatureDataSetOperation(select_set))
        return ds

    def set_weight_field(self, weight_field):
        if weight_field is None:
            raise ValueError("Weight field name cannot be None.")
        ds = self._duplicate_data_set(self._bind_endpoint)
        self._perform_operation(ds, WeightDataSetOperation(weight_field))
        return ds

    def set_label_field(self, label_field):
        if label_field is None:
            raise ValueError("Label field name cannot be None.")
        ds = self._duplicate_data_set(self._bind_endpoint)
        self._perform_operation(ds, LabelDataSetOperation(label_field))
        return ds

    def set_continuity(self, **kwargs):
        ds = self._duplicate_data_set(self._bind_endpoint)
        self._perform_operation(ds, FieldContinuityDataSetOperation(kwargs))
        return ds

    def store_odps(self, table_name, partition_name=None):
        logger.debug('Operation step DataSet.store_odps(\'%s%s\') called.' %
                     (table_name, ', ' + partition_name if partition_name is not None else ''))

        odps_node = OdpsTargetNode(table_name, partition_name)
        self._context()._dag.add_node(odps_node)
        self._context()._dag.add_link(self._bind_node, self._bind_output, odps_node, "input")

        self._context()._run(self._bind_node)

    def to_xml(self):
        data_set = Element('dataset', {
            'uuid': str(self._data_uuid)
        })
        uplinks = SubElement(data_set, 'uplinks')
        operations = SubElement(data_set, 'operations')
        for uplink_obj in self._uplink:
            uplink = SubElement(uplinks, 'uplink')
            uplink.text = str(uplink_obj._data_uuid)
        for operation in self._operations:
            operations.append(operation.to_xml())
        return data_set

    def _link_node(self, node, input_port):
        self._context()._dag.add_node(node)
        self._context()._dag.add_data_input(self, node, input_port)

    def _duplicate_data_set(self, endpoint):
        ds = DataSet(self._context(), endpoint, fields=self._fields)
        for attr, value in iteritems(vars(self)):
            if not hasattr(ds, attr):
                setattr(ds, attr, value)
        ds.__class__ = self.__class__
        ds._uplink.append(self)
        return ds

    @staticmethod
    def _perform_operation(ds, op):
        if ds._uplink:
            source_fields = [ul._fields for ul in ds._uplink]
            ds._fields = op.execute(source_fields)
        ds._operations.append(op)

    def _append_fields(self, fields):
        ds = self._duplicate_data_set(self._bind_endpoint)
        self._perform_operation(ds, StaticFieldChangeOperation(fields, is_append=True))
        return ds
