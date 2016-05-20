#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from .core import AbstractXMLRemoteModel
from .. import serializers, errors, compat
from ..compat import Enum


class TaskType(Enum):
    UnknownTask = 'UNKNOWN'
    SQLTask = 'SQL'


def _get_task_type(name):
    try:
        return TaskType(name)
    except ValueError:
        return TaskType.UnknownTask


class Task(AbstractXMLRemoteModel):

    __slots__ = 'name', 'comment', 'properties'

    _type_indicator = 'type'

    name = serializers.XMLNodeField('Name')
    type = serializers.XMLTagField('.', parse_callback=lambda s: _get_task_type(s.upper()))
    comment = serializers.XMLNodeField('Comment')
    properties = serializers.XMLNodePropertiesField('Config', 'Property',
                                                    key_tag='Name', value_tag='Value')

    def __new__(cls, *args, **kwargs):
        typo = kwargs.get('type')

        if typo is not None:
            task_cls = globals().get(typo.name, cls)
        else:
            task_cls = cls

        return object.__new__(task_cls)

    def set_property(self, key, value):
        if self.properties is None:
            self.properties = compat.OrderedDict()
        self.properties[key] = value

    def serialize(self):
        if self.type == TaskType.UnknownTask:
            raise errors.OdpsError('Unknown task type')
        return super(Task, self).serialize()


def format_cdata(query):
    stripped_query = query.strip()
    if not stripped_query.endswith(';'):
        stripped_query += ';'
    return '<![CDATA[%s]]>' % stripped_query


class SQLTask(Task):
    __slots__ = '_anonymous_sql_task_name',

    _root = 'SQL'
    _anonymous_sql_task_name = 'AnonymousSQLTask'

    query = serializers.XMLNodeField('Query', serialize_callback=format_cdata)

    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = SQLTask._anonymous_sql_task_name
        super(SQLTask, self).__init__(**kwargs)

    def serial(self):
        if self.properties is None:
            self.properties = compat.OrderedDict()

        key = 'settings'
        if key not in self.properties:
            self.properties[key] = '{"odps.sql.udf.strict.mode": "true"}'

        return super(SQLTask, self).serial()

try:
    from odps.internal.models.tasks import *
except ImportError:
    pass
