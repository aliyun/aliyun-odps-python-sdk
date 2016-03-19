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
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import absolute_import

import os
import sys
from hashlib import md5
import pickle
import shutil

from .config import options
from .core import ODPS
from .errors import InteractiveError
from .models import Schema
from .utils import to_binary, build_pyodps_dir
from .df.backends.frame import ResultFrame


DEFAULT_ROOM_NAME = 'default'
ODPS_FILE_NAME = '__ODPS__'
INFO_FILE_NAME = '__INFO__'
OBJECT_FILE_NAME = '__OBJ__'


class Room(object):
    def __init__(self, room_name):
        self._room_name = room_name
        self._room_dir = _get_room_dir(self._room_name)

        self._inited = False
        self._init()

    def _init(self):
        if self._inited:
            return

        odps_file = os.path.join(self._room_dir, ODPS_FILE_NAME)
        if not os.path.exists(odps_file):
            raise InteractiveError(
                'This room(%s) is not configured' % self._room_name)

        with open(odps_file, 'rb') as f:
            try:
                access_id, access_key, default_project, \
                    endpoint, tunnel_endpoint = pickle.load(f)
            except pickle.UnpicklingError:
                raise InteractiveError(
                    'Failed to enter a room: %s' % self._room_name)
            options.access_id = access_id
            options.access_key = access_key
            options.end_point = endpoint
            options.default_project = default_project
            options.tunnel_endpoint = tunnel_endpoint

        self._inited = True

    @property
    def odps(self):
        return ODPS(options.access_id, options.access_key, options.default_project,
                    endpoint=options.end_point, tunnel_endpoint=options.tunnel_endpoint)

    def __getattr__(self, attr):
        try:
            return self.fetch(attr)
        except InteractiveError:
            return super(Room, self).__getattr__(attr)

    def __getitem__(self, item):
        return self.fetch(item)

    def _obj_store_dir(self, name):
        fn = md5(to_binary(name)).hexdigest()
        return os.path.join(self._room_dir, fn)

    def store(self, name, obj, desc=None):
        path = self._obj_store_dir(name)

        if os.path.exists(path):
            raise InteractiveError('%s already exists' % name)

        os.makedirs(path)
        with open(os.path.join(path, INFO_FILE_NAME), 'wb') as f:
            pickle.dump((name, desc), f)

        with open(os.path.join(path, OBJECT_FILE_NAME), 'wb') as f:
            pickle.dump(obj, f, protocol=1)

    def fetch(self, name):
        path = self._obj_store_dir(name)

        if not os.path.exists(path):
            raise InteractiveError('%s does not exist' % name)

        with open(os.path.join(path, OBJECT_FILE_NAME), 'rb') as f:
            return pickle.load(f)

    def drop(self, name):
        path = self._obj_store_dir(name)

        if os.path.exists(path):
            shutil.rmtree(path)

    def list_stores(self):
        results = []
        for obj_dir in os.listdir(self._room_dir):
            info_path = os.path.join(self._room_dir, obj_dir, INFO_FILE_NAME)
            if os.path.exists(info_path):
                with open(info_path, 'rb') as f:
                    results.append(list(pickle.load(f)))

        return results

    def display(self):
        schema = Schema.from_lists(['name', 'desc'], ['string'] * 2)
        frame = ResultFrame(self.list_stores(), schema=schema)
        try:
            import pandas as pd

            if isinstance(frame.values, pd.DataFrame):
                df = frame.values

                df.columns.name = self._room_name
                frame._values = df.set_index('name')
        except ImportError:
            pass

        return frame


def _get_root_dir():
    rooms_dir = build_pyodps_dir('rooms')

    return os.path.join(rooms_dir, str(sys.version_info[0]))


def _get_room_dir(room_name, mkdir=False):
    rooms_dir = _get_root_dir()

    room_name = md5(to_binary(room_name)).hexdigest()
    room_dir = os.path.join(rooms_dir, room_name)
    if not os.path.exists(room_dir) and mkdir:
        os.makedirs(room_dir)

    return room_dir


def setup(access_id, access_key, default_project,
          endpoint=None, tunnel_endpoint=None,
          room=DEFAULT_ROOM_NAME):
    room_dir = _get_room_dir(room, mkdir=True)
    odps_file = os.path.join(room_dir, ODPS_FILE_NAME)

    if os.path.exists(odps_file):
        raise InteractiveError(
            'This room(%s) has been configured before, '
            'you can teardown it first' % room)

    obj = (access_id, access_key, default_project,
           endpoint, tunnel_endpoint)

    with open(odps_file, 'wb') as f:
        pickle.dump(obj, f)

    with open(os.path.join(room_dir, INFO_FILE_NAME), 'wb') as f:
        f.write(to_binary(room))


def enter(room=DEFAULT_ROOM_NAME):
    return Room(room)


def teardown(room=DEFAULT_ROOM_NAME):
    room_dir = _get_room_dir(room)
    if os.path.exists(room_dir):
        shutil.rmtree(room_dir)


def list_rooms():
    root = _get_root_dir()

    results = []
    for room_dir in os.listdir(root):
        info_path = os.path.join(root, room_dir, INFO_FILE_NAME)
        if os.path.exists(info_path):
            with open(info_path) as f:
                results.append(f.read())

    return results
