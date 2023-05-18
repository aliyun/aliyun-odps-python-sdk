#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import shutil
import tempfile

import pytest

from ..compat import StringIO
from ..config import options
from ..errors import InteractiveError
from ..inter import Room, list_rooms
from ..lib import cloudpickle
from ..models import TableSchema
from .core import tn


@pytest.fixture(autouse=True)
def install_cloud_unpickler():
    cloudpickle.CloudUnpickler(StringIO('abcdefg'))


def test_room():
    from ..inter import enter, setup, teardown

    access_id = 'test_access_id'
    access_key = 'test_access_key'
    project = 'test_default_project'
    endpoint = 'test_endpoint'

    test_room = '__test'

    teardown(test_room)

    setup(access_id, access_key, project, endpoint=endpoint, room=test_room)
    enter(test_room)

    assert access_id == options.account.access_id
    assert access_key == options.account.secret_access_key
    assert project == options.default_project
    assert endpoint == options.endpoint
    assert options.tunnel.endpoint is None

    pytest.raises(InteractiveError,
        lambda: setup(access_id, access_key, project, room=test_room))

    assert test_room in list_rooms()

    teardown(test_room)
    pytest.raises(InteractiveError, lambda: enter(test_room))


def test_room_stores(odps):
    class FakeRoom(Room):
        def _init(self):
            return

    room = FakeRoom('__test')
    room._room_dir = tempfile.mkdtemp()

    try:
        s = TableSchema.from_lists(['name', 'id'], ['string', 'bigint'])
        table_name = tn('pyodps_test_room_stores')
        odps.delete_table(table_name, if_exists=True)
        t = odps.create_table(table_name, s)
        data = [['name1', 1], ['name2', 2]]
        with t.open_writer() as writer:
            writer.write(data)

        del t

        t = odps.get_table(table_name)
        assert t.table_schema.names == ['name', 'id']

        try:
            room.store('table', t)

            t2 = room['table']
            assert t2.name == table_name

            with t2.open_reader() as reader:
                values = [r.values for r in reader]
                assert data == values

            assert room.list_stores() == [['table', None]]
        finally:
            t.drop()
    finally:
        shutil.rmtree(room._room_dir)
