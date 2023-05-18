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

import itertools

from ... import errors
from ...compat import ConfigParser
from ...tests.core import tn
from ...utils import to_text
from .. import Resource

FUNCTION_CONTENT = """\
from odps.udf import annotate

@annotate("bigint,bigint->bigint")
class MyPlus(object):

   def evaluate(self, arg0, arg1):
       if None in (arg0, arg1):
           return None
       return arg0 + arg1
"""


def test_functions(odps):
    assert odps.get_project().functions is odps.get_project().functions

    functions_model = odps.get_project().functions

    functions = list(functions_model.iterate(maxitems=400))
    size = len(functions)
    assert size >= 0

    for i, function in zip(itertools.count(1), functions):
        if i > 50:
            break
        try:
            assert all(map(lambda r: type(r) != Resource, function.resources)) is True
        except errors.ODPSError:
            continue


def test_function_exists(odps):
    non_exists_function = 'a_non_exists_function'
    assert odps.exist_function(non_exists_function) is False


def test_function(odps):
    functions = odps.list_functions()
    try:
        function = next(functions)
    except StopIteration:
        return

    assert function is odps.get_function(function.name)

    assert function._getattr('name') is not None
    assert function._getattr('owner') is not None
    assert function._getattr('creation_time') is not None
    assert function._getattr('class_type') is not None
    assert function._getattr('_resources') is not None


def test_create_delete_update_function(config, odps):
    try:
        secondary_project = config.get('test', 'secondary_project')
        secondary_user = config.get('test', 'secondary_user')
    except ConfigParser.NoOptionError:
        secondary_project = secondary_user = None

    test_resource_name = tn('pyodps_t_tmp_test_function_resource') + '.py'
    test_resource_name2 = tn('pyodps_t_tmp_test_function_resource2') + '.py'
    test_resource_name3 = tn('pyodps_t_tmp_test_function_resource3') + '.py'
    test_function_name = tn('pyodps_t_tmp_test_function')
    test_function_name3 = tn('pyodps_t_tmp_test_function3')

    try:
        odps.delete_resource(test_resource_name)
    except errors.NoSuchObject:
        pass
    try:
        odps.delete_resource(test_resource_name2)
    except errors.NoSuchObject:
        pass
    try:
        odps.delete_function(test_function_name)
    except errors.NoSuchObject:
        pass
    try:
        odps.delete_function(test_function_name3)
    except errors.NoSuchObject:
        pass

    if secondary_project:
        try:
            odps.delete_resource(test_resource_name3, project=secondary_project)
        except errors.NoSuchObject:
            pass

    test_resource = odps.create_resource(
        test_resource_name, 'py', file_obj=FUNCTION_CONTENT
    )

    test_function = odps.create_function(
        test_function_name,
        class_type=test_resource_name.split('.', 1)[0]+'.MyPlus',
        resources=[test_resource]
    )

    assert test_function.name is not None
    assert test_function.owner is not None
    assert test_function.creation_time is not None
    assert test_function.class_type is not None
    assert len(test_function.resources) == 1

    with odps.open_resource(name=test_resource_name, mode='r') as fp:
        assert to_text(fp.read()) == to_text(FUNCTION_CONTENT)

    assert test_function.owner != secondary_user

    test_resource2 = odps.create_resource(
        test_resource_name2, 'file', file_obj='Hello World'
    )
    test_function.resources.append(test_resource2)
    if secondary_user:
        test_function.owner = secondary_user
    test_function.update()

    test_function_id = id(test_function)
    del test_function.project.functions[test_function.name]
    test_function = odps.get_function(test_function_name)
    assert test_function_id != id(test_function)
    assert len(test_function.resources) == 2
    if secondary_user:
        assert test_function.owner == secondary_user

    test_resource3 = None
    test_function3 = None
    if secondary_project:
        test_resource3 = odps.create_resource(
            test_resource_name3, 'py', file_obj=FUNCTION_CONTENT, project=secondary_project
        )

        test_function3 = odps.create_function(
            test_function_name3,
            class_type=test_resource_name3.split('.', 1)[0]+'.MyPlus',
            resources=[test_resource3]
        )

        assert test_function3.name == test_function_name3
        assert test_function3.owner is not None
        assert test_function3.creation_time is not None
        assert test_function3.class_type is not None
        assert len(test_function3.resources) == 1
        assert test_function3.resources[0].name == test_resource_name3
        assert test_function3.resources[0].project.name == secondary_project

    test_resource.drop()
    test_resource2.drop()
    if test_resource3:
        test_resource3.drop()
    if test_function3:
        test_function3.drop()
    test_function.drop()
