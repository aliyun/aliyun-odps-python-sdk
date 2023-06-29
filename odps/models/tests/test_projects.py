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

from datetime import datetime

import pytest

from ...compat import six
from .. import Projects, Project


def test_projects_exists(odps):
    not_exists_project_name = 'a_not_exists_project'
    assert odps.exist_project(not_exists_project_name) is False

    assert odps.exist_project(odps.project) is True


def test_project(odps):
    assert odps.get_project() is odps.get_project()
    assert Projects(client=odps.rest) is Projects(client=odps.rest)

    del odps._projects[odps.project]
    project = odps.get_project()

    assert project.name == odps.project

    assert project._getattr("owner") is None
    assert project._getattr("type") is None
    assert project._getattr("comment") is None
    assert project._getattr("creation_time") is None
    assert project._getattr("last_modified_time") is None
    assert project._getattr("project_group_name") is None
    assert project._getattr("properties") is None
    assert project._getattr("_extended_properties") is None
    assert project._getattr("_state") is None
    assert project._getattr("clusters") is None

    assert project.is_loaded is False

    assert isinstance(project.extended_properties, dict)
    assert isinstance(project.owner, six.string_types)
    assert project.type == Project.ProjectType.MANAGED
    assert isinstance(project.creation_time, datetime)
    assert isinstance(project.last_modified_time, datetime)
    assert isinstance(project.properties, dict)
    assert len(project.properties) > 0
    assert len(project.extended_properties) > 0
    assert isinstance(project.state, six.string_types)
    assert isinstance(project.status, Project.ProjectStatus)
    assert project.status == Project.ProjectStatus.AVAILABLE

    assert project.is_loaded is True

    with pytest.raises(KeyError):
        project.get_property("non_exist_property")
    assert project.get_property("non_exist_property", None) is None


def test_list_projects(odps):
    projects = [next(odps.list_projects(max_items=1)) for _ in range(2)]
    assert len(projects) > 1
    assert isinstance(projects[0], Project)
