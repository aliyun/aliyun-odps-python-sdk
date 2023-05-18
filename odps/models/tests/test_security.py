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

import json
from datetime import datetime

import pytest

from ...config import options
from ...errors import NoSuchObject, ODPSError, SecurityQueryError
from ...tests.core import tn, global_locked

TEST_ROLE_NAME = tn('test_role_name')

TEST_PROJECT_POLICY_STRING = """
{
    "Statement": [{
        "Action": ["odps:*"],
        "Effect": "Allow",
        "Principal": ["*"],
        "Resource": ["acs:odps:*:projects/#project#"]},
    {
        "Action": ["odps:*"],
        "Effect": "Allow",
        "Principal": ["*"],
        "Resource": ["acs:odps:*:projects/#project#/tables/*"]},
    {
        "Action": ["odps:*"],
        "Effect": "Allow",
        "Principal": ["*"],
        "Resource": ["acs:odps:*:projects/#project#/functions/*"]},
    {
        "Action": ["odps:*"],
        "Effect": "Allow",
        "Principal": ["*"],
        "Resource": ["acs:odps:*:projects/#project#/resources/*"]},
    {
        "Action": ["odps:Read"],
        "Effect": "Allow",
        "Principal": ["*"],
        "Resource": ["acs:odps:*:projects/#project#/packages/base_common.datamask"]}],
    "Version": "1"
}
"""

TEST_ROLE_POLICY_STRING = """
{
    "Statement": [{
        "Action": ["odps:Read",
            "odps:List"],
        "Effect": "Allow",
        "Resource": ["acs:odps:*:projects/#project#"]},
    {
        "Action": ["odps:Describe",
            "odps:Select"],
        "Effect": "Allow",
        "Resource": ["acs:odps:*:projects/#project#/tables/*"]},
    {
        "Action": ["odps:Read"],
        "Effect": "Allow",
        "Resource": ["acs:odps:*:projects/#project#/packages/base_common.datamask"]},
    {
        "Action": ["odps:Write",
            "odps:CreateTable",
            "odps:CreateInstance",
            "odps:CreateFunction",
            "odps:CreateResource",
            "odps:CreateJob",
            "odps:CreateVolume"],
        "Effect": "Deny",
        "Resource": ["acs:odps:*:projects/#project#"]},
    {
        "Action": ["odps:Alter",
            "odps:Update",
            "odps:Drop"],
        "Effect": "Deny",
        "Resource": ["acs:odps:*:projects/#project#/tables/*"]}],
    "Version": "1"
}
"""


@pytest.fixture()
def project(odps):
    return odps.get_project()


def safe_delete_role(project, role):
    try:
        project.roles.delete(role)
    except NoSuchObject:
        pass


def safe_delete_user(project, user):
    try:
        user = project.users[user]
        for r in user.roles:
            try:
                r.revoke_from(user)
            except ODPSError:
                pass
        project.users.delete(user)
    except NoSuchObject:
        pass


@global_locked
def test_project_methods(project):
    cur_user = project.current_user
    assert cur_user.id is not None and cur_user.display_name is not None

    old_policy = project.policy
    policy_json = json.loads(TEST_PROJECT_POLICY_STRING.replace('#project#', project.name))
    project.policy = policy_json
    project.reload()
    assert json.dumps(project.policy) == json.dumps(policy_json)
    project.policy = old_policy

    sec_options = project.security_options
    label_sec = sec_options.label_security

    sec_options.label_security = True
    sec_options.update()
    sec_options.reload()
    assert sec_options.label_security

    sec_options.label_security = False
    sec_options.update()
    sec_options.reload()
    assert not sec_options.label_security

    sec_options.label_security = label_sec
    sec_options.update()


def test_roles(odps, project):
    safe_delete_role(project, TEST_ROLE_NAME)

    role = project.roles.create(TEST_ROLE_NAME)
    assert TEST_ROLE_NAME in [r.name for r in project.roles]
    assert TEST_ROLE_NAME in project.roles
    assert role in project.roles
    assert 'non_exist_role_name' not in project.roles

    policy_json = json.loads(TEST_ROLE_POLICY_STRING.replace('#project#', project.name))
    role.policy = policy_json
    role.reload()
    assert json.dumps(role.policy) == json.dumps(policy_json)

    project.roles.delete(TEST_ROLE_NAME)
    assert TEST_ROLE_NAME not in project.roles


@global_locked('odps_project_user')
def test_users(config, project):
    secondary_user = config.get('test', 'secondary_user')
    if not secondary_user:
        return

    safe_delete_user(project, secondary_user)

    project.users.create(secondary_user)
    assert secondary_user in project.users
    assert secondary_user in [user.display_name for user in project.users]
    assert 'non_exist_user' not in project.users

    project.users.delete(secondary_user)
    assert secondary_user not in project.users


@global_locked('odps_project_user')
def test_user_role(config, project):
    secondary_user = config.get('test', 'secondary_user')
    if not secondary_user:
        return

    safe_delete_user(project, secondary_user)
    safe_delete_role(project, TEST_ROLE_NAME)

    role = project.roles.create(TEST_ROLE_NAME)
    user = project.users.create(secondary_user)

    role.grant_to(user)
    assert user in role.users
    assert role in user.roles
    role.revoke_from(user)
    assert user not in role.users
    assert role not in user.roles

    user.grant_role(role)
    assert user in role.users
    assert role in user.roles
    user.revoke_role(role)
    assert user not in role.users
    assert role not in user.roles

    project.users.delete(secondary_user)
    project.roles.delete(role)


def test_security_query(odps, project):
    assert 'ALIYUN' in odps.run_security_query('LIST ACCOUNTPROVIDERS')
    assert 'ALIYUN' in odps.execute_security_query('LIST ACCOUNTPROVIDERS')

    inst = odps.run_security_query(
        'INSTALL PACKAGE %s.non_exist_package' % project.name
    )
    assert isinstance(inst, project.AuthQueryInstance)

    with pytest.raises(SecurityQueryError):
        inst.wait_for_success()
    assert inst.is_terminated
    assert not inst.is_successful

    with pytest.raises(SecurityQueryError):
        odps.execute_security_query(
            'INSTALL PACKAGE %s.non_exist_package' % project.name
        )


def test_generate_auth_token(odps, project):
    with pytest.raises(SecurityQueryError):
        project.generate_auth_token(None, "not_supported", 1)

    policy = {
        "Version": "1",
        "Statement": [
            {
                "Action": ["odps:*"],
                "Resource": "acs:odps:*:*",
                "Effect": "Allow"
            }
        ]
    }
    token = project.generate_auth_token(policy, "bearer", 5)

    try:
        from ... import ODPS
        from ...accounts import BearerTokenAccount

        account = BearerTokenAccount(token)
        account._last_modified_time = datetime.now()
        new_odps = ODPS(
            project=odps.project,
            endpoint=odps.endpoint,
            account=account,
        )
        instance = new_odps.run_sql("select * from dual")
        assert instance.get_logview_address()
    finally:
        options.account = options.default_project = options.endpoint = None
