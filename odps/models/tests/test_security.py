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

from odps.errors import NoSuchObject, ODPSError, SecurityQueryError
from odps.tests.core import TestBase, tn, global_locked
from odps.compat import unittest

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


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.project = self.odps.get_project()

    def safe_delete_role(self, role):
        try:
            self.project.roles.delete(role)
        except NoSuchObject:
            pass

    def safe_delete_user(self, user):
        try:
            user = self.project.users[user]
            for r in user.roles:
                try:
                    r.revoke_from(user)
                except ODPSError:
                    pass
            self.project.users.delete(user)
        except NoSuchObject:
            pass

    @global_locked
    def testProjectMethods(self):
        cur_user = self.project.current_user
        assert cur_user.id is not None and cur_user.display_name is not None

        old_policy = self.project.policy
        policy_json = json.loads(TEST_PROJECT_POLICY_STRING.replace('#project#', self.project.name))
        self.project.policy = policy_json
        self.project.reload()
        assert json.dumps(self.project.policy) == json.dumps(policy_json)
        self.project.policy = old_policy

        sec_options = self.project.security_options
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

    def testRoles(self):
        self.safe_delete_role(TEST_ROLE_NAME)

        role = self.project.roles.create(TEST_ROLE_NAME)
        assert TEST_ROLE_NAME in [r.name for r in self.project.roles]
        assert TEST_ROLE_NAME in self.project.roles
        assert role in self.project.roles
        assert 'non_exist_role_name' not in self.project.roles

        policy_json = json.loads(TEST_ROLE_POLICY_STRING.replace('#project#', self.project.name))
        role.policy = policy_json
        role.reload()
        assert json.dumps(role.policy) == json.dumps(policy_json)

        self.project.roles.delete(TEST_ROLE_NAME)
        assert TEST_ROLE_NAME not in self.project.roles

    @global_locked('odps_project_user')
    def testUsers(self):
        secondary_user = self.config.get('test', 'secondary_user')
        if not secondary_user:
            return

        self.safe_delete_user(secondary_user)

        self.project.users.create(secondary_user)
        assert secondary_user in self.project.users
        assert secondary_user in [user.display_name for user in self.project.users]
        assert 'non_exist_user' not in self.project.users

        self.project.users.delete(secondary_user)
        assert secondary_user not in self.project.users

    @global_locked('odps_project_user')
    def testUserRole(self):
        secondary_user = self.config.get('test', 'secondary_user')
        if not secondary_user:
            return

        self.safe_delete_user(secondary_user)
        self.safe_delete_role(TEST_ROLE_NAME)

        role = self.project.roles.create(TEST_ROLE_NAME)
        user = self.project.users.create(secondary_user)

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

        self.project.users.delete(secondary_user)
        self.project.roles.delete(role)

    def testSecurityQuery(self):
        assert 'ALIYUN' in self.odps.run_security_query('LIST ACCOUNTPROVIDERS')
        assert 'ALIYUN' in self.odps.execute_security_query('LIST ACCOUNTPROVIDERS')

        inst = self.odps.run_security_query(
            'INSTALL PACKAGE %s.non_exist_package' % self.project.name
        )
        assert isinstance(inst, self.project.AuthQueryInstance)

        with self.assertRaises(SecurityQueryError):
            inst.wait_for_success()
        assert inst.is_terminated
        assert not inst.is_successful

        with self.assertRaises(SecurityQueryError):
            self.odps.execute_security_query(
                'INSTALL PACKAGE %s.non_exist_package' % self.project.name
            )


if __name__ == '__main__':
    unittest.main()
