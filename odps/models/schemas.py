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

from ..errors import InternalServerError, NoSuchObject
from .core import Iterable
from .schema import Schema


class Schemas(Iterable):
    def __iter__(self):
        return self.iterate()

    def resource(self, client=None):
        return self.parent.resource(client)

    def iterate(self):
        inst = self.parent.odps.execute_sql("SHOW SCHEMAS IN %s" % self.parent.name)
        schema_names = inst.get_task_results().get("AnonymousSQLTask").split("\n")
        for schema_name in schema_names:
            yield Schema(name=schema_name, parent=self, client=self._client)

    def create(self, schema_name, async_=False):
        inst = self.parent.odps.run_sql(
            "CREATE SCHEMA %s.%s" % (self.parent.name, schema_name)
        )
        if not async_:
            inst.wait_for_success()
            return Schema(name=schema_name, parent=self, client=self._client)
        return inst

    def delete(self, schema_name, async_=False):
        if isinstance(schema_name, Schema):
            schema_name = schema_name.name
        inst = self.parent.odps.run_sql(
            "DROP SCHEMA %s.%s" % (self.parent.name, schema_name)
        )
        if not async_:
            return inst.wait_for_success()
        return inst

    def _get(self, item):
        return Schema(name=item, parent=self, client=self._client)

    def __contains__(self, item):
        try:
            next(self._get(item).functions)
        except StopIteration:
            pass
        except NoSuchObject:
            return False
        except InternalServerError as ex:
            if "invalid schema name" in str(ex).lower():
                return False
            raise

        return True
