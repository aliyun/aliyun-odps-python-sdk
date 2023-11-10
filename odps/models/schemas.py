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

import logging

from .. import serializers
from ..compat import six
from ..errors import InternalServerError, InvalidParameter, MethodNotAllowed, NoSuchObject
from ..utils import with_wait_argument
from .core import Iterable
from .schema import Schema

logger = logging.getLogger(__name__)
_project_has_schema_api = dict()


def with_schema_api_fallback(fallback_fun, is_iter=False):
    def decorator(fun):
        @six.wraps(fun)
        def wrapper(self, *args, **kwargs):
            key = (self.parent.odps.endpoint, self.parent.name)
            kw = kwargs.copy()
            try:
                self._check_schema_api()
                result = fun(self, *args, **kw)
                _project_has_schema_api[key] = True
                return result
            except (MethodNotAllowed, InvalidParameter):
                _project_has_schema_api[key] = False
                return fallback_fun(self, *args, **kwargs)

        @six.wraps(fun)
        def iter_wrapper(self, *args, **kwargs):
            key = (self.parent.odps.endpoint, self.parent.name)
            kw = kwargs.copy()
            try:
                self._check_schema_api()
                for item in fun(self, *args, **kw):
                    yield item
                _project_has_schema_api[key] = True
            except (MethodNotAllowed, InvalidParameter):
                _project_has_schema_api[key] = False
                for item in fallback_fun(self, *args, **kwargs):
                    yield item

        return iter_wrapper if is_iter else wrapper
    return decorator


class Schemas(Iterable):
    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems')
    schemas = serializers.XMLNodesReferencesField(Schema, 'Schema')

    def __iter__(self):
        return self.iterate()

    def resource(self, client=None, endpoint=None):
        return self.parent.resource(client, endpoint=endpoint)

    def _check_schema_api(self):
        key = (self.parent.odps.endpoint, self.parent.name)
        if not _project_has_schema_api.get(key, True):
            raise MethodNotAllowed("Schema API not supported")

    def _iterate_legacy(self, name=None, owner=None):
        if name is not None or owner is not None:
            raise ValueError("Iterating schemas with name or owner not supported on current service")
        inst = self.parent.odps.execute_sql("SHOW SCHEMAS IN %s" % self.parent.name)
        schema_names = inst.get_task_results().get("AnonymousSQLTask").split("\n")
        for schema_name in schema_names:
            yield Schema(name=schema_name, parent=self, client=self._client)

    @with_schema_api_fallback(fallback_fun=_iterate_legacy, is_iter=True)
    def iterate(self, name=None, owner=None):
        params = {'expectmarker': 'true'}
        if name is not None:
            params['name'] = name
        if owner is not None:
            params['owner'] = owner
        schema_name = self._get_schema_name()
        if schema_name is not None:
            params['curr_schema'] = schema_name

        def _it():
            last_marker = params.get('marker')
            if 'marker' in params and \
                (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource() + "/schemas"
            resp = self._client.get(url, params=params)

            r = Schemas.parse(self._client, resp, obj=self)
            params['marker'] = r.marker

            return r.schemas

        while True:
            schemas = _it()
            if schemas is None:
                break
            for schema in schemas:
                yield schema

    @with_wait_argument
    def _create_legacy(self, obj=None, async_=False, **kwargs):
        schema_name = kwargs.pop("schema_name", obj)
        if isinstance(obj, Schema):
            schema_name = obj.name
        inst = self.parent.odps.run_sql(
            "CREATE SCHEMA %s.%s" % (self.parent.name, schema_name)
        )
        if not async_:
            inst.wait_for_success()
            return Schema(name=schema_name, parent=self, client=self._client)
        return inst

    @with_schema_api_fallback(fallback_fun=_create_legacy)
    def create(self, obj=None, **kwargs):
        kwargs.pop("async_", None)
        kwargs.pop("wait", None)

        if isinstance(obj, six.string_types):
            kwargs["name"] = obj
            obj = None
        schema = obj or Schema(parent=self, client=self._client, **kwargs)

        if schema.parent is None:
            schema._parent = self
        if schema._client is None:
            schema._client = self._client

        headers = {'Content-Type': 'application/xml'}
        data = schema.serialize()

        resource = self.resource() + "/schemas"
        self._client.post(resource, data, headers=headers)
        return schema

    @with_wait_argument
    def _delete_legacy(self, schema_name, async_=False):
        if isinstance(schema_name, Schema):
            schema_name = schema_name.name
        inst = self.parent.odps.run_sql(
            "DROP SCHEMA %s.%s" % (self.parent.name, schema_name)
        )
        if not async_:
            return inst.wait_for_success()
        return inst

    @with_schema_api_fallback(fallback_fun=_delete_legacy)
    def delete(self, schema_name, async_=False):
        if isinstance(schema_name, Schema):
            schema_name = schema_name.name
        resource = self.resource() + "/schemas/" + schema_name
        self._client.delete(resource)

    def _get(self, item):
        if isinstance(item, Schema):
            return item
        return Schema(name=item, parent=self, client=self._client)

    def _contains_legacy(self, item):
        try:
            next(self._get(item).functions)
        except StopIteration:
            pass
        except NoSuchObject:
            return False
        except InvalidParameter as ex:
            if "NoSuchObjectException" in str(ex):
                return False
            raise
        except InternalServerError as ex:
            if "invalid schema name" in str(ex).lower():
                return False
            raise

        return True

    @with_schema_api_fallback(fallback_fun=_contains_legacy)
    def __contains__(self, item):
        schema = self._get(item)
        try:
            schema.reload()
            return True
        except NoSuchObject:
            return False
