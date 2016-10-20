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

import ast
import keyword
import sys
from . import six


if six.PY3:
    def isvalidattr(ident):
        return not keyword.iskeyword(ident) and ident.isidentifier()
else:
    def isvalidattr(ident):
        """Determines, if string is valid Python identifier."""

        # Smoke test - if it's not string, then it's not identifier, but we don't
        # want to just silence exception. It's better to fail fast.
        if not isinstance(ident, str):
            raise TypeError('expected str, but got {!r}'.format(type(ident)))

        # Resulting AST of simple identifier is <Module [<Expr <Name "foo">>]>
        try:
            root = ast.parse(ident)
        except SyntaxError:
            return False

        if not isinstance(root, ast.Module):
            return False

        if len(root.body) != 1:
            return False

        if not isinstance(root.body[0], ast.Expr):
            return False

        if not isinstance(root.body[0].value, ast.Name):
            return False

        if root.body[0].value.id != ident:
            return False

        return True


if six.PY3:
    def dir2(obj):
        return object.__dir__(obj)
else:
    # http://www.quora.com/How-dir-is-implemented-Is-there-any-PEP-related-to-that
    def get_attrs(obj):
        import types
        if not hasattr(obj, '__dict__'):
            return []  # slots only
        if not isinstance(obj.__dict__, (dict, types.DictProxyType)):
            raise TypeError("%s.__dict__ is not a dictionary"
                            "" % obj.__name__)
        return obj.__dict__.keys()


    def dir2(obj):
        attrs = set()
        if not hasattr(obj, '__bases__'):
            # obj is an instance
            if not hasattr(obj, '__class__'):
                # slots
                return sorted(get_attrs(obj))
            klass = obj.__class__
            attrs.update(get_attrs(klass))
        else:
            # obj is a class
            klass = obj

        for cls in klass.__bases__:
            attrs.update(get_attrs(cls))
            attrs.update(dir2(cls))
        attrs.update(get_attrs(obj))
        return sorted(list(attrs))


if tuple(sys.version_info[:2]) < (3, 5):
    def raise_exc(ex_type, ex_value, tb):
        glb = dict(ex_type=ex_type, ex=ex_value, tb=tb)
        six.exec_('raise ex_type, ex, tb', glb, locals())
else:
    def raise_exc(ex_type, ex_value, tb):
        raise ex_value
