# encoding: utf-8
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from ...utils import camel_to_underline
from ...compat import six, getargspec
from .core import PipelineStep
from ..expr.mixin import merge_data


class SimpleDataFrameStep(PipelineStep):
    def __init__(self, method, **kwargs):
        super(SimpleDataFrameStep, self).__init__(camel_to_underline(self.__class__.__name__), list(six.iterkeys(kwargs)),
                                                  ['output', ])
        self._method = method
        [setattr(self, k, v) for k, v in six.iteritems(kwargs)]

    def transform(self, *args, **kwargs):
        attr_vals = dict((k, getattr(self, k)) for k in self._param_names)
        return getattr(args[0], self._method)(**attr_vals)


class SimpleFunctionStep(PipelineStep):
    def __init__(self, func, **kwargs):
        argtuple = getargspec(func)
        defaults = [None, ] * (len(argtuple.args) - len(argtuple.defaults) - 1) + list(argtuple.defaults)
        new_kwargs = dict(zip(argtuple.args[1:], defaults))
        new_kwargs.update(kwargs)
        super(SimpleFunctionStep, self).__init__(camel_to_underline(self.__class__.__name__), list(six.iterkeys(new_kwargs)),
                                                 ['output', ])

        self._func = func
        [setattr(self, k, v) for k, v in six.iteritems(new_kwargs)]

    def transform(self, *args, **kwargs):
        attr_vals = dict((k, getattr(self, k)) for k in self._param_names)
        return self._func(args[0], **attr_vals)


class AppendID(SimpleDataFrameStep):
    def __init__(self, id_col_name):
        super(AppendID, self).__init__('append_id', id_col_name=id_col_name)


class Sample(SimpleDataFrameStep):
    def __init__(self, ratio, prob_field=None, replace=False):
        super(Sample, self).__init__('sample', ratio=ratio, prob_field=prob_field, replace=replace)


class MergeColumns(PipelineStep):
    def __init__(self, *data_frames, **kwargs):
        super(MergeColumns, self).__init__('merge_data', ['auto_rename', ], ['output', ])
        self.auto_rename = kwargs.get('auto_rename')

    def transform(self, *args, **kwargs):
        return merge_data(*args, **{'auto_rename': self.auto_rename})
