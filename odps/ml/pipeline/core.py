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

from collections import namedtuple, Iterable
from copy import deepcopy
from types import MethodType

from ...compat import OrderedDict, reduce, six
from ...utils import camel_to_underline, is_namedtuple
from ..expr.mixin import merge_data
from ..utils import FieldRole


class PipelineBit(object):
    def __init__(self, step, output):
        self._step = step
        self._output = output


class PipelineStep(object):
    def __init__(self, step_name, param_names=None, output_names=None):
        self._step_name = step_name
        self._ml_uplink = []
        self._param_names = [camel_to_underline(n) for n in param_names] if param_names is not None else []
        self._output_names = [camel_to_underline(n) for n in output_names] if output_names is not None else []
        self._pipeline = None
        self._input_args = []
        self._input_kwargs = dict()

    def __repr__(self):
        return self.__class__.__name__ + '(' + ', '.join('%s=%s' % (p, getattr(self, p)) for p in self._param_names
                                                         if p in dir(self) and getattr(self, p) is not None) + ')'

    def link(self, *args, **kwargs):
        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
        else:
            name = None
        self._input_args = [PipelineBit(v._final_step, v._final_step._output_names[0]) if isinstance(v, Pipeline) else v for v in args]
        self._input_kwargs = dict((k, PipelineBit(v._final_step, v._final_step._output_names[0]))
                                  if isinstance(v, Pipeline) else v for k, v in six.iteritems(kwargs))
        inputs = set(self._input_args) | set(six.itervalues(self._input_kwargs))
        first_input = next(iter(inputs))
        self._pipeline = first_input._step._pipeline
        self._ml_uplink = [bit._step for bit in inputs]
        self._pipeline._add_step(self, name=name)

        out_tuple = namedtuple('OutTuple', ' '.join(self._output_names))
        return out_tuple(**dict((out, PipelineBit(self, out)) for out in self._output_names))

    def _eval(self, method_name, result_dict, *args, **kwargs):
        if not (args or kwargs):
            args = [result_dict[b._step][b._output] for b in self._input_args]
            kwargs = dict((k, result_dict[b._step][b._output]) for k, b in six.iteritems(self._input_kwargs))
        results = getattr(self, method_name)(*args, **kwargs)

        # check namedtuple
        if is_namedtuple(results):
            result_dict[self] = results._asdict()
        else:
            if not isinstance(results, Iterable):
                results = (results, )
            result_dict[self] = OrderedDict(zip(self._output_names, results))


class GroupedStep(PipelineStep):
    def __init__(self, **kwargs):
        name = kwargs.get('name')
        super(GroupedStep, self).__init__(name, output_names=['output'])

        self._steps = []
        self._step_dict = dict()
        self._locked = False

    def __repr__(self):
        return self.__class__.__name__ + '(steps=[' + ', '.join('("%s", %s)' % step for step in self._steps) + '])'

    def __getitem__(self, item):
        if isinstance(item, six.integer_types):
            return self._steps[item][1]
        else:
            item = camel_to_underline(item)
            return self._step_dict[item]

    def __getattr__(self, item):
        if '__' not in item or item.startswith('_'):
            return object.__getattribute__(self, item)
        else:
            step_name, _ = item.split('__', 1)
            if step_name not in self._step_dict:
                raise AttributeError
            return object.__getattribute__(self, item)

    def __setattr__(self, item, value):
        if '__' not in item or item.startswith('_'):
            object.__setattr__(self, item, value)
        else:
            step_name, _ = item.split('__', 1)
            if step_name not in self._step_dict:
                raise AttributeError
            object.__setattr__(self, item, value)

    def _prop_setter_generator(self, step_obj, prop_name):
        def _setter(self, val):
            setattr(step_obj, prop_name, val)
            return self

        return MethodType(_setter, self)

    @staticmethod
    def _prop_generator(step_obj, prop_name):
        def _getter(self):
            return getattr(step_obj, prop_name)

        def _setter(self, val):
            setattr(step_obj, prop_name, val)

        return property(fget=_getter, fset=_setter)

    def _add_step(self, step, name=None):
        if self._locked:
            raise RuntimeError('This object has already been locked.')
        if isinstance(step, GroupedStep):
            step._locked = True

        if name is None:
            new_name = camel_to_underline(step._step_name)
        else:
            new_name = camel_to_underline(name)

        if step in self._step_dict:
            raise ValueError('Step names must be unique.')
        self._output_names = deepcopy(step._output_names)
        self._steps.append((new_name, step))
        self._step_dict[new_name] = step

        cls = type(self)
        if not hasattr(cls, '__perinstance'):
            cls = type(cls.__name__, (cls,), {})
            cls.__perinstance = True

        for step_prop in step._param_names:
            if step_prop in dir(step):
                prop_name = camel_to_underline(new_name) + '__' + camel_to_underline(step_prop)
                setattr(cls, 'set_' + prop_name, self._prop_setter_generator(step, step_prop))
                setattr(cls, prop_name, self._prop_generator(step, step_prop))
                self._param_names.append(prop_name)

        self.__class__ = cls

        return step


class Pipeline(GroupedStep):
    """
    Pipeline defines a sequence of algorithms that can be treated as one algorithm.

    :Examples:
    >>> # define a name-default algorithm sequence
    >>> Pipeline(Normalize(), LogisticRegression())
    >>> # define a named step in the algorithm sequence
    >>> Pipeline(('norm', Normalize()), LogisticRegression())
    >>> # define an output-specified step in the algorithm sequence
    >>> Pipeline(SplitWord(), (DocWordCount(), 'multi'), Word2Vec())
    """
    def __init__(self, *steps, **kwargs):
        name = kwargs.get('name')
        super(Pipeline, self).__init__(name=name if name is not None else 'pipeline')

        self._initial_step = None
        self._final_step = None

        last_step = last_output = None
        for name, step, output in self._normalize_step_input(steps):
            ret_step = self._add_step(step, name=name)
            ret_step._ml_uplink = [last_step, ] if last_step is not None else []
            ret_step._input_args = [PipelineBit(last_step, last_output or last_step._output_names[0])]\
                if last_step is not None else None
            ret_step._pipeline = self
            last_step = ret_step
            last_output = output

        self._final_output = last_output
        self._step_dict = dict(self._steps)

    @staticmethod
    def _normalize_step_input(steps):
        # merge input arrays
        listed = map(lambda s: [s, ] if isinstance(s, (tuple, PipelineStep)) else s, steps)
        condensed = reduce(lambda a, b: a + b, listed, [])

        def normalize_tuple(tp):
            if isinstance(tp, PipelineStep):
                tp = (tp, )
            output_name = None
            if isinstance(tp[0], PipelineStep):
                step = tp[0]
                step_name = camel_to_underline(step._step_name)
                if len(tp) >= 2:
                    output_name = tp[1]
            else:
                step_name = camel_to_underline(tp[0])
                step = tp[1]
                if len(tp) >= 3:
                    output_name = tp[2]
            return step_name, step, output_name

        return list(map(normalize_tuple, condensed))

    def _add_step(self, step, name=None):
        step = super(Pipeline, self)._add_step(step, name=name)
        if self._initial_step is None:
            self._initial_step = step
        self._final_step = step
        return step

    def _assert_valid(self):
        if not self._steps:
            raise ValueError('Cannot execute without any steps defined.')
        zero_input_nodes = [sobj for sname, sobj in self._steps if not sobj._ml_uplink]
        if len(zero_input_nodes) > 1:
            raise ValueError('Multiple input steps is not allowed.')
        rev_dict = dict((s, []) for _, s in self._steps)
        [rev_dict[ul].append(dl) for _, dl in self._steps for ul in dl._ml_uplink]
        mid_items = [s for s, dl in six.iteritems(rev_dict) if dl]
        last_items = [s for s, dl in six.iteritems(rev_dict) if not dl]
        if len(last_items) > 1:
            raise ValueError('Multiple output steps is not allowed.')
        for mid_item in mid_items:
            if 'transform' not in dir(mid_item):
                raise ValueError('Intermediate steps must have transform() method.')

    def _eval_steps(self, step, method_name, result_dict, args, kwargs):
        # run first step
        if step == self._initial_step:
            # initial step needs original args
            if len(self._steps) == 1:
                # if we only have one step, go directly
                self._initial_step._eval(method_name, result_dict, *args, **kwargs)
            else:
                self._initial_step._eval('transform', result_dict, *args, **kwargs)
        else:
            # pass raw args on
            [self._eval_steps(s, 'transform', result_dict, args, kwargs) for s in step._ml_uplink]
            if step != self._final_step:
                step._eval('transform', result_dict)
            else:
                step._eval(method_name, result_dict)

    def _eval_pl(self, method_name, *args, **kwargs):
        self._assert_valid()
        if method_name not in dir(self._final_step):
            raise ValueError('Final step does not contains a method called ' + method_name + '.')
        result_dict = dict()
        self._eval_steps(self._final_step, method_name, result_dict, args, kwargs)
        ret_val = result_dict[self._final_step]

        ret_type = namedtuple('PipelineResult', list(six.iterkeys(ret_val)))
        ret_val = ret_type(**ret_val)

        if len(ret_val) == 1:
            return ret_val[0]
        else:
            return ret_val

    def _eval(self, method_name, result_dict, *args, **kwargs):
        result = self._eval_pl(method_name, *args, **kwargs)
        if is_namedtuple(result):
            result_dict[self] = result._asdict()
        else:
            if not isinstance(result, Iterable):
                result = (result, )
            result_dict[self] = OrderedDict(zip(self._output_names, result))

    def train(self, df):
        return self._eval_pl('train', df)

    def transform(self, *args, **kwargs):
        return self._eval_pl('transform', *args, **kwargs)


class FeatureUnion(GroupedStep):
    """
    FeatureUnion defines a group of feature transform algorithms whose results are eventually merged into one.

    :Examples:
    >>> FeatureUnion(Normalize(), Standardize())
    """
    def __init__(self, *steps, **kwargs):
        name = kwargs.get('name')
        self._include_roles = kwargs.get('include_roles', [FieldRole.LABEL, FieldRole.WEIGHT])
        super(FeatureUnion, self).__init__(name=name if name is not None else 'feature_union')

        self._append_col_dict = dict()
        self.auto_rename = kwargs.get('auto_rename') or True

        for name, step, output, cols in self._normalize_step_input(steps):
            ret_step = self._add_step(step, name=name)
            ret_step._input_args = None
            ret_step._pipeline = self
            self._append_col_dict[name] = cols

        self._step_dict = dict(self._steps)
        self._param_names.append('auto_rename_col')

    @staticmethod
    def _normalize_step_input(steps):
        # merge input arrays
        listed = map(lambda s: [s, ] if isinstance(s, (tuple, PipelineStep)) else s, steps)
        condensed = reduce(lambda a, b: a + b, listed, [])

        def normalize_tuple(tp):
            if isinstance(tp, PipelineStep):
                tp = (tp, )
            output_name = None
            if isinstance(tp[0], PipelineStep):
                step = tp[0]
                step_name = camel_to_underline(step._step_name)
                if len(tp) >= 2:
                    output_name = tp[1]
            else:
                step_name = camel_to_underline(tp[0])
                step = tp[1]
                if len(tp) >= 3:
                    output_name = tp[2]
            if output_name and '[' in output_name:
                output_name, col_str = output_name.split('[', 1)
                cols = [c.strip() for c in col_str.strip(']').split(',')]
            else:
                cols = None
            return step_name, step, output_name, cols

        return list(map(normalize_tuple, condensed))

    def _assert_valid(self):
        if not self._steps:
            raise ValueError('Cannot execute without any steps defined.')
        for _, step in self._steps:
            if 'transform' not in dir(step):
                raise ValueError('Steps must have transform() method.')

    def _eval_fu(self, method_name, *args, **kwargs):
        self._assert_valid()
        result_dict = dict()
        [step._eval('transform', result_dict, *args, **kwargs) for _, step in self._steps]

        processed = reduce(lambda a, b: [(a[0], v) for v in six.itervalues(a[1])] + [(a[0], v) for v in six.itervalues(b[1])],
                           six.iteritems(result_dict))
        results = []
        role_exported = dict((r, False) for r in self._include_roles)
        for src, result in processed:
            user_inclusive = set(self._append_col_dict[src]) if src in self._append_col_dict else set()
            for r in six.iterkeys(role_exported):
                role_fields = [f.name for f in result._ml_fields if r in f.role]
                if role_fields and not role_exported[r]:
                    user_inclusive.update(role_fields)
                    role_exported[r] = True
            exclude_fields = [f.name for f in result._ml_fields
                              if FieldRole.FEATURE not in f.role and f.name not in user_inclusive]
            if exclude_fields:
                results.append((result, exclude_fields, True))
            else:
                results.append(result)

        return merge_data(*results, auto_rename=self.auto_rename)

    def _eval(self, method_name, result_dict, *args, **kwargs):
        result = self._eval_fu(method_name, *args, **kwargs)
        if is_namedtuple(result):
            result_dict[self] = result._asdict()
        else:
            if not isinstance(result, Iterable):
                result = (result, )
            result_dict[self] = OrderedDict(zip(self._output_names, result))

    def transform(self, *args, **kwargs):
        return self._eval_fu('transform', *args, **kwargs)
