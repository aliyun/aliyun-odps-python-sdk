# encoding: utf-8
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

from abc import ABCMeta, abstractmethod
from copy import copy
from functools import partial

from .classification import *
from .regression import *
from ...compat import six


class BaseScorer(six.with_metaclass(ABCMeta, object)):
    def __init__(self, func, sign, kwargs):
        self._func = func
        self._sign = sign
        self._kwargs = kwargs

    @abstractmethod
    def __call__(self, df, col_true=None, col_pred=None, sample_weight=None):
        pass


class PredictScorer(BaseScorer):
    def __call__(self, df, col_true=None, col_pred=None, sample_weight=None, **kwargs):
        kw = copy(self._kwargs)
        kw.update(kwargs)
        if col_true:
            kw['col_true'] = col_true
        if col_pred:
            kw['col_pred'] = col_pred
        if kw.get('execute_now', True):
            if sample_weight is not None:
                return self._sign * self._func(df, sample_weight=sample_weight, **kw)
            else:
                return self._sign * self._func(df, **kw)
        else:
            if sample_weight is not None:
                expr = self._func(df, sample_weight=sample_weight, **kw)
            else:
                expr = self._func(df, **kw)
            old_callback = getattr(expr, '_result_callback', None) or (lambda v: v)
            expr._result_callback = lambda v: self._sign * old_callback(v)
            return expr


def make_scorer(func, greater_is_better=True, **kwargs):
    sign = 1 if greater_is_better else -1
    return PredictScorer(func, sign, kwargs)

mean_squared_error_scorer = make_scorer(mean_squared_error,
                                        greater_is_better=False)
mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                         greater_is_better=False)
mean_absolute_percentage_scorer = make_scorer(mean_absolute_percentage_error,
                                              greater_is_better=False)

accuracy_scorer = make_scorer(accuracy_score)
roc_auc_scorer = make_scorer(roc_auc_score)

SCORERS = dict(mean_absolute_error=mean_absolute_error_scorer,
               mean_squared_error=mean_squared_error_scorer,
               mean_absolute_percentage_error=mean_absolute_percentage_scorer,
               accuracy=accuracy_scorer,
               roc_auc=roc_auc_scorer)

for name, metric in [('precision', precision_score),
                     ('recall', recall_score), ('f1', f1_score)]:
    SCORERS[name] = make_scorer(metric)
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        SCORERS[qualified_name] = make_scorer(partial(metric, pos_label=None,
                                                      average=average))
