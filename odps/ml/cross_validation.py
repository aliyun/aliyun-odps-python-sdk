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

import time

from ..df import RandomScalar, Delay
from ..compat import six
from .utils import get_function_args
from .metrics import SCORERS

try:
    import numpy as np
    _has_numpy = True
except ImportError:
    _has_numpy = False


def k_fold(n_folds=3):
    def _calc(df):
        seed = int(time.time())
        rand_col = '_rand_key_%s' % seed
        old_columns = df.schema.names
        rand_df = df[df, RandomScalar(seed).rename(rand_col)].cache()

        for idx in range(n_folds):
            left_df = rand_df[idx * 1.0 / n_folds < getattr(rand_df, rand_col) <= (idx + 1) * 1.0 / n_folds].__getitem__(old_columns)
            right_df = rand_df[(getattr(rand_df, rand_col) > (idx + 1) * 1.0 / n_folds) |
                               (getattr(rand_df, rand_col) <= idx * 1.0 / n_folds)].__getitem__(old_columns)
            yield (left_df, right_df)
    return _calc


def train_test(train_size=None, test_size=None):
    if train_size and test_size and train_size + test_size < 1.0e-10:
        raise ValueError('Sum of train size and test size must be 1.0.')
    if train_size and (train_size < 0 or train_size >= 1.0):
        raise ValueError('Illegal train size.')
    if test_size and (test_size < 0 or test_size >= 1.0):
        raise ValueError('Illegal train size.')
    if test_size:
        train_size = 1.0 - test_size
    if not (train_size or test_size):
        train_size = 0.75

    def _calc(df):
        return tuple(df.split(train_size))

    return _calc


def cross_val_score(trainer, df, col_true=None, col_pred=None, col_score=None, scoring=None, cv=None,
                    fit_params=None, ui=None, async_=False, n_parallel=1, timeout=None,
                    close_and_notify=True, **kw):
    async_ = kw.get('async', async_)
    if cv is None:
        cv = k_fold()
    elif isinstance(cv, six.integer_types):
        cv = k_fold(cv)
    if scoring is None:
        scoring = SCORERS['accuracy']

    if fit_params:
        [setattr(trainer, param, val) for param, val in six.iteritems(fit_params)]

    kwargs = dict(col_true=col_true, col_pred=col_pred)
    if 'col_score' in get_function_args(scoring):
        kwargs['col_score'] = col_score

    delay = Delay()
    futures = []
    for train_df, test_df in cv(df):
        model = trainer.train(train_df)
        predicted = model.predict(test_df)
        metrics_expr = scoring(df=predicted, execute_now=False, **kwargs)
        futures.append(metrics_expr.execute(delay=delay))

    delay.execute(ui=ui, async_=async_, n_parallel=n_parallel, timeout=timeout, close_and_notify=close_and_notify)

    results = [f.result() for f in futures]
    if _has_numpy:
        return np.array(results)
    else:
        return results
