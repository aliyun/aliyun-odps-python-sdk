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

import logging

from ..enums import FieldRole
from ...runner import adapter_from_df

logger = logging.getLogger(__name__)


def _get_field_name_by_role(df, role):
    adapter = adapter_from_df(df)
    fields = [f for f in adapter._fields if role in f.role]
    if not fields:
        raise ValueError('Input df does not contain a field with role %s.' % role.name)
    return fields[0].name


def _run_evaluation_node(df, col_true, col_pred):
    from . import _customize
    eval_fun = getattr(_customize, '_eval_regression')
    return eval_fun(df, label_col=col_true, predict_col=col_pred)


def mean_squared_error(df, col_true, col_pred=None):
    """
    Compute mean squared error of a predicted data set.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true value
    :type col_true: str
    :param col_true: column name of predicted value, 'prediction_score' by default.
    :type col_pred: str
    :return: Mean squared error
    :rtype: float
    """
    if not col_pred:
        col_pred = _get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['mse']


def mean_absolute_error(df, col_true, col_pred=None):
    """
    Compute mean absolute error of a predicted data set.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true value
    :type col_true: str
    :param col_true: column name of predicted value, 'prediction_score' by default.
    :type col_pred: str
    :return: Mean absolute error
    :rtype: float
    """
    if not col_pred:
        col_pred = _get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['mae']


def mean_absolute_percentage_error(df, col_true, col_pred=None):
    """
    Compute mean absolute percentage error of a predicted data set.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true value
    :type col_true: str
    :param col_true: column name of predicted value, 'prediction_score' by default.
    :type col_pred: str
    :return: Mean absolute percentage error
    :rtype: float
    """
    if not col_pred:
        col_pred = _get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['mape']


def total_sum_of_squares(df, col_true, col_pred=None):
    """
    Compute total sum of squares (SST) of a predicted data set.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true value
    :type col_true: str
    :param col_true: column name of predicted value, 'prediction_score' by default.
    :type col_pred: str
    :return: Mean absolute percentage error
    :rtype: float
    """
    if not col_pred:
        col_pred = _get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['sst']


def explained_sum_of_squares(df, col_true, col_pred=None):
    """
    Compute explained sum of squares (SSE) of a predicted data set.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true value
    :type col_true: str
    :param col_true: column name of predicted value, 'prediction_score' by default.
    :type col_pred: str
    :return: Mean absolute percentage error
    :rtype: float
    """
    if not col_pred:
        col_pred = _get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['sse']


def r2_score(df, col_true, col_pred=None):
    """
    Compute determination coefficient (R2) of a predicted data set.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true value
    :type col_true: str
    :param col_true: column name of predicted value, 'prediction_score' by default.
    :type col_pred: str
    :return: Mean absolute percentage error
    :rtype: float
    """
    if not col_pred:
        col_pred = _get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['r2']


def multi_corr(df, col_true, col_pred=None):
    """
    Compute multiple correlation coefficient (R) of a predicted data set.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true value
    :type col_true: str
    :param col_true: column name of predicted value, 'prediction_score' by default.
    :type col_pred: str
    :return: Mean absolute percentage error
    :rtype: float
    """
    if not col_pred:
        col_pred = _get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['r']


def rooted_mean_squared_error(df, col_true, col_pred=None):
    """
    Compute rooted mean squared error (RMSE) of a predicted data set.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true value
    :type col_true: str
    :param col_true: column name of predicted value, 'prediction_score' by default.
    :type col_pred: str
    :return: Mean absolute percentage error
    :rtype: float
    """
    if not col_pred:
        col_pred = _get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['rmse']


def mean_absolute_deviation(df, col_true, col_pred=None):
    """
    Compute mean absolute deviation (MAD) of a predicted data set.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true value
    :type col_true: str
    :param col_true: column name of predicted value, 'prediction_score' by default.
    :type col_pred: str
    :return: Mean absolute percentage error
    :rtype: float
    """
    if not col_pred:
        col_pred = _get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['mad']


def residual_histogram(df, col_true, col_pred=None):
    """
    Compute histogram of residuals of a predicted data set.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true value
    :type col_true: str
    :param col_true: column name of predicted value, 'prediction_score' by default.
    :type col_pred: str
    :return: histograms for every columns, containing histograms and bins.
    """
    if not col_pred:
        col_pred = _get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['hist']
