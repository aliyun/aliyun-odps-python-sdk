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

import logging

from ..enums import FieldRole
from .utils import get_field_name_by_role, metrics_result

logger = logging.getLogger(__name__)


def _run_evaluation_node(df, col_true, col_pred, execute_now=True, result_callback=None):
    from . import _customize
    eval_fun = getattr(_customize, '_eval_regression')
    return eval_fun(df, label_col=col_true, predict_col=col_pred,
                    execute_now=execute_now, _result_callback=result_callback)


@metrics_result(_run_evaluation_node)
def mean_squared_error(df, col_true, col_pred=None):
    """
    Compute mean squared error of a predicted DataFrame.

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
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['mse']


@metrics_result(_run_evaluation_node)
def mean_absolute_error(df, col_true, col_pred=None):
    """
    Compute mean absolute error of a predicted DataFrame.

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
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['mae']


@metrics_result(_run_evaluation_node)
def mean_absolute_percentage_error(df, col_true, col_pred=None):
    """
    Compute mean absolute percentage error of a predicted DataFrame.

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
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['mape']


@metrics_result(_run_evaluation_node)
def total_sum_of_squares(df, col_true, col_pred=None):
    """
    Compute total sum of squares (SST) of a predicted DataFrame.

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
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['sst']


@metrics_result(_run_evaluation_node)
def explained_sum_of_squares(df, col_true, col_pred=None):
    """
    Compute explained sum of squares (SSE) of a predicted DataFrame.

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
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['sse']


@metrics_result(_run_evaluation_node)
def r2_score(df, col_true, col_pred=None):
    """
    Compute determination coefficient (R2) of a predicted DataFrame.

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
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['r2']


@metrics_result(_run_evaluation_node)
def multi_corr(df, col_true, col_pred=None):
    """
    Compute multiple correlation coefficient (R) of a predicted DataFrame.

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
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['r']


@metrics_result(_run_evaluation_node)
def rooted_mean_squared_error(df, col_true, col_pred=None):
    """
    Compute rooted mean squared error (RMSE) of a predicted DataFrame.

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
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['rmse']


@metrics_result(_run_evaluation_node)
def mean_absolute_deviation(df, col_true, col_pred=None):
    """
    Compute mean absolute deviation (MAD) of a predicted DataFrame.

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
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['mad']


@metrics_result(_run_evaluation_node)
def residual_histogram(df, col_true, col_pred=None):
    """
    Compute histogram of residuals of a predicted DataFrame.

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
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_VALUE)
    return _run_evaluation_node(df, col_true, col_pred)['hist']
