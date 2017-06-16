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
from collections import namedtuple

from ... import options, compat
from . import _customize
from .utils import get_field_name_by_role, detect_metrics_fallback, metrics_result

try:
    import numpy as np
except ImportError:
    from warnings import warn
    warn('Numpy not installed. Metrics module will not be available.')
    np = None

from ..enums import FieldRole

logger = logging.getLogger(__name__)


def _run_cm_node(df, col_true, col_pred, execute_now=True, result_callback=None):
    detect_metrics_fallback(df)
    if not options.ml.use_old_metrics:
        if result_callback:
            cb = lambda v: result_callback(v.confusion_matrix)
        else:
            cb = lambda v: v.confusion_matrix

        eval_fun = getattr(_customize, 'eval_multi_class')
        eval_result = eval_fun(df, label_col=col_true, predict_col=col_pred,
                               execute_now=execute_now, _result_callback=cb)
        return eval_result
    else:
        cb = result_callback or (lambda v: v)
        cm_fun = getattr(_customize, '_confusion_matrix')
        return cm_fun(df, label_col=col_true, predict_col=col_pred,
                      execute_now=execute_now, _result_callback=cb)


def _run_roc_node(df, pos_label, col_true, col_pred, col_scores, execute_now=True, result_callback=None):
    detect_metrics_fallback(df)
    RocResult = _customize.RocResult

    if not options.ml.use_old_metrics:
        if result_callback:
            cb = lambda v: result_callback(RocResult(v.thresh, v.tp, v.fn, v.tn, v.fp))
        else:
            cb = lambda v: RocResult(v.thresh, v.tp, v.fn, v.tn, v.fp)

        eval_fun = getattr(_customize, 'eval_binary_class')
        eval_result = eval_fun(df, good_value=pos_label, label_col=col_true, score_col=col_scores,
                               execute_now=execute_now, _result_callback=cb)
        return eval_result
    else:
        result_callback = result_callback or (lambda v: v)
        roc_fun = getattr(_customize, '_roc')
        return roc_fun(df, good_value=pos_label, label_col=col_true, predict_col=col_pred,
                       score_col=col_scores, execute_now=execute_now, _result_callback=result_callback)


@metrics_result(_run_cm_node)
def confusion_matrix(df, col_true=None, col_pred=None):
    """
    Compute confusion matrix of a predicted DataFrame.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true label
    :type col_true: str
    :param col_true: column name of predicted label, 'prediction_result' by default.
    :type col_pred: str
    :return: Confusion matrix and mapping list for classes

    :Example:

    >>> predicted = model.predict(input_data)
    >>> cm, mapping = confusion_matrix(predicted, 'category')
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    return _run_cm_node(df, col_true, col_pred)


@metrics_result(_run_cm_node)
def accuracy_score(df, col_true=None, col_pred=None, normalize=True):
    """
    Compute accuracy of a predicted DataFrame.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param col_true: column name of true label
    :type col_true: str
    :param col_true: column name of predicted label, 'prediction_result' by default.
    :type col_pred: str
    :param normalize: denoting if the output is normalized between [0, 1]
    :type normalize: bool
    :return: Accuracy value
    :rtype: float
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    mat, _ = _run_cm_node(df, col_true, col_pred)
    if np is not None:
        acc_count = np.sum(np.diag(mat))
        if not normalize:
            return acc_count
        else:
            return acc_count * 1.0 / np.sum(mat)
    else:
        diag_sum = mat_sum = 0
        mat_size = len(mat)
        for i in compat.irange(mat_size):
            for j in compat.irange(mat_size):
                if i == j:
                    diag_sum += mat[i][j]
                mat_sum += mat[i][j]
        if not normalize:
            return diag_sum
        else:
            return diag_sum * 1.0 / mat_sum


@metrics_result(_run_cm_node)
def precision_score(df, col_true=None, col_pred='precision_result', pos_label=1, average=None):
    r"""
    Compute precision of a predicted DataFrame. Precision is defined as :math:`\frac{TP}{TP + TN}`

    :Parameters:
        - **df** - predicted data frame
        - **col_true** - column name of true label
        - **col_pred** - column name of predicted label, 'prediction_result' by default.
        - **pos_label** - denote the desired class label when ``average`` == `binary`
        - **average** - denote the method to compute average.
    :Returns:
        Precision score
    :Return type:
        float or numpy.array[float]

    The parameter ``average`` controls the behavior of the function.

    - When ``average`` == None (by default), precision of every class is given as a list.

    - When ``average`` == 'binary', precision of class specified in ``pos_label`` is given.

    - When ``average`` == 'micro', STP / (STP + STN) is given, where STP and STN are summations of TP and TN for every class.

    - When ``average`` == 'macro', average precision of all the class is given.

    - When ``average`` == `weighted`, average precision of all the class weighted by support of every true classes is given.

    :Example:

    Assume we have a table named 'predicted' as follows:

    ======== ===================
    label    prediction_result
    ======== ===================
    0        0
    1        2
    2        1
    0        0
    1        0
    2        1
    ======== ===================

    Different options of ``average`` parameter outputs different values:

.. code-block:: python

    >>> precision_score(predicted, 'label', average=None)
    array([ 0.66...,  0.        ,  0.        ])
    >>> precision_score(predicted, 'label', average='macro')
    0.22
    >>> precision_score(predicted, 'label', average='micro')
    0.33
    >>> precision_score(predicted, 'label', average='weighted')
    0.22
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    mat, label_list = _run_cm_node(df, col_true, col_pred)
    class_dict = dict((label, idx) for idx, label in enumerate(label_list))
    tps = np.diag(mat)
    pred_count = np.sum(mat, axis=0)
    if average is None:
        return tps * 1.0 / pred_count
    elif average == 'binary':
        class_idx = class_dict[pos_label]
        return tps[class_idx] * 1.0 / pred_count[class_idx]
    elif average == 'micro':
        return np.sum(tps) / np.sum(pred_count)
    elif average == 'macro':
        return np.mean(tps * 1.0 / pred_count)
    elif average == 'weighted':
        support = np.sum(mat, axis=1)
        return np.sum(tps * 1.0 / pred_count * support) / np.sum(support)


@metrics_result(_run_cm_node)
def recall_score(df, col_true=None, col_pred='precision_result', pos_label=1, average=None):
    r"""
    Compute recall of a predicted DataFrame. Precision is defined as :math:`\frac{TP}{TP + FP}`

    :Parameters:
        - **df* - predicted data frame
        - **col_true** - column name of true label
        - **col_pred** - column name of predicted label, 'prediction_result' by default.
        - **pos_label** - denote the desired class label when ``average`` == `binary`
        - **average** - denote the method to compute average.
    :Returns:
        Recall score
    :Return type:
        float | numpy.array[float]

    The parameter ``average`` controls the behavior of the function.

    * When ``average`` == None (by default), recall of every class is given as a list.

    * When ``average`` == 'binary', recall of class specified in ``pos_label`` is given.

    * When ``average`` == 'micro', STP / (STP + SFP) is given, where STP and SFP are summations of TP and FP for every class.

    * When ``average`` == 'macro', average recall of all the class is given.

    * When ``average`` == `weighted`, average recall of all the class weighted by support of every true classes is given.

    :Example:

    Assume we have a table named 'predicted' as follows:

    ======== ===================
    label    prediction_result
    ======== ===================
    0        1
    1        2
    2        1
    1        1
    1        0
    2        2
    ======== ===================

    Different options of ``average`` parameter outputs different values:

.. code-block:: python

    >>> recall_score(predicted, 'label', average=None)
    array([ 0.        ,  0.33333333,  0.5       ])
    >>> recall_score(predicted, 'label', average='macro')
    0.27
    >>> recall_score(predicted, 'label', average='micro')
    0.33
    >>> recall_score(predicted, 'label', average='weighted')
    0.33
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    mat, label_list = _run_cm_node(df, col_true, col_pred)
    class_dict = dict((label, idx) for idx, label in enumerate(label_list))
    tps = np.diag(mat)
    supp_count = np.sum(mat, axis=0)
    if average is None:
        return tps * 1.0 / supp_count
    elif average == 'binary':
        class_idx = class_dict[pos_label]
        return tps[class_idx] * 1.0 / supp_count[class_idx]
    elif average == 'micro' or average == 'weighted':
        return np.sum(tps) / np.sum(supp_count)
    elif average == 'macro':
        return np.mean(tps * 1.0 / supp_count)


@metrics_result(_run_cm_node)
def fbeta_score(df, col_true=None, col_pred='precision_result', beta=1.0, pos_label=1, average=None):
    r"""
    Compute f-beta score of a predicted DataFrame. f-beta is defined as

    .. math::

        \frac{1 + \beta^2 \cdot precision \cdot recall}{\beta^2 \cdot precision + recall}

    F-beta score is a generalization of f-1 score.

    :Parameters:
        - **df** - predicted data frame
        - **col_true** - column name of true label
        - **col_pred** - column name of predicted label, 'prediction_result' by default.
        - **pos_label** - denote the desired class label when ``average`` == `binary`
        - **average** - denote the method to compute average.
    :Returns:
        Recall score
    :Return type:
        float | numpy.array[float]

    The parameter ``average`` controls the behavior of the function.

    * When ``average`` == None (by default), f-beta of every class is given as a list.

    * When ``average`` == 'binary', f-beta of class specified in ``pos_label`` is given.

    * When ``average`` == 'micro', f-beta of overall precision and recall is given, where overall precision and recall are computed in micro-average mode.

    * When ``average`` == 'macro', average f-beta of all the class is given.

    * When ``average`` == `weighted`, average f-beta of all the class weighted by support of every true classes is given.

    :Example:

    Assume we have a table named 'predicted' as follows:

    ======== ===================
    label    prediction_result
    ======== ===================
    0        1
    1        2
    2        1
    1        1
    1        0
    2        2
    ======== ===================

    Different options of ``average`` parameter outputs different values:

.. code-block:: python

    >>> fbeta_score(predicted, 'label', average=None, beta=0.5)
    array([ 0.        ,  0.33333333,  0.5       ])
    >>> fbeta_score(predicted, 'label', average='macro', beta=0.5)
    0.27
    >>> fbeta_score(predicted, 'label', average='micro', beta=0.5)
    0.33
    >>> fbeta_score(predicted, 'label', average='weighted', beta=0.5)
    0.33
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    mat, label_list = _run_cm_node(df, col_true, col_pred)
    class_dict = dict((label, idx) for idx, label in enumerate(label_list))
    tps = np.diag(mat)
    pred_count = np.sum(mat, axis=0)
    supp_count = np.sum(mat, axis=1)
    beta2 = beta ** 2

    precision = tps * 1.0 / pred_count
    recall = tps * 1.0 / supp_count
    ppr = precision * beta2 + recall
    ppr[ppr == 0] = 1e-6

    fbeta = (1 + beta2) * precision * recall / ppr

    if average is None:
        return fbeta
    elif average == 'binary':
        class_idx = class_dict[pos_label]
        return fbeta[class_idx]
    elif average == 'micro':
        g_precision = np.sum(tps) * 1.0 / np.sum(supp_count)
        g_recall = np.sum(tps) * 1.0 / np.sum(pred_count)
        return (1 + beta2) * g_precision * g_recall / (beta2 * g_precision + g_recall)
    elif average == 'macro':
        return np.mean(fbeta)
    elif average == 'weighted':
        return sum(fbeta * supp_count) / sum(supp_count)


@metrics_result(_run_roc_node)
def f1_score(df, col_true=None, col_pred='precision_result', pos_label=1, average=None):
    r"""
    Compute f-1 score of a predicted DataFrame. f-1 is defined as

    .. math::

        \frac{2 \cdot precision \cdot recall}{precision + recall}


    :Parameters:
        - **df** - predicted data frame
        - **col_true** - column name of true label
        - **col_pred** - column name of predicted label, 'prediction_result' by default.
        - **pos_label** - denote the desired class label when ``average`` == `binary`
        - **average** - denote the method to compute average.
    :Returns:
        Recall score
    :Return type:
        float | numpy.array[float]

    The parameter ``average`` controls the behavior of the function.

    * When ``average`` == None (by default), f-1 of every class is given as a list.

    * When ``average`` == 'binary', f-1 of class specified in ``pos_label`` is given.

    * When ``average`` == 'micro', f-1 of overall precision and recall is given, where overall precision and recall are computed in micro-average mode.

    * When ``average`` == 'macro', average f-1 of all the class is given.

    * When ``average`` == `weighted`, average f-1 of all the class weighted by support of every true classes is given.


    :Example:

    Assume we have a table named 'predicted' as follows:

    ======== ===================
    label    prediction_result
    ======== ===================
    0        1
    1        2
    2        1
    1        1
    1        0
    2        2
    ======== ===================

    Different options of ``average`` parameter outputs different values:

.. code-block:: python

    >>> f1_score(predicted, 'label', average=None)
    array([ 0.        ,  0.33333333,  0.5       ])
    >>> f1_score(predicted, 'label', average='macro')
    0.27
    >>> f1_score(predicted, 'label', average='micro')
    0.33
    >>> f1_score(predicted, 'label', average='weighted')
    0.33
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    return fbeta_score(df, col_true, col_pred, pos_label=pos_label, average=average)


@metrics_result(_run_roc_node)
def roc_curve(df, col_true=None, col_pred=None, col_scores=None, pos_label=1):
    r"""
    Compute true positive rate (TPR), false positive rate (FPR) and threshold from predicted DataFrame.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param pos_label: positive label
    :type pos_label: str
    :param col_true: true column
    :type col_true: str
    :param col_pred: predicted column, 'prediction_result' if absent.
    :type col_pred: str
    :param col_scores: score column, 'prediction_score' if absent.
    :type col_scores: str

    :return: False positive rate, true positive rate and threshold, in numpy array format.

    :Example:

    >>> import matplotlib.pyplot as plt
    >>> fpr, tpr, thresh = roc_curve(predicted, "class")
    >>> plt.plot(fpr, tpr)
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    if not col_scores:
        col_scores = get_field_name_by_role(df, FieldRole.PREDICTED_SCORE)
    thresh, tp, fn, tn, fp = _run_roc_node(df, pos_label, col_true, col_pred, col_scores)

    if np is not None:
        tpr = tp * 1.0 / (tp + fn)
        fpr = fp * 1.0 / (fp + tn)
    else:
        tpr = [tp[i] * 1.0 / (tp[i] + fn[i]) for i in range(len(tp))]
        fpr = [fp[i] * 1.0 / (fp[i] + tn[i]) for i in range(len(fp))]

    roc_result = namedtuple('ROCResult', 'fpr tpr thresh')
    return roc_result(fpr=fpr, tpr=tpr, thresh=thresh)


@metrics_result(_run_roc_node)
def gain_chart(df, col_true=None, col_pred=None, col_scores=None, pos_label=1):
    r"""
    Compute positive proportion, true positive rate (TPR) and threshold from predicted DataFrame. The trace can be plotted as a cumulative gain chart

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param pos_label: positive label
    :type pos_label: str
    :param col_true: true column
    :type col_true: str
    :param col_pred: predicted column, 'prediction_result' if absent.
    :type col_pred: str
    :param col_scores: score column, 'prediction_score' if absent.
    :type col_scores: str

    :return: positive proportion, true positive rate and threshold, in numpy array format.

    :Example:

    >>> import matplotlib.pyplot as plt
    >>> depth, tpr, thresh = gain_chart(predicted)
    >>> plt.plot(depth, tpr)
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    if not col_scores:
        col_scores = get_field_name_by_role(df, FieldRole.PREDICTED_SCORE)
    thresh, tp, fn, tn, fp = _run_roc_node(df, pos_label, col_true, col_pred, col_scores)

    depth = (tp + fp) * 1.0 / (tp + fp + tn + fn)
    tpr = tp * 1.0 / (tp + fn)

    gain_result = namedtuple('GainChartResult', 'depth tpr thresh')
    return gain_result(depth=depth, tpr=tpr, thresh=thresh)


@metrics_result(_run_roc_node)
def lift_chart(df, col_true=None, col_pred=None, col_scores=None, pos_label=1):
    r"""
    Compute life value, true positive rate (TPR) and threshold from predicted DataFrame.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param pos_label: positive label
    :type pos_label: str
    :param col_true: true column
    :type col_true: str
    :param col_pred: predicted column, 'prediction_result' if absent.
    :type col_pred: str
    :param col_scores: score column, 'prediction_score' if absent.
    :type col_scores: str

    :return: lift value, true positive rate and threshold, in numpy array format.

    :Example:

    >>> import matplotlib.pyplot as plt
    >>> depth, lift, thresh = lift_chart(predicted)
    >>> plt.plot(depth, lift)
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    if not col_scores:
        col_scores = get_field_name_by_role(df, FieldRole.PREDICTED_SCORE)
    thresh, tp, fn, tn, fp = _run_roc_node(df, pos_label, col_true, col_pred, col_scores)

    depth = (tp + fp) * 1.0 / (tp + fp + tn + fn)
    tpr = tp * 1.0 / (tp + fn)
    lift = tpr / depth

    lift_result = namedtuple('LiftResult', 'depth lift thresh')
    return lift_result(depth=depth, lift=lift, thresh=thresh)


def auc(tpr, fpr):
    """
    Calculate AUC value from true positive rate (TPR) and false positive rate (FPR)\
    with trapezoidal rule.

    Note that calculation on DataFrames should use ``roc_auc_score`` instead.

    :param tpr: True positive rate array
    :param fpr: False positive rate array
    :return: AUC value
    :rtype: float
    """
    return abs(np.trapz(tpr, fpr))


@metrics_result(_run_roc_node)
def roc_auc_score(df, col_true=None, col_pred=None, col_scores=None, pos_label=1):
    """
    Compute Area Under the Curve (AUC) from prediction scores with trapezoidal rule.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param pos_label: positive label
    :type pos_label: str
    :param col_true: true column
    :type col_true: str
    :param col_pred: predicted column, 'prediction_result' if absent.
    :type col_pred: str
    :param col_scores: score column, 'prediction_score' if absent.
    :type col_scores: str
    :return: AUC value
    :rtype: float
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    if not col_scores:
        col_scores = get_field_name_by_role(df, FieldRole.PREDICTED_SCORE)
    thresh, tp, fn, tn, fp = _run_roc_node(df, pos_label, col_true, col_pred, col_scores)
    tpr = tp * 1.0 / (tp + fn)
    fpr = fp * 1.0 / (fp + tn)
    return auc(tpr, fpr)


@metrics_result(_run_roc_node)
def precision_recall_curve(df, col_true=None, col_pred=None, col_scores=None, pos_label=1):
    """
    Compute precision and recall value with different thresholds. These precision and recall\
      values can be used to plot a precision-recall curve.

    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param pos_label: positive label
    :type pos_label: str
    :param col_true: true column
    :type col_true: str
    :param col_pred: predicted column, 'prediction_result' if absent.
    :type col_pred: str
    :param col_scores: score column, 'prediction_score' if absent.
    :type col_scores: str
    :return: precision, recall and threshold, in numpy arrays.
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    if not col_scores:
        col_scores = get_field_name_by_role(df, FieldRole.PREDICTED_SCORE)
    thresh, tp, fn, tn, fp = _run_roc_node(df, pos_label, col_true, col_pred, col_scores)

    precisions = np.squeeze(np.asarray(tp * 1.0 / (tp + fp)))
    recalls = np.squeeze(np.asarray(tp * 1.0 / (tp + fn)))

    result_type = namedtuple('PrecisionRecallResult', 'precisions recalls thresh')
    return result_type(precisions=precisions, recalls=recalls, thresh=thresh)


@metrics_result(_run_roc_node)
def average_precision_score(df, col_true=None, col_pred=None, col_scores=None, pos_label=1):
    """
    Compute average precision score, i.e., the area under precision-recall curve.
    
    Note that this method will trigger the defined flow to execute.

    :param df: predicted data frame
    :type df: DataFrame
    :param pos_label: positive label
    :type pos_label: str
    :param col_true: true column
    :type col_true: str
    :param col_pred: predicted column, 'prediction_result' if absent.
    :type col_pred: str
    :param col_scores: score column, 'prediction_score' if absent.
    :type col_scores: str
    :return: Average precision score
    :rtype: float
    """
    if not col_pred:
        col_pred = get_field_name_by_role(df, FieldRole.PREDICTED_CLASS)
    if not col_scores:
        col_scores = get_field_name_by_role(df, FieldRole.PREDICTED_SCORE)
    thresh, tp, fn, tn, fp = _run_roc_node(df, pos_label, col_true, col_pred, col_scores)

    precisions = np.squeeze(np.asarray(tp * 1.0 / (tp + fp)))
    recalls = np.squeeze(np.asarray(tp * 1.0 / (tp + fn)))

    return np.trapz(precisions, recalls)
