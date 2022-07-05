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

from .utils import metrics_result


def _run_evaluation_node(df, model, cols, execute_now=True, result_callback=None):
    from . import _customize
    eval_fun = getattr(_customize, '_eval_clustering')
    return eval_fun(df, model, cols=cols, execute_now=execute_now, _result_callback=result_callback)


@metrics_result(_run_evaluation_node)
def calinhara_score(df, model, cols=None):
    r"""
    Calculate Calinski-Harabasz coefficient for a clustering model.

    Calinski-Harabasz coefficient, also known as VRC (variance ratio criterion), is defined as

    .. math::

        VRC_k=\frac{SS_B}{SS_W}\times\frac{n-k}{k-1}

    Where :math:`SS_B` is the inter-cluster variance matrix, :math:`SS_W` is the intra-cluster variance matrix, while :math:`N` is the number of instances and :math:`k` is the number of cluster centers.

    :math:`SS_B` is defined as follows

    .. math::

        SS_B=\sum_{i=1}^k n_i \left\lVert m_i - \overline{m} \right\rVert^2

    While :math:`SS_W` is defined as follows

    .. math::

        SS_W = \sum_{i=1}^k \sum_{x \in c_i} \left\lVert x - m_i \right\rVert^2

    :param df: input DataFrame
    :param model: model generated by a clustering algorithm. Currently only k-Means is supported.
    :param cols: selected columns. Feature columns on ``df`` by default.
    """
    return _run_evaluation_node(df, model, cols)['calinhara']
