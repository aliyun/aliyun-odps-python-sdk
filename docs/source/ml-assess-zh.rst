.. _ml_assess:

============
结果评估
============

PyODPS ML 提供了若干种方法用于评估算法的运行结果。涉及的评估方法如下：

+-------------+-------------------------------------------------------------+
|对应算法类型 |评估方法                                                     |
+=============+=============================================================+
|分类         |accuracy_score, average_precision_score, confusion_matrix,   |
|             |f1_score, fbeta_score, precision_recall_curve,               |
|             |precision_score, recall_score, roc_auc_score, roc_curve,     |
|             |lift_chart, gain_chart                                       |
+-------------+-------------------------------------------------------------+
|回归         |mean_squared_error, mean_absolute_error,                     |
|             |mean_absolute_percentage_error                               |
+-------------+-------------------------------------------------------------+

分类评估算法需要使用预测结果集中具有 LABEL、PREDICTED_LABEL 作用的字段。回归评估算法需要使用预测结果集中 具有 LABEL 和
PREDICTED_VALUE 作用的字段，ROC、AUC 等算法还需要 PREDICTED_SCORE 字段。这些字段作用在调用 predict 方法时都会被设置。
因而，在调用时，一般将算法名作为函数名，将带有原始标签和预测结果标签的数据集传入，即可获得评估结果，如下所示：

.. code-block:: python

    acc = accuracy_score(predicted)

如果这些字段名不存在，也可以手动通过参数传递。需要注意的的是，一部分算法不提供 PREDICTED_SCORE，计算 AUC、ROC 等评价指标时
可能出错。具体请参考算法 API 文档。

一些评估方法会调用相同的中间结果。例如，accuracy_score 和 precision_score 都调用了混淆矩阵的结果。为此，PyODPS ML 设计了
避免重复调用的机制。如果在计算 precision_score 前，已经计算了 accuracy_score，那么 precision_score 将直接利用 accuracy_score
求得的混淆矩阵，而不是另起炉灶重新计算一遍。

可以使用 matplotlib 等包绘制 PyODPS ML 评估算法返回的坐标数据，例如下面绘制 ROC 曲线的例子：

.. code-block:: python

    fpr, tpr, thresh = roc_curve(predicted)
    plt.plot(fpr, tpr)
