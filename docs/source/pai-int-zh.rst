.. _pai:

========
PAI
========

PAI 是 ODPS 提供的算法支持。用户可以使用 PAI 利用 ODPS 内置的算法支持来进行数据挖掘和机器学习。

在 pyodps 中，用户主要通过 ``DataSet`` 和 ``TrainedModel`` 对象来调用 PAI 提供的方法。``DataSet``
提供了对数据集的基本操作，而 ``TrainedModel`` 提供了对模型的基本操作。

基本操作
--------

在使用 PAI 前，需要创建 ``PAIContext`` 上下文对象。``PAIContext`` 可以从 ``ODPS`` 对象创建，如

>>> from odps import ODPS
>>> from odps.internal.pai import PAIContext
>>> odps = ODPS('**your-access-id**', '**your-secret-access-key**', '**your-default-project**',
                endpoint='**your-end-point**')
>>> context = PAIContext(odps)

此后，可以从该上下文对象中取出一个数据集如下：

>>> # 无分区的表
>>> dataset1 = context.odps_data('table_name')
>>> # 有分区的表
>>> dataset2 = context.odps_data('table_name', 'partition_name')

对数据集可以使用拆分操作如下，0.6为拆分比例，返回值为两个数据集组成的数组：

>>> splited = dataset1.split(0.6)

此后，可以将某个数据集保存回 ODPS：

>>> splited[0].store_odps('splited_table')

上下文中也可以获取模型：

>>> model = context.odps_model('model_name')

用户可以获取模型的 PMML 表示并打印之：

>>> print(model.export_pmml())

也可以将模型保存为 ODPS 的 OfflineModel：

>>> model.store_odps('saved_model_name')

延迟执行
--------

为了使 PAI 流程的搭建更加连贯，PAI SDK 采用了延迟执行的设计。对于数据存储、数据收集的操作，PAI SDK 将会
立即执行整个流程。对于其他操作，例如数据变换、训练、预测过程，在代码执行时，并不会直接运行具体的步骤，而
是建立数据集间的连接关系。

例如下列的代码：

>>> dataset = context.odps_data(IONOSPHERE_TABLE)
>>> splited = dataset.split(0.6)
>>>
>>> labeled_data = splited[0].set_label_field("class")
>>> lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
>>> model = lr.train(labeled_data)
>>> model.store_odps("testOutModel")
>>>
>>> predicted = model.predict(splited[1])
>>> predicted.store_odps("testOut")
>>>
>>> fpr, tpr, thresh = roc_curve(predicted, "class")

``model`` 和 ``predicted`` 上的 ``store_odps`` 方法在执行时，分别会执行整个流程。即，对于
``model.store_odps`` ，会执行下列流程：

#. dataset.split
#. splited[0].set_label_field
#. LogisticRegression
#. lr.train
#. model.store_odps

对于 ``predicted.store_odps`` ，会执行下列流程：

#. model.predict
#. predicted.store_odps

使用算法
--------

分类算法
~~~~~~~~~
PAI 提供了数个分类算法。这些算法以 ``BaseSupervisedAlgorithm`` 为基类，可通过 ``odps.pai.algorithms.classifiers``
包进行引用。classifiers中的每个类对应一个具体的算法，算法参数可参考 PAI 命令手册。

每个算法均打包为一个类。每个类的初始化原型均为

>>> def __init__(self, **kwargs):
>>>     # initialization

kwargs 中给出算法的参数。在类初始化后，也可以通过 set_param 方法设置各个参数。这一系列的 set
方法均返回算法对象本身，因而 set 方法可以连续调用。例如下列逻辑回归算法的示例：

>>> lr = LogisticRegression(epsilon=0.001).set_max_iter(50)

如果不需要 Fluent 样式的调用，设置算法参数也可以直接向属性赋值，获得的效果是一样的：

>>> lr.max_iter = 50

与 PAI 命令使用驼峰式命名作为参数名称， PAI SDK 使用的参数名称为下划线连接的格式。

分类算法拥有一个 ``train`` 方法，输入为一个数据集，该数据集需要事先通过 ``set_label_field`` 方法
设置标签列。 ``select_fields`` 方法可以进一步地选择所需要的列。方法返回一个 ``TrainedModel`` 对象，
可以用于数据的预测。

结果评估
--------
PAI SDK 提供了混淆矩阵、ROC等方法用于评估算法的运行结果。不同类型的算法评估组件列表如下：

+-------+------------------------------------------+
| 类型  | 包名                                     |
+=======+==========================================+
| 分类  | odps.internal.pai.metrics.classification |
+-------+------------------------------------------+

关于每一种评估方法具体的使用细节，请参考各个方法的 References。
