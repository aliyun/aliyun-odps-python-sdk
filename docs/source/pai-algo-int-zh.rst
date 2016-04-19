.. _pai_algo:

==============
调用算法
==============

算法是 PAI SDK 服务的核心。ODPS 中内嵌的算法通过 PAI SDK 的包装以更方便的形式提供给用户。

在 PAI SDK 中，算法依照其返回的对象分为 3 类：变换算法、训练算法以及度量算法。不论是哪种算法，都由算法参数、输入和输出三个部分
组成。算法参数是 ODPS 中的各个算法提供的用于调整和优化算法执行的参数，例如迭代次数、终止条件、正则化系数等等。输入和输出则是
算法待处理和完成处理的数据或者模型。输入数据中会包含一些元数据，例如特征列、标签列的选择等等。如果输入为 DataSet，那么输入参数
的元数据由 DataSet 的属性提供，在算法调用前，需要在 DataSet 上确保字段作用、连续性、稀疏性等设置正确。

PAI SDK 中的算法分类列表如下，使用前需要加载指定的包。详细的算法文档请参考 API 手册。

+-----------+---------+-----------------------+---------------------------------------------------------------+
|返回值分类 |算法类型 |包名                   |算法列表                                                       |
+===========+=========+=======================+===============================================================+
|变换       |分类     |odps.pai.classifiers   |KNN                                                            |
|           +---------+-----------------------+---------------------------------------------------------------+
|           |聚类     |odps.pai.clustering    |KMeans, EMCluster                                              |
|           +---------+-----------------------+---------------------------------------------------------------+
|           |预处理   |odps.pai.preprocessing |normalize, standardize, pivot, unpivot, modify_abnormal        |
|           +---------+-----------------------+---------------------------------------------------------------+
|           |统计     |odps.pai.statistics    |PrinCompAnalysis, feature_stats, replace_woe                   |
|           +---------+-----------------------+---------------------------------------------------------------+
|           |NLP      |odps.pai.nlp           |TFIDF, DocWordStat, SplitWord, TripleLDA, Word2Vec,            |
|           |         |                       |TextNormalize                                                  |
|           +---------+-----------------------+---------------------------------------------------------------+
|           |复杂网络 |odps.pai.network       |NodeDensity, MaximalConnectedComponent, TriangleCount,         |
|           |         |                       |EdgeDensity, PageRank, LabelPropagationClustering,             |
|           |         |                       |LabelPropagationClassification, KCore, SSSP                    |
|           +---------+-----------------------+---------------------------------------------------------------+
|           |推荐     |odps.pai.recommend     |Etrec, AssociationAnalysis                                     |
+-----------+---------+-----------------------+---------------------------------------------------------------+
|训练       |分类     |odps.pai.classifiers   |Xgboost, RandomForests, LogisticRegression, LinearSVM, GBDTLR, |
|           |         |                       |NaiveBayes, `KernelSVM`, `Discriminant`, `C5`                  |
|           +---------+-----------------------+---------------------------------------------------------------+
|           |回归     |odps.pai.regression    |Xgboost, GBDT, LinearRegression, `LinearSVR`,                  |
|           |         |                       |`LassoRegression`, `RidgeRegression`                           |
|           +---------+-----------------------+---------------------------------------------------------------+
|           |推荐     |odps.pai.recommend     |`AlsCF`, `SvdCF`                                               |
+-----------+---------+-----------------------+---------------------------------------------------------------+
|度量       |统计     |odps.pai.statistics    |histograms, pearson, t-test                                    |
|           +---------+-----------------------+---------------------------------------------------------------+
|           |复杂网络 |odps.pai.network       |modularity                                                     |
|           +---------+-----------------------+---------------------------------------------------------------+
|           |评估     |odps.pai.metrics       |accuracy_score, average_precision_score, confusion_matrix,     |
|           |         |                       |f1_score, fbeta_score, precision_recall_curve,                 |
|           |         |                       |precision_score, recall_score, roc_auc_score, roc_curve,       |
|           |         |                       |lift_chart, gain_chart, mean_squared_error,                    |
|           |         |                       |mean_absolute_error, mean_absolute_percentage_error            |
+-----------+---------+-----------------------+---------------------------------------------------------------+

调用方法
==========

PAI SDK 中的算法有两种可能的调用方式。以大写字母开头的算法需要先利用算法参数创建算法对象，再调用该对象的方法，传入需要处理的
数据集，如

.. code-block:: python

    labeled = ds.label_field('label')
    algo = LogisticRegression(epsilon=0.01)
    model = algo.train(labeled)

对于算法对象而言，除了在初始化时指定参数外，也可以通过 set 方法或属性值指定参数，如

.. code-block:: python

    # 构造后调用 set 方法
    lr = LogisticRegression().set_epsilon(0.01).set_max_iter(1000)
    # 构造后为属性赋值
    lr = LogisticRegression()
    lr.epsilon, lr.max_iter = 0.01, 1000

全小写并使用下划线连接的算法名可以直接以函数的形式进行调用。调用时，将输入数据作为参数列表的开头，算法参数以关键字参数的形式
传入即可调用。例如下面的例子中，ds 为输入数据集，interval_num 为算法参数。

.. code-block:: python

    hist, bins = histograms(ds, interval_num=20)

变换
============

使用对象定义的变换算法拥有一个 transform 方法，输入和输出为 1 个或多个数据集，具体的输入 / 输出个数可参考算法文档。通常，
在使用 transform 方法前需要指定特征字段，对于一些特定的算法可能还需要指定别的字段作用，或通过参数传递字段名称。

如果输出只有一个，则 transform 方法的输出即为 DataSet，如下面的 KNN 例子所示：

.. code-block:: python

    labeled = ds.label_field('class')
    predicted = KNN(k=5).transform(labeled, to_be_predict)
    predicted.store_odps('predicted')

如果有多个输出，transform 方法会返回一个 ``namedtuple`` 对象，其中的元素为输出数据集，可直接赋值，也可以根据 API 文档给出的
输出名称进行引用。如下面的例子：

.. code-block:: python

    # TripleLDA 有6个输出，我们只取第一个
    topic_word, _, _, _, _, _ = TripleLDA(topic_num=10).transform(word_count_ds)
    # K-Means 有两个输出，我们先取 namedtuple，再利用名称取出各个数据集
    out_tuple = KMeans(k=3).transform(ds)
    idx_ds = out_tuple.index # 样本归属
    centroid_ds = out_tuple.centroid # 聚类中心点

使用函数定义的算法对象则只需要将算法

训练
===========

训练算法拥有一个 train 方法，输入一个或多个数据集，输出一个模型对象。具体的输入 / 输出可参考算法文档。通常，在使用 train 方法
时，需要指定 DataSet 的特征字段和标签字段。对于稀疏数据，需要指定分隔符。决策树类的算法可能还需要指定字段是否连续。关于如何指定
字段作用请参考 :ref:`字段作用 <pai_field_role>` 一节。

输出的模型对象拥有 predict 方法，可以预测一个目标数据集。关于模型及预测的详细信息，可以参见 :ref:`模型 <pai_models>` 一节。

一个完整的使用训练算法的例子如下：

.. code-block:: python

    # 使用训练集训练模型
    model = LogisticRegression(epsilon=0.01).train(train)
    # 对测试集进行预测
    predicted = model.predict(test)

度量
==========

度量算法大都使用函数进行包装，但输出为数值或多个对象组成的 tuple。用户可以根据这些返回值获取数据集的信息，例如准确率、AUC、
皮尔森系数等统计信息，也可以通过返回的数组进行绘图等操作。

需要注意的是，由于要获取数据集的统计数据，度量算法会触发整个链路的执行。

我们以计算并绘制直方图作为使用度量算法的例子：

.. code-block:: python

    from odps.pai.statistics import *
    # 调用算法
    hists = histograms(ionosphere)
    # 获取一列的直方图数据
    hist, bins = hists['a04']
    # 绘图
    plt.bar(bins[:-1], hist, width=bins[1] - bins[0])
