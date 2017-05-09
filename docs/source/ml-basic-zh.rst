.. _ml_basic:
.. py:currentmodule:: odps.ml


============
基本概念
============

PyODPS ML 中用户需要接触的对象有三种，分别是 `DataFrame`、`Model` 和 `Algorithm`，分别对应数据、模型和算法。

在使用 PyODPS ML 之前，需要初始化 :class:`ODPS` 对象。

===========
使用数据
===========

PyODPS ML 使用 :class:`DataFrame` 维护数据。在 DataFrame 的基础上，PyODPS ML 增加了在算法中使用的元数据信息，包括

=========== ==========================================================================================
属性         说明
=========== ==========================================================================================
字段类型      字段在ODPS中的类型
字段作用      字段在算法中的作用，包括特征、权重、标签等通用类型以及图算法中的起始顶点、终止顶点等特殊类型
字段连续性     字段的连续性，默认对 BigInt、String 和 DateTime 为离散，其余为连续，对GBDT等算法有意义
字段稀疏性     指定字段是否稀疏，以及稀疏字段的 KV 和项目间分隔符
=========== ==========================================================================================

加入这些信息的目的在于将数据本身的属性和算法参数相分离。用户在某个数据上调用不同类型的算法时，无需反复考虑哪些字段需要
作为算法的输入，PyODPS ML 将帮助用户完成上述所有的步骤。如果对某字段调用相关方法后，后续没有在该字段上调用与之冲突的设置
方法或算法节点，那么该字段上的设置将不会变化。

生成数据集
==============

PyODPS ML 中获取 DataFrame 的方法与一般的 DataFrame 相同，即

.. code-block:: python

    df = DataFrame(o.get_table('ionosphere'))

如果表为分区表，可以使用 filter_partition 方法指定输入的分区名。该方法针对 XFlow 进行了优化。

.. code-block:: python

    df = DataFrame(o.get_table('ionosphere_parted')).filter_partition('pt=20160101')

也可以使用 roles 方法指定字段的用途，如

.. code-block:: python

    df = DataFrame(o.get_table('iris')).roles(label='category')

DataFrame 也可以通过其他 DataFrame 通过变换得到。这一变换可以通过调用 DataFrame 上的方法完成，，也可以调用用 PyODPS ML 中
的算法获得。大多数方法调用后都会返回一个新的 DataFrame 对象。除了标准的 DataFrame 方法外，PyODPS ML 还增加了下述方法：

================ ==========================================================================================
方法类型          方法名
================ ==========================================================================================
字段作用设置       select_features, exclude_fields, label_field, weight_field, xxxx_fields
字段连续性设置     set_continuous, set_discrete
字段稀疏性设置     set_key_value, erase_key_value
数据变换           append_id, split, sql_transform
数据摘要           calc_summary
================ ==========================================================================================

获取字段列表
============
字段列表可以通过 DataFrame 的 dtypes 属性获得，通过 Jupyter Notebook 可以直接看到字段作用：

.. parsed-literal::

    odps.Schema {
      sepal_length            float64        FEATURE
      sepal_width             float64        FEATURE
      petal_length            float64        FEATURE
      petal_width             float64        FEATURE
      category                string         LABEL
    }

保存数据集
===========

使用 DataFrame 的相关方法可以将数据保存回 ODPS，如

.. code-block:: python

    df.persist('result_table')

.. _ml_field_role:

字段作用
===========

PyODPS ML 中，可以指定各个字段的作用，这些作用将在 DataFrame 传入的算法中使用。一个字段可以有多个作用，用于针对不同的算法。

PyODPS ML 中的字段作用有：

=================== ========== ============== ===========================
作用名               定义于     排除特征        描述
=================== ========== ============== ===========================
FEATURE              通用                      特征字段
LABEL                通用       是             标签字段
WEIGHT               通用       是             权重字段
PREDICTED_CLASS      通用       否             预测类别（算法自动标注）
PREDICTED_SCORE      通用       否             预测类别分值（算法自动标注）
PREDICTED_VALUE      通用       否             预测值（算法自动标注）
FROM_VERTEX          复杂网络   是             源顶点
TO_VERTEX            复杂网络   是             目标顶点
VERTEX_ID            复杂网络   是             顶点 ID
VERTEX_LABEL         复杂网络   是             顶点标签
FROM_VERTEX_LABEL    复杂网络   是             源点标签字段
TO_VERTEX_LABEL      复杂网络   是             目标点标签字段
VERTEX_WEIGHT        复杂网络   否             顶点权重字段
EDGE_WEIGHT          复杂网络   否             边权重字段
DOC_ID               NLP        是             文档 ID 字段
DOC_CONTENT          NLP        否             文档内容字段
WORD                 NLP        否             单词字段
WORD_COUNT           NLP        否             单词计数字段
REC_USER_ID          推荐       是             用户 ID 字段
REC_ITEM             推荐       是             商品字段
REC_SEQUENCE         推荐       否             事务顺序字段
REC_PAYLOAD          推荐       否             商品计数字段
TS_GROUP             时间序列   是             时间序列分组字段
TS_SEQ               时间序列   是             时间序列序号字段
TS_VALUE             时间序列   否             时间序列值字段
=================== ========== ============== ===========================

需要注意的是，在 DataFrame 上标注这些内容并不意味着算法一定支持这些标签，需要参考算法文档是否有相关字段选择参数再作判断。

PyODPS ML 默认一个 DataFrame 上的所有字段均为特征字段。xxx_field 方法可以将某个字段设为其他作用，而 exclude_fields 会
将字段排除出特征。大多数 xxx_field 方法会在设置字段作用的同时将该字段排除出特征字段，但这并不是肯定的，如上表所示。

设置字段作用的例子如下：

.. code-block:: python

    df = DataFrame(o.get_table('iris'))
    labeled = dataset.label_field('category')

此外，为了简便标签的设置，也可以使用 roles 方法进行设置，例子如下：

.. code-block:: python

    marked_df = df.roles(label='category')

稀疏数据
=========================
部分算法支持使用 Key-Value 格式输入稀疏数据，这在高维情形下会非常有用。用户可以通过 `DataFrame.key_value` 标注一个字符串
字段为稀疏字段，并指定其分隔符。设置方法为指定列名，同时指定分隔符。也可以使用 `DataFrame.erase_key_value` 清除字段上的
Key-Value 标注。

.. code-block:: python

    kv_df = df.key_value('f1 f2', kv=':', item=',')

由于算法的限制，PyODPS ML 仅支持对一个数据集采用一种分隔符。对于不支持稀疏的算法（其参数中无 Delimiter 选项），使用稀疏字段会
导致算法执行出错。

连续性
=========================
对于决策树等算法，字段连续性决定了该字段在算法中的处理方式。为了方便地处理连续性，PyODPS ML 规定，默认 double 和 bigint 类型字段
为连续字段，其他字段为离散字段。如果需要手工设置字段连续性，可以使用 `DataFrame.continuous` 和
`DataFrame.discrete` 这两个方法。使用方式如下：

.. code-block:: python

    new_df = df.continuous('f1 f2')
    new_df2 = df.discrete('f1 f2')


.. _ml_models:

===========
模型
===========

模型是 PyODPS ML 中训练算法输出的训练结果。根据算法的不同，PyODPS ML 提供了不同类型的模型，其中包括 PMML 模型（即 ODPS 线下模型）、
、表模型等。所有的模型都以 :class:`MLModel` 为基类，提供 ``predict`` 方法用于在数据集上进行预测。

PMML 模型
==========
PMML 模型（:class:`PmmlModel`）针对 ODPS 中的线下模型（OfflineModel），提供了模型载入、存储和预测的功能。

PyODPS ML 中无法显示创建一个模型。模型需要通过算法生成，例如下面通过逻辑回归算法生成一个模型：

.. code-block:: python

    pmml_model = LogisticRegression().train(df)

生成模型后，可将其存储为 ODPS 线下模型（OfflineModel），此后可使用 PmmlModel 的构造函数从 ODPS 中重新读取：

.. code-block:: python

    pmml_model.persist('model_name')
    reloaded_model = PmmlModel(o.get_offline_model('model_name'))

使用 :func:`PmmlModel.execute` 方法可以获取模型的 Pmml，该方法返回一个 :class:`PmmlResult` 对象，可获取其 pmml 属性：

.. code-block:: python

    result = pmml_model.execute()
    print(result.pmml)

目前，PyODPS ML 支持对结果中的随机森林模型以及逻辑回归模型进行可视化。

对于随机森林模型，:class:`PmmlResult` 中可通过下标读取每一颗决策树。在决策树中，可以通过 root 属性获得根节点，并对
决策树进行遍历。在 Jupyter Notebook 中，也可以直接对节点进行可视化，如下面的代码，在模型中获取 ID 为 0 的决策树。如果安装有
GraphViz，那么将显示 SVG 格式的决策树，否则将显示文本格式的决策树：

.. code-block:: python

    result = model.execute()
    result[0]

对于逻辑回归模型，迭代 :func:`PmmlResult` 方法可以获得各个计算公式。

可以使用模型的 :func:`PmmlModel.predict` 方法对数据集进行预测操作。该方法的输出为一个新的 DataFrame，除了原有列之外，还会附加
三个新字段。不同算法对这些字段的定义可能会不同。常见的预测结果列见下表：

==================== ======== ====================================================
 字段名               类型      注释
==================== ======== ====================================================
 prediction_result    string   分类算法预测标签，回归算法不适用
 prediction_score     double   分类算法权重值，对回归算法为预测结果
 prediction_detail    string   分类算法各个类别的权重值，回归算法不适用
==================== ======== ====================================================

预测时，只需要将需要预测的数据集作为参数并设置其特征即可，默认使用全部字段作为特征：

.. code-block:: python

    predicted = pmml_model.predict(input_df.exclude_fields('label'))

表模型
==========
表模型（:class:`TablesModel`）为 PyODPS ML 为方便部分将 ODPS 表作为输出的算法而设计，对应 ODPS 中的一张或几张表。这些表的表名
组成为 ``otm_模型名__表后缀``。例如，当模型名为 output_model，其中包含一张后缀为 model 的表时，该表在 ODPS 中的实际名称为
otm_output_model__model。

与 PMML 模型类似，PyODPS ML 无法显示创建一个表模型，需要通过使用 TablesModel 的算法输出，例如下面通过核 SVM 算法生成一个表模型：

.. code-block:: python

    tables_model = KernelSVM().train(input_df)

生成模型后，可存储为 ODPS 表，调用方法为 :func:`TablesModel.persist` ：

.. code-block:: python

    tables_model.persist('model_name')

可通过 list_tables_model 函数列出某个 Project 内的所有 TablesModel，也可以通过 TablesModel 的构造函数进行
载入：

.. code-block:: python

    model = o.get_tables_model('model_prefix')
    tables_model = TablesModel(model)

表模型也拥有 predict 方法，可对数据集进行预测，但输出列不确定，一部分分类算法不支持输出 predict 列，具体需要参考各算法文档。

推荐模型
==========
推荐模型建立在表模型基础上，除了正常的 predict 方法外，还拥有 recommend 方法，可计算推荐结果。
该模型也可使用 :func:`TablesModel` 的构造函数进行加载，PyODPS ML 会自动判别类型。


===========
执行
===========


延迟执行
============

在 PyODPS ML 中，我们将每个算法看作一个 Node，每个 Node 有若干个输入和输出，我们称之为 Port 。不同 Node 间通过数据的流动相连，
形成一个有向无环图。在 PyODPS ML 中，每个输出 Port 可以唯一绑定一个 DataFrame 或者 MLModel，而每一行用户代码都会通过 DataFrame
提供的上游 Node 信息将该 Node 与下游 Node 相连。

PyODPS ML 不会立即执行每一个 Node 对应的操作，而是等到 IO、Collect 或者 Metrics 操作被执行时，方才执行先前相关的操作。如下面的
代码段：

.. code-block:: python

    df1, df2 = DataFrame(o.get_table('iris')).split(0.5)
    df1.std_scale().persist('iris_part_std')
    df2.min_max_scale()

代码中的标准化（std_scale）操作会被执行，因为 df1 这条链路上执行了 persist 操作。而归一化（min_max_scale）操作则不会被执行，
因为其链路中并不存在任何触发执行的代码。

采用延迟执行的好处有三。首先，对于存在分支的流程，延迟执行能帮助 PyODPS ML 决定哪些步骤可以并行化，从而能够尽可能地利用计算资源。
其次，对于多个输出的情形，例如 TripleLDA，如果用户书写了下面的代码

.. code-block:: python

    word_stats, _, _, _, _, _ = TripleLDA(topic_num=2).transform(freq)

PyODPS ML 可以使用 GC 获得真正需要的输出个数，从而避免了多余的输出操作。最后，延迟执行也能够帮助用户更快地搭建算法流程。

如果需要某个步骤立即执行，也可以在相应的 DataFrame 或 Model 上执行 persist() 方法。此时，该数据对象对应的节点及所有依赖节点都将被执行。
