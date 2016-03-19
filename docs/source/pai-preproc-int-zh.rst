.. _pai_preproc:
.. py:currentmodule:: odps.pai


==============
处理数据
==============

在 PAI SDK 中，除了可以使用 :ref:`DataFrame <df>` 处理数据，也可以使用 PAI SDK 自带的工具进行一些简单的数据预处理。

采样与拆分
=============
:func:`DataSet.sample` 方法支持对数据进行采样。该方法支持按数据权重采样，也支持放回采样。调用方法为

.. code-block:: python

    # 普通采样
    result1 = dataset.sample(0.6)
    # 放回采样
    result2 = dataset.sample(0.6, replace=True)
    # 指定权重字段
    result3 = dataset.sample(0.6, prob_field='p', replace=True)

:func:`DataSet.split` 方法支持按比例拆分数据集，调用方法为

.. code-block:: python

    left, right = dataset.split(0.6)

ID 追加
=============
:func:`DataSet.append_id` 方法支持向数据集追加一列，内容为每行数据的行号。可指定输出列的名称，默认为``append_id``。

.. code-block:: python

    result = dataset.append_id('append_col_name')

数据合并
=============
使用 :func:`DataSet.merge_fields` 可将数据按行合并，要求输入数据集的行数相同。该方法既可以作为 DataSet 的成员调用，也可以单独
调用。参数 `auto_rename_col` 可指定是否按照 ``tn_col`` 的形式重命名列，保证输出表的列均不相同。

下面给出的是数据合并的两种调用方法：

.. code-block:: python

    merged1 = merge_fields(ds1, ds2)
    merged2 = ds1.merge_fields(ds2)

使用 SQL
=============
使用 :func:`DataSet.sql_transform` 可通过一条 SQL 语句对数据进行变换。输入的 SQL 语句仅允许 SELECT，当前 DataSet 需要使用
$input 进行指代。方法返回一个 DataSet。

使用 SQL 的例子如下：

.. code-block:: python

    iris_ds = context.odps_data('iris')
    transformed = iris_ds.sql_transform("select * from $input where category in ('Iris Setosa', 'Iris Virginica');")

归一化与标准化
==============
.. py:currentmodule:: odps.pai.preprocess

归一化和标准化模块位于 ``odps.pai.preprocess`` 中，使用时需要 Import。

归一化的作用是使得数据落入[0, 1]区间内，函数为 :func:`normalize`。参数可控制是否保留原始字段（追加``_orig``）以及选择需要
归一化的列。

标准化的作用是使用标准正态分布对原始数据进行线性变换，使数据的均值为0，标准差为1。函数为 :func:`standardize`，参数与归一化相似。

归一化与标准化的例子如下：

.. code-block:: python

    norm_result = normalize(ds, keep_original=True)
    std_result = standardize(ds, keep_original=True)

行列变换
=============
.. py:currentmodule:: odps.pai.preprocess

PAI SDK 支持对数据进行行列变换。行列转换模块位于 ``odps.pai.preprocess`` 中，使用时需要 Import。

行转列的函数为 :func:`pivot`，参数依次为新列名对应列、数值对应列和标识对应列。例如，对于下列数据集scores：

========== ============== ============ =============
grade      name           course       score
========== ============== ============ =============
1          Tom            english      90
1          Tom            math         80
1          John           english      100
1          John           math         90
1          Tom            physics
1          John           physics      90
========== ============== ============ =============

执行下列变换：

.. code-block:: python

    scores_pv = pivot(scores, 'course', 'score', ['grade', 'name'])

返回的 scores_pv 为

========== ============== =============== ============= ===============
grade      name           pivot_english   pivot_math    pivot_physics
========== ============== =============== ============= ===============
1          Tom            90              80
1          John           100             90            90
========== ============== =============== ============= ===============

列转行的函数为 :func:`unpivot`，参数依次为合并字段列表、标志字段列表、原字段名列名以及值列名。

例如，针对上面生成的 scores_pv 表，执行下列变换：

.. code-block:: python

    scores_reconstruct = unpivot(scores_pv, ['pivot_' + c for c in ['english', 'math', 'physics']],
                                 ['grade', 'name'], 'course', 'score')

生成的结果为

========== ============== ============ =============
grade      name           course       score
========== ============== ============ =============
1          Tom            english      90
1          Tom            math         80
1          John           english      100
1          John           math         90
1          Tom            physics
1          John           physics      90
========== ============== ============ =============

缺失值和异常值处理
====================
.. py:currentmodule:: odps.pai.preprocess

PAI SDK 支持对数据进行异常值处理。异常值处理模块位于 ``odps.pai.preprocess`` 中，使用时需要 Import。

异常值处理函数为 :func:`modify_abnormal`。其参数为若干个配置项组成的数组，指定针对不同的情形，应当作何处理。各种配置的用法
如下表

+-------------------+---------------------------------------+-----------------------------------------------------------+
|类名               | 说明                                  | 用法                                                      |
+===================+=======================================+===========================================================+
|ReplaceNull        | 替换 Null 值                          | ReplaceNull('col_name', new_value)                        |
+-------------------+---------------------------------------+-----------------------------------------------------------+
|ReplaceEmpty       | 替换空字符串                          | ReplaceEmpty('col_name', new_value)                       |
+-------------------+---------------------------------------+-----------------------------------------------------------+
|ReplaceNullEmpty   | 替换 Null 值及空字符串                | ReplaceNullEmpty('col_name', new_value)                   |
+-------------------+---------------------------------------+-----------------------------------------------------------+
|ReplaceCustom      | 将给定的值替换成另一个                | ReplaceCustom('col_name', old_value, new_value)           |
+-------------------+---------------------------------------+-----------------------------------------------------------+
|ReplacePercentile  | 替换某一百分比区间外的值为两端点值    | ReplacePercentile('col_name', low_range, high_range,      |
|                   |                                       | low_value, high_value)                                    |
+-------------------+---------------------------------------+-----------------------------------------------------------+
|ReplaceConfidence  | 替换落于正态置信区间外的值为两端点值  | ReplaceCustom('col_name', confidence, low_value,          |
|                   |                                       | high_value)                                               |
+-------------------+---------------------------------------+-----------------------------------------------------------+
|ReplaceZScore      | 替换落于 Z-Score 区间外的值为两端点值 | ReplaceCustom('col_name', low_range, high_range,          |
|                   |                                       | low_value, high_value)                                    |
+-------------------+---------------------------------------+-----------------------------------------------------------+

使用这些配置进行异常值处理时，可以只使用单个条件，如

.. code-block:: python

    result = modify_abnormal(ds, ReplaceNull('col_name', 0))

也可以传入多个条件组成的数组，如

.. code-block:: python

    result = modify_abnormal(ds, [ReplaceEmpty('col_name', 'EMPTY'), ReplacePercentile('col_name', 5, 0, 95, 100)])
