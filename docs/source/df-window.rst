.. _dfwindow:

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))

窗口函数
========

DataFrame API也支持使用窗口函数：

.. code:: python

    >>> grouped = iris.groupby('name')
    >>> grouped.mutate(grouped.sepallength.cumsum(), grouped.sort('sepallength').row_number()).head(10)
              name  sepallength_sum  row_number
    0  Iris-setosa            250.3           1
    1  Iris-setosa            250.3           2
    2  Iris-setosa            250.3           3
    3  Iris-setosa            250.3           4
    4  Iris-setosa            250.3           5
    5  Iris-setosa            250.3           6
    6  Iris-setosa            250.3           7
    7  Iris-setosa            250.3           8
    8  Iris-setosa            250.3           9
    9  Iris-setosa            250.3          10

窗口函数可以使用在列选择中：

.. code:: python

    >>> iris['name', 'sepallength', iris.groupby('name').sort('sepallength').sepallength.cumcount()].head(5)
              name  sepallength  sepallength_count
    0  Iris-setosa          4.3                  1
    1  Iris-setosa          4.4                  2
    2  Iris-setosa          4.4                  3
    3  Iris-setosa          4.4                  4
    4  Iris-setosa          4.5                  5

窗口函数按标量聚合时，和分组聚合的处理方式一致。


.. code:: python

    >>> from odps.df import Scalar
    >>> iris.groupby(Scalar(1)).sort('sepallength').sepallength.cumcount()

DataFrame API支持的窗口函数包括：

============= =================================================================================
 窗口函数      说明
============= =================================================================================
 cumsum        计算累积和
 cummean       计算累积均值
 cummedian     计算累积中位数
 cumstd        计算累积标准差
 cummax        计算累积最大值
 cummin        计算累积最小值
 cumcount      计算累积和
 lag           按偏移量取当前行之前第几行的值，如当前行号为rn，则取行号为rn-offset的值
 lead          按偏移量取当前行之后第几行的值，如当前行号为rn则取行号为rn+offset的值
 rank          计算排名
 dense_rank    计算连续排名
 percent_rank  计算一组数据中某行的相对排名
 row_number    计算行号，从1开始
 qcut          将分组数据按顺序切分成n片，并返回当前切片值，如果切片不均匀，默认增加第一个切片的分布
 nth_value     返回分组中的第n个值
 cume_dist     计算分组中值小于等于当前值的行数占分组总行数的比例
============= =================================================================================

其中，rank、dense_rank、percent_rank 和 row_number 支持下列参数：

============= ============================================================================
 参数名        说明
============= ============================================================================
 sort          排序关键字，默认为空
 ascending     排序时，是否依照升序排序，默认 True
============= ============================================================================

lag 和 lead 除 rank 的参数外，还支持下列参数：

============= ============================================================================
 参数名        说明
============= ============================================================================
 offset        取数据的行距离当前行的行数
 default       当 offset 指定的行不存在时的返回值
============= ============================================================================

而 cumsum、cummax、cummin、cummean、cummedian、cumcount 和 cumstd 除 rank 的上述参数外，还支持下列参数：

============= ============================================================================
 参数名        说明
============= ============================================================================
 unique        是否排除重复值，默认 False
 preceding     窗口范围起点
 following     窗口范围终点
============= ============================================================================
