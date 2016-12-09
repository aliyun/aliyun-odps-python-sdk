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

============= ============================================================================
 窗口函数      说明
============= ============================================================================
 cumsum
 cummean
 cummedian
 cumstd
 cummax
 cummin
 cumcount
 lag           按偏移量取当前行之前第几行的值，如当前行号为rn，则取行号为rn-offset的值
 lead          按偏移量取当前行之后第几行的值，如当前行号为rn则取行号为rn+offset的值
 rank          计算排名
 dense_rank    计算连续排名
 percent_rank  计算一组数据中某行的相对排名
 row_number    计算行号，从1开始
============= ============================================================================
