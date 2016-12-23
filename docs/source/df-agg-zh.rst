.. _dfagg:

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))

聚合操作
========

首先，我们可以使用\ ``describe``\ 函数，来查看DataFrame里数字列的数量、最大值、最小值、平均值以及标准差是多少。

.. code:: python

    >>> print(iris.describe())
              type  sepal_length  sepal_width  petal_length  petal_width
    0        count    150.000000   150.000000    150.000000   150.000000
    1         mean      5.843333     3.054000      3.758667     1.198667
    2  std(ddof=0)      0.825301     0.432147      1.758529     0.760613
    3          min      4.300000     2.000000      1.000000     0.100000
    4          max      7.900000     4.400000      6.900000     2.500000

我们可以使用单列来执行聚合操作：

.. code:: python

    >>> iris.sepallength.max()
    7.9

    >>> iris.name.cat(sep=',')
    u'Iris-setosa,Iris-versicolor,Iris-virginica'


支持的聚合操作包括：

================ ========================
 聚合操作         说明
================ ========================
 count（或size）  数量
 nunique          不重复值数量
 min              最小值
 max              最大值
 sum              求和
 mean             均值
 median           中位数
 var              方差
 std              标准差
 moment           n 阶中心矩（或 n 阶矩）
 skew             样本偏度（无偏估计）
 kurtosis         样本峰度（无偏估计）
 cat              按sep做字符串连接操作
================ ========================

分组聚合
--------

DataFrame
API提供了groupby来执行分组操作，分组后的一个主要操作就是通过调用agg或者aggregate方法，来执行聚合操作。

.. code:: python

    >>> iris.groupby('name').agg(iris.sepallength.max(), smin=iris.sepallength.min())
                  name  sepallength_max  smin
    0      Iris-setosa              5.8   4.3
    1  Iris-versicolor              7.0   4.9
    2   Iris-virginica              7.9   4.9

最终的结果列中会包含分组的列，以及聚合的列。

DataFrame 提供了一个\ ``value_counts``\ 操作，能返回按某列分组后，每个组的个数从大到小排列的操作。

我们使用 groupby 表达式可以写成：

.. code:: python

    >>> iris.groupby('name').agg(count=iris.name.count()).sort('count', ascending=False).head(5)
                  name  count
    0   Iris-virginica     50
    1  Iris-versicolor     50
    2      Iris-setosa     50

使用value\_counts就很简单了：

.. code:: python

    >>> iris['name'].value_counts().head(5)
                  name  count
    0   Iris-virginica     50
    1  Iris-versicolor     50
    2      Iris-setosa     50

对于聚合后的单列操作，我们也可以直接取出列名。但此时只能使用聚合函数。

.. code:: python

    >>> iris.groupby('name').petallength.sum()
       petallength_sum
    0             73.2
    1            213.0
    2            277.6

.. code:: python

    >>> iris.groupby('name').agg(iris.petallength.notnull().sum())
                  name  petallength_sum
    0      Iris-setosa               50
    1  Iris-versicolor               50
    2   Iris-virginica               50


分组时也支持对常量进行分组，但是需要使用Scalar初始化。

.. code:: python

    >>> from odps.df import Scalar
    >>> iris.groupby(Scalar(1)).petallength.sum()
       petallength_sum
    0            563.8

编写自定义聚合
--------------

对字段调用agg或者aggregate方法来调用自定义聚合。自定义聚合需要提供一个类，这个类需要提供以下方法：

* buffer()：返回一个mutable的object（比如list、dict），buffer大小不应随数据而递增。
* __call__(buffer, val)：将值聚合到中间buffer。
* merge(buffer, pbuffer)：讲pbuffer聚合到buffer中。
* getvalue(buffer)：返回最终值。

让我们看一个计算平均值的例子。

.. code-block:: python

    class Agg(object):

        def buffer(self):
            return [0.0, 0]

        def __call__(self, buffer, val):
            buffer[0] += val
            buffer[1] += 1

        def merge(self, buffer, pbuffer):
            buffer[0] += pbuffer[0]
            buffer[1] += pbuffer[1]

        def getvalue(self, buffer):
            if buffer[1] == 0:
                return 0.0
            return buffer[0] / buffer[1]

.. code:: python

    >>> iris.sepalwidth.agg(Agg)
    3.0540000000000007

如果最终类型和输入类型发生了变化，则需要指定类型。

.. code:: python

    >>> iris.sepalwidth.agg(Agg, 'float')


自定义聚合也可以用在分组聚合中。

.. code:: python

    >>> iris.groupby('name').sepalwidth.agg(Agg)
       petallength_aggregation
    0                    3.418
    1                    2.770
    2                    2.974

HyperLogLog 计数
----------------

DataFrame 提供了对列进行 HyperLogLog 计数的接口 ``hll_count``，这个接口是个近似的估计接口，
当数据量很大时，能较快的对数据的唯一个数进行估计。

这个接口在对比如海量用户UV进行计算时，能很快得出估计值。

.. code:: python

    >>> df = DataFrame(pd.DataFrame({'a': np.random.randint(100000, size=100000)}))
    >>> df.a.hll_count()
    63270
    >>> df.a.nunique()
    63250

提供 ``splitter`` 参数会对每个字段进行分隔，再计算唯一数。
