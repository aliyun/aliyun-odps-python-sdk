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
    0  count    150.000000   150.000000    150.000000   150.000000
    1   mean      5.843333     3.054000      3.758667     1.198667
    2    std      0.828066     0.433594      1.764420     0.763161
    3    min      4.300000     2.000000      1.000000     0.100000
    4    max      7.900000     4.400000      6.900000     2.500000

我们可以使用单列来执行聚合操作：

.. code:: python

    >>> iris.sepallength.max()
    7.9

如果要在消除重复后的列上进行聚合，可以先调用 ``unique`` 方法，再调用相应的聚合函数：

.. code:: python

    >>> iris.name.unique().cat(sep=',')
    u'Iris-setosa,Iris-versicolor,Iris-virginica'


如果所有列支持同一种聚合操作，也可以直接在整个 DataFrame 上执行聚合操作：

.. code:: python

    >>> iris.exclude('category').mean()
       sepal_length  sepal_width  petal_length  petal_width
    1      5.843333     3.054000      3.758667     1.198667

需要注意的是，在 DataFrame 上执行 count 获取的是 DataFrame 的总行数：

.. code:: python

    >>> iris.count()
    150

PyODPS 支持的聚合操作包括：

================ ==================================
 聚合操作         说明
================ ==================================
 count（或size）  数量
 nunique          不重复值数量
 min              最小值
 max              最大值
 sum              求和
 mean             均值
 median           中位数
 quantile(p)      p分位数，仅在整数值下可取得准确值
 var              方差
 std              标准差
 moment           n 阶中心矩（或 n 阶矩）
 skew             样本偏度（无偏估计）
 kurtosis         样本峰度（无偏估计）
 cat              按sep做字符串连接操作
 tolist           组合为 list
================ ==================================

需要注意的是，与 Pandas 不同，对于列上的聚合操作，不论是在 ODPS 还是 Pandas 后端下，PyODPS DataFrame
都会忽略空值。这一逻辑与 SQL 类似。

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

* buffer()：返回一个mutable的object（比如 list、dict），buffer大小不应随数据而递增。
* __call__(buffer, *val)：将值聚合到中间 buffer。
* merge(buffer, pbuffer)：将 pbuffer 聚合到 buffer 中。
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

当对多列调用自定义聚合，可以使用agg方法。

.. code-block:: python

    class Agg(object):

        def buffer(self):
            return [0.0, 0.0]

        def __call__(self, buffer, val1, val2):
            buffer[0] += val1
            buffer[1] += val2

        def merge(self, buffer, pbuffer):
            buffer[0] += pbuffer[0]
            buffer[1] += pbuffer[1]

        def getvalue(self, buffer):
            if buffer[1] == 0:
                return 0.0
            return buffer[0] / buffer[1]

.. code:: python

    >>> from odps.df import agg
    >>> to_agg = agg([iris.sepalwidth, iris.sepallength], Agg, rtype='float')  # 对两列调用自定义聚合
    >>> iris.groupby('name').agg(val=to_agg)
                  name       val
    0      Iris-setosa  0.682781
    1  Iris-versicolor  0.466644
    2   Iris-virginica  0.451427

要调用 ODPS 上已经存在的 UDAF，指定函数名即可。

.. code:: python

    >>> iris.groupby('name').agg(iris.sepalwidth.agg('your_func'))  # 对单列聚合
    >>> to_agg = agg([iris.sepalwidth, iris.sepallength], 'your_func', rtype='float')
    >>> iris.groupby('name').agg(to_agg.rename('val'))  # 对多列聚合

.. warning::
    目前，受限于 Python UDF，自定义聚合无法支持将 list / dict 类型作为初始输入或最终输出结果。

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
