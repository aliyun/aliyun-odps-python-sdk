.. _dfagg:

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))

聚合操作
========

首先，我们可以使用\ ``describe``\ 函数，来查看DataFrame里数字列的数量、最大值、最小值、平均值以及标准差是多少。

.. code:: python

    print(iris.describe())


.. code:: python

       sepallength_count  sepallength_min  sepallength_max  sepallength_mean  \
    0                150              4.3              7.9          5.843333   
    
       sepallength_std  sepalwidth_count  sepalwidth_min  sepalwidth_max  \
    0         0.825301               150               2             4.4   
    
       sepalwidth_mean  sepalwidth_std  petallength_count  petallength_min  \
    0            3.054        0.432147                150                1   
    
       petallength_max  petallength_mean  petallength_std  petalwidth_count  \
    0              6.9          3.758667         1.758529               150   
    
       petalwidth_min  petalwidth_max  petalwidth_mean  petalwidth_std  
    0             0.1             2.5         1.198667        0.760613  


我们可以使用单列来执行聚合操作：

.. code:: python

    iris.sepallength.max()




.. code:: python

    7.9



支持的聚合操作包括：

.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <tr>
        <th>聚合操作</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>count（或size）</td>
        <td>数量</td>
      </tr>
      <tr>
        <td>nunique</td>
        <td>不重复值数量</td>
      </tr>
      <tr>
        <td>min</td>
        <td>最小值</td>
      </tr>
      <tr>
        <td>max</td>
        <td>最大值</td>
      </tr>
      <tr>
       <td>sum</td>
       <td>求和</td>
      </tr>
      <tr>
        <td>mean</td>
        <td>均值</td>
      </tr>
      <tr>
        <td>median</td>
        <td>中位数</td>
      </tr>
      <tr>
        <td>var</td>
        <td>方差</td>
      </tr>
      <tr>
        <td>std</td>
        <td>标准差</td>
      </tr>
    </table>
    </div>

分组聚合
========

DataFrame
API提供了groupby来执行分组操作，分组后的一个主要操作就是通过调用agg或者aggregate方法，来执行聚合操作。

.. code:: python

    iris.groupby('name').agg(iris.sepallength.max(), smin=iris.sepallength.min())




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>sepallength_max</th>
          <th>smin</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-setosa</td>
          <td>5.8</td>
          <td>4.3</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-versicolor</td>
          <td>7.0</td>
          <td>4.9</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-virginica</td>
          <td>7.9</td>
          <td>4.9</td>
        </tr>
      </tbody>
    </table>
    </div>



最终的结果列中会包含分组的列，以及聚合的列。

DataFrame
API提供了一个\ ``value_counts``\ 操作，能返回按某列分组后，每个组的个数从大到小排列的操作。

我们使用groupby表达式可以写成：

.. code:: python

    iris.groupby('name').agg(count=iris.name.count()).sort('count', ascending=False).head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>count</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-virginica</td>
          <td>50</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-versicolor</td>
          <td>50</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-setosa</td>
          <td>50</td>
        </tr>
      </tbody>
    </table>
    </div>



使用value\_counts就很简单了：

.. code:: python

    iris['name'].value_counts().head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>count</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-virginica</td>
          <td>50</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-versicolor</td>
          <td>50</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-setosa</td>
          <td>50</td>
        </tr>
      </tbody>
    </table>
    </div>



对于聚合后的单列操作，我们也可以直接取出列名。但此时只能使用聚合函数。

.. code:: python

    iris.groupby('name').petallength.sum()




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>petallength_sum</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>73.2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>213.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>277.6</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    iris.groupby('name').agg(iris.petallength.notnull().sum())




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>petallength_sum</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-setosa</td>
          <td>50</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-versicolor</td>
          <td>50</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-virginica</td>
          <td>50</td>
        </tr>
      </tbody>
    </table>
    </div>



分组时也支持对常量进行分组，但是需要使用Scalar初始化。

.. code:: python

    from odps.df import Scalar

.. code:: python

    iris.groupby(Scalar(1)).petallength.sum()




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>petallength_sum</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>563.8</td>
        </tr>
      </tbody>
    </table>
    </div>


编写自定义聚合
===================

对字段调用agg或者aggregate方法来调用自定义聚合。自定义聚合需要提供一个类，这个类需要提供以下方法：

* buffer()：返回一个mutable的object（比如list、dict），buffer大小不应随数据而递增。
* __call__(buffer, val)：将值聚合到中间buffer。
* merge(buffer, pbuffer)：讲pbuffer聚合到buffer中。
* getvalue(buffer)：返回最终值。

让我们看一个计算平均值的例子。

.. code:: python

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

    iris.sepalwidth.agg(Agg)


.. code:: python

    3.0540000000000007

如果最终类型和输入类型发生了变化，则需要指定类型。

.. code:: python

    iris.sepalwidth.agg(Agg, 'float')


自定义聚合也可以用在分组聚合中。

.. code:: python

    iris.groupby('name').sepalwidth.agg(Agg)

.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>petallength_aggregation</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3.418</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.770</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2.974</td>
        </tr>
      </tbody>
    </table>
    </div>


HyperLogLog 计数
==================

PyODPS DataFrame提供了对列进行HyperLogLog计数的接口 ``hll_count``，这个接口是个近似的估计接口，
当数据量很大时，能较快的对数据的唯一个数进行估计。

这个接口在对比如海量用户UV进行计算时，能很快得出估计值。

.. code:: python

    df = DataFrame(pd.DataFrame({'a': np.random.randint(100000, size=100000)}))
    df.a.hll_count()

.. code:: python

    63270

.. code:: python

    df.a.nunique()

.. code:: python

    63250

提供 ``splitter`` 参数会对每个字段进行分隔，再计算唯一数。