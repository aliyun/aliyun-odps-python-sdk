.. _dfbasic:

创建DataFrame
=============

创建DataFrame非常简单，只需将Table对象传入，或者传入pandas DataFrame对象即可。

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))

.. code:: python

    import pandas as pd
    import numpy as np
    df = DataFrame(pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc')))

``dtypes``\ 可以用来查看DataFrame的字段和类型。

.. code:: python

    iris.dtypes




.. parsed-literal::

    odps.Schema {
      sepallength           float64       
      sepalwidth            float64       
      petallength           float64       
      petalwidth            float64       
      name                  string        
    }



基本概念
========

PyOdps
DataFrame中包括三个基本对象：\ ``Collection``\ ，\ ``Sequence``\ ，\ ``Scalar``\ ，分别表示表结构（或者二维结构）、列（一维结构）、标量。

类型系统
========

PyOdps
DataFrame包括自己的类型系统，在使用Table初始化的时候，ODPS的类型会被进行转换。这样做的好处是，能支持更多的计算后端。目前，DataFrame的执行后端支持ODPS
SQL和pandas。

PyOdps DataFrame包括以下类型：

``int8``\ ，\ ``int16``\ ，\ ``int32``\ ，\ ``int64``\ ，\ ``float32``\ ，\ ``float64``\ ，\ ``boolean``\ ，\ ``string``\ ，\ ``decimal``\ ，\ ``datetime``

ODPS的字段和DataFrame的类型映射关系如下：

.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <tr>
        <th>ODPS类型</th>
        <th>DataFrame类型</th>
      </tr>
      <tr>
        <td>bigint</td>
        <td>int64</td>
      </tr>
      <tr>
        <td>double</td>
        <td>float64</td>
      </tr>
      <tr>
        <td>string</td>
        <td>string</td>
      </tr>
      <tr>
        <td>datetime</td>
        <td>datetime</td>
      </tr>
      <tr>
        <td>boolean</td>
        <td>boolean</td>
      </tr>
      <tr>
        <td>decimal</td>
        <td>decimal</td>
      </tr>
    </table>
    </div>

目前DataFrame不支持ODPS中的array和map类型，未来的版本会支持。

延迟执行
========

DataFrame上的所有操作并不会立即执行，只有当用户显式调用\ ``execute``\ 方法，或者一些立即执行的方法时（内部调用的就是\ ``execute``\ ），才会真正去执行。

这些立即执行的方法包括：

.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <tr>
        <th>方法</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>persist</td>
        <td>将执行结果保存到ODPS表</td>
      </tr>
      <tr>
        <td>head</td>
        <td>查看开头N行数据，这个方法会执行所有结果，并取开头N行数据</td>
      </tr>
      <tr>
        <td>tail</td>
        <td>查看结尾N行数据，这个方法会执行所有结果，并取结尾N行数据</td>
      </tr>
      <tr>
        <td>to_pandas</td>
        <td>转化为pandas DataFrame或者Series，wrap参数为True的时候，返回PyOdps DataFrame对象</td>
      </tr>
      <tr>
        <td>plot，hist，boxplot</td>
        <td>画图有关</td>
      </tr>
    </table>
    </div>

**注意**\ ：在交互式环境下，PyOdps
DataFrame会在打印或者repr的时候，调用\ ``execute``\ 方法，这样省去了用户手动去调用execute。

.. code:: python

    iris[iris.sepallength < 5][:5]




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength</th>
          <th>sepalwidth</th>
          <th>petallength</th>
          <th>petalwidth</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.6</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>3.4</td>
          <td>1.4</td>
          <td>0.3</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4.4</td>
          <td>2.9</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



如果想关闭自动调用执行，则需要手动设置

.. code:: python

    from odps import options
    options.interactive = False

.. code:: python

    iris[iris.sepallength < 5][:5]




.. parsed-literal::

    Collection: ref_0
      odps.Table
        name: odps_test_sqltask_finance.`pyodps_iris`
        schema:
          sepallength           : double      
          sepalwidth            : double      
          petallength           : double      
          petalwidth            : double      
          name                  : string      
    
    Collection: ref_1
      Filter[collection]
        collection: ref_0
        predicate:
          Less[sequence(boolean)]
            sepallength = Column[sequence(float64)] 'sepallength' from collection ref_0
            Scalar[int8]
              5
    
    Slice[collection]
      collection: ref_1
      stop:
        Scalar[int8]
          5



此时打印或者repr对象，会显示整棵抽象语法树。


运行时显示详细信息
==================

有时，用户需要查看运行时instance的logview时，需要修改全局配置：

.. code:: python

    from odps import options
    options.verbose = True

.. code:: python

    iris[iris.sepallength < 5].exclude('sepallength')[:5].execute()


.. parsed-literal::

    Sql compiled:
    SELECT t1.`sepalwidth`, t1.`petallength`, t1.`petalwidth`, t1.`name` 
    FROM odps_test_sqltask_finance.`pyodps_iris` t1 
    WHERE t1.`sepallength` < 5 
    LIMIT 5
    logview:
    http://logview




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepalwidth</th>
          <th>petallength</th>
          <th>petalwidth</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3.4</td>
          <td>1.4</td>
          <td>0.3</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2.9</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



用户可以指定自己的日志记录函数，比如像这样：

.. code:: python

    my_logs = []
    def my_logger(x):
        my_logs.append(x)
        
    options.verbose_log = my_logger

.. code:: python

    iris[iris.sepallength < 5].exclude('sepallength')[:5].execute()




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepalwidth</th>
          <th>petallength</th>
          <th>petalwidth</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3.4</td>
          <td>1.4</td>
          <td>0.3</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2.9</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    print(my_logs)


.. code:: python

    ['Sql compiled:', 'SELECT t1.`sepalwidth`, t1.`petallength`, t1.`petalwidth`, t1.`name` \nFROM odps_test_sqltask_finance.`pyodps_iris` t1 \nWHERE t1.`sepallength` < 5 \nLIMIT 5', 'logview:', u'http://logview']



缓存中间Collection计算结果
=============================


DataFrame的计算过程中，一些Collection被多处使用，或者用户需要查看中间过程的执行结果，
这时用户可以使用 ``cache``\ 标记某个collection需要被优先计算。

值得注意的是，``cache``\ 延迟执行，调用cache不会触发立即计算。


.. code:: python

    cached = iris[iris.sepalwidth < 3.5].cache()
    df = cached['sepallength', 'name'].head(3)
    df




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>4.9</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.7</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.6</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    cached.head(3)  # 由于cached已经被计算，所以能立刻取到计算结果




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength</th>
          <th>sepalwidth</th>
          <th>petallength</th>
          <th>petalwidth</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.6</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>


关于列名
========

在DataFrame的计算过程中，一个Sequence是必须要有名字，在很多情况下，DataFrame
API会起一个名字。比如：

.. code:: python

    iris.groupby('name').sepalwidth.max()




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepalwidth_max</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>4.4</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.4</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.8</td>
        </tr>
      </tbody>
    </table>
    </div>



可以看到，\ ``sepalwidth``\ 取最大值后被命名为\ ``sepalwidth_max``\ 。还有一些操作，比如一个Sequence做加法，加上一个Scalar，这时，会被命名为这个Sequence的名字。其它情况下，需要用户去自己命名。

.. code:: python

    (iris.sepalwidth + iris.petalwidth).rename('width_sum').head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>width_sum</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3.7</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.4</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3.3</td>
        </tr>
        <tr>
          <th>4</th>
          <td>3.8</td>
        </tr>
      </tbody>
    </table>
    </div>


