.. _dfquery:

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))

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



列选择和筛选
============

取出单列

.. code:: python

    iris.sepallength.head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    iris['sepallength'].head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
        </tr>
      </tbody>
    </table>
    </div>



取出多列

.. code:: python

    iris['name', 'sepallength'].head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>sepallength</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-setosa</td>
          <td>5.1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-setosa</td>
          <td>4.9</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-setosa</td>
          <td>4.7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Iris-setosa</td>
          <td>4.6</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Iris-setosa</td>
          <td>5.0</td>
        </tr>
      </tbody>
    </table>
    </div>



排除某些列，取出剩下的列

.. code:: python

    iris.exclude('sepallength', 'petallength')[:5]




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepalwidth</th>
          <th>petalwidth</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>3.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.0</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.2</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3.1</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>3.6</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



有时候，我们需要在原有的列基础上，多一列值，则可以（记住选择列不能重名，因此我们需要用rename来重新命名）：

.. code:: python

    iris[iris, (iris.sepalwidth + 1).rename('sepalwidthplus1')].head(5)




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
          <th>sepalwidthplus1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.1</td>
          <td>3.5</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
          <td>4.5</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
          <td>4.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
          <td>4.2</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
          <td>4.1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>3.6</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
          <td>4.6</td>
        </tr>
      </tbody>
    </table>
    </div>



我们也可以像这样，去除原有的一列，再计算出新的列（这里不需要担心重名了）。

.. code:: python

    iris[iris.exclude('sepalwidth'), iris.sepalwidth * 2].head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength</th>
          <th>petallength</th>
          <th>petalwidth</th>
          <th>name</th>
          <th>sepalwidth</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.1</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
          <td>7.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
          <td>6.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
          <td>6.4</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
          <td>6.2</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
          <td>7.2</td>
        </tr>
      </tbody>
    </table>
    </div>



我们也可以显式调用select方法，作用是一样的。

.. code:: python

    iris.select('name', sepalwidthminus1=iris.sepalwidth - 1).head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>sepalwidthminus1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-setosa</td>
          <td>2.5</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-setosa</td>
          <td>2.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-setosa</td>
          <td>2.2</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Iris-setosa</td>
          <td>2.1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Iris-setosa</td>
          <td>2.6</td>
        </tr>
      </tbody>
    </table>
    </div>



**注意**\ ，我们用来选择的collection或者sequence，来源必须是上一步的collection，比如：

.. code:: python

    iris['name', 'petallength'][[iris.name, ]].head(5)

就是错误的。

一种方法是分成两步写：

.. code:: python

    df = iris['name', 'petallength']
    df[[df.name, ]].head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



另一种方法是，我们可以传入一个lambda表达式，它接收一个参数，参数传递过来的就是上一步的结果。

.. code:: python

    iris['name', 'petallength'][[lambda x: x.name]].head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



过滤数据
========

我们试着查询\ ``sepallength``\ 大于5的几条数据。

.. code:: python

    iris[iris.sepallength > 5].head(5)




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
          <td>5.1</td>
          <td>3.5</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>5.4</td>
          <td>3.9</td>
          <td>1.7</td>
          <td>0.4</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5.4</td>
          <td>3.7</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.8</td>
          <td>4.0</td>
          <td>1.2</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.7</td>
          <td>4.4</td>
          <td>1.5</td>
          <td>0.4</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



多个查询条件：

.. code:: python

    iris[(iris.sepallength < 5) & (iris['petallength'] > 1.5)].head(5)




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
          <td>4.8</td>
          <td>3.4</td>
          <td>1.6</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.8</td>
          <td>3.4</td>
          <td>1.9</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.6</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.8</td>
          <td>3.1</td>
          <td>1.6</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4.9</td>
          <td>2.4</td>
          <td>3.3</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
      </tbody>
    </table>
    </div>



或条件：

.. code:: python

    iris[(iris.sepalwidth < 2.5) | (iris.sepalwidth > 4)].head(5)




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
          <td>5.7</td>
          <td>4.4</td>
          <td>1.5</td>
          <td>0.4</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>5.2</td>
          <td>4.1</td>
          <td>1.5</td>
          <td>0.1</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5.5</td>
          <td>4.2</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.5</td>
          <td>2.3</td>
          <td>1.3</td>
          <td>0.3</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.5</td>
          <td>2.3</td>
          <td>4.0</td>
          <td>1.3</td>
          <td>Iris-versicolor</td>
        </tr>
      </tbody>
    </table>
    </div>



非条件：

.. code:: python

    iris[~(iris.sepalwidth > 3)].head(5)




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
          <td>4.4</td>
          <td>2.9</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.8</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.1</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.3</td>
          <td>3.0</td>
          <td>1.1</td>
          <td>0.1</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>3.0</td>
          <td>1.6</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



我们也可以显式调用filter方法，提供多个与条件

.. code:: python

    iris.filter(iris.sepalwidth > 3.5, iris.sepalwidth < 4).head(5)




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
          <td>5.0</td>
          <td>3.6</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>5.4</td>
          <td>3.9</td>
          <td>1.7</td>
          <td>0.4</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5.4</td>
          <td>3.7</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.4</td>
          <td>3.9</td>
          <td>1.3</td>
          <td>0.4</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.7</td>
          <td>3.8</td>
          <td>1.7</td>
          <td>0.3</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



同样对于连续的操作，我们可以使用lambda表达式

.. code:: python

    iris[iris.sepalwidth > 3.8]['name', lambda x: x.sepallength + 1]




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>sepallength</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-setosa</td>
          <td>6.4</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-setosa</td>
          <td>6.8</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-setosa</td>
          <td>6.7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Iris-setosa</td>
          <td>6.4</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Iris-setosa</td>
          <td>6.2</td>
        </tr>
        <tr>
          <th>5</th>
          <td>Iris-setosa</td>
          <td>6.5</td>
        </tr>
      </tbody>
    </table>
    </div>



限制条数
========

.. code:: python

    iris[:3]




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
          <td>5.1</td>
          <td>3.5</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



值得注意的是，目前切片仅支持不支持start和step。我们也可以使用limit方法

.. code:: python

    iris.limit(3)




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
          <td>5.1</td>
          <td>3.5</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



**另外，切片操作只能作用在collection上，不能作用于sequence。**

保存DataFrame的执行结果为新的表
===============================

我们可以调用\ ``persist``\ 方法，参数为表名。返回一个新的DataFrame对象

.. code:: python

    iris2 = iris[iris.sepalwidth < 2.5].persist('pyodps_iris2')

.. code:: python

    iris2.head(5)




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
          <td>4.5</td>
          <td>2.3</td>
          <td>1.3</td>
          <td>0.3</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>5.5</td>
          <td>2.3</td>
          <td>4.0</td>
          <td>1.3</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.9</td>
          <td>2.4</td>
          <td>3.3</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.0</td>
          <td>2.0</td>
          <td>3.5</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>4</th>
          <td>6.0</td>
          <td>2.2</td>
          <td>4.0</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
      </tbody>
    </table>
    </div>



``persist``\ 可以传入partitions参数，这样会创建一个表，它的分区是partitions所指定的字段。

.. code:: python

    iris3 = iris[iris.sepalwidth < 2.5].persist('pyodps_iris3', partitions=['name'])

.. code:: python

    iris3.data




.. parsed-literal::

    odps.Table
      name: odps_test_sqltask_finance.`pyodps_iris3`
      schema:
        sepallength           : double      
        sepalwidth            : double      
        petallength           : double      
        petalwidth            : double      
      partitions:
        name                  : string      


