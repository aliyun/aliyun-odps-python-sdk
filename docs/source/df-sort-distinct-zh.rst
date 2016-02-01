.. _dfsortdistinct:

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))

排序
====

排序操作只能作用于Collection。我们只需要调用sort或者sort\_values方法。

.. code:: python

    iris.sort('sepalwidth').head(5)




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
          <td>2.0</td>
          <td>3.5</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>1</th>
          <td>6.2</td>
          <td>2.2</td>
          <td>4.5</td>
          <td>1.5</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>2</th>
          <td>6.0</td>
          <td>2.2</td>
          <td>5.0</td>
          <td>1.5</td>
          <td>Iris-virginica</td>
        </tr>
        <tr>
          <th>3</th>
          <td>6.0</td>
          <td>2.2</td>
          <td>4.0</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
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



如果想要降序排列，则可以使用参数\ ``ascending``\ ，并设为False。

.. code:: python

    iris.sort('sepalwidth', ascending=False).head(5)




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
          <td>5.5</td>
          <td>4.2</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5.2</td>
          <td>4.1</td>
          <td>1.5</td>
          <td>0.1</td>
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
          <td>5.4</td>
          <td>3.9</td>
          <td>1.3</td>
          <td>0.4</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



也可以这样调用，来进行降序排列：

.. code:: python

    iris.sort(-iris.sepalwidth).head(5)




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
          <td>5.5</td>
          <td>4.2</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5.2</td>
          <td>4.1</td>
          <td>1.5</td>
          <td>0.1</td>
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
          <td>5.4</td>
          <td>3.9</td>
          <td>1.3</td>
          <td>0.4</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



多字段排序也很简单：

.. code:: python

    iris.sort(['sepalwidth', 'petallength']).head(5)




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
          <td>2.0</td>
          <td>3.5</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>1</th>
          <td>6.0</td>
          <td>2.2</td>
          <td>4.0</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>2</th>
          <td>6.2</td>
          <td>2.2</td>
          <td>4.5</td>
          <td>1.5</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>3</th>
          <td>6.0</td>
          <td>2.2</td>
          <td>5.0</td>
          <td>1.5</td>
          <td>Iris-virginica</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4.5</td>
          <td>2.3</td>
          <td>1.3</td>
          <td>0.3</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



多字段排序时，如果是升序降序不同，\ ``ascending``\ 参数可以传入一个列表，长度必须等同于排序的字段，它们的值都是boolean类型

.. code:: python

    iris.sort(['sepalwidth', 'petallength'], ascending=[True, False]).head(5)




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
          <td>2.0</td>
          <td>3.5</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>1</th>
          <td>6.0</td>
          <td>2.2</td>
          <td>5.0</td>
          <td>1.5</td>
          <td>Iris-virginica</td>
        </tr>
        <tr>
          <th>2</th>
          <td>6.2</td>
          <td>2.2</td>
          <td>4.5</td>
          <td>1.5</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>3</th>
          <td>6.0</td>
          <td>2.2</td>
          <td>4.0</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>4</th>
          <td>6.3</td>
          <td>2.3</td>
          <td>4.4</td>
          <td>1.3</td>
          <td>Iris-versicolor</td>
        </tr>
      </tbody>
    </table>
    </div>



下面效果是一样的：

.. code:: python

    iris.sort(['sepalwidth', -iris.petallength]).head(5)




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
          <td>2.0</td>
          <td>3.5</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>1</th>
          <td>6.0</td>
          <td>2.2</td>
          <td>5.0</td>
          <td>1.5</td>
          <td>Iris-virginica</td>
        </tr>
        <tr>
          <th>2</th>
          <td>6.2</td>
          <td>2.2</td>
          <td>4.5</td>
          <td>1.5</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>3</th>
          <td>6.0</td>
          <td>2.2</td>
          <td>4.0</td>
          <td>1.0</td>
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>4</th>
          <td>6.3</td>
          <td>2.3</td>
          <td>4.4</td>
          <td>1.3</td>
          <td>Iris-versicolor</td>
        </tr>
      </tbody>
    </table>
    </div>



去重
====

去重同样也只能在Collection上调用，用户可以调用distinct方法。

.. code:: python

    iris[['name']].distinct()




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
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-virginica</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    iris.distinct('name')




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
          <td>Iris-versicolor</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-virginica</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    iris.distinct('name', 'sepallength').head(3)




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
          <td>4.3</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-setosa</td>
          <td>4.4</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-setosa</td>
          <td>4.5</td>
        </tr>
      </tbody>
    </table>
    </div>



