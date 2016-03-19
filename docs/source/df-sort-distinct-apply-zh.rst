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

去重在Collection上，用户可以调用distinct方法。

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


在Sequence上，用户可以调用unique，但是记住，调用unique的Sequence不能用在列选择中。


.. code:: python

    iris.name.unique()



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


下面的代码是错误的用法。


.. code:: python

    iris[iris.name, iris.name.unique()]  # 错误的



对一行数据使用自定义函数
========================


要对一行数据使用自定义函数，可以使用apply方法，axis参数必须为1，表示在行上操作。


apply的自定义函数接收一个参数，为上一步Collection的一行数据，用户可以通过属性、或者偏移取得一个字段的数据。


.. code:: python

    iris.apply(lambda row: row.sepallength + row.sepalwidth, axis=1, reduce=True, types='float').rename('sepaladd').head(3)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepaladd</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>8.6</td>
        </tr>
        <tr>
          <th>1</th>
          <td>7.9</td>
        </tr>
        <tr>
          <th>2</th>
          <td>7.9</td>
        </tr>
      </tbody>
    </table>
    </div>



``reduce``\ 为True时，表示返回结果为Sequence，否则返回结果为Collection。
``names``\ 和 ``types``\ 参数分别指定返回的Sequence或Collection的字段名和类型。
如果类型不指定，将会默认为string类型。

在apply的自定义函数中，reduce为False时，也可以使用 ``yield``\ 关键字来返回多行结果。


.. code:: python

    iris.count()




.. code:: python

    150



.. code:: python

    def handle(row):
        yield row.sepallength - row.sepalwidth, row.sepallength + row.sepalwidth
        yield row.petallength - row.petalwidth, row.petallength + row.petalwidth

    iris.apply(handle, axis=1, names=['iris_add', 'iris_sub'], types=['float', 'float']).count()




.. code:: python

    300


我们也可以在函数上来注释返回的字段和类型，这样就不需要在函数调用时再指定。


.. code:: python

    from odps.df import output

    @output(['iris_add', 'iris_sub'], ['float', 'float'])
    def handle(row):
        yield row.sepallength - row.sepalwidth, row.sepallength + row.sepalwidth
        yield row.petallength - row.petalwidth, row.petallength + row.petalwidth

    iris.apply(handle, axis=1).count()


.. code:: python

    300


MapReduce API
==============


PyOdps DataFrame也支持MapReduce API，用户可以分别编写map和reduce函数。我们来看个简单的wordcount的例子。

.. code:: python

    words_df




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sentence</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Hello World</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Hello Python</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Life is short I use Python</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    def mapper(row):
        for word in row[0].split():
            yield word.lower(), 1

    def reducer(keys):
        cnt = [0]
        def h(row, done):  # done表示这个key已经迭代结束
            cnt[0] += row[1]
            if done:
                yield keys[0], cnt[0]
        return h

    words_df.map_reduce(mapper, reducer, group=['word', ],
                        mapper_output_names=['word', 'cnt'],
                        mapper_output_types=['string', 'int'],
                        reducer_output_names=['word', 'cnt'],
                        reducer_output_types=['string', 'int'])




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>word</th>
          <th>cnt</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>hello</td>
          <td>2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>i</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>is</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>life</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>python</td>
          <td>2</td>
        </tr>
        <tr>
          <th>5</th>
          <td>short</td>
          <td>1</td>
        </tr>
        <tr>
          <th>6</th>
          <td>use</td>
          <td>1</td>
        </tr>
        <tr>
          <th>7</th>
          <td>world</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>


其中对于reducer来说，会稍微有些不同。它需要接收聚合的keys初始化，并能继续处理按这些keys聚合的每行数据。
第2个参数表示这些keys相关的所有行是不是都迭代完成。

这里写成函数闭包的方式，主要为了方便，当然我们也能写成callable的类。


.. code:: python

    class reducer(object):
        def __init__(self, keys):
            self.cnt = 0

        def __call__(self, row, done):  # done表示这个key已经迭代结束
            self.cnt += row.cnt
            if done:
                yield row.word, self.cnt


使用 ``output``\ 来注释会让代码更简单些。


.. code:: python

    from odps.df import output

    @output(['word', 'cnt'], ['string', 'int'])
    def mapper(row):
        for word in row[0].split():
            yield word.lower(), 1

    @output(['word', 'cnt'], ['string', 'int'])
    def reducer(keys):
        cnt = [0]
        def h(row, done):  # done表示这个key已经迭代结束
            cnt[0] += row.cnt
            if done:
                yield keys.word, cnt[0]
        return h

    words_df.map_reduce(mapper, reducer, group='word')




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>word</th>
          <th>cnt</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>hello</td>
          <td>2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>i</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>is</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>life</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>python</td>
          <td>2</td>
        </tr>
        <tr>
          <th>5</th>
          <td>short</td>
          <td>1</td>
        </tr>
        <tr>
          <th>6</th>
          <td>use</td>
          <td>1</td>
        </tr>
        <tr>
          <th>7</th>
          <td>world</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>


有时候我们在迭代的时候需要按某些列排序，则可以使用 ``sort``\ 参数，来指定按哪些列排序。