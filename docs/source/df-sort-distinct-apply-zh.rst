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


采样
=======


要对一个collection的数据采样，可以调用 ``sample`` 方法。

.. code:: python

    iris.sample(parts=10)  # 分成10份，默认取第0份
    iris.sample(parts=10, i=0)  # 手动指定取第0份
    iris.sample(parts=10, i=[2, 5])   # 分成10份，取第2和第5份
    iris.sample(parts=10, columns=['name', 'sepalwidth'])  # 根据name和sepalwidth的值做采样


用Apply对所有行或者所有列调用自定义函数
=============================================


对一行数据使用自定义函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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


对所有列调用自定义聚合
~~~~~~~~~~~~~~~~~~~~~~~

调用apply方法，当我们不指定axis，或者axis为0的时候，我们可以通过传入一个自定义聚合类来对所有sequence进行聚合操作。

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

    iris.exclude('name').apply(Agg)


.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength_aggregation</th>
          <th>sepalwidth_aggregation</th>
          <th>petallength_aggregation</th>
          <th>petalwidth_aggregation</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.843333</td>
          <td>3.054</td>
          <td>3.758667</td>
          <td>1.198667</td>
        </tr>
      </tbody>
    </table>
    </div>


引用资源
~~~~~~~~~~~~~


类似于对 ``map`` 方法的resources参数，每个resource可以是ODPS上的资源（表资源或文件资源），或者引用一个collection作为资源。

对于axis为1，也就是在行上操作，我们需要写一个函数闭包或者callable的类。
而对于列上的聚合操作，我们只需在 \_\_init\_\_ 函数里读取资源即可。


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

    import pandas as pd

    stop_words = DataFrame(pd.DataFrame({'stops': ['is', 'a', 'I']}))


.. code:: python

    @output(['sentence'], ['string'])
    def filter_stops(resources):
        stop_words = set([r[0] for r in resources[0]])

        def h(row):
            return ' '.join(w for w in row[0].split() if w not in stop_words),
        return h

    words_df.apply(filter_stops, axis=1, resources=[stop_words])


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
          <td>Life short use Python</td>
        </tr>
      </tbody>
    </table>
    </div>


可以看到这里的stop_words是存放于本地，但在真正执行时会被上传到ODPS作为资源引用。




MapReduce API
==============


PyODPS DataFrame也支持MapReduce API，用户可以分别编写map和reduce函数。我们来看个简单的wordcount的例子。


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


group参数用来指定reduce按哪些字段做分组，如果不指定，会按全部字段做分组。

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


引用资源
~~~~~~~~~~~~~

在MapReduce API里，我们能分别指定mapper和reducer所要引用的资源。

如下面的例子，我们对mapper里的单词做停词过滤，在reducer里对白名单的单词数量加5。

.. code:: python

    white_list_file = o.create_resource('pyodps_white_list_words', 'file', file_obj='Python\nWorld')

.. code:: python

    @output(['word', 'cnt'], ['string', 'int'])
    def mapper(resources):
        stop_words = set(r[0].strip() for r in resources[0])

        def h(row):
            for word in row[0].split():
                if word not in stop_words:
                    yield word, 1
        return h

    @output(['word', 'cnt'], ['string', 'int'])
    def reducer(resources):
        d = dict()
        d['white_list'] = set(word.strip() for word in resources[0])
        d['cnt'] = 0

        def inner(keys):
            d['cnt'] = 0
            def h(row, done):
                d['cnt'] += row.cnt
                if done:
                    if row.word in d['white_list']:
                        d['cnt'] += 5
                    yield keys.word, d['cnt']
            return h
        return inner

    words_df.map_reduce(mapper, reducer, group='word',
                        mapper_resources=[stop_words], reducer_resources=[white_list_file])

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
          <td>life</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>python</td>
          <td>7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>world</td>
          <td>6</td>
        </tr>
        <tr>
          <th>4</th>
          <td>short</td>
          <td>1</td>
        </tr>
        <tr>
          <th>5</th>
          <td>use</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>


布隆过滤器
==============


PyODPS DataFrame提供了 ``bloom_filter`` 接口来进行布隆过滤器的计算。

给定某个collection，和它的某个列计算的sequence1，我们能对另外一个sequence2进行布隆过滤，sequence1不在sequence2中的一定会过滤，
但可能不能完全过滤掉不存在于sequence2中的数据，这也是一种近似的方法。

这样的好处是能快速对collection进行快速过滤一些无用数据。

这在大规模join的时候，一边数据量远大过另一边数据，而大部分并不会join上的场景很有用。
比如，我们在join用户的浏览数据和交易数据时，用户的浏览大部分不会带来交易，我们可以利用交易数据先对浏览数据进行布隆过滤，
然后再join能很好提升性能。

.. code:: python

    df1 = DataFrame(pd.DataFrame({'a': ['name1', 'name2', 'name3', 'name1'], 'b': [1, 2, 3, 4]}))
    df1

.. code:: python

           a  b
    0  name1  1
    1  name2  2
    2  name3  3
    3  name1  4

.. code:: python

    df2 = DataFrame(pd.DataFrame({'a': ['name1']}))
    df2

.. code:: python

           a
    0  name1

.. code:: python

    df1.bloom_filter('a', df2.a) # 这里第0个参数可以是个计算表达式如: df1.a + '1'

.. code:: python

           a  b
    0  name1  1
    1  name1  4

这里由于数据量很小，df1中的a为name2和name3的行都被正确过滤掉了，当数据量很大的时候，可能会有一定的数据不能被过滤。

如之前提的join场景中，少量不能过滤并不能并不会影响正确性，但能较大提升join的性能。

我们可以传入 ``capacity`` 和 ``error_rate`` 来设置数据的量以及错误率，默认值是 ``3000`` 和 ``0.01``。
要注意，调大 ``capacity`` 或者减小 ``error_rate`` 会增加内存的使用，所以应当根据实际情况选择一个合理的值。