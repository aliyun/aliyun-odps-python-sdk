.. _dfsortdistinct:

排序、去重、采样、数据变换
=========================

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))

排序
-----

排序操作只能作用于Collection。我们只需要调用sort或者sort\_values方法。

.. code:: python

    >>> iris.sort('sepalwidth').head(5)
       sepallength  sepalwidth  petallength  petalwidth             name
    0          5.0         2.0          3.5         1.0  Iris-versicolor
    1          6.2         2.2          4.5         1.5  Iris-versicolor
    2          6.0         2.2          5.0         1.5   Iris-virginica
    3          6.0         2.2          4.0         1.0  Iris-versicolor
    4          5.5         2.3          4.0         1.3  Iris-versicolor

如果想要降序排列，则可以使用参数\ ``ascending``\ ，并设为False。

.. code:: python

    >>> iris.sort('sepalwidth', ascending=False).head(5)
       sepallength  sepalwidth  petallength  petalwidth         name
    0          5.7         4.4          1.5         0.4  Iris-setosa
    1          5.5         4.2          1.4         0.2  Iris-setosa
    2          5.2         4.1          1.5         0.1  Iris-setosa
    3          5.8         4.0          1.2         0.2  Iris-setosa
    4          5.4         3.9          1.3         0.4  Iris-setosa

也可以这样调用，来进行降序排列：

.. code:: python

    >>> iris.sort(-iris.sepalwidth).head(5)
       sepallength  sepalwidth  petallength  petalwidth         name
    0          5.7         4.4          1.5         0.4  Iris-setosa
    1          5.5         4.2          1.4         0.2  Iris-setosa
    2          5.2         4.1          1.5         0.1  Iris-setosa
    3          5.8         4.0          1.2         0.2  Iris-setosa
    4          5.4         3.9          1.3         0.4  Iris-setosa

多字段排序也很简单：

.. code:: python

    >>> iris.sort(['sepalwidth', 'petallength']).head(5)
       sepallength  sepalwidth  petallength  petalwidth             name
    0          5.0         2.0          3.5         1.0  Iris-versicolor
    1          6.0         2.2          4.0         1.0  Iris-versicolor
    2          6.2         2.2          4.5         1.5  Iris-versicolor
    3          6.0         2.2          5.0         1.5   Iris-virginica
    4          4.5         2.3          1.3         0.3      Iris-setosa

多字段排序时，如果是升序降序不同，\ ``ascending``\ 参数可以传入一个列表，长度必须等同于排序的字段，它们的值都是boolean类型

.. code:: python

    >>> iris.sort(['sepalwidth', 'petallength'], ascending=[True, False]).head(5)
       sepallength  sepalwidth  petallength  petalwidth             name
    0          5.0         2.0          3.5         1.0  Iris-versicolor
    1          6.0         2.2          5.0         1.5   Iris-virginica
    2          6.2         2.2          4.5         1.5  Iris-versicolor
    3          6.0         2.2          4.0         1.0  Iris-versicolor
    4          6.3         2.3          4.4         1.3  Iris-versicolor

下面效果是一样的：

.. code:: python

    >>> iris.sort(['sepalwidth', -iris.petallength]).head(5)
       sepallength  sepalwidth  petallength  petalwidth             name
    0          5.0         2.0          3.5         1.0  Iris-versicolor
    1          6.0         2.2          5.0         1.5   Iris-virginica
    2          6.2         2.2          4.5         1.5  Iris-versicolor
    3          6.0         2.2          4.0         1.0  Iris-versicolor
    4          6.3         2.3          4.4         1.3  Iris-versicolor


.. note::

    由于 ODPS 要求排序必须指定个数，所以在 ODPS 后端执行时，
    会通过 ``options.df.odps.sort.limit`` 指定排序个数，这个值默认是 10000，
    如果要排序尽量多的数据，可以把这个值设到较大的值。不过注意，此时可能会导致 OOM。

去重
-----

去重在Collection上，用户可以调用distinct方法。

.. code:: python

    >>> iris[['name']].distinct()
                  name
    0      Iris-setosa
    1  Iris-versicolor
    2   Iris-virginica

.. code:: python

    >>> iris.distinct('name')
                  name
    0      Iris-setosa
    1  Iris-versicolor
    2   Iris-virginica

.. code:: python

    >>> iris.distinct('name', 'sepallength').head(3)
              name  sepallength
    0  Iris-setosa          4.3
    1  Iris-setosa          4.4
    2  Iris-setosa          4.5

在Sequence上，用户可以调用unique，但是记住，调用unique的Sequence不能用在列选择中。

.. code:: python

    >>> iris.name.unique()
                  name
    0      Iris-setosa
    1  Iris-versicolor
    2   Iris-virginica


下面的代码是错误的用法。

.. code:: python

    >>> iris[iris.name, iris.name.unique()]  # 错误的


采样
------


要对一个 collection 的数据采样，可以调用 ``sample`` 方法。PyODPS 支持四种采样方式。

.. warning::
    除了按份数采样外，其余方法如果要在 ODPS DataFrame 上执行，需要 Project 支持 XFlow，否则，这些方法只能在
    Pandas DataFrame 后端上执行。

- 按份数采样

在这种采样方式下，数据被分为 ``parts`` 份，可选择选取的份数序号。

.. code:: python

    >>> iris.sample(parts=10)  # 分成10份，默认取第0份
    >>> iris.sample(parts=10, i=0)  # 手动指定取第0份
    >>> iris.sample(parts=10, i=[2, 5])   # 分成10份，取第2和第5份
    >>> iris.sample(parts=10, columns=['name', 'sepalwidth'])  # 根据name和sepalwidth的值做采样

- 按比例 / 条数采样

在这种采样方式下，用户指定需要采样的数据条数或采样比例。指定 ``replace`` 参数为 True 可启用放回采样。

.. code:: python

    >>> iris.sample(n=100)  # 选取100条数据
    >>> iris.sample(frac=0.3)  # 采样30%的数据

- 按权重列采样

在这种采样方式下，用户指定权重列和数据条数 / 采样比例。指定 ``replace`` 参数为 True 可启用放回采样。

.. code:: python

    >>> iris.sample(n=100, weights='sepal_length')
    >>> iris.sample(n=100, weights='sepal_width', replace=True)

- 分层采样

在这种采样方式下，用户指定用于分层的标签列，同时为需要采样的每个标签指定采样比例（ ``frac`` 参数）或条数
（ ``n`` 参数）。暂不支持放回采样。

.. code:: python

    >>> iris.sample(strata='category', n={'Iris Setosa': 10, 'Iris Versicolour': 10})
    >>> iris.sample(strata='category', frac={'Iris Setosa': 0.5, 'Iris Versicolour': 0.4})

数据缩放
--------

DataFrame 支持通过最大/最小值或平均值/标准差对数据进行缩放。例如，对数据

.. code:: python

        name  id  fid
    0  name1   4  5.3
    1  name2   2  3.5
    2  name2   3  1.5
    3  name1   4  4.2
    4  name1   3  2.2
    5  name1   3  4.1

使用 min_max_scale 方法进行归一化：

.. code:: python

    >>> df.min_max_scale(columns=['fid'])
        name  id       fid
    0  name1   4  1.000000
    1  name2   2  0.526316
    2  name2   3  0.000000
    3  name1   4  0.710526
    4  name1   3  0.184211
    5  name1   3  0.684211

min_max_scale 还支持使用 feature_range 参数指定输出值的范围，例如，如果我们需要使输出值在 (-1, 1)
范围内，可使用

.. code:: python

    >>> df.min_max_scale(columns=['fid'], feature_range=(-1, 1))
        name  id       fid
    0  name1   4  1.000000
    1  name2   2  0.052632
    2  name2   3 -1.000000
    3  name1   4  0.421053
    4  name1   3 -0.631579
    5  name1   3  0.368421

如果需要保留原始值，可以使用 preserve 参数。此时，缩放后的数据将会以新增列的形式追加到数据中，
列名默认为原列名追加“_scaled”后缀，该后缀可使用 suffix 参数更改。例如，

.. code:: python

    >>> df.min_max_scale(columns=['fid'], preserve=True)
        name  id  fid  fid_scaled
    0  name1   4  5.3    1.000000
    1  name2   2  3.5    0.526316
    2  name2   3  1.5    0.000000
    3  name1   4  4.2    0.710526
    4  name1   3  2.2    0.184211
    5  name1   3  4.1    0.684211

min_max_scale 也支持使用 group 参数指定一个或多个分组列，在分组列中分别取最值进行缩放。例如，

.. code:: python

    >>> df.min_max_scale(columns=['fid'], group=['name'])
        name  id       fid
    0  name1   4  1.000000
    1  name1   4  0.645161
    2  name1   3  0.000000
    3  name1   3  0.612903
    4  name2   2  1.000000
    5  name2   3  0.000000

可见结果中，name1 和 name2 两组均按组中的最值进行了缩放。

std_scale 可依照标准正态分布对数据进行调整。例如，

.. code:: python

    >>> df.std_scale(columns=['fid'])
        name  id       fid
    0  name1   4  1.436467
    1  name2   2  0.026118
    2  name2   3 -1.540938
    3  name1   4  0.574587
    4  name1   3 -0.992468
    5  name1   3  0.496234

std_scale 同样支持 preserve 参数保留原始列以及使用 group 进行分组，具体请参考 min_max_scale，此处不再赘述。

空值处理
--------

DataFrame 支持筛去空值以及填充空值的功能。例如，对数据

.. code:: python

       id   name   f1   f2   f3   f4
    0   0  name1  1.0  NaN  3.0  4.0
    1   1  name1  2.0  NaN  NaN  1.0
    2   2  name1  3.0  4.0  1.0  NaN
    3   3  name1  NaN  1.0  2.0  3.0
    4   4  name1  1.0  NaN  3.0  4.0
    5   5  name1  1.0  2.0  3.0  4.0
    6   6  name1  NaN  NaN  NaN  NaN

使用 dropna 可删除 subset 中包含空值的行：

.. code:: python

    >>> df.dropna(subset=['f1', 'f2', 'f3', 'f4'])
       id   name   f1   f2   f3   f4
    0   5  name1  1.0  2.0  3.0  4.0

如果行中包含非空值则不删除，可以使用 how='all'：

.. code:: python

    >>> df.dropna(how='all', subset=['f1', 'f2', 'f3', 'f4'])
       id   name   f1   f2   f3   f4
    0   0  name1  1.0  NaN  3.0  4.0
    1   1  name1  2.0  NaN  NaN  1.0
    2   2  name1  3.0  4.0  1.0  NaN
    3   3  name1  NaN  1.0  2.0  3.0
    4   4  name1  1.0  NaN  3.0  4.0
    5   5  name1  1.0  2.0  3.0  4.0

你也可以使用 thresh 参数来指定行中至少要有多少个非空值。例如：

.. code:: python

    >>> df.dropna(thresh=3, subset=['f1', 'f2', 'f3', 'f4'])
       id   name   f1   f2   f3   f4
    0   0  name1  1.0  NaN  3.0  4.0
    2   2  name1  3.0  4.0  1.0  NaN
    3   3  name1  NaN  1.0  2.0  3.0
    4   4  name1  1.0  NaN  3.0  4.0
    5   5  name1  1.0  2.0  3.0  4.0

使用 fillna 可使用常数或已有的列填充未知值。下面给出了使用常数填充的例子：

.. code:: python

    >>> df.fillna(100, subset=['f1', 'f2', 'f3', 'f4'])
       id   name     f1     f2     f3     f4
    0   0  name1    1.0  100.0    3.0    4.0
    1   1  name1    2.0  100.0  100.0    1.0
    2   2  name1    3.0    4.0    1.0  100.0
    3   3  name1  100.0    1.0    2.0    3.0
    4   4  name1    1.0  100.0    3.0    4.0
    5   5  name1    1.0    2.0    3.0    4.0
    6   6  name1  100.0  100.0  100.0  100.0

你也可以使用一个已有的列来填充未知值。例如：

.. code:: python

    >>> df.fillna(df.f2, subset=['f1', 'f2', 'f3', 'f4'])
       id   name   f1   f2   f3   f4
    0   0  name1  1.0  NaN  3.0  4.0
    1   1  name1  2.0  NaN  NaN  1.0
    2   2  name1  3.0  4.0  1.0  4.0
    3   3  name1  1.0  1.0  2.0  3.0
    4   4  name1  1.0  NaN  3.0  4.0
    5   5  name1  1.0  2.0  3.0  4.0
    6   6  name1  NaN  NaN  NaN  NaN

特别地，DataFrame 提供了向前 / 向后填充的功能。通过指定 method 参数为下列值可以达到目的：

================== ==============
 取值               含义
================== ==============
 bfill / backfill   向前填充
 ffill / pad        向后填充
================== ==============

例如：

.. code:: python

    >>> df.fillna(method='bfill', subset=['f1', 'f2', 'f3', 'f4'])
       id   name   f1   f2   f3   f4
    0   0  name1  1.0  3.0  3.0  4.0
    1   1  name1  2.0  1.0  1.0  1.0
    2   2  name1  3.0  4.0  1.0  NaN
    3   3  name1  1.0  1.0  2.0  3.0
    4   4  name1  1.0  3.0  3.0  4.0
    5   5  name1  1.0  2.0  3.0  4.0
    6   6  name1  NaN  NaN  NaN  NaN
    >>> df.fillna(method='ffill', subset=['f1', 'f2', 'f3', 'f4'])
       id   name   f1   f2   f3   f4
    0   0  name1  1.0  1.0  3.0  4.0
    1   1  name1  2.0  2.0  2.0  1.0
    2   2  name1  3.0  4.0  1.0  1.0
    3   3  name1  NaN  1.0  2.0  3.0
    4   4  name1  1.0  1.0  3.0  4.0
    5   5  name1  1.0  2.0  3.0  4.0
    6   6  name1  NaN  NaN  NaN  NaN

你也可以使用 ffill / bfill 函数来简化代码。ffill 等价于 fillna(method='ffill')，
bfill 等价于 fillna(method='bfill')

对所有行/列调用自定义函数
------------------------

.. _dfudtfapp:

对一行数据使用自定义函数
~~~~~~~~~~~~~~~~~~~~~~~

要对一行数据使用自定义函数，可以使用 apply 方法，axis 参数必须为 1，表示在行上操作。

apply 的自定义函数接收一个参数，为上一步 Collection 的一行数据，用户可以通过属性、或者偏移取得一个字段的数据。

.. code:: python

    >>> iris.apply(lambda row: row.sepallength + row.sepalwidth, axis=1, reduce=True, types='float').rename('sepaladd').head(3)
       sepaladd
    0       8.6
    1       7.9
    2       7.9

``reduce``\ 为 True 时，表示返回结果为Sequence，否则返回结果为Collection。
``names``\ 和 ``types``\ 参数分别指定返回的Sequence或Collection的字段名和类型。
如果类型不指定，将会默认为string类型。

在 apply 的自定义函数中，reduce 为 False 时，也可以使用 ``yield``\ 关键字来返回多行结果。

.. code:: python

    >>> iris.count()
    150
    >>>
    >>> def handle(row):
    >>>     yield row.sepallength - row.sepalwidth, row.sepallength + row.sepalwidth
    >>>     yield row.petallength - row.petalwidth, row.petallength + row.petalwidth
    >>>
    >>> iris.apply(handle, axis=1, names=['iris_add', 'iris_sub'], types=['float', 'float']).count()
    300

我们也可以在函数上来注释返回的字段和类型，这样就不需要在函数调用时再指定。


.. code:: python

    >>> from odps.df import output
    >>>
    >>> @output(['iris_add', 'iris_sub'], ['float', 'float'])
    >>> def handle(row):
    >>>     yield row.sepallength - row.sepalwidth, row.sepallength + row.sepalwidth
    >>>     yield row.petallength - row.petalwidth, row.petallength + row.petalwidth
    >>>
    >>> iris.apply(handle, axis=1).count()
    300

也可以使用 map-only 的 map_reduce，和 axis=1 的apply操作是等价的。

.. code:: python

    >>> iris.map_reduce(mapper=handle).count()
    300

如果想调用 ODPS 上已经存在的 UDTF，则函数指定为函数名即可。

.. code:: python

    >>> iris['name', 'sepallength'].apply('your_func', axis=1, names=['name2', 'sepallength2'], types=['string', 'float'])

使用 apply 对行操作，且 ``reduce``\ 为 False 时，可以使用 :ref:`dflateralview` 与已有的行结合，用于后续聚合等操作。

.. code:: python
  
    >>> from odps.df import output
    >>>
    >>> @output(['iris_add', 'iris_sub'], ['float', 'float'])
    >>> def handle(row):
    >>>     yield row.sepallength - row.sepalwidth, row.sepallength + row.sepalwidth
    >>>     yield row.petallength - row.petalwidth, row.petallength + row.petalwidth
    >>>
    >>> iris[iris.category, iris.apply(handle, axis=1)]

对所有列调用自定义聚合
~~~~~~~~~~~~~~~~~~~~~~~

调用apply方法，当我们不指定axis，或者axis为0的时候，我们可以通过传入一个自定义聚合类来对所有sequence进行聚合操作。

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

    >>> iris.exclude('name').apply(Agg)
       sepallength_aggregation  sepalwidth_aggregation  petallength_aggregation  petalwidth_aggregation
    0                 5.843333                   3.054                 3.758667                1.198667

.. warning::
    目前，受限于 Python UDF，自定义函数无法支持将 list / dict 类型作为初始输入或最终输出结果。

引用资源
~~~~~~~~~~~~~

类似于对 :ref:`map <map>` 方法的resources参数，每个resource可以是ODPS上的资源（表资源或文件资源），或者引用一个collection作为资源。

对于axis为1，也就是在行上操作，我们需要写一个函数闭包或者callable的类。
而对于列上的聚合操作，我们只需在 \_\_init\_\_ 函数里读取资源即可。


.. code:: python

    >>> words_df
                         sentence
    0                 Hello World
    1                Hello Python
    2  Life is short I use Python
    >>>
    >>> import pandas as pd
    >>> stop_words = DataFrame(pd.DataFrame({'stops': ['is', 'a', 'I']}))
    >>>
    >>> @output(['sentence'], ['string'])
    >>> def filter_stops(resources):
    >>>     stop_words = set([r[0] for r in resources[0]])
    >>>     def h(row):
    >>>         return ' '.join(w for w in row[0].split() if w not in stop_words),
    >>>     return h
    >>>
    >>> words_df.apply(filter_stops, axis=1, resources=[stop_words])
                    sentence
    0            Hello World
    1           Hello Python
    2  Life short use Python

可以看到这里的stop_words是存放于本地，但在真正执行时会被上传到ODPS作为资源引用。


使用第三方Python库
~~~~~~~~~~~~~~~~~~~~

使用方法类似 :ref:`map中使用第三方Python库 <third_party_library>` 。

可以在全局指定使用的库：

.. code:: python

    >>> from odps import options
    >>> options.df.libraries = ['six.whl', 'python_dateutil.whl']

或者在立即执行的方法中，局部指定：

.. code:: python

    >>> df.apply(my_func, axis=1).to_pandas(libraries=['six.whl', 'python_dateutil.whl'])

.. warning::
    由于字节码定义的差异，Python 3 下使用新语言特性（例如 ``yield from`` ）时，代码在使用 Python 2.7 的 ODPS
    Worker 上执行时会发生错误。因而建议在 Python 3 下使用 MapReduce API 编写生产作业前，先确认相关代码是否能正常
    执行。

.. _map_reduce:

MapReduce API
--------------


PyODPS DataFrame也支持MapReduce API，用户可以分别编写map和reduce函数（map_reduce可以只有mapper或者reducer过程）。
我们来看个简单的wordcount的例子。


.. code:: python

    >>> def mapper(row):
    >>>     for word in row[0].split():
    >>>         yield word.lower(), 1
    >>>
    >>> def reducer(keys):
    >>>     cnt = [0]
    >>>     def h(row, done):  # done表示这个key已经迭代结束
    >>>         cnt[0] += row[1]
    >>>         if done:
    >>>             yield keys[0], cnt[0]
    >>>     return h
    >>>
    >>> words_df.map_reduce(mapper, reducer, group=['word', ],
    >>>                     mapper_output_names=['word', 'cnt'],
    >>>                     mapper_output_types=['string', 'int'],
    >>>                     reducer_output_names=['word', 'cnt'],
    >>>                     reducer_output_types=['string', 'int'])
         word  cnt
    0   hello    2
    1       i    1
    2      is    1
    3    life    1
    4  python    2
    5   short    1
    6     use    1
    7   world    1

group参数用来指定reduce按哪些字段做分组，如果不指定，会按全部字段做分组。

其中对于reducer来说，会稍微有些不同。它需要接收聚合的keys初始化，并能继续处理按这些keys聚合的每行数据。
第2个参数表示这些keys相关的所有行是不是都迭代完成。

这里写成函数闭包的方式，主要为了方便，当然我们也能写成callable的类。

.. code-block:: python

    class reducer(object):
        def __init__(self, keys):
            self.cnt = 0

        def __call__(self, row, done):  # done表示这个key已经迭代结束
            self.cnt += row.cnt
            if done:
                yield row.word, self.cnt

使用 ``output``\ 来注释会让代码更简单些。

.. code:: python

    >>> from odps.df import output
    >>>
    >>> @output(['word', 'cnt'], ['string', 'int'])
    >>> def mapper(row):
    >>>     for word in row[0].split():
    >>>         yield word.lower(), 1
    >>>
    >>> @output(['word', 'cnt'], ['string', 'int'])
    >>> def reducer(keys):
    >>>     cnt = [0]
    >>>     def h(row, done):  # done表示这个key已经迭代结束
    >>>         cnt[0] += row.cnt
    >>>         if done:
    >>>             yield keys.word, cnt[0]
    >>>     return h
    >>>
    >>> words_df.map_reduce(mapper, reducer, group='word')
         word  cnt
    0   hello    2
    1       i    1
    2      is    1
    3    life    1
    4  python    2
    5   short    1
    6     use    1
    7   world    1

有时候我们在迭代的时候需要按某些列排序，则可以使用 ``sort``\ 参数，来指定按哪些列排序，升序降序则通过 ``ascending``\ 参数指定。
``ascending`` 参数可以是一个bool值，表示所有的 ``sort``\ 字段是相同升序或降序，
也可以是一个列表，长度必须和 ``sort``\ 字段长度相同。


指定combiner
~~~~~~~~~~~~~~

combiner表示在map_reduce API里表示在mapper端，就先对数据进行聚合操作，它的用法和reducer是完全一致的，但不能引用资源。
并且，combiner的输出的字段名和字段类型必须和mapper完全一致。

上面的例子，我们就可以使用reducer作为combiner来先在mapper端对数据做初步的聚合，减少shuffle出去的数据量。

.. code:: python

    >>> words_df.map_reduce(mapper, reducer, combiner=reducer, group='word')

引用资源
~~~~~~~~~~~~~

在MapReduce API里，我们能分别指定mapper和reducer所要引用的资源。

如下面的例子，我们对mapper里的单词做停词过滤，在reducer里对白名单的单词数量加5。

.. code:: python

    >>> white_list_file = o.create_resource('pyodps_white_list_words', 'file', file_obj='Python\nWorld')
    >>>
    >>> @output(['word', 'cnt'], ['string', 'int'])
    >>> def mapper(resources):
    >>>     stop_words = set(r[0].strip() for r in resources[0])
    >>>     def h(row):
    >>>         for word in row[0].split():
    >>>             if word not in stop_words:
    >>>                 yield word, 1
    >>>     return h
    >>>
    >>> @output(['word', 'cnt'], ['string', 'int'])
    >>> def reducer(resources):
    >>>     d = dict()
    >>>     d['white_list'] = set(word.strip() for word in resources[0])
    >>>     d['cnt'] = 0
    >>>     def inner(keys):
    >>>         d['cnt'] = 0
    >>>         def h(row, done):
    >>>             d['cnt'] += row.cnt
    >>>             if done:
    >>>                 if row.word in d['white_list']:
    >>>                     d['cnt'] += 5
    >>>                 yield keys.word, d['cnt']
    >>>         return h
    >>>     return inner
    >>>
    >>> words_df.map_reduce(mapper, reducer, group='word',
    >>>                     mapper_resources=[stop_words], reducer_resources=[white_list_file])
         word  cnt
    0   hello    2
    1    life    1
    2  python    7
    3   world    6
    4   short    1
    5     use    1

使用第三方Python库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


使用方法类似 :ref:`map中使用第三方Python库 <third_party_library>` 。

可以在全局指定使用的库：

.. code:: python

    >>> from odps import options
    >>> options.df.libraries = ['six.whl', 'python_dateutil.whl']


或者在立即执行的方法中，局部指定：

.. code:: python

    >>> df.map_reduce(mapper=my_mapper, reducer=my_reducer, group='key').execute(libraries=['six.whl', 'python_dateutil.whl'])


.. warning::
    由于字节码定义的差异，Python 3 下使用新语言特性（例如 ``yield from`` ）时，代码在使用 Python 2.7 的 ODPS
    Worker 上执行时会发生错误。因而建议在 Python 3 下使用 MapReduce API 编写生产作业前，先确认相关代码是否能正常
    执行。


重排数据
----------

有时候我们的数据在集群上分布可能是不均匀的，我们需要对数据重排。调用 ``reshuffle`` 接口即可。


.. code:: python

    >>> df1 = df.reshuffle()


默认会按随机数做哈希来分布。也可以指定按那些列做分布，且可以指定重排后的排序顺序。


.. code:: python

    >>> df1.reshuffle('name', sort='id', ascending=False)


布隆过滤器
----------


PyODPS DataFrame提供了 ``bloom_filter`` 接口来进行布隆过滤器的计算。

给定某个collection，和它的某个列计算的sequence1，我们能对另外一个sequence2进行布隆过滤，sequence1不在sequence2中的一定会过滤，
但可能不能完全过滤掉不存在于sequence2中的数据，这也是一种近似的方法。

这样的好处是能快速对collection进行快速过滤一些无用数据。

这在大规模join的时候，一边数据量远大过另一边数据，而大部分并不会join上的场景很有用。
比如，我们在join用户的浏览数据和交易数据时，用户的浏览大部分不会带来交易，我们可以利用交易数据先对浏览数据进行布隆过滤，
然后再join能很好提升性能。

.. code:: python

    >>> df1 = DataFrame(pd.DataFrame({'a': ['name1', 'name2', 'name3', 'name1'], 'b': [1, 2, 3, 4]}))
    >>> df1
           a  b
    0  name1  1
    1  name2  2
    2  name3  3
    3  name1  4
    >>> df2 = DataFrame(pd.DataFrame({'a': ['name1']}))
    >>> df2
           a
    0  name1
    >>> df1.bloom_filter('a', df2.a) # 这里第0个参数可以是个计算表达式如: df1.a + '1'
           a  b
    0  name1  1
    1  name1  4

这里由于数据量很小，df1中的a为name2和name3的行都被正确过滤掉了，当数据量很大的时候，可能会有一定的数据不能被过滤。

如之前提的join场景中，少量不能过滤并不能并不会影响正确性，但能较大提升join的性能。

我们可以传入 ``capacity`` 和 ``error_rate`` 来设置数据的量以及错误率，默认值是 ``3000`` 和 ``0.01``。

.. note::
    要注意，调大 ``capacity`` 或者减小 ``error_rate`` 会增加内存的使用，所以应当根据实际情况选择一个合理的值。


.. _dfpivot:

透视表（pivot_table）
---------------------

PyODPS DataFrame提供透视表的功能。我们通过几个例子来看使用。


.. code:: python

    >>> df
         A    B      C  D  E
    0  foo  one  small  1  3
    1  foo  one  large  2  4
    2  foo  one  large  2  5
    3  foo  two  small  3  6
    4  foo  two  small  3  4
    5  bar  one  large  4  5
    6  bar  one  small  5  3
    7  bar  two  small  6  2
    8  bar  two  large  7  1


最简单的透视表必须提供一个 ``rows`` 参数，表示按一个或者多个字段做取平均值的操作。

.. code:: python

    >>> df['A', 'D', 'E'].pivot_table(rows='A')
         A  D_mean  E_mean
    0  bar     5.5    2.75
    1  foo     2.2    4.40

rows可以提供多个，表示按多个字段做聚合。

.. code:: python

    >>> df.pivot_table(rows=['A', 'B', 'C'])
         A    B      C  D_mean  E_mean
    0  bar  one  large     4.0     5.0
    1  bar  one  small     5.0     3.0
    2  bar  two  large     7.0     1.0
    3  bar  two  small     6.0     2.0
    4  foo  one  large     2.0     4.5
    5  foo  one  small     1.0     3.0
    6  foo  two  small     3.0     5.0

我们可以指定 ``values`` 来显示指定要计算的列。

.. code:: python

    >>> df.pivot_table(rows=['A', 'B'], values='D')
         A    B    D_mean
    0  bar  one  4.500000
    1  bar  two  6.500000
    2  foo  one  1.666667
    3  foo  two  3.000000

计算值列时，默认会计算平均值，用户可以指定一个或者多个聚合函数。

.. code:: python

    >>> df.pivot_table(rows=['A', 'B'], values=['D'], aggfunc=['mean', 'count', 'sum'])
         A    B    D_mean  D_count  D_sum
    0  bar  one  4.500000        2      9
    1  bar  two  6.500000        2     13
    2  foo  one  1.666667        3      5
    3  foo  two  3.000000        2      6

我们也可以把原始数据的某一列的值，作为新的collection的列。 **这也是透视表最强大的地方。**

.. code:: python

    >>> df.pivot_table(rows=['A', 'B'], values='D', columns='C')
         A    B  large_D_mean  small_D_mean
    0  bar  one           4.0           5.0
    1  bar  two           7.0           6.0
    2  foo  one           2.0           1.0
    3  foo  two           NaN           3.0

我们可以提供 ``fill_value`` 来填充空值。

.. code:: python

    >>> df.pivot_table(rows=['A', 'B'], values='D', columns='C', fill_value=0)
         A    B  large_D_mean  small_D_mean
    0  bar  one             4             5
    1  bar  two             7             6
    2  foo  one             2             1
    3  foo  two             0             3


Key-Value 字符串转换
---------------------

DataFrame 提供了将 Key-Value 对展开为列，以及将普通列转换为 Key-Value 列的功能。

我们的数据为

.. code:: python

    >>> df
        name               kv
    0  name1  k1=1,k2=3,k5=10
    1  name1    k1=7.1,k7=8.2
    2  name2    k2=1.2,k3=1.5
    3  name2      k9=1.1,k2=1

可以通过 extract_kv 方法将 Key-Value 字段展开：

.. code:: python

    >>> df.extract_kv(columns=['kv'], kv_delim='=', item_delim=',')
       name   kv_k1  kv_k2  kv_k3  kv_k5  kv_k7  kv_k9
    0  name1    1.0    3.0    NaN   10.0    NaN    NaN
    1  name1    7.0    NaN    NaN    NaN    8.2    NaN
    2  name2    NaN    1.2    1.5    NaN    NaN    NaN
    3  name2    NaN    1.0    NaN    NaN    NaN    1.1

其中，需要展开的字段名由 columns 指定，Key 和 Value 之间的分隔符，以及 Key-Value 对之间的分隔符分别由
kv_delim 和 item_delim 这两个参数指定，默认分别为半角冒号和半角逗号。输出的字段名为原字段名和 Key
值的组合，通过“_”相连。缺失值默认为 None，可通过 ``fill_value`` 选择需要填充的值。例如，相同的 df，

.. code:: python

    >>> df.extract_kv(columns=['kv'], kv_delim='=', fill_value=0)
       name   kv_k1  kv_k2  kv_k3  kv_k5  kv_k7  kv_k9
    0  name1    1.0    3.0    0.0   10.0    0.0    0.0
    1  name1    7.0    0.0    0.0    0.0    8.2    0.0
    2  name2    0.0    1.2    1.5    0.0    0.0    0.0
    3  name2    0.0    1.0    0.0    0.0    0.0    1.1

DataFrame 也支持将多列数据转换为一个 Key-Value 列。例如，

.. code:: python

    >>> df
       name    k1   k2   k3    k5   k7   k9
    0  name1  1.0  3.0  NaN  10.0  NaN  NaN
    1  name1  7.0  NaN  NaN   NaN  8.2  NaN
    2  name2  NaN  1.2  1.5   NaN  NaN  NaN
    3  name2  NaN  1.0  NaN   NaN  NaN  1.1

可通过 to_kv 方法转换为 Key-Value 表示的格式：

.. code:: python

    >>> df.to_kv(columns=['k1', 'k2', 'k3', 'k5', 'k7', 'k9'], kv_delim='=')
        name               kv
    0  name1  k1=1,k2=3,k5=10
    1  name1    k1=7.1,k7=8.2
    2  name2    k2=1.2,k3=1.5
    3  name2      k9=1.1,k2=1
