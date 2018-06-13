.. _dfbasic:
.. currentmodule:: odps.df

基本概念
========

在使用 DataFrame 时，你需要了解三个对象上的操作：\ ``Collection``\ (``DataFrame``) ，\ ``Sequence``\ ，\ ``Scalar``\ 。
这三个对象分别表示表结构（或者二维结构）、列（一维结构）、标量。需要注意的是，这些对象仅在使用 Pandas 数据创建后会包含实际数据，
而在 ODPS 表上创建的对象中并不包含实际的数据，而仅仅包含对这些数据的操作，实质的存储和计算会在 ODPS 中进行。

创建 DataFrame
--------------

通常情况下，你唯一需要直接创建的 Collection 对象是 :class:`DataFrame`，这一对象用于引用数据源，可能是一个 ODPS 表，
ODPS 分区，Pandas DataFrame或sqlalchemy.Table（数据库表）。
使用这几种数据源时，相关的操作相同，这意味着你可以不更改数据处理的代码，仅仅修改输入/输出的指向，
便可以简单地将小数据量上本地测试运行的代码迁移到 ODPS 上，而迁移的正确性由 PyODPS 来保证。

创建 DataFrame 非常简单，只需将 Table 对象、 pandas DataFrame 对象或者 sqlalchemy Table 对象传入即可。

.. code:: python

    >>> from odps.df import DataFrame
    >>>
    >>> # 从 ODPS 表创建
    >>> iris = DataFrame(o.get_table('pyodps_iris'))
    >>> iris2 = o.get_table('pyodps_iris').to_df()  # 使用表的to_df方法
    >>>
    >>> # 从 ODPS 分区创建
    >>> pt_df = DataFrame(o.get_table('partitioned_table').get_partition('pt=20171111'))
    >>> pt_df2 = o.get_table('partitioned_table').get_partition('pt=20171111').to_df()  # 使用分区的to_df方法
    >>>
    >>> # 从 Pandas DataFrame 创建
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = DataFrame(pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc')))
    >>>
    >>> # 从 sqlalchemy Table 创建
    >>> engine = sqlalchemy.create_engine('mysql://root:123456@localhost/movielens')
    >>> metadata = sqlalchemy.MetaData(bind=engine) # 需要绑定到engine
    >>> table = sqlalchemy.Table('top_users', metadata, extend_existing=True, autoload=True)
    >>> users = DataFrame(table)

在用 pandas DataFrame 初始化时，对于 numpy object 类型或者 string 类型，PyODPS DataFrame 会尝试推断类型，
如果一整列都为空，则会报错。这时，用户可以指定 `unknown_as_string` 为True，会将这些列指定为string类型。
用户也可以指定 as_type 参数。若类型为基本类型，会在创建 PyODPS DataFrame 时进行强制类型转换。
如果 Pandas DataFrame 中包含 list 或者 dict 列，该列的类型不会被推断，必须手动使用 as_type 指定。
as_type 参数类型必须是dict。

.. code:: python

    >>> df2 = DataFrame(df, unknown_as_string=True, as_type={'null_col2': 'float'})
    >>> df2.dtypes
    odps.Schema {
      sepallength           float64
      sepalwidth            float64
      petallength           float64
      petalwidth            float64
      name                  string
      null_col1             string   # 无法识别，通过unknown_as_string设置成string类型
      null_col2             float64  # 强制转换成float类型
    }
    >>> df4 = DataFrame(df3, as_type={'list_col': 'list<int64>'})
    >>> df4.dtypes
    odps.Schema {
      id        int64
      list_col  list<int64>  # 无法识别且无法自动转换，通过 as_type 设置
    }

Sequence
--------

:class:`SequenceExpr` 代表了二维数据集中的一列。你不应当手动创建 SequenceExpr，而应当从一个 Collection 中获取。


获取列
~~~~~~~

你可以使用 collection.column_name 取出一列，例如

.. code:: python

    >>> iris.sepallength.head(5)
       sepallength
    0          5.1
    1          4.9
    2          4.7
    3          4.6
    4          5.0


如果列名存储在一个字符串变量中，除了使用 getattr(df, 'column_name') 达到相同的效果外，也可以使用 df[column_name]
的形式，例如

.. code:: python

    >>> iris['sepallength'].head(5)
       sepallength
    0          5.1
    1          4.9
    2          4.7
    3          4.6
    4          5.0

列类型
~~~~~~

DataFrame包括自己的类型系统，在使用Table初始化的时候，ODPS的类型会被进行转换。这样做的好处是，能支持更多的计算后端。
目前，DataFrame的执行后端支持ODPS SQL、pandas以及数据库（MySQL和Postgres）。

PyODPS DataFrame 包括以下类型：

``int8``\ ，\ ``int16``\ ，\ ``int32``\ ，\ ``int64``\ ，\ ``float32``\ ，\ ``float64``\ ，\ ``boolean``\ ，\
``string``\ ，\ ``decimal``\ ，\ ``datetime``\ ，\ ``list``\ ，\ ``dict``

ODPS的字段和DataFrame的类型映射关系如下：

============================= ============================
 ODPS类型                      DataFrame类型
============================= ============================
 bigint                        int64
 double                        float64
 string                        string
 datetime                      datetime
 boolean                       boolean
 decimal                       decimal
 array<value_type>             list<value_type>
 map<key_type, value_type>     dict<key_type, value_type>
============================= ============================

list 和 dict 必须填写其包含值的类型，否则会报错。目前 DataFrame 暂不支持 MaxCompute 2.0 中新增的
Timestamp 及 Struct 类型，未来的版本会支持。

在 Sequence 中可以通过 sequence.dtype 获取数据类型：

.. code:: python

    >>> iris.sepallength.dtype
    float64

如果要修改一列的类型，可以使用 astype 方法。该方法输入一个类型，并返回类型转换后的 Sequence。例如，

.. code:: python

    >>> iris.sepallength.astype('int')
       sepallength
    0            5
    1            4
    2            4
    3            4
    4            5


列名
~~~~

在 DataFrame 的计算过程中，一个 Sequence 必须要有列名。在很多情况下，DataFrame 会起一个名字。比如：

.. code:: python

    >>> iris.groupby('name').sepalwidth.max()
       sepalwidth_max
    0             4.4
    1             3.4
    2             3.8

可以看到，\ ``sepalwidth``\ 取最大值后被命名为\ ``sepalwidth_max``\ 。还有一些操作，比如一个 Sequence
做加法，加上一个 Scalar，这时，会被命名为这个 Sequence 的名字。其它情况下，需要用户去自己命名。

Sequence 提供 rename 方法对一列进行重命名，用法示例如下：

.. code:: python

    >>> iris.sepalwidth.rename('sepal_width').head(5)
       sepal_width
    0          3.5
    1          3.0
    2          3.2
    3          3.1
    4          3.6


简单的列变换
~~~~~~~~~~~

你可以对一个 Sequence 进行运算，返回一个新的 Sequence，正如对简单的 Python 变量进行运算一样。对数值列，
Sequence 支持四则运算，而对字符串则支持字符串相加等操作。例如，

.. code:: python

    >>> (iris.sepallength + 5).head(5)
       sepallength
    0         10.1
    1          9.9
    2          9.7
    3          9.6
    4         10.0

而

.. code:: python

    >>> (iris.sepallength + iris.sepalwidth).rename('sum_sepal').head(5)
       sum_sepal
    0        8.6
    1        7.9
    2        7.9
    3        7.7
    4        8.6

注意到两列参与运算，因而 PyODPS 无法确定最终显示的列名，需要手动指定。详细的列变换说明，请参见 :ref:`dfelement`。

Collection
----------
DataFrame 中所有二维数据集上的操作都属于 :class:`CollectionExpr`，可视为一张 ODPS 表或一张电子表单，DataFrame
对象也是 CollectionExpr 的特例。CollectionExpr 中包含针对二维数据集的列操作、筛选、变换等大量操作。

获取类型
~~~~~~~~

``dtypes``\ 可以用来获取 CollectionExpr 中所有列的类型。``dtypes`` 返回的是 :ref:`Schema类型 <table_schema>` 。

.. code:: python

    >>> iris.dtypes
    odps.Schema {
      sepallength           float64
      sepalwidth            float64
      petallength           float64
      petalwidth            float64
      name                  string
    }


列选择和增删
~~~~~~~~~~~~

如果要从一个 CollectionExpr 中选取部分列，产生新的数据集，可以使用 expr[columns] 语法。例如，

.. code:: python

    >>> iris['name', 'sepallength'].head(5)
              name  sepallength
    0  Iris-setosa          5.1
    1  Iris-setosa          4.9
    2  Iris-setosa          4.7
    3  Iris-setosa          4.6
    4  Iris-setosa          5.0

.. note::

    **注意**\ ：如果需要选择的列只有一列，需要在 columns 后加上逗号或者显示标记为列表，例如
    df[df.sepal_length, ] 或 df[[df.sepal_length]]，否则返回的将是一个 Sequence 对象，而不是 Collection。

如果想要在新的数据集中排除已有数据集的某些列，可使用 exclude 方法：

.. code:: python

    >>> iris.exclude('sepallength', 'petallength')[:5]
       sepalwidth  petalwidth         name
    0         3.5         0.2  Iris-setosa
    1         3.0         0.2  Iris-setosa
    2         3.2         0.2  Iris-setosa
    3         3.1         0.2  Iris-setosa
    4         3.6         0.2  Iris-setosa

0.7.2 以后的 PyODPS 支持另一种写法，即在数据集上直接排除相应的列：

.. code:: python

    >>> del iris['sepallength']
    >>> del iris['petallength']
    >>> iris[:5]
       sepalwidth  petalwidth         name
    0         3.5         0.2  Iris-setosa
    1         3.0         0.2  Iris-setosa
    2         3.2         0.2  Iris-setosa
    3         3.1         0.2  Iris-setosa
    4         3.6         0.2  Iris-setosa

如果我们需要在已有 collection 中添加某一列变换的结果，也可以使用 expr[expr, new_sequence] 语法，
新增的列会作为新 collection 的一部分。

下面的例子将 iris 中的 sepalwidth 列加一后重命名为 sepalwidthplus1 并追加到数据集末尾，形成新的数据集：

.. code:: python

    >>> iris[iris, (iris.sepalwidth + 1).rename('sepalwidthplus1')].head(5)
       sepallength  sepalwidth  petallength  petalwidth         name  \
    0          5.1         3.5          1.4         0.2  Iris-setosa
    1          4.9         3.0          1.4         0.2  Iris-setosa
    2          4.7         3.2          1.3         0.2  Iris-setosa
    3          4.6         3.1          1.5         0.2  Iris-setosa
    4          5.0         3.6          1.4         0.2  Iris-setosa

       sepalwidthplus1
    0              4.5
    1              4.0
    2              4.2
    3              4.1
    4              4.6

使用 `df[df, new_sequence]` 需要注意的是，变换后的列名与原列名可能相同，如果需要与原 collection 合并，
请将该列重命名。

0.7.2 以后版本的 PyODPS 支持直接在当前数据集中追加，写法为

.. code:: python

    >>> iris['sepalwidthplus1'] = iris.sepalwidth + 1
    >>> iris.head(5)
       sepallength  sepalwidth  petallength  petalwidth         name  \
    0          5.1         3.5          1.4         0.2  Iris-setosa
    1          4.9         3.0          1.4         0.2  Iris-setosa
    2          4.7         3.2          1.3         0.2  Iris-setosa
    3          4.6         3.1          1.5         0.2  Iris-setosa
    4          5.0         3.6          1.4         0.2  Iris-setosa

       sepalwidthplus1
    0              4.5
    1              4.0
    2              4.2
    3              4.1
    4              4.6

我们也可以先将原列通过 exclude 方法进行排除，再将变换后的新列并入，而不必担心重名。

.. code:: python

    >>> iris[iris.exclude('sepalwidth'), iris.sepalwidth * 2].head(5)
       sepallength  petallength  petalwidth         name  sepalwidth
    0          5.1          1.4         0.2  Iris-setosa         7.0
    1          4.9          1.4         0.2  Iris-setosa         6.0
    2          4.7          1.3         0.2  Iris-setosa         6.4
    3          4.6          1.5         0.2  Iris-setosa         6.2
    4          5.0          1.4         0.2  Iris-setosa         7.2

对于 0.7.2 以后版本的 PyODPS，如果想在当前数据集上直接覆盖，则可以写

.. code:: python

    >>> iris['sepalwidth'] = iris.sepalwidth * 2
    >>> iris.head(5)
       sepallength  sepalwidth  petallength  petalwidth         name
    0          5.1         7.0          1.4         0.2  Iris-setosa
    1          4.9         6.0          1.4         0.2  Iris-setosa
    2          4.7         6.4          1.3         0.2  Iris-setosa
    3          4.6         6.2          1.5         0.2  Iris-setosa
    4          5.0         7.2          1.4         0.2  Iris-setosa

增删列以创建新 collection 的另一种方法是调用 select 方法，将需要选择的列作为参数输入。如果需要重命名，使用 keyword
参数输入，并将新的列名作为参数名即可。

.. code:: python

    >>> iris.select('name', sepalwidthminus1=iris.sepalwidth - 1).head(5)
              name  sepalwidthminus1
    0  Iris-setosa               2.5
    1  Iris-setosa               2.0
    2  Iris-setosa               2.2
    3  Iris-setosa               2.1
    4  Iris-setosa               2.6

此外，我们也可以传入一个 lambda 表达式，它接收一个参数，接收上一步的结果。在执行时，PyODPS
会检查这些 lambda 表达式，传入上一步生成的 collection 并将其替换为正确的列。

.. code:: python

    >>> iris['name', 'petallength'][[lambda x: x.name]].head(5)
              name
    0  Iris-setosa
    1  Iris-setosa
    2  Iris-setosa
    3  Iris-setosa
    4  Iris-setosa

此外，在 0.7.2 以后版本的 PyODPS 中，支持对数据进行条件赋值，例如

.. code:: python

    >>> iris[iris.sepallength > 5.0, 'sepalwidth'] = iris.sepalwidth * 2
    >>> iris.head(5)
       sepallength  sepalwidth  petallength  petalwidth         name
    0          5.1        14.0          1.4         0.2  Iris-setosa
    1          4.9         6.0          1.4         0.2  Iris-setosa
    2          4.7         6.4          1.3         0.2  Iris-setosa
    3          4.6         6.2          1.5         0.2  Iris-setosa
    4          5.0         7.2          1.4         0.2  Iris-setosa

引入常数和随机数
~~~~~~~~~~~~~~~~
DataFrame 支持在 collection 中追加一列常数。追加常数需要使用 :class:`Scalar`，引入时需要手动指定列名，如

.. code:: python

    >>> from odps.df import Scalar
    >>> iris[iris, Scalar(1).rename('id')][:5]
       sepallength  sepalwidth  petallength  petalwidth         name  id
    0          5.1         3.5          1.4         0.2  Iris-setosa   1
    1          4.9         3.0          1.4         0.2  Iris-setosa   1
    2          4.7         3.2          1.3         0.2  Iris-setosa   1
    3          4.6         3.1          1.5         0.2  Iris-setosa   1
    4          5.0         3.6          1.4         0.2  Iris-setosa   1


如果需要指定一个空值列，可以使用 :class:`NullScalar`，需要提供字段类型。

.. code:: python

    >>> from odps.df import NullScalar
    >>> iris[iris, NullScalar('float').rename('fid')][:5]
       sepal_length  sepal_width  petal_length  petal_width     category   fid
    0           5.1          3.5           1.4          0.2  Iris-setosa  None
    1           4.9          3.0           1.4          0.2  Iris-setosa  None
    2           4.7          3.2           1.3          0.2  Iris-setosa  None
    3           4.6          3.1           1.5          0.2  Iris-setosa  None
    4           5.0          3.6           1.4          0.2  Iris-setosa  None


在 PyODPS 0.7.12 及以后版本中，引入了简化写法：

.. code:: python

    >>> iris['id'] = 1
    >>> iris
       sepallength  sepalwidth  petallength  petalwidth         name  id
    0          5.1         3.5          1.4         0.2  Iris-setosa   1
    1          4.9         3.0          1.4         0.2  Iris-setosa   1
    2          4.7         3.2          1.3         0.2  Iris-setosa   1
    3          4.6         3.1          1.5         0.2  Iris-setosa   1
    4          5.0         3.6          1.4         0.2  Iris-setosa   1


需要注意的是，这种写法无法自动识别空值的类型，所以在增加空值列时，仍然要使用

.. code:: python

    >>> iris['null_col'] = NullScalar('float')
    >>> iris
       sepallength  sepalwidth  petallength  petalwidth         name  null_col
    0          5.1         3.5          1.4         0.2  Iris-setosa      None
    1          4.9         3.0          1.4         0.2  Iris-setosa      None
    2          4.7         3.2          1.3         0.2  Iris-setosa      None
    3          4.6         3.1          1.5         0.2  Iris-setosa      None
    4          5.0         3.6          1.4         0.2  Iris-setosa      None


DataFrame 也支持在 collection 中增加一列随机数列，该列类型为 float，范围为 0 - 1，每行数值均不同。
追加随机数列需要使用 :class:`RandomScalar`，参数为随机数种子，可省略。

.. code:: python

    >>> from odps.df import RandomScalar
    >>> iris[iris, RandomScalar().rename('rand_val')][:5]
       sepallength  sepalwidth  petallength  petalwidth         name  rand_val
    0          5.1         3.5          1.4         0.2  Iris-setosa  0.000471
    1          4.9         3.0          1.4         0.2  Iris-setosa  0.799520
    2          4.7         3.2          1.3         0.2  Iris-setosa  0.834609
    3          4.6         3.1          1.5         0.2  Iris-setosa  0.106921
    4          5.0         3.6          1.4         0.2  Iris-setosa  0.763442

过滤数据
~~~~~~~~

Collection 提供了数据过滤的功能，

我们试着查询\ ``sepallength``\ 大于5的几条数据。

.. code:: python

    >>> iris[iris.sepallength > 5].head(5)
       sepallength  sepalwidth  petallength  petalwidth         name
    0          5.1         3.5          1.4         0.2  Iris-setosa
    1          5.4         3.9          1.7         0.4  Iris-setosa
    2          5.4         3.7          1.5         0.2  Iris-setosa
    3          5.8         4.0          1.2         0.2  Iris-setosa
    4          5.7         4.4          1.5         0.4  Iris-setosa

多个查询条件：

.. code:: python

    >>> iris[(iris.sepallength < 5) & (iris['petallength'] > 1.5)].head(5)
       sepallength  sepalwidth  petallength  petalwidth             name
    0          4.8         3.4          1.6         0.2      Iris-setosa
    1          4.8         3.4          1.9         0.2      Iris-setosa
    2          4.7         3.2          1.6         0.2      Iris-setosa
    3          4.8         3.1          1.6         0.2      Iris-setosa
    4          4.9         2.4          3.3         1.0  Iris-versicolor


或条件：

.. code:: python

    >>> iris[(iris.sepalwidth < 2.5) | (iris.sepalwidth > 4)].head(5)
       sepallength  sepalwidth  petallength  petalwidth             name
    0          5.7         4.4          1.5         0.4      Iris-setosa
    1          5.2         4.1          1.5         0.1      Iris-setosa
    2          5.5         4.2          1.4         0.2      Iris-setosa
    3          4.5         2.3          1.3         0.3      Iris-setosa
    4          5.5         2.3          4.0         1.3  Iris-versicolor


.. note::

    **记住，与和或条件必须使用&和|，不能使用and和or。**


非条件：

.. code:: python

    >>> iris[~(iris.sepalwidth > 3)].head(5)
       sepallength  sepalwidth  petallength  petalwidth         name
    0          4.9         3.0          1.4         0.2  Iris-setosa
    1          4.4         2.9          1.4         0.2  Iris-setosa
    2          4.8         3.0          1.4         0.1  Iris-setosa
    3          4.3         3.0          1.1         0.1  Iris-setosa
    4          5.0         3.0          1.6         0.2  Iris-setosa


我们也可以显式调用filter方法，提供多个与条件

.. code:: python

    >>> iris.filter(iris.sepalwidth > 3.5, iris.sepalwidth < 4).head(5)
       sepallength  sepalwidth  petallength  petalwidth         name
    0          5.0         3.6          1.4         0.2  Iris-setosa
    1          5.4         3.9          1.7         0.4  Iris-setosa
    2          5.4         3.7          1.5         0.2  Iris-setosa
    3          5.4         3.9          1.3         0.4  Iris-setosa
    4          5.7         3.8          1.7         0.3  Iris-setosa


同样对于连续的操作，我们可以使用lambda表达式

.. code:: python

    >>> iris[iris.sepalwidth > 3.8]['name', lambda x: x.sepallength + 1]
              name  sepallength
    0  Iris-setosa          6.4
    1  Iris-setosa          6.8
    2  Iris-setosa          6.7
    3  Iris-setosa          6.4
    4  Iris-setosa          6.2
    5  Iris-setosa          6.5

对于Collection，如果它包含一个列是boolean类型，则可以直接使用该列作为过滤条件。


.. code:: python

    >>> df.dtypes
    odps.Schema {
      a boolean
      b int64
    }
    >>> df[df.a]
          a  b
    0  True  1
    1  True  3

因此，记住对Collection取单个squence的操作时，只有boolean列是合法的，即对Collection作过滤操作。


.. code:: python

    >>> df[df.a, ]       # 取列操作
    >>> df[[df.a]]       # 取列操作
    >>> df.select(df.a)  # 显式取列
    >>> df[df.a]         # a列是boolean列，执行过滤操作
    >>> df.a             # 取单列
    >>> df['a']          # 取单列

同时，我们也支持Pandas中的\ ``query``\方法，用查询语句来做数据的筛选，在表达式中直接使用列名如\ ``sepallength``\进行操作，
另外在查询语句中\ ``&``\和\ ``and``\都表示与操作，\ ``|``\和\ ``or``\都表示或操作。


.. code:: python

    >>> iris.query("(sepallength < 5) and (petallength > 1.5)").head(5)
       sepallength  sepalwidth  petallength  petalwidth             name
    0          4.8         3.4          1.6         0.2      Iris-setosa
    1          4.8         3.4          1.9         0.2      Iris-setosa
    2          4.7         3.2          1.6         0.2      Iris-setosa
    3          4.8         3.1          1.6         0.2      Iris-setosa
    4          4.9         2.4          3.3         1.0  Iris-versicolor

当表达式中需要使用到本地变量时，需要在该变量前加一个\ ``@``\ 前缀。


.. code:: python

    >>> var = 4
    >>> iris.query("(iris.sepalwidth < 2.5) | (sepalwidth > @var)").head(5)
       sepallength  sepalwidth  petallength  petalwidth             name
    0          5.7         4.4          1.5         0.4      Iris-setosa
    1          5.2         4.1          1.5         0.1      Iris-setosa
    2          5.5         4.2          1.4         0.2      Iris-setosa
    3          4.5         2.3          1.3         0.3      Iris-setosa
    4          5.5         2.3          4.0         1.3  Iris-versicolor

目前\ ``query``\支持的语法包括：

======================== ==========================================================================================
 语法                     说明
======================== ==========================================================================================
 name                     没有 \ ``@``\ 前缀的都当做列名处理，有前缀的会获取本地变量
 operator                 支持部分运算符：\ ``+``\，\ ``-``\，\ ``*``\，\ ``/``\，\ ``//``\，\ ``%``\，\ ``**``\,
                          \ ``==``\，\ ``!=``\，\ ``<``\，\ ``<=``\，\ ``>``\，\ ``>=``\，\ ``in``\，\ ``not in``\
 bool                     与或非操作，其中 \ ``&``\ 和 \ ``and``\ 表示与，\ ``|``\ 和 \ ``or``\ 表示或
 attribute                取对象属性
 index, slice, Subscript  切片操作
======================== ==========================================================================================

.. _dflateralview:

并列多行输出
~~~~~~~~~~~~
对于 list 及 map 类型的列，explode 方法会将该列转换为多行输出。使用 apply 方法也可以输出多行。
为了进行聚合等操作，常常需要将这些输出和原表中的列合并。此时可以使用 DataFrame 提供的并列多行输出功能，
写法为将多行输出函数生成的集合与原集合中的列名一起映射。

并列多行输出的例子如下：

.. code:: python

    >>> df
       id         a             b
    0   1  [a1, b1]  [a2, b2, c2]
    1   2      [c1]      [d2, e2]
    >>> df[df.id, df.a.explode(), df.b]
       id   a             b
    0   1  a1  [a2, b2, c2]
    1   1  b1  [a2, b2, c2]
    2   2  c1      [d2, e2]
    >>> df[df.id, df.a.explode(), df.b.explode()]
       id   a   b
    0   1  a1  a2
    1   1  a1  b2
    2   1  a1  c2
    3   1  b1  a2
    4   1  b1  b2
    5   1  b1  c2
    6   2  c1  d2
    7   2  c1  e2

如果多行输出方法对某个输入不产生任何输出，默认输入行将不在最终结果中出现。如果需要在结果中出现该行，可以设置
``keep_nulls=True``。此时，与该行并列的值将输出为空值：

.. code:: python

    >>> df
       id         a
    0   1  [a1, b1]
    1   2        []
    >>> df[df.id, df.a.explode()]
       id   a
    0   1  a1
    1   1  b1
    >>> df[df.id, df.a.explode(keep_nulls=True)]
       id     a
    0   1    a1
    1   1    b1
    2   2  None

关于 explode 使用并列输出的具体文档可参考 :ref:`dfcollections`，对于 apply 方法使用并列输出的例子可参考 :ref:`dfudtfapp`。


限制条数
~~~~~~~~

.. code:: python

    >>> iris[:3]
       sepallength  sepalwidth  petallength  petalwidth         name
    0          5.1         3.5          1.4         0.2  Iris-setosa
    1          4.9         3.0          1.4         0.2  Iris-setosa
    2          4.7         3.2          1.3         0.2  Iris-setosa

值得注意的是，目前切片对于ODPS SQL后端不支持start和step。我们也可以使用limit方法

.. code:: python

    >>> iris.limit(3)
       sepallength  sepalwidth  petallength  petalwidth         name
    0          5.1         3.5          1.4         0.2  Iris-setosa
    1          4.9         3.0          1.4         0.2  Iris-setosa
    2          4.7         3.2          1.3         0.2  Iris-setosa

.. note::

    **另外，切片操作只能作用在collection上，不能作用于sequence。**

执行
-----

.. _df_delay_execute:

延迟执行
~~~~~~~~

DataFrame上的所有操作并不会立即执行，只有当用户显式调用\ ``execute``\ 方法，或者一些立即执行的方法时（内部调用的就是\ ``execute``\ ），才会真正去执行。

这些立即执行的方法包括：

===================== ============================================================================================== ========================================================================
 方法                  说明                                                                                           返回值
===================== ============================================================================================== ========================================================================
 persist              将执行结果保存到ODPS表                                                                          PyODPS DataFrame
 execute              执行并返回全部结果                                                                              ResultFrame
 head                 查看开头N行数据，这个方法会执行所有结果，并取开头N行数据                                        ResultFrame
 tail                 查看结尾N行数据，这个方法会执行所有结果，并取结尾N行数据                                        ResultFrame
 to_pandas            转化为 Pandas DataFrame 或者 Series，wrap 参数为 True 的时候，返回 PyODPS DataFrame 对象        wrap为True返回PyODPS DataFrame，False（默认）返回pandas DataFrame
 plot，hist，boxplot  画图有关
===================== ============================================================================================== ========================================================================

.. note::

    **注意**\ ：在交互式环境下，PyODPS
    DataFrame会在打印或者repr的时候，调用\ ``execute``\ 方法，这样省去了用户手动去调用execute。

.. code:: python

    >>> iris[iris.sepallength < 5][:5]
       sepallength  sepalwidth  petallength  petalwidth         name
    0          4.9         3.0          1.4         0.2  Iris-setosa
    1          4.7         3.2          1.3         0.2  Iris-setosa
    2          4.6         3.1          1.5         0.2  Iris-setosa
    3          4.6         3.4          1.4         0.3  Iris-setosa
    4          4.4         2.9          1.4         0.2  Iris-setosa

如果想关闭自动调用执行，则需要手动设置

.. code:: python

    >>> from odps import options
    >>> options.interactive = False
    >>>
    >>> iris[iris.sepallength < 5][:5]
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


.. note::

    ResultFrame是结果集合，不能参与后续计算。


ResultFrame可以迭代取出每条记录：


.. code:: python

    >>> result = iris.head(3)
    >>> for r in result:
    >>>     print(list(r))
    [5.0999999999999996, 3.5, 1.3999999999999999, 0.20000000000000001, u'Iris-setosa']
    [4.9000000000000004, 3.0, 1.3999999999999999, 0.20000000000000001, u'Iris-setosa']
    [4.7000000000000002, 3.2000000000000002, 1.3, 0.20000000000000001, u'Iris-setosa']


ResultFrame 也支持在安装有 pandas 的前提下转换为 pandas DataFrame 或使用 pandas 后端的 PyODPS DataFrame：


.. code:: python

    >>> pd_df = iris.head(3).to_pandas()  # 返回 pandas DataFrame
    >>> wrapped_df = iris.head(3).to_pandas(wrap=True)  # 返回使用 Pandas 后端的 PyODPS DataFrame


保存执行结果为 ODPS 表
~~~~~~~~~~~~~~~~~~~~~~

对 Collection，我们可以调用\ ``persist``\ 方法，参数为表名。返回一个新的DataFrame对象

.. code:: python

    >>> iris2 = iris[iris.sepalwidth < 2.5].persist('pyodps_iris2')
    >>> iris2.head(5)
       sepallength  sepalwidth  petallength  petalwidth             name
    0          4.5         2.3          1.3         0.3      Iris-setosa
    1          5.5         2.3          4.0         1.3  Iris-versicolor
    2          4.9         2.4          3.3         1.0  Iris-versicolor
    3          5.0         2.0          3.5         1.0  Iris-versicolor
    4          6.0         2.2          4.0         1.0  Iris-versicolor

``persist``\ 可以传入partitions参数，这样会创建一个表，它的分区是partitions所指定的字段。

.. code:: python

    >>> iris3 = iris[iris.sepalwidth < 2.5].persist('pyodps_iris3', partitions=['name'])
    >>> iris3.data
    odps.Table
      name: odps_test_sqltask_finance.`pyodps_iris3`
      schema:
        sepallength           : double
        sepalwidth            : double
        petallength           : double
        petalwidth            : double
      partitions:
        name                  : string


如果想写入已经存在的表的某个分区，``persist``\ 可以传入partition参数，指明写入表的哪个分区（如ds=******）。
这时要注意，该DataFrame的每个字段都必须在该表存在，且类型相同。drop_partition和create_partition参数只有在此时有效,
分别表示是否要删除（如果分区存在）或创建（如果分区不存在）该分区。

.. code:: python

    >>> iris[iris.sepalwidth < 2.5].persist('pyodps_iris4', partition='ds=test', drop_partition=True, create_partition=True)

写入表时，还可以指定表的生命周期，如下列语句将表的生命周期指定为10天：

.. code:: python

    >>> iris[iris.sepalwidth < 2.5].persist('pyodps_iris5', lifecycle=10)

如果数据源中没有 ODPS 对象，例如数据源仅为 Pandas，在 persist 时需要手动指定 ODPS 入口对象，
或者将需要的入口对象标明为全局对象，如：

.. code:: python

    >>> # 假设入口对象为 o
    >>> # 指定入口对象
    >>> df.persist('table_name', odps=o)
    >>> # 或者可将入口对象标记为全局
    >>> o.to_global()
    >>> df.persist('table_name')

保存执行结果为 Pandas DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们可以使用 ``to_pandas``\ 方法，如果wrap参数为True，将返回PyODPS DataFrame对象。

.. code:: python

    >>> type(iris[iris.sepalwidth < 2.5].to_pandas())
    pandas.core.frame.DataFrame
    >>> type(iris[iris.sepalwidth < 2.5].to_pandas(wrap=True))
    odps.df.core.DataFrame


立即运行设置运行参数
~~~~~~~~~~~~~~~~~~~

对于立即执行的方法，比如 ``execute``、``persist``、``to_pandas`` 等，可以设置运行时参数（仅对ODPS SQL后端有效 ）。

一种方法是设置全局参数。详细参考 :ref:`SQL设置运行参数 <sql_hints>` 。

也可以在这些立即执行的方法上，使用 ``hints`` 参数。这样，这些参数只会作用于当前的计算过程。


.. code:: python

    >>> iris[iris.sepallength < 5].to_pandas(hints={'odps.sql.mapper.split.size': 16})


运行时显示详细信息
~~~~~~~~~~~~~~~~~~

有时，用户需要查看运行时instance的logview时，需要修改全局配置：

.. code:: python

    >>> from odps import options
    >>> options.verbose = True
    >>>
    >>> iris[iris.sepallength < 5].exclude('sepallength')[:5].execute()
    Sql compiled:
    SELECT t1.`sepalwidth`, t1.`petallength`, t1.`petalwidth`, t1.`name`
    FROM odps_test_sqltask_finance.`pyodps_iris` t1
    WHERE t1.`sepallength` < 5
    LIMIT 5
    logview:
    http://logview

       sepalwidth  petallength  petalwidth         name
    0         3.0          1.4         0.2  Iris-setosa
    1         3.2          1.3         0.2  Iris-setosa
    2         3.1          1.5         0.2  Iris-setosa
    3         3.4          1.4         0.3  Iris-setosa
    4         2.9          1.4         0.2  Iris-setosa

用户可以指定自己的日志记录函数，比如像这样：

.. code:: python

    >>> my_logs = []
    >>> def my_logger(x):
    >>>     my_logs.append(x)
    >>>
    >>> options.verbose_log = my_logger
    >>>
    >>> iris[iris.sepallength < 5].exclude('sepallength')[:5].execute()
       sepalwidth  petallength  petalwidth         name
    0         3.0          1.4         0.2  Iris-setosa
    1         3.2          1.3         0.2  Iris-setosa
    2         3.1          1.5         0.2  Iris-setosa
    3         3.4          1.4         0.3  Iris-setosa
    4         2.9          1.4         0.2  Iris-setosa

    >>> print(my_logs)
    ['Sql compiled:', 'SELECT t1.`sepalwidth`, t1.`petallength`, t1.`petalwidth`, t1.`name` \nFROM odps_test_sqltask_finance.`pyodps_iris` t1 \nWHERE t1.`sepallength` < 5 \nLIMIT 5', 'logview:', u'http://logview']

缓存中间Collection计算结果
~~~~~~~~~~~~~~~~~~~~~~~~~~

DataFrame的计算过程中，一些Collection被多处使用，或者用户需要查看中间过程的执行结果，
这时用户可以使用 ``cache``\ 标记某个collection需要被优先计算。

.. note::

    值得注意的是，``cache``\ 延迟执行，调用cache不会触发立即计算。

.. code:: python

    >>> cached = iris[iris.sepalwidth < 3.5].cache()
    >>> df = cached['sepallength', 'name'].head(3)
    >>> df
       sepallength         name
    0          4.9  Iris-setosa
    1          4.7  Iris-setosa
    2          4.6  Iris-setosa
    >>> cached.head(3)  # 由于cached已经被计算，所以能立刻取到计算结果
       sepallength         name
    0          4.9  Iris-setosa
    1          4.7  Iris-setosa
    2          4.6  Iris-setosa

.. intinclude:: df-seahawks-int.rst


异步和并行执行
~~~~~~~~~~~~~~~

DataFrame 支持异步操作，对于立即执行的方法，包括 ``execute``、``persist``、``head``、``tail``、``to_pandas`` （其他方法不支持），
传入 ``async`` 参数，即可以将一个操作异步执行，``timeout`` 参数指定超时时间，
异步返回的是 `Future <https://docs.python.org/3/library/concurrent.futures.html#future-objects>`_ 对象。

.. code-block:: python

    >>> future = iris[iris.sepal_width < 10].head(10, async=True)
    >>> future.done()
    True
    >>> future.result()
       sepal_length  sepal_width  petal_length  petal_width     category
    0           5.1          3.5           1.4          0.2  Iris-setosa
    1           4.9          3.0           1.4          0.2  Iris-setosa
    2           4.7          3.2           1.3          0.2  Iris-setosa
    3           4.6          3.1           1.5          0.2  Iris-setosa
    4           5.0          3.6           1.4          0.2  Iris-setosa
    5           5.4          3.9           1.7          0.4  Iris-setosa
    6           4.6          3.4           1.4          0.3  Iris-setosa
    7           5.0          3.4           1.5          0.2  Iris-setosa
    8           4.4          2.9           1.4          0.2  Iris-setosa
    9           4.9          3.1           1.5          0.1  Iris-setosa


DataFrame 的并行执行可以使用多线程来并行，单个 expr 的执行可以通过 ``n_parallel`` 参数来指定并发度。
比如，当一个 DataFrame 的执行依赖的多个 cache 的 DataFrame 能够并行执行时，该参数就会生效。

.. code-block:: python

    >>> expr1 = iris.groupby('category').agg(value=iris.sepal_width.sum()).cache()
    >>> expr2 = iris.groupby('category').agg(value=iris.sepal_length.mean()).cache()
    >>> expr3 = iris.groupby('category').agg(value=iris.petal_length.min()).cache()
    >>> expr = expr1.union(expr2).union(expr3)
    >>> future = expr.execute(n_parallel=3, async=True, timeout=2)  # 并行和异步执行，2秒超时，返回Future对象
    >>> future.result()
              category    value
    0      Iris-setosa    5.006
    1  Iris-versicolor    5.936
    2   Iris-virginica    6.588
    3      Iris-setosa  170.900
    4  Iris-versicolor  138.500
    5   Iris-virginica  148.700
    6      Iris-setosa    1.000
    7  Iris-versicolor    3.000
    8   Iris-virginica    4.500


当同时执行多个 expr 时，我们可以用多线程执行，但会面临一个问题，
比如两个 DataFrame 有共同的依赖，这个依赖将会被执行两遍。

现在我们提供了新的 ``Delay API``，
来将立即执行的操作（包括 ``execute``、``persist``、``head``、``tail``、``to_pandas``，其他方法不支持）变成延迟操作，
并返回 `Future <https://docs.python.org/3/library/concurrent.futures.html#future-objects>`_ 对象。
当用户触发delay执行的时候，会去寻找共同依赖，按用户给定的并发度执行，并支持异步执行。

.. code-block:: python

    >>> from odps.df import Delay
    >>> delay = Delay()  # 创建Delay对象
    >>>
    >>> df = iris[iris.sepal_width < 5].cache()  # 有一个共同的依赖
    >>> future1 = df.sepal_width.sum().execute(delay=delay)  # 立即返回future对象，此时并没有执行
    >>> future2 = df.sepal_width.mean().execute(delay=delay)
    >>> future3 = df.sepal_length.max().execute(delay=delay)
    >>> delay.execute(n_parallel=3)  # 并发度是3，此时才真正执行。
    |==========================================|   1 /  1  (100.00%)        21s
    >>> future1.result()
    458.10000000000014
    >>> future2.result()
    3.0540000000000007


可以看到上面的例子里，共同依赖的对象会先执行，然后再以并发度为3分别执行future1到future3。
当 ``n_parallel`` 为1时，执行时间会达到37s。

``delay.execute`` 也接受 ``async`` 操作来指定是否异步执行，当异步的时候，也可以指定 ``timeout`` 参数来指定超时时间。
