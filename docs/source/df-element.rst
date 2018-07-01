.. _dfelement:

列运算
=======

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))
    lens = DataFrame(o.get_table('pyodps_ml_100k_lens'))


对于一个Sequence来说，对它加上一个常量、或者执行sin函数的这类操作时，是作用于每个元素上的。接下来会详细说明。

NULL相关（isnull，notnull，fillna）
-----------------------------------

DataFrame
API提供了几个和NULL相关的内置函数，比如isnull来判断是否某字段是NULL，notnull则相反，fillna是将NULL填充为用户指定的值。

.. code:: python

    >>> iris.sepallength.isnull().head(5)
       sepallength
    0        False
    1        False
    2        False
    3        False
    4        False

逻辑判断（ifelse，switch）
--------------------------

``ifelse``\ 作用于boolean类型的字段，当条件成立时，返回第0个参数，否则返回第1个参数。

.. code:: python

    >>> (iris.sepallength > 5).ifelse('gt5', 'lte5').rename('cmp5').head(5)
       cmp5
    0   gt5
    1  lte5
    2  lte5
    3  lte5
    4  lte5


switch用于多条件判断的情况。

.. code:: python

    >>> iris.sepallength.switch(4.9, 'eq4.9', 5.0, 'eq5.0', default='noeq').rename('equalness').head(5)
       equalness
    0       noeq
    1      eq4.9
    2       noeq
    3       noeq
    4      eq5.0

.. code:: python

    >>> from odps.df import switch
    >>> switch(iris.sepallength == 4.9, 'eq4.9', iris.sepallength == 5.0, 'eq5.0', default='noeq').rename('equalness').head(5)
       equalness
    0       noeq
    1      eq4.9
    2       noeq
    3       noeq
    4      eq5.0

PyODPS 0.7.8 以上版本支持根据条件修改数据集某一列的一部分值，写法为：

.. code:: python

    >>> iris[iris.sepallength > 5, 'cmp5'] = 'gt5'
    >>> iris[iris.sepallength <= 5, 'cmp5'] = 'lte5'
    >>> iris.head(5)
       cmp5
    0   gt5
    1  lte5
    2  lte5
    3  lte5
    4  lte5

数学运算
--------

对于数字类型的字段，支持+，-，\*，/等操作，也支持log、sin等数学计算。

.. code:: python

    >>> (iris.sepallength * 10).log().head(5)
       sepallength
    0     3.931826
    1     3.891820
    2     3.850148
    3     3.828641
    4     3.912023

.. code:: python

    >>> fields = [iris.sepallength,
    >>>           (iris.sepallength / 2).rename('sepallength除以2'),
    >>>           (iris.sepallength ** 2).rename('sepallength的平方')]
    >>> iris[fields].head(5)
       sepallength  sepallength除以2  sepallength的平方
    0          5.1              2.55             26.01
    1          4.9              2.45             24.01
    2          4.7              2.35             22.09
    3          4.6              2.30             21.16
    4          5.0              2.50             25.00


算术运算支持的操作包括：

========== ===================================
 算术操作   说明
========== ===================================
 abs        绝对值
 sqrt       平方根
 sin
 sinh
 cos
 cosh
 tan
 tanh
 arccos
 arccosh
 arcsin
 arcsinh
 arctan
 arctanh
 exp        指数函数
 expm1      指数减1
 log        传入参数表示底是几
 log2
 log10
 log1p      log(1+x)
 radians    给定角度计算弧度
 degrees    给定弧度计算角度
 ceil       不小于输入值的最小整数
 floor      向下取整，返回比输入值小的整数值
 trunc      将输入值截取到指定小数点位置
========== ===================================

对于sequence，也支持其于其他sequence或者scalar的比较。

.. code:: python

    >>> (iris.sepallength < 5).head(5)
       sepallength
    0        False
    1         True
    2         True
    3         True
    4        False

值得主意的是，DataFrame
API不支持连续操作，比如\ ``3 <= iris.sepallength <= 5``\ ，但是提供了between这个函数来进行是否在某个区间的判断。

.. code:: python

    >>> (iris.sepallength.between(3, 5)).head(5)
       sepallength
    0        False
    1         True
    2         True
    3         True
    4         True

默认情况下，between包含两边的区间，如果计算开区间，则需要设inclusive=False。

.. code:: python

    >>> (iris.sepallength.between(3, 5, inclusive=False)).head(5)
       sepallength
    0        False
    1         True
    2         True
    3         True
    4        False

String 相关操作
--------------

DataFrame API提供了一系列针对string类型的Sequence或者Scalar的操作。

.. code:: python

    >>> fields = [
    >>>     iris.name.upper().rename('upper_name'),
    >>>     iris.name.extract('Iris(.*)', group=1)
    >>> ]
    >>> iris[fields].head(5)
        upper_name     name
    0  IRIS-SETOSA  -setosa
    1  IRIS-SETOSA  -setosa
    2  IRIS-SETOSA  -setosa
    3  IRIS-SETOSA  -setosa
    4  IRIS-SETOSA  -setosa

string相关操作包括：

============= ===========================================================================================================================================================================
 string 操作   说明
============= ===========================================================================================================================================================================
 capitalize
 contains      包含某个字符串，如果 regex 参数为 True，则是包含某个正则表达式，默认为 True
 count         指定字符串出现的次数
 endswith      以某个字符串结尾
 startswith    以某个字符串开头
 extract       抽取出某个正则表达式，如果 group 不指定，则返回满足整个 pattern 的子串；否则，返回第几个 group
 find          返回第一次出现的子串位置，若不存在则返回-1
 rfind         从右查找返回子串第一次出现的位置，不存在则返回-1
 replace       将某个 pattern 的子串全部替换成另一个子串， ``n`` 参数若指定，则替换n次
 get           返回某个位置上的字符串
 len           返回字符串的长度
 ljust         若未达到指定的 ``width`` 的长度，则在右侧填充 ``fillchar`` 指定的字符串（默认空格）
 rjust         若未达到指定的 ``width`` 的长度，则在左侧填充 ``fillchar`` 指定的字符串（默认空格）
 lower         变为全部小写
 upper         变为全部大写
 lstrip        在左侧删除空格（包括空行符）
 rstrip        在右侧删除空格（包括空行符）
 strip         在左右两侧删除空格（包括空行符）
 split         将字符串按分隔符拆分为若干个字符串（返回 list<string> 类型）
 pad           在指定的位置（left，right 或者 both）用指定填充字符（用 ``fillchar`` 指定，默认空格）来对齐
 repeat        重复指定 ``n`` 次
 slice         切片操作
 swapcase      对调大小写
 title         同 str.title
 zfill         长度没达到指定 ``width`` ，则左侧填充0
 isalnum       同 str.isalnum
 isalpha       同 str.isalpha
 isdigit       是否都是数字，同 str.isdigit
 isspace       是否都是空格，同 str.isspace
 islower       是否都是小写，同 str.islower
 isupper       是否都是大写，同 str.isupper
 istitle       同 str.istitle
 isnumeric     同 str.isnumeric
 isdecimal     同 str.isdecimal
 todict        将字符串按分隔符拆分为一个 dict，传入的两个参数分别为项目分隔符和 Key-Value 分隔符（返回 dict<string, string> 类型）
 strptime      按格式化读取成时间，时间格式和Python标准库相同，详细参考 `Python 时间格式化 <https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior>`_
============= ===========================================================================================================================================================================

时间相关操作
------------

对于datetime类型Sequence或者Scalar，可以调用时间相关的内置函数。

.. code:: python

    >>> df = lens[[lens.unix_timestamp.astype('datetime').rename('dt')]]
    >>> df[df.dt,
    >>>    df.dt.year.rename('year'),
    >>>    df.dt.month.rename('month'),
    >>>    df.dt.day.rename('day'),
    >>>    df.dt.hour.rename('hour')].head(5)
                        dt  year  month  day  hour
    0  1998-04-08 11:02:00  1998      4    8    11
    1  1998-04-08 10:57:55  1998      4    8    10
    2  1998-04-08 10:45:26  1998      4    8    10
    3  1998-04-08 10:25:52  1998      4    8    10
    4  1998-04-08 10:44:19  1998      4    8    10

与时间相关的属性包括：

============== ===========================================================================================================================================================
 时间相关属性   说明
============== ===========================================================================================================================================================
 year
 month
 day
 hour
 minute
 second
 weekofyear     返回日期位于那一年的第几周。周一作为一周的第一天
 weekday        返回日期当前周的第几天
 dayofweek      同 weekday
 strftime       格式化时间，时间格式和 Python 标准库相同，详细参考 `Python 时间格式化 <https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior>`_
============== ===========================================================================================================================================================

PyODPS 也支持时间的加减操作，比如可以通过以下方法得到前3天的日期。两个日期列相减得到相差的毫秒数。


.. code:: python

    >>> df
                               a                          b
    0 2016-12-06 16:43:12.460001 2016-12-06 17:43:12.460018
    1 2016-12-06 16:43:12.460012 2016-12-06 17:43:12.460021
    2 2016-12-06 16:43:12.460015 2016-12-06 17:43:12.460022
    >>> from odps.df import day
    >>> df.a - day(3)
                               a
    0 2016-12-03 16:43:12.460001
    1 2016-12-03 16:43:12.460012
    2 2016-12-03 16:43:12.460015
    >>> (df.b - df.a).dtype
    int64
    >>> (df.b - df.a).rename('a')
             a
    0  3600000
    1  3600000
    2  3600000


支持的时间类型包括：

============= =======
 属性          说明
============= =======
 year
 month
 day
 hour
 minute
 second
 millisecond
============= =======

.. _dfcollections:

集合类型相关操作
----------------
PyODPS 支持的集合类型有 List 和 Dict。这两个类型都可以使用下标获取集合中的某个项目，另有 len 方法，可获得集合的大小。

同时，两种集合均有 explode 方法，用于展开集合中的内容。对于 List，explode 默认返回一列，当传入参数 pos 时，
将返回两列，其中一列为值在数组中的编号（类似 Python 的 enumerate 函数）。对于 Dict，explode 会返回两列，
分别表示 keys 及 values。explode 中也可以传入列名，作为最后生成的列。

示例如下：

.. code:: python

    >>> df
       id         a                            b
    0   1  [a1, b1]  {'a2': 0, 'b2': 1, 'c2': 2}
    1   2      [c1]           {'d2': 3, 'e2': 4}
    >>> df[df.id, df.a[0], df.b['b2']]
       id   a    b
    0   1  a1    1
    1   2  c1  NaN
    >>> df[df.id, df.a.len(), df.b.len()]
       id  a  b
    0   1  2  3
    1   2  1  2
    >>> df.a.explode()
        a
    0  a1
    1  b1
    2  c1
    >>> df.a.explode(pos=True)
       a_pos   a
    0      0  a1
    1      1  b1
    2      0  c1
    >>> # 指定列名
    >>> df.a.explode(['pos', 'value'], pos=True)
       pos value
    0    0    a1
    1    1    b1
    2    0    c1
    >>> df.b.explode()
      b_key  b_value
    0    a2        0
    1    b2        1
    2    c2        2
    3    d2        3
    4    e2        4
    >>> # 指定列名
    >>> df.b.explode(['key', 'value'])
      key  value
    0  a2      0
    1  b2      1
    2  c2      2
    3  d2      3
    4  e2      4

explode 也可以和 :ref:`dflateralview` 结合，以将原有列和 explode 的结果相结合，例子如下：

.. code:: python

    >>> df[df.id, df.a.explode()]
       id   a
    0   1  a1
    1   1  b1
    2   2  c1
    >>> df[df.id, df.a.explode(), df.b.explode()]
       id   a b_key  b_value
    0   1  a1    a2        0
    1   1  a1    b2        1
    2   1  a1    c2        2
    3   1  b1    a2        0
    4   1  b1    b2        1
    5   1  b1    c2        2
    6   2  c1    d2        3
    7   2  c1    e2        4


除了下标、len 和 explode 两个共有方法以外，List 还支持下列方法：

============= ==================================
 list 操作     说明
============= ==================================
 contains(v)   列表是否包含某个元素
 sort          返回排序后的列表（返回值为 List）
============= ==================================

Dict 还支持下列方法：

============= ==================================
 dict 操作     说明
============= ==================================
 keys          获取 Dict keys（返回值为 List）
 values        获取 Dict values（返回值为 List）
============= ==================================


其他元素操作（isin，notin，cut）
-------------------------------

``isin``\ 给出Sequence里的元素是否在某个集合元素里。\ ``notin``\ 是相反动作。

.. code:: python

    >>> iris.sepallength.isin([4.9, 5.1]).rename('sepallength').head(5)
       sepallength
    0         True
    1         True
    2        False
    3        False
    4        False


cut提供离散化的操作，可以将Sequence的数据拆成几个区段。

.. code:: python

    >>> iris.sepallength.cut(range(6), labels=['0-1', '1-2', '2-3', '3-4', '4-5']).rename('sepallength_cut').head(5)
       sepallength_cut
    0             None
    1              4-5
    2              4-5
    3              4-5
    4              4-5

``include_under``\ 和\ ``include_over``\ 可以分别包括向下和向上的区间。

.. code:: python

    >>> labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-']
    >>> iris.sepallength.cut(range(6), labels=labels, include_over=True).rename('sepallength_cut').head(5)
       sepallength_cut
    0               5-
    1              4-5
    2              4-5
    3              4-5
    4              4-5

.. _map:

使用自定义函数
--------------

DataFrame函数支持对Sequence使用map，它会对它的每个元素调用自定义函数。比如：

.. code:: python

    >>> iris.sepallength.map(lambda x: x + 1).head(5)
       sepallength
    0          6.1
    1          5.9
    2          5.7
    3          5.6
    4          6.0

.. warning::
    目前，受限于 Python UDF，自定义函数无法支持将 list / dict 类型作为输入或输出。

如果map前后，Sequence的类型发生了变化，则需要显式指定map后的类型。

.. code:: python

    >>> iris.sepallength.map(lambda x: 't'+str(x), 'string').head(5)
       sepallength
    0         t5.1
    1         t4.9
    2         t4.7
    3         t4.6
    4         t5.0

如果在函数中包含闭包，需要注意的是，函数外闭包变量值的变化会引起函数内该变量值的变化。例如，

.. code:: python

    >>> dfs = []
    >>> for i in range(10):
    >>>     dfs.append(df.sepal_length.map(lambda x: x + i))

结果为 dfs 中每个 SequenceExpr 均为 ``df.sepal_length + 9``。为解决此问题，可以将函数作为另一函数的返回值，或者使用
partial，如

.. code:: python

    >>> dfs = []
    >>> def get_mapper(i):
    >>>     return lambda x: x + i
    >>> for i in range(10):
    >>>     dfs.append(df.sepal_length.map(get_mapper(i)))

或

.. code:: python

    >>> import functools
    >>> dfs = []
    >>> for i in range(10):
    >>>     dfs.append(df.sepal_length.map(functools.partial(lambda v, x: x + v, i)))

map也支持使用现有的UDF函数，传入的参数是str类型（函数名）或者 :ref:`Function对象 <functions>` 。

map传入Python函数的实现使用了ODPS Python UDF，因此，如果用户所在的Project不支持Python
UDF，则map函数无法使用。除此以外，所有 Python UDF 的限制在此都适用。

目前，第三方库（包含C）只能使用\ ``numpy``\ ，第三方库使用参考 :ref:`使用第三方Python库 <third_party_library>`。

除了调用自定义函数，DataFrame还提供了很多内置函数，这些函数中部分使用了map函数来实现，因此，如果\ **用户所在Project未开通Python
UDF，则这些函数也就无法使用（注：阿里云公共服务暂不提供Python UDF支持）**\ 。


.. warning::
    由于字节码定义的差异，Python 3 下使用新语言特性（例如 ``yield from`` ）时，代码在使用 Python 2.7 的 ODPS
    Worker 上执行时会发生错误。因而建议在 Python 3 下使用 MapReduce API 编写生产作业前，先确认相关代码是否能正常
    执行。

.. _function_resource:

引用资源
~~~~~~~~~

自定义函数也能读取ODPS上的资源（表资源或文件资源），或者引用一个collection作为资源。
此时，自定义函数需要写成函数闭包或callable的类。

.. code:: python

    >>> file_resource = o.create_resource('pyodps_iris_file', 'file', file_obj='Iris-setosa')
    >>>
    >>> iris_names_collection = iris.distinct('name')[:2]
    >>> iris_names_collection
           sepallength
    0      Iris-setosa
    1  Iris-versicolor

.. code:: python

    >>> def myfunc(resources):  # resources按调用顺序传入
    >>>     names = set()
    >>>     fileobj = resources[0] # 文件资源是一个file-like的object
    >>>     for l in fileobj:
    >>>         names.add(l)
    >>>     collection = resources[1]
    >>>     for r in collection:
    >>>         names.add(r.name)  # 这里可以通过字段名或者偏移来取
    >>>     def h(x):
    >>>         if x in names:
    >>>             return True
    >>>         else:
    >>>             return False
    >>>     return h
    >>>
    >>> df = iris.distinct('name')
    >>> df = df[df.name,
    >>>         df.name.map(myfunc, resources=[file_resource, iris_names_collection], rtype='boolean').rename('isin')]
    >>>
    >>> df
                  name   isin
    0      Iris-setosa   True
    1  Iris-versicolor   True
    2   Iris-virginica  False

注：分区表资源在读取时不包含分区字段。

.. _third_party_library:

使用第三方Python库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

现在用户可以把第三方 Python 包作为资源上传到 MaxCompute，支持的格式有 whl、egg、zip 以及 tar.gz。
在全局或者在立即执行的方法时，指定需要使用的包文件。即可以在自定义函数中使用第三方库。

值得注意的是，第三方库的依赖库，也必须指定，否则依然会有导入错误。

下面我们会以 python-dateutil 这个包作为例子。

首先，我们可以使用pip download命令，下载包以及其依赖到某个路径。
这里下载后会出现两个包：six-1.10.0-py2.py3-none-any.whl和python_dateutil-2.5.3-py2.py3-none-any.whl
（这里注意需要下载支持linux环境的包）

.. code-block:: shell

    $ pip download python-dateutil -d /to/path/

然后我们分别把两个文件上传到ODPS资源

.. code:: python

    >>> # 这里要确保资源名的后缀是正确的文件类型
    >>> odps.create_resource('six.whl', 'file', file_obj=open('six-1.10.0-py2.py3-none-any.whl', 'rb'))
    >>> odps.create_resource('python_dateutil.whl', 'file', file_obj=open('python_dateutil-2.5.3-py2.py3-none-any.whl', 'rb'))

现在我们有个DataFrame，只有一个string类型字段。

.. code:: python

    >>> df
                   datestr
    0  2016-08-26 14:03:29
    1  2015-08-26 14:03:29

全局配置使用到的三方库：

.. code:: python

    >>> from odps import options
    >>>
    >>> def get_year(t):
    >>>     from dateutil.parser import parse
    >>>     return parse(t).strftime('%Y')
    >>>
    >>> options.df.libraries = ['six.whl', 'python_dateutil.whl']
    >>> df.datestr.map(get_year)
       datestr
    0     2016
    1     2015

或者，通过立即运行方法的 ``libraries`` 参数指定：


.. code:: python

    >>> def get_year(t):
    >>>     from dateutil.parser import parse
    >>>     return parse(t).strftime('%Y')
    >>>
    >>> df.datestr.map(get_year).execute(libraries=['six.whl', 'python_dateutil.whl'])
       datestr
    0     2016
    1     2015

PyODPS 默认支持执行纯 Python 且不含文件操作的第三方库。在较新版本的 MaxCompute 服务下，PyODPS
也支持执行带有二进制代码或带有文件操作的 Python 库。这些库的后缀必须是 cp27-cp27m-manylinux1_x86_64，
以 archive 格式上传，whl 后缀的包需要重命名为 zip。同时，作业需要开启 ``odps.isolation.session.enable``
选项，或者在 Project 级别开启 Isolation。下面的例子展示了如何上传并使用 scipy 中的特殊函数：

.. code:: python

    >>> # 对于含有二进制代码的包，必须使用 Archive 方式上传资源，whl 后缀需要改为 zip
    >>> odps.create_resource('scipy.zip', 'archive', file_obj=open('scipy-0.19.0-cp27-cp27m-manylinux1_x86_64.whl', 'rb'))
    >>>
    >>> # 如果 Project 开启了 Isolation，下面的选项不是必需的
    >>> options.sql.settings = { 'odps.isolation.session.enable': True }
    >>>
    >>> def psi(value):
    >>>     # 建议在函数内部 import 第三方库，以防止不同操作系统下二进制包结构差异造成执行错误
    >>>     from scipy.special import psi
    >>>     return float(psi(value))
    >>>
    >>> df.float_col.map(psi).execute(libraries=['scipy.zip'])


对于只提供源码的二进制包，可以在 Linux Shell 中打包成 Wheel 再上传，Mac 和 Windows 中生成的 Wheel
包无法在 MaxCompute 中使用：

.. code-block:: shell

    python setup.py bdist_wheel


使用计数器
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from odps.udf import get_execution_context

    def h(x):
        ctx = get_execution_context()
        counters = ctx.get_counters()
        counters.get_counter('df', 'add_one').increment(1)
        return x + 1

    df.field.map(h)

logview 的 JSONSummary 中即可找到计数器值。


调用ODPS内建或者已定义函数
------------------------------------

要想调用 ODPS 上的内建或者已定义函数，来生成列，我们可以使用 ``func`` 接口，该接口默认函数返回值为 String，
可以用 rtype 参数指定返回值。

.. code:: python

    >>> from odps.df import func
    >>>
    >>> iris[iris.name, func.rand(rtype='float').rename('rand')][:4]
    >>> iris[iris.name, func.rand(10, rtype='float').rename('rand')][:4]
    >>> # 调用 ODPS 上定义的 UDF，列名无法确定时需要手动指定
    >>> iris[iris.name, func.your_udf(iris.sepalwidth, iris.sepallength, rtype='float').rename('new_col')]
    >>> # 从其他 Project 调用 UDF，也可通过 name 参数指定列名
    >>> iris[iris.name, func.your_udf(iris.sepalwidth, iris.sepallength, rtype='float', project='udf_project',
    >>>                               name='new_col')]

.. note::
    注意：在使用 Pandas 后端时，不支持执行带有 ``func`` 的表达式。
