.. _dfelement:

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))
    lens = DataFrame(o.get_table('pyodps_ml_100k_lens'))


对于一个Sequence来说，对它加上一个常量、或者执行sin函数的这类操作时，是作用于每个元素上的。接下来会详细说明。

.. _map:

使用自定义函数
==============

DataFrame函数支持对Sequence使用map，它会对它的每个元素调用自定义函数。比如：

.. code:: python

    iris.sepallength.map(lambda x: x + 1).head(5)




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
          <td>6.1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>5.9</td>
        </tr>
        <tr>
          <th>2</th>
          <td>5.7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.6</td>
        </tr>
        <tr>
          <th>4</th>
          <td>6.0</td>
        </tr>
      </tbody>
    </table>
    </div>



如果map前后，Sequence的类型发生了变化，则需要显式指定map后的类型。

.. code:: python

    iris.sepallength.map(lambda x: 't'+str(x), 'string').head(5)




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
          <td>t5.1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>t4.9</td>
        </tr>
        <tr>
          <th>2</th>
          <td>t4.7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>t4.6</td>
        </tr>
        <tr>
          <th>4</th>
          <td>t5.0</td>
        </tr>
      </tbody>
    </table>
    </div>



map也支持使用现有的UDF函数，传入的参数是str类型（函数名）或者 :ref:`Function对象 <functions>` 。

map传入Python函数的实现使用了ODPS Python UDF，因此，如果用户所在的Project不支持Python
UDF，则map函数无法使用。除此以外，所有Python
UDF的限制在此都适用。

目前，第三方库（包含C）只能使用\ ``numpy``\ ，纯Python库使用参考 :ref:`使用第三方纯Python库 <third_party_library>`。

除了调用自定义函数，DataFrame还提供了很多内置函数，这些函数中部分使用了map函数来实现，因此，如果\ **用户所在Project未开通Python
UDF，则这些函数也就无法使用（注：阿里云公共服务暂不提供Python UDF支持）**\ 。

.. _function_resource:

引用资源
~~~~~~~~~~~~~

自定义函数也能读取ODPS上的资源（表资源或文件资源），或者引用一个collection作为资源。
此时，自定义函数需要写成函数闭包或callable的类。

.. code:: python

    file_resource = o.create_resource('pyodps_iris_file', 'file', file_obj='Iris-setosa')

.. code:: python

    iris_names_collection = iris.distinct('name')[:2]
    iris_names_collection

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
          <td>Iris-setosa</th>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-versicolor</th>
        </tr>
      </tbody>
    </table>
    </div>

.. code:: python

    def myfunc(resources):  # resources按调用顺序传入
        names = set()

        fileobj = resources[0] # 文件资源是一个file-like的object
        for l in fileobj:
            names.add(l)

        collection = resources[1]
        for r in collection:
            names.add(r.name)  # 这里可以通过字段名或者偏移来取

        def h(x):
            if x in names:
                return True
            else:
                return False

        return h

    df = iris.distinct('name')
    df = df[df.name,
            df.name.map(myfunc, resources=[file_resource, iris_names_collection], rtype='boolean').rename('isin')]

    df


.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>isin</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Iris-setosa</th>
          <td>True</th>
        </tr>
        <tr>
          <th>1</th>
          <td>Iris-versicolor</th>
          <td>True</th>
        </tr>
        <tr>
          <th>2</th>
          <td>Iris-virginica</th>
          <td>False</th>
        </tr>
      </tbody>
    </table>
    </div>


.. _third_party_library:

使用第三方纯Python库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


现在用户可以把第三方Python包作为资源上传到ODPS，支持的格式有whl、egg、zip以及tar.gz。
在全局或者在立即执行的方法时，指定需要使用的包文件。即可以在自定义函数中使用第三方库。

值得注意的是，第三方库的依赖库，也必须指定，否则依然会有导入错误。

下面我们会以 python-dateutil 这个包作为例子。

首先，我们可以使用pip download命令，下载包以及其依赖到某个路径。
这里下载后会出现两个包：six-1.10.0-py2.py3-none-any.whl和python_dateutil-2.5.3-py2.py3-none-any.whl
（这里注意需要下载支持linux环境的包）

.. code-block:: shell

    pip download python-dateutil -d /to/path/



然后我们分别把两个文件上传到ODPS资源

.. code:: python

    # 这里要确保资源名的后缀是正确的文件类型
    odps.create_resource('six.whl', 'file', file_obj=open('six-1.10.0-py2.py3-none-any.whl'))
    odps.create_resource('python_dateutil.whl', 'file', file_obj=open('python_dateutil-2.5.3-py2.py3-none-any.whl'))


现在我们有个DataFrame，只有一个string类型字段。




.. code:: python

    df



.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>datestr</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2016-08-26 14:03:29</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2015-08-26 14:03:29</td>
        </tr>
      </tbody>
    </table>
    </div>



全局配置使用到的三方库：


.. code:: python

    from odps import options

    def get_year(t):
        from dateutil.parser import parse
        return parse(t).strftime('%Y')

    options.df.libraries = ['six.whl', 'python_dateutil.whl']
    df.datestr.map(get_year)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>datestr</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2016</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2015</td>
        </tr>
      </tbody>
    </table>
    </div>



或者，通过立即运行方法的 ``libraries`` 参数指定：


.. code:: python

    def get_year(t):
        from dateutil.parser import parse
        return parse(t).strftime('%Y')

    df.datestr.map(get_year).execute(libraries=['six.whl', 'python_dateutil.whl'])




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>datestr</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2016</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2015</td>
        </tr>
      </tbody>
    </table>
    </div>




NULL相关（isnull，notnull，fillna）
=======================================

DataFrame
API提供了几个和NULL相关的内置函数，比如isnull来判断是否某字段是NULL，notnull则相反，fillna是将NULL填充为用户指定的值。

.. code:: python

    iris.sepallength.isnull().head(5)




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
          <td>False</td>
        </tr>
        <tr>
          <th>1</th>
          <td>False</td>
        </tr>
        <tr>
          <th>2</th>
          <td>False</td>
        </tr>
        <tr>
          <th>3</th>
          <td>False</td>
        </tr>
        <tr>
          <th>4</th>
          <td>False</td>
        </tr>
      </tbody>
    </table>
    </div>



逻辑判断（ifelse，switch）
==============================

``ifelse``\ 作用于boolean类型的字段，当条件成立时，返回第0个参数，否则返回第1个参数。

.. code:: python

    (iris.sepallength > 5).ifelse('gt5', 'lte5').rename('cmp5').head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>cmp5</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>gt5</td>
        </tr>
        <tr>
          <th>1</th>
          <td>lte5</td>
        </tr>
        <tr>
          <th>2</th>
          <td>lte5</td>
        </tr>
        <tr>
          <th>3</th>
          <td>lte5</td>
        </tr>
        <tr>
          <th>4</th>
          <td>lte5</td>
        </tr>
      </tbody>
    </table>
    </div>



switch用于多条件判断的情况。

.. code:: python

    iris.sepallength.switch(4.9, 'eq4.9', 5.0, 'eq5.0', default='noeq').rename('equalness').head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>equalness</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>noeq</td>
        </tr>
        <tr>
          <th>1</th>
          <td>eq4.9</td>
        </tr>
        <tr>
          <th>2</th>
          <td>noeq</td>
        </tr>
        <tr>
          <th>3</th>
          <td>noeq</td>
        </tr>
        <tr>
          <th>4</th>
          <td>eq5.0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    from odps.df import switch
    
    switch(iris.sepallength == 4.9, 'eq4.9', iris.sepallength == 5.0, 'eq5.0', default='noeq').rename('equalness').head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>equalness</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>noeq</td>
        </tr>
        <tr>
          <th>1</th>
          <td>eq4.9</td>
        </tr>
        <tr>
          <th>2</th>
          <td>noeq</td>
        </tr>
        <tr>
          <th>3</th>
          <td>noeq</td>
        </tr>
        <tr>
          <th>4</th>
          <td>eq5.0</td>
        </tr>
      </tbody>
    </table>
    </div>



数学运算
========

对于数字类型的字段，支持+，-，\*，/等操作，也支持log、sin等数学计算。

.. code:: python

    (iris.sepallength * 10).log().head(5)




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
          <td>3.931826</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.891820</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.850148</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3.828641</td>
        </tr>
        <tr>
          <th>4</th>
          <td>3.912023</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    fields = [iris.sepallength,
              (iris.sepallength / 2).rename('sepallength除以2'), 
              (iris.sepallength ** 2).rename('sepallength的平方')]
    iris[fields].head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength</th>
          <th>sepallength除以2</th>
          <th>sepallength的平方</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.1</td>
          <td>2.55</td>
          <td>26.01</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>2.45</td>
          <td>24.01</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>2.35</td>
          <td>22.09</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>2.30</td>
          <td>21.16</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>2.50</td>
          <td>25.00</td>
        </tr>
      </tbody>
    </table>
    </div>



算术运算支持的操作包括：

.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <tr>
        <th>算术操作</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>abs</td>
        <td>绝对值</td>
      </tr>
      <tr>
        <td>sqrt</td>
        <td>平方根</td>
      </tr>
      <tr>
        <td>sin</td>
        <td></td>
      </tr>
      <tr>
        <td>sinh</td>
        <td></td>
      </tr>
      <tr>
        <td>cos</td>
        <td></td>
      </tr>
      <tr>
        <td>cosh</td>
        <td></td>
      </tr>
      <tr>
        <td>tan</td>
        <td></td>
      </tr>
      <tr>
        <td>tanh</td>
        <td></td>
      </tr>
      <tr>
        <td>arccos</td>
        <td></td>
      </tr>
      <tr>
        <td>arccosh</td>
        <td></td>
      </tr>
      <tr>
        <td>arcsin</td>
        <td></td>
      </tr>
      <tr>
        <td>arcsinh</td>
        <td></td>
      </tr>
      <tr>
        <td>arctan</td>
        <td></td>
      </tr>
      <tr>
        <td>arctanh</td>
        <td></td>
      </tr>
      <tr>
        <td>exp</td>
        <td>指数函数</td>
      </tr>
      <tr>
        <td>expm1</td>
        <td>指数减1</td>
      </tr>
      <tr>
        <td>log</td>
        <td>传入参数表示底是几</td>
      </tr>
      <tr>
        <td>log2</td>
        <td></td>
      </tr>
      <tr>
        <td>log10</td>
        <td></td>
      </tr>
      <tr>
        <td>log1p</td>
        <td>log(1+x)</td>
      </tr>
      <tr>
        <td>radians</td>
        <td>给定角度计算弧度</td>
      </tr>
      <tr>
        <td>degrees</td>
        <td>给定弧度计算角度</td>
      </tr>
      <tr>
        <td>ceil</td>
        <td>不小于输入值的最小整数</td>
      </tr>
      <tr>
        <td>floor</td>
        <td>向下取整，返回比输入值小的整数值。</td>
      </tr>
      <tr>
        <td>trunc</td>
        <td>将输入值截取到指定小数点位置</td>
      </tr>
    </table>
    </div>

对于sequence，也支持其于其他sequence或者scalar的比较。

.. code:: python

    (iris.sepallength < 5).head(5)




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
          <td>False</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
        </tr>
        <tr>
          <th>2</th>
          <td>True</td>
        </tr>
        <tr>
          <th>3</th>
          <td>True</td>
        </tr>
        <tr>
          <th>4</th>
          <td>False</td>
        </tr>
      </tbody>
    </table>
    </div>



值得主意的是，DataFrame
API不支持连续操作，比如\ ``3 <= iris.sepallength <= 5``\ ，但是提供了between这个函数来进行是否在某个区间的判断。

.. code:: python

    (iris.sepallength.between(3, 5)).head(5)




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
          <td>False</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
        </tr>
        <tr>
          <th>2</th>
          <td>True</td>
        </tr>
        <tr>
          <th>3</th>
          <td>True</td>
        </tr>
        <tr>
          <th>4</th>
          <td>True</td>
        </tr>
      </tbody>
    </table>
    </div>



默认情况下，between包含两边的区间，如果计算开区间，则需要设inclusive=False。

.. code:: python

    (iris.sepallength.between(3, 5, inclusive=False)).head(5)




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
          <td>False</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
        </tr>
        <tr>
          <th>2</th>
          <td>True</td>
        </tr>
        <tr>
          <th>3</th>
          <td>True</td>
        </tr>
        <tr>
          <th>4</th>
          <td>False</td>
        </tr>
      </tbody>
    </table>
    </div>



String相关操作
==============

DataFrame API提供了一系列针对string类型的Sequence或者Scalar的操作。

.. code:: python

    fields = [
        iris.name.upper().rename('upper_name'),
        iris.name.extract('Iris(.*)', group=1)
    ]
    iris[fields].head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>upper_name</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>IRIS-SETOSA</td>
          <td>-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>IRIS-SETOSA</td>
          <td>-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>IRIS-SETOSA</td>
          <td>-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>IRIS-SETOSA</td>
          <td>-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>IRIS-SETOSA</td>
          <td>-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



string相关操作包括：

.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <tr>
        <th>string操作</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>capitalize</td>
        <td></td>
      </tr>
      <tr>
        <td>contains</td>
        <td>包含某个字符串，如果regex参数为True，则是包含某个正则表达式</td>
      </tr>
      <tr>
        <td>count</td>
        <td>指定字符串出现的次数</td>
      </tr>
      <tr>
        <td>endswith</td>
        <td>以某个字符串结尾</td>
      </tr>
      <tr>
        <td>startswith</td>
        <td>以某个字符串开头</td>
      </tr>
      <tr>
        <td>extract</td>
        <td>抽取出某个正则表达式，如果group不指定，则返回满足整个pattern的子串；否则，返回第几个group</td>
      </tr>
      <tr>
        <td>find</td>
        <td>返回第一次出现的子串位置，若不存在则返回-1</td>
      </tr>
      <tr>
        <td>rfind</td>
        <td>从右查找返回子串第一次出现的位置，不存在则返回-1</td>
      </tr>
      <tr>
        <td>replace</td>
        <td>将某个pattern的子串全部替换成另一个子串，<code class="docutils literal">n</code>参数若指定，则替换n次</td>
      </tr>
      <tr>
        <td>get</td>
        <td>返回某个位置上的字符串</td>
      </tr>
      <tr>
        <td>ljust</td>
        <td>若未达到指定的<code class="docutils literal">width</code>的长度，则在右侧填充<code class="docutils literal">fillchar</code>指定的字符串（默认空格）</td>
      </tr>
      <tr>
        <td>rjust</td>
        <td>若未达到指定的<code class="docutils literal">width</code>的长度，则在左侧填充<code class="docutils literal">fillchar</code>指定的字符串（默认空格）</td>
      </tr>
      <tr>
        <td>lower</td>
        <td>变为全部小写</td>
      </tr>
      <tr>
        <td>upper</td>
        <td>变为全部大写</td>
      </tr>
      <tr>
        <td>lstrip</td>
        <td>在左侧删除空格（包括空行符）</td>
      </tr>
      <tr>
        <td>rstrip</td>
        <td>在右侧删除空格（包括空行符）</td>
      </tr>
      <tr>
        <td>strip</td>
        <td>在左右两侧删除空格（包括空行符）</td>
      </tr>
      <tr>
        <td>pad</td>
        <td>在指定的位置（left，right或者both）用指定填充字符（用<code class="docutils literal">fillchar</code>指定，默认空格）来对齐</td>
      </tr>
      <tr>
        <td>repeat</td>
        <td>重复指定<code class="docutils literal">n</code>次</td>
      </tr>
      <tr>
        <td>slice</td>
        <td>切片操作</td>
      </tr>
      <tr>
        <td>swapcase</td>
        <td>对调大小写</td>
      </tr>
      <tr>
        <td>title</td>
        <td>同str.title</td>
      </tr>
      <tr>
        <td>zfill</td>
        <td>长度没达到指定<code class="docutils literal">width</code>，则左侧填充0</td>
      </tr>
      <tr>
        <td>isalnum</td>
        <td>同str.isalnum</td>
      </tr>
      <tr>
        <td>isalpha</td>
        <td>同str.isalpha</td>
      </tr>
      <tr>
        <td>isdigit</td>
        <td>是否都是数字，同str.isdigit</td>
      </tr>
      <tr>
        <td>isspace</td>
        <td>是否都是空格，同str.isspace</td>
      </tr>
      <tr>
        <td>islower</td>
        <td>是否都是小写，同str.islower</td>
      </tr>
      <tr>
        <td>isupper</td>
        <td>是否都是大写，同str.isupper</td>
      </tr>
      <tr>
        <td>istitle</td>
        <td>同str.istitle</td>
      </tr>
      <tr>
        <td>isnumeric</td>
        <td>同str.isnumeric</td>
      </tr>
      <tr>
        <td>isdecimal</td>
        <td>同str.isdecimal</td>
      </tr>
      <tr>
        <td>strptime</td>
        <td>按格式化读取成时间，时间格式和Python标准库相同，详细参考<a href='https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior'>Python时间格式化</a></td>
      </tr>
    </table>
    </div>

时间相关操作
============

对于datetime类型Sequence或者Scalar，可以调用时间相关的内置函数。

.. code:: python

    df = lens[[lens.unix_timestamp.astype('datetime').rename('dt')]]
    df[df.dt, 
       df.dt.year.rename('year'), 
       df.dt.month.rename('month'), 
       df.dt.day.rename('day'), 
       df.dt.hour.rename('hour')].head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>dt</th>
          <th>year</th>
          <th>month</th>
          <th>day</th>
          <th>hour</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1998-04-08 11:02:00</td>
          <td>1998</td>
          <td>4</td>
          <td>8</td>
          <td>11</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1998-04-08 10:57:55</td>
          <td>1998</td>
          <td>4</td>
          <td>8</td>
          <td>10</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1998-04-08 10:45:26</td>
          <td>1998</td>
          <td>4</td>
          <td>8</td>
          <td>10</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1998-04-08 10:25:52</td>
          <td>1998</td>
          <td>4</td>
          <td>8</td>
          <td>10</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1998-04-08 10:44:19</td>
          <td>1998</td>
          <td>4</td>
          <td>8</td>
          <td>10</td>
        </tr>
      </tbody>
    </table>
    </div>



与时间相关的属性包括：

.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <tr>
        <th>时间相关属性</th>
        <th>说明</th>
      </tr>
      <tr>
        <td>year</td>
        <td></td>
      </tr>
      <tr>
        <td>month</td>
        <td></td>
      </tr>
      <tr>
        <td>month</td>
        <td></td>
      </tr>
      <tr>
        <td>month</td>
        <td></td>
      </tr>
      <tr>
        <td>day</td>
        <td></td>
      </tr>
      <tr>
        <td>hour</td>
        <td></td>
      </tr>
      <tr>
        <td>minute</td>
        <td></td>
      </tr>
      <tr>
        <td>second</td>
        <td></td>
      </tr>
      <tr>
        <td>weekofyear</td>
        <td>返回日期位于那一年的第几周。周一作为一周的第一天。</td>
      </tr>
      <tr>
        <td>weekday</td>
        <td>返回日期当前周的第几天。</td>
      </tr>
      <tr>
        <td>dayofweek</td>
        <td>同weekday</td>
      </tr>
      <tr>
        <td>strftime</td>
        <td>格式化时间，时间格式和Python标准库相同，详细参考<a href='https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior'>Python时间格式化</a></td>
      </tr>
    </table>
    </div>

其他元素操作（isin，notin，cut）
======================================

``isin``\ 给出Sequence里的元素是否在某个集合元素里。\ ``notin``\ 是相反动作。

.. code:: python

    iris.sepallength.isin([4.9, 5.1]).rename('sepallength').head(5)




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
          <td>True</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
        </tr>
        <tr>
          <th>2</th>
          <td>False</td>
        </tr>
        <tr>
          <th>3</th>
          <td>False</td>
        </tr>
        <tr>
          <th>4</th>
          <td>False</td>
        </tr>
      </tbody>
    </table>
    </div>



cut提供离散化的操作，可以将Sequence的数据拆成几个区段。

.. code:: python

    iris.sepallength.cut(range(6), labels=['0-1', '1-2', '2-3', '3-4', '4-5']).rename('sepallength_cut').head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength_cut</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>None</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4-5</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4-5</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4-5</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4-5</td>
        </tr>
      </tbody>
    </table>
    </div>



``include_under``\ 和\ ``include_over``\ 可以分别包括向下和向上的区间。

.. code:: python

    labels=['0-1', '1-2', '2-3', '3-4', '4-5', '5-']
    iris.sepallength.cut(range(6), labels=labels, include_over=True).rename('sepallength_cut').head(5)




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength_cut</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5-</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4-5</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4-5</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4-5</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4-5</td>
        </tr>
      </tbody>
    </table>
    </div>


要想调用ODPS上的无参或者常数参的内建函数，我们可以使用 ``BuiltinFunction`` 类来完成。


.. code:: python

    from odps.df import BuiltinFunction

    iris[iris.name, BuiltinFunction('rand', rtype='float').rename('rand')][:4]
    iris[iris.name, BuiltinFunction('rand', rtype='float', args=(10, )).rename('rand')][:4]
