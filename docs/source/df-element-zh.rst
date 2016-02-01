.. _dfelement:

.. code:: python

    from odps.df import DataFrame

.. code:: python

    iris = DataFrame(o.get_table('pyodps_iris'))
    lens = DataFrame(o.get_table('pyodps_ml_100k_lens'))

元素级别操作
============

对于一个Sequence来说，对它加上一个常量、或者执行sin函数的这类操作时，是作用于每个元素上的。接下来会详细说明。

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



map的实现使用了ODPS Python UDF，因此，如果用户所在的Project不支持Python
UDF，则map函数无法使用。除此以外，所有Python
UDF的限制在此都适用。因此目前，第三方库只能使用\ ``numpy``\ 。

除了调用自定义函数，DataFrame还提供了很多内置函数，这些函数中部分使用了map函数来实现，因此，如果\ **用户所在Project未开通Python
UDF，则这些函数也就无法使用**\ 。

NULL相关
========

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



逻辑判断
========

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
        <td>将某个pattern的子串全部替换成另一个子串，``n``参数若指定，则替换n次</td>
      </tr>
      <tr>
        <td>get</td>
        <td>返回某个位置上的字符串</td>
      </tr>
      <tr>
        <td>ljust</td>
        <td>若未达到指定的``width``的长度，则在右侧填充``fillchar``指定的字符串（默认空格）</td>
      </tr>
      <tr>
        <td>rjust</td>
        <td>若未达到指定的``width``的长度，则在左侧填充``fillchar``指定的字符串（默认空格）</td>
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
        <td>在指定的位置（left，right或者both）用指定填充字符（用``fillchar``指定，默认空格）来对齐</td>
      </tr>
      <tr>
        <td>repeat</td>
        <td>重复指定``n``次</td>
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
        <td>长度没达到指定``width``，则左侧填充0</td>
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
    </table>
    </div>

其他操作
========

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


