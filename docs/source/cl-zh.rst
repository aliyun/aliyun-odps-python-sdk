.. _cl:

********************
命令行和 IPython 增强
********************

命令行增强
==========

PyODPS 提供了命令行下的增强工具。首先，用户可以在任何地方配置了帐号以后，下次就无需再次输入帐号信息。

.. code:: python

    from odps.inter import setup, enter, teardown

接着就可以配置帐号

.. code:: python

    setup('**your-access_id**', '**your-access-key**', '**your-project**', endpoint='**your-endpoint**)

在不指定\ ``room``\ 这个参数时，会被配置到叫做\ ``default``\ 的room里。

以后，在任何命令行打开的地方，都可以直接调用：

.. code:: python

    room = enter()

我们可以拿到ODPS的入口：

.. code:: python

    o = room.odps

.. code:: python

    o.get_table('dual')




.. parsed-literal::

    odps.Table
      name: odps_test_sqltask_finance.`dual`
      schema:
        c_int_a                 : bigint          
        c_int_b                 : bigint          
        c_double_a              : double          
        c_double_b              : double          
        c_string_a              : string          
        c_string_b              : string          
        c_bool_a                : boolean         
        c_bool_b                : boolean         
        c_datetime_a            : datetime        
        c_datetime_b            : datetime        



我们可以把常用的ODPS表或者资源都可以存放在room里。

.. code:: python

    room.store('存储表示例', o.get_table('dual'), desc='简单的表存储示例')

我们可以调用\ ``display``\ 方法，来把已经存储的对象以表格的形式打印出来：

.. code:: python

    room.display()




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>default</th>
          <th>desc</th>
        </tr>
        <tr>
          <th>name</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>存储表示例</th>
          <td>简单的表存储示例</td>
        </tr>
        <tr>
          <th>iris</th>
          <td>安德森鸢尾花卉数据集</td>
        </tr>
      </tbody>
    </table>
    </div>



我们通过\ ``room['存储表示例']``\ ，或者像\ ``room.iris``\ ，就可以取出来存储的对象了。

.. code:: python

    room['存储表示例']




.. parsed-literal::

    odps.Table
      name: odps_test_sqltask_finance.`dual`
      schema:
        c_int_a                 : bigint          
        c_int_b                 : bigint          
        c_double_a              : double          
        c_double_b              : double          
        c_string_a              : string          
        c_string_b              : string          
        c_bool_a                : boolean         
        c_bool_b                : boolean         
        c_datetime_a            : datetime        
        c_datetime_b            : datetime        



删除也很容易，只需要调用drop方法

.. code:: python

    room.drop('存储表示例')

.. code:: python

    room.display()




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>default</th>
          <th>desc</th>
        </tr>
        <tr>
          <th>name</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>iris</th>
          <td>安德森鸢尾花卉数据集</td>
        </tr>
      </tbody>
    </table>
    </div>



要删除某个room，只需要调用teardown就可以了，不传参数时删除默认room。

::

    teardown()

IPython增强
===========

PyODPS 还提供了 IPython 的插件，来更方便得操作 ODPS。

首先，针对命令行增强，也有相应的命令。让我们先加载插件：

.. code:: python

    %load_ext odps




.. code:: python

    %enter




.. parsed-literal::

    <odps.inter.Room at 0x11341df10>



.. code:: python

    room = _

这样我们就取到了我们的默认帐号所在的room。

.. code:: python

    o = room.odps

.. code:: python

    o.get_table('dual')




.. parsed-literal::

    odps.Table
      name: odps_test_sqltask_finance.`dual`
      schema:
        c_int_a                 : bigint          
        c_int_b                 : bigint          
        c_double_a              : double          
        c_double_b              : double          
        c_string_a              : string          
        c_string_b              : string          
        c_bool_a                : boolean         
        c_bool_b                : boolean         
        c_datetime_a            : datetime        
        c_datetime_b            : datetime        



.. code:: python

    %stores




.. raw:: html

    <div style='padding-bottom: 30px'>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>default</th>
          <th>desc</th>
        </tr>
        <tr>
          <th>name</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>iris</th>
          <td>安德森鸢尾花卉数据集</td>
        </tr>
      </tbody>
    </table>
    </div>


SQL命令
---------


PyODPS 还提供了 SQL 插件，来执行 ODPS SQL。下面是单行 SQL：

.. code:: python

    %sql select * from pyodps_iris limit 5




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
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>3.6</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



多行SQL可以使用\ ``%%sql``\ 的命令

.. code:: python

    %%sql
    
    select * from pyodps_iris 
    where sepallength < 5 
    limit 5




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


如果想执行参数化SQL查询，则需要替换的参数可以使用\ ``:参数``\ 的方式。


.. code:: python

    In [1]: %load_ext odps

    In [2]: mytable = 'dual'

    In [3]: %sql select * from :mytable
    |==========================================|   1 /  1  (100.00%)         2s
    Out[3]:
       c_int_a  c_int_b  c_double_a  c_double_b  c_string_a  c_string_b c_bool_a  \
    0        0        0       -1203           0           0       -1203     True

      c_bool_b         c_datetime_a         c_datetime_b
    0    False  2012-03-30 23:59:58  2012-03-30 23:59:59

设置SQL运行时参数，可以通过 ``%set`` 设置到全局，或者在sql的cell里用SET进行局部设置。

.. code:: python

    In [17]: %%sql
             set odps.sql.mapper.split.size = 16;
             select * from pyodps_iris;

这个会局部设置，不会影响全局的配置。

.. code:: python

   In [18]: %set odps.sql.mapper.split.size = 16

这样设置后，后续运行的SQL都会使用这个设置。


持久化 pandas DataFrame 到 ODPS 表
----------------------------------


PyODPS 还提供把 pandas DataFrame 上传到 ODPS 表的命令:

.. code:: python

    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc'))

.. code:: python

    %persist df pyodps_pandas_df

这里的第0个参数\ ``df``\ 是前面的变量名，\ ``pyodps_pandas_df``\ 是ODPS表名。
