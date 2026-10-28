.. _cl:

************
交互体验增强
************

命令行增强
==========

PyODPS 提供了命令行下的增强工具。首先，用户可以在任何地方配置了帐号以后，下次就无需再次输入帐号信息。

.. code:: python

    from odps.inter import setup, enter, teardown

接着就可以配置帐号

.. code:: python

    import os
    # 保证 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量设置为用户 Access Key ID，
    # ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量设置为用户 Access Key Secret
    # 不建议直接使用 Access Key ID / Access Key Secret 字符串
    setup(
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        '**your-project**',
        endpoint='**your-endpoint**',
    )

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

.. note::

    **注意**\ ：在重新 setup room 后，ODPS 入口对象并不会自动替换，需要再次调用 enter() 以获得新的 Room 对象。

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


此时全局会包含o和odps变量，即ODPS入口。


.. code:: python

    o.get_table('dual')
    odps.get_table('dual')




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


对象名补全
----------
PyODPS 拓展了 IPython 原有的代码补全功能，支持在书写 ``o.get_xxx`` 这样的语句时，自动补全对象名。

例如，在 IPython 中输入下列语句（<tab>不是实际输入的字符，而是当所有输入完成后，将光标移动到相应位置，
并按 Tab 键）：

.. code-block:: python

    o.get_table(<tab>

如果已知需要补全对象的前缀，也可以使用

.. code-block:: python

    o.get_table('tabl<tab>

IPython 会自动补全前缀为 tabl 的表。

对象名补全也支持补全不同 Project 下的对象名。下列用法都被支持：

.. code-block:: python

    o.get_table(project='project_name', name='tabl<tab>
    o.get_table('tabl<tab>', project='project_name')

如果匹配的对象有多个，IPython 会给出一个列表，其最大长度由 ``options.completion_size`` 给出，
默认为 10。


SQL命令
---------

.. _sqlinter:

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

.. _sqlparam:

如果想执行参数化SQL查询，则需要替换的参数可以使用\ ``:参数``\ 的方式。


.. code:: python

    In [1]: %load_ext odps

    In [2]: var = '0'

    In [3]: %sql select * from table_name where c_string_a = :var
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

Jupyter 增强
============

PyODPS 针对 Jupyter 下的探索性数据分析进行了增强，包括结果探索功能以及进度展示功能。

.. note::

    使用这些功能需要安装 ``ipywidgets`` 7.0 或以上版本，并正确安装和启用 PyODPS 提供的 Jupyter 前端扩展
    （安装包中同时提供了 JupyterLab 3.0+ 与经典 Jupyter 6.0+ 的扩展）。选项 ``display.notebook_widget`` 默认为
    ``None``，表示自动判断：除 DSW 与 VSCode 内置的 Jupyter 外，交互式组件默认开启。将其显式设置为 ``False``
    可关闭交互式组件，设置为 ``True`` 则强制开启：

    .. code:: python

        from odps import options

        options.display.notebook_widget = False

结果探索
--------
PyODPS 为 DataFrame 结果提供了交互式的数据探索功能。当执行结果为 DataFrame 时，PyODPS 会将其转换为 pandas
DataFrame，并以交互式组件的形式展示。组件顶端为模式选择区，除数据表外，还可以选择柱状图、饼图、折线图和散点图。

在数据表模式下，结果以分页表格的形式展示，每页至多显示 10 行，可通过底部的翻页按钮在数据中导航，如下图：

.. image:: _static/dfview-data-grid.png

在绘图模式下，单击右上角的配置按钮（齿轮图标）可以修改图表设置：

- 分组：按列的取值将数据拆分为多个系列，仅柱状图、折线图和散点图支持；
- 键：作为 X 轴类目（饼图中为扇区）的列，柱状图和饼图支持选择多个键，折线图和散点图仅支持一个键，且可选择
  ``(Row ID)`` 表示以行号作为 X 轴；
- 值：作为度量的列。柱状图和饼图会对值列做聚合，可在字段下方选择聚合函数（支持 count、nunique、min、max、
  sum、mean、median、var、std），其中 sum、mean、median、var、std 仅对数值列有意义，用于其他类型的列时会
  自动回退为 count；折线图和散点图直接使用所选列的值，不做聚合。

下图为使用默认字段选择（即前三个字段）绘制的散点图：

.. image:: _static/dfview-scatter.png

在配置面板中将 category 设置为分组列，X 轴选择为 petal_length，Y 轴选择为 petal_width，即可得到按鸢尾花品种
分组的 petal_length - petal_width 散点图；可见在 petal_length - petal_width 维度下，数据对 category 有较好的
区分度：

.. image:: _static/dfview-scatter-opts.png

对于柱状图和饼图，值字段的聚合函数可在配置面板中修改。下图为将饼图的键设置为 category，值字段设置为 petal_length
并将聚合函数选择为 sum 的结果：

.. image:: _static/dfview-pie-aggsel.png

修改配置后单击 Refresh 即可重新计算并绘制图表。请求尚未返回时，图表区域显示 Loading...；当前显示的数据来自旧配置
时，则显示 Updating...。

.. note::

    **注意**\ ：使用此功能需要安装 Pandas，并保证 ipywidgets 被正确安装。

进度展示
--------

大型作业执行通常需要较长的时间，因而 PyODPS 提供了进度展示功能。当 DataFrame 作业或通过 ``%sql`` 编写的 SQL
语句在 Jupyter 中执行作业时，会显示当前正在执行的作业组链接（例如 ``Executing: summary``）及总体进度，如下图：

.. image:: _static/progress_example.png

单击作业组链接即可打开对话框，查看该组中每个 Instance 的状态、各 Task 的 Stage 执行进度以及 Logview 链接。
对话框底部给出了 Stage 进度条三种颜色对应的状态，如下图：

.. image:: _static/task_details_dialog.png
