.. _d2:

=======================
DataWorks 用户使用指南
=======================


新建工作流节点
===============

在工作流节点中会包含PYODPS节点。新建即可。


.. image:: _static/d2-node-zh.png


ODPS入口
===========


DataWorks 的 PyODPS 节点中，将会包含一个全局的变量 ``odps`` 或者 ``o`` ，即 ODPS 入口。用户不需要手动定义 ODPS 入口。

.. code-block:: python

    print(o.exist_table('pyodps_iris'))


执行SQL
==========

可以参考 :ref:`执行SQL文档 <execute_sql>` 。


DataFrame
============

执行
--------

在 DataWorks 的环境里， :ref:`DataFrame <df>` 的执行需要显式调用 :ref:`立即执行的方法（如execute，head等） <df_delay_execute>` 。

.. code-block:: python

    from odps.df import DataFrame

    iris = DataFrame(o.get_table('pyodps_iris'))
    for record in iris[iris.sepal_width < 3].execute():  # 调用立即执行的方法
        # 处理每条record


如果用户想在print的时候调用立即执行，需要打开 ``options.interactive`` 。

.. code-block:: python

    from odps import options
    from odps.df import DataFrame

    options.interactive = True  # 在开始打开开关

    iris = DataFrame(o.get_table('pyodps_iris'))
    print(iris.sepal_width.sum())  # 这里print的时候会立即执行


打印详细信息
----------------

通过设置 ``options.verbose`` 选项。在 DataWorks 上，默认已经处于打开状态，运行过程会打印 logview 等详细过程。


获取调度参数
==============

在全局包括一个 ``args`` 对象，可以在这个中获取，它是一个dict类型。

比如在节点基本属性 -> 参数中设置 ``ds=${yyyymmdd}`` ，则可以：

.. code-block:: python

    args['ds']


.. code-block:: python

    '20161116'


受限功能
=========

DataWorks 上现在已经包含 numpy 和 pandas，而由于缺少 matplotlib 等包，所以如下功能可能会受限。


- DataFrame的plot函数


DataFrame自定义函数由于 Python 沙箱的原因，
第三方库支持所有的纯 Python 库（参考 :ref:`第三方纯 Python 库支持 <third_party_library>` ），
以及numpy，因此不能直接使用 pandas。

由于兼容性的原因，在 DataWorks 中，`options.tunnel.use_instance_tunnel` 默认设置为 False。如果需要全局开启 Instance Tunnel，
需要手动将该值设置为 True。

由于实现的原因，Python 的 atexit 包不被支持，请使用 try - finally 结构实现相关功能。

使用限制
===========

在 DataWorks 上使用 PyODPS，为了防止对 DataWorks 的 gateway 造成压力，对内存和 CPU 都有限制。这个限制由 DataWorks 统一管理。

如果看到 **Got killed** ，即内存使用超限，进程被 kill。因此，尽量避免本地的数据操作。

通过 PyODPS 起的 SQL 和 DataFrame 任务（除 to_pandas) 不受此限制。