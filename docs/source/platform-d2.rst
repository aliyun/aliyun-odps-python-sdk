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


.. Note::
    o 变量在 PyODPS 节点执行前已经提前赋值。除非确定必须这么做，请不要手动设置该变量，这将导致 ODPS 入口被改写，
    并可能使节点在生产调度时由于默认账号发生变化从而导致权限错误。

执行SQL
==========

可以参考 :ref:`执行SQL文档 <execute_sql>` 。

.. note::
    Dataworks 上默认没有打开 instance tunnel，即 instance.open_reader 默认走 result 接口（最多一万条）。
    打开 instance tunnel，通过 reader.count 能取到记录数，如果要迭代获取全部数据，则需要关闭 limit 限制。

要想全局打开，则

.. code-block:: python

    options.tunnel.use_instance_tunnel = True
    options.tunnel.limit_instance_tunnel = False  # 关闭 limit 读取全部数据

    with instance.open_reader() as reader:
        # 能通过 instance tunnel 读取全部数据


或者通过在 open_reader 上添加 ``tunnel=True``，来仅对这次 open_reader 打开 instance tunnel；
添加 ``limit=False``，来关闭 limit 限制从而能下载全部数据。

.. code-block:: python

    with instance.open_reader(tunnel=True, limit=False) as reader:
        # 这次 open_reader 会走 instance tunnel 接口，且能读取全部数据


需要注意的是，部分 Project 可能对 Tunnel 下载全部数据设置了限制，因而打开这些选项可能会导致权限错误。
此时应当与 Project Owner 协调，或者考虑将数据处理放在 MaxCompute 中，而不是下载到本地。

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

与 DataWorks 中的 SQL 节点不同，为了避免侵入代码，PyODPS 节点 **不会** 在代码中替换 ${param_name}
这样的字符串，而是在执行代码前，在全局变量中增加一个名为 ``args`` 的 dict，调度参数可以在此获取。例如，
在节点基本属性 -> 参数中设置 ``ds=${yyyymmdd}`` ，则可以通过下面的方式在代码中获取此参数

.. code-block:: python

    print('ds=' + args['ds'])

上面的 print 语句将在 DataWorks 窗口中输出

.. code-block:: text

    ds=20161116

特别地，如果要获取名为 ``ds=${yyyymmdd}`` 的分区，则可以使用

.. code-block:: python

    o.get_table('table_name').get_partition('ds=' + args['ds'])

关于如何使用调度参数的详细例子可以参考 `DataWorks 文档 <https://help.aliyun.com/document_detail/417492.htm>`_ 。

.. note::
    args 变量在 PyODPS 节点执行前已经提前赋值，请不要手动设置该变量，这将导致调度参数被改写。

    SQL 节点中可用的 ${param_name} 写法不能在 PyODPS 节点中使用，
    即便在某些情况下它似乎输出了正确的结果。

.. _dw_3rdparty_lib:

使用三方包
==========
DataWorks 节点预装了下面的三方包，版本列表如下：

==================== ================== ==================
包名                  Python 2 节点版本    Python 3 节点版本
==================== ================== ==================
requests             2.11.1             2.26.0
numpy                1.16.6             1.18.1
pandas               0.24.2             1.0.5
scipy                0.19.0             1.3.0
scikit_learn         0.18.1             0.22.1
pyarrow              0.16.0             2.0.0
lz4                  2.1.4              3.1.10
zstandard            0.14.1             0.17.0
==================== ================== ==================

如果你需要使用上面列表中不存在的包，DataWorks 节点提供了 ``load_resource_package`` 方法，支持从
MaxCompute 资源下载三方包。使用 ``pyodps-pack`` 打包后，可以直接使用 ``load_resource_package``
方法加载三方包，此后就可以 import 包中的内容。关于 ``pyodps-pack`` 的文档可见 :ref:`制作和使用三方包 <pyodps_pack>`。

.. note::

    如果为 Python 2 节点打包，请在打包时为 ``pyodps-pack`` 增加 ``--dwpy27`` 参数。

例如，使用下面的命令打包

.. code-block:: bash

    pyodps-pack -o ipaddress-bundle.tar.gz ipaddress

并上传 / 提交 ``ipaddress-bundle.tar.gz`` 为资源后，可以在 PyODPS 3 节点中按照下面的方法使用
ipaddress 包：

.. code-block:: python

    load_resource_package("ipaddress-bundle.tar.gz")
    import ipaddress

DataWorks 限制下载的包总大小为 100MB。如果你需要跳过预装包的打包，可以在打包时使用 ``pyodps-pack`` 提供的
``--exclude`` 参数。例如，下面的打包方法排除了 DataWorks 环境中存在的 numpy 和 pandas 包。

.. code-block:: bash

    pyodps-pack -o bundle.tar.gz --exclude numpy --exclude pandas <your_package>

使用其他账号
============

.. note::

    ``as_account`` 方法从 PyODPS 0.11.3 开始支持。如果你的 DataWorks 未部署该版本，则无法使用该方法。
    如果你使用的是独享资源组，可以考虑升级资源组中的 PyODPS 版本，具体可见 `该文档 <https://help.aliyun.com/document_detail/144824.htm>`_ 。

在某些情形下你可能希望使用其他账号（而非平台提供的账号）访问 MaxCompute。此时，可以使用 ODPS 入口对象的 ``as_account``
方法创建一个使用新账号的入口对象，该对象与系统默认提供的 ``o`` 实例彼此独立。例如：

.. code-block:: python

    import os
    # 确保 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量设置为 Access Key ID，
    # ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量设置为 Access Key Secret，
    # 不建议直接使用 Access Key ID / Access Key Secret 字符串
    new_odps = o.as_account(
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    )

问题诊断
=========
如果你的代码在执行中卡死且没有任何输出，你可以在代码头部增加以下注释，DataWorks 每隔 30 秒将输出所有线程的堆栈：

.. code-block:: python

    # -*- dump_traceback: true -*-

受限功能
=========

由于缺少 matplotlib 等包，所以如下功能可能会受限。

- DataFrame的plot函数

DataFrame 自定义函数需要提交到 MaxCompute 执行。由于 Python 沙箱的原因，第三方库只支持所有的纯 Python 库以及 Numpy，
因此不能直接使用 Pandas，可参考 :ref:`第三方库支持 <third_party_library>` 上传和使用所需的库。DataWorks
中执行的非自定义函数代码可以使用平台预装的 Numpy 和 Pandas。其他带有二进制代码的三方包不被支持。

由于兼容性的原因，在 DataWorks 中，`options.tunnel.use_instance_tunnel` 默认设置为 False。如果需要全局开启 Instance Tunnel，
需要手动将该值设置为 True。

由于实现的原因，Python 的 atexit 包不被支持，请使用 try - finally 结构实现相关功能。

使用限制
===========

在 DataWorks 上使用 PyODPS，为了防止对 DataWorks 的 gateway 造成压力，对内存和 CPU 都有限制。这个限制由 DataWorks 统一管理。

如果看到 **Got killed** ，即内存使用超限，进程被 kill。因此，尽量避免本地的数据操作。

通过 PyODPS 起的 SQL 和 DataFrame 任务（除 to_pandas) 不受此限制。