.. _mars-quickstart:

**************
Mars 快速开始
**************


准备 ODPS 入口
----------------

ODPS 入口是 MaxCompute 所有操作的基础，:ref:`首先获取 ODPS 对象 <odps_entry>`。


提交 Mars 作业
----------------

Mars 目前有两种模式提交作业，一种是 Job 模式，编写一个函数提交执行。另一种是 cluster 模式，拉起 Mars 集群，通过 Mars session 提交作业。


Job 模式
~~~~~~~~~~~~~~~

Job 模式中我们只需要写一个处理数据的函数并提交即可，简单易操作，数据可以是 ODPS 表。并且在这种模式下，读写表性能会明显优于客户端使用 tunnel 读写表。
对于 DataWorks 用户来说，这种方式不会受限于 DataWorks 节点的资源，可以指定作业运行所需要的 CPU 以及内存，函数提交后会在 Mars 服务端执行。

首先需要编写函数：

.. code:: python

    def func(s_name, d_name):
        df = o.to_mars_dataframe(s_name).to_pandas()
        o.persist_mars_dataframe(df, d_name)


与 PyODPS 代码相比，在函数内只需要修改读写表的接口，就可以轻松地改写 PyODPS 代码作为 Mars job 提交，并且在读写表上获得比较大的性能提升。


当我们完成函数编写之后，只需要通过 run_mars_job 接口提交这个函数就可以完成执行。

.. code:: python

    o.run_mars_job(func, args=('source_table_name', 'des_table_name'),
                   worker_cpu=4, worker_mem=8)


如果任务数据很大，内存或者 CPU 要求比较高，可以通过参数 worker_cpu, worker_mem 配置(单位分别是核数、Gb)。

.. code:: python

    o.run_mars_job(func, args=('source_table_name', 'des_table_name'), worker_cpu=8, worker_mem=32)

传入以上参数时，服务端会使用8核32G的资源完成计算。

更加详细的用法可以参考 :ref:`Job 模式 <job_mode>`。


使用 Mars 集群
~~~~~~~~~~~~~~~

除了 job 模式外，用户也可以先拉起 Mars 集群，通过创建的默认 session 提交任务到 Mars 集群。在交互式环境下用户可以多次提交 Mars 任务，进行数据探索等，不需要使用时停止 Mars 集群即可。


首先拉起 Mars 集群，只需要运行如下代码。

.. code:: python

    >>> from odps import options
    >>> options.verbose = True  # 在 dataworks pyodps3 里已经设置，所以不需要前两行代码
    >>> client = o.create_mars_cluster(2, 4, 16)

这个例子里指定了 worker 数量为 2 的集群，每个 worker 是4核、16G 内存的配置。

首先可以通过 ``to_mars_dataframe`` 接口提交一个读表任务。

.. code:: python

    >>> df = o.to_mars_dataframe('test_mars')
    >>> df.head(6).execute()
           col1  col2
    0        0    0
    1        0    1
    2        0    2
    3        1    0
    4        1    1
    5        1    2

如果需要将读的数据上传到其他表里，可以通过 ``o.persist_mars_dataframe(df, 'table_name')`` 将 Mars DataFrame 保存成 MaxCompute 表。

.. code:: ipython

    >>> df2 = df + 1
    >>> o.persist_mars_dataframe(df2, 'test_mars_persist')  # 保存 Mars DataFrame
    >>> o.get_table('test_mars_persist').to_df().head(6)  # 通过 PyODPS DataFrame 查看数据
           col1  col2
    0        1    1
    1        1    2
    2        1    3
    3        2    1
    4        2    2
    5        2    3


当你不再需要运行其他作业时，可以通过调用 ``client.stop_server()`` 手动释放 Mars 集群：

.. code:: python

    client.stop_server()


更多的介绍可以参考 :ref:`创建 Mars 集群相关内容 <cluster_mode>`。
