.. _mars:

*****************
Mars 使用指南
*****************

准备工作
============
在具备使用Mars能力之前，用户需要在本地安装Mars，分别通过以下命令安装：

.. code:: shell

    pip install pymars==0.4.0rc1 # install mars
    pip install pyarrow==0.11.1


Mars 集群控制
===============

PyODPS 提供了Mars的使用接口，用户拉起集群后，就可以使用Mars服务。

创建集群
----------

在拿到ODPS入口后，调用接口：

.. code:: python

    >>> client = o.create_mars_cluster(5, 4, 16, min_worker_num=3)


在调用时可以指定运行的worker数量以及配置，上述调用表明需要创建worker数量为5的集群，每个worker需要4核、16G的配置，
\ ``min_worker_num``\ 是指最少运行worker数量，当集群worker数量达到该值便返回，默认情况会等所有worker资源都申请得到时才会返回。

.. note::

    申请的单个worker资源最好大于1G，CPU核数与内存大小（单位为G）的比例推荐为1：4，例如4核、16G的单个worker资源配置。

如果已经申请创建过Mars集群，可以通过传入创建集群任务的instance id获取client。

.. code:: python

    >>> client = o.create_mars_cluster(instance_id=**instance-id**)


获取 LogView 地址
------------------

通过调用 ``get_logview_address`` 方法即可。

.. code:: python

    >>> print(client.get_logview_address())

也可以通过设置``options.verbose``为 True，可以自动打印Logview等信息。


释放集群
----------

当我们不再使用Mars服务时，需要手动调用 ``stop_server`` 方法，及时释放资源。

.. code:: python

    >>> client.stop_server()


提交作业
============

当创建Mars集群时，会自动创建session作为提交作业的默认session，用户可以通过成员变量 ``client.session`` 获取，
并提交Mars作业

.. code:: python

    >>> import mars.tensor as mt
    >>> session = client.session
    >>> t = mt.random.rand(100, 100)
    >>> session.run(t)

作业运行完成后，客户端会从Web拉取结果，如果不需要拉取，可以设置参数 ``wait=False``，
运行过程中可以通过Web UI查看作业运行情况以及每个worker的运行状态，Web的地址获取方式：

.. code:: python

    >>> print(client.endpoint)


当Mars集群创建完成，获取到Web地址后，也可以手动创建session，使用该session提交作业

.. code:: python

    >>> from mars.session import new_session
    >>> mars_url = client.endpoint
    >>> session = new_session(mars_url)


ODPS 数据读写
==============

目前，Mars支持通过直接读写ODPS数据。


读表
----------

用户可以从 ODPS 表创建 Mars DataFrame 并进行后续计算。

.. code:: python

    >>> df = o.to_mars_dataframe('test_mars')
    >>> df.iloc[:6].execute()
           col1  col2
    0        0    0
    1        0    1
    2        0    2
    3        1    0
    4        1    1
    5        1    2


写表
----------

通过Mars计算后的DataFrame结果也可以写入ODPS表。

.. code:: python

    >>> df = o.to_mars_dataframe('test_mars')
    >>> df2 = df + 1
    >>> o.persist_mars_dataframe(df2, 'test_mars_persist')
    >>> o.get_table('test_mars_persist').to_df().head(6)
           col1  col2
    0        1    1
    1        1    2
    2        1    3
    3        2    1
    4        2    2
    5        2    3
