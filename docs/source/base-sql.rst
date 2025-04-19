.. _sql:

SQL
=====

PyODPS支持ODPS SQL的查询，并可以读取执行的结果。 :meth:`~odps.ODPS.execute_sql` /
:meth:`~odps.ODPS.execute_sql_interactive` / :meth:`~odps.ODPS.run_sql` /
:meth:`~odps.ODPS.run_sql_interactive` 方法的返回值是 :ref:`运行实例 <instances>` 。

.. note::

    并非所有在 ODPS Console 中可以执行的命令都是 ODPS 可以接受的 SQL 语句。
    在调用非 DDL / DML 语句时，请使用其他方法，例如 GRANT / REVOKE 等语句请使用
    :meth:`~odps.ODPS.run_security_query` 方法，PAI 命令请使用
    :meth:`~odps.ODPS.run_xflow` 或 :meth:`~odps.ODPS.execute_xflow` 方法。

.. _execute_sql:

执行 SQL
--------

你可以使用 :meth:`~odps.ODPS.execute_sql` 方法以同步方式执行 SQL。调用时，该方法会阻塞直至 SQL 执行完成，并返回一个
:class:`~odps.models.Instance` 实例。如果 SQL 执行报错，该方法会抛出以 ``odps.errors.ODPSError`` 为基类的错误。

.. code-block:: python

   >>> o.execute_sql('select * from dual')  # 同步的方式执行，会阻塞直到SQL执行完成

你也可以使用非阻塞方式异步执行 SQL。调用时，该方法在将 SQL 提交到 MaxCompute 后即返回 Instance
实例。你需要使用 :meth:`~odps.models.Instance.wait_for_success` 方法等待该 SQL 执行完成。\
同样地，如果 instance 出现错误，:meth:`~odps.models.Instance.wait_for_success` 会抛出以
``odps.errors.ODPSError`` 为基类的错误。

.. code-block:: python

   >>> instance = o.run_sql('select * from dual')  # 异步的方式执行
   >>> print(instance.get_logview_address())  # 获取logview地址
   >>> instance.wait_for_success()  # 阻塞直到完成

关于如何操作 run_sql / execute_sql 返回的 Instance 实例，可以参考 :ref:`运行实例 <instances>` 。

使用 MCQA 执行 SQL
-------------------
`MCQA <https://help.aliyun.com/document_detail/180701.html>`_ 是 MaxCompute 提供的查询加速功能，
支持使用独立资源池对中小规模数据进行加速。PyODPS 从 0.11.4.1 开始支持以下列方式通过 MCQA 执行 SQL
，同时需要 MaxCompute 具备 MCQA 的支持。

你可以使用 :meth:`~odps.ODPS.execute_sql_interactive` 通过 MCQA 执行 SQL 并返回 MCQA Instance。如果
MCQA 无法执行相应的 SQL ，会自动回退到传统模式。此时，函数返回的 Instance 为回退后的 Instance。

.. code-block:: python

    >>> o.execute_sql_interactive('select * from dual')

如果不希望回退，可以指定参数 ``fallback=False``。也可以指定为回退策略（或回退策略的组合，使用逗号分隔的字符串）。
可用的策略名如下。默认策略为 ``all`` （即 ``generic,unsupported,upgrading,noresource,timeout`` ）。

* ``generic`` ：指定时，表示发生未知错误时回退到传统模式。
* ``noresource`` ：指定时，表示发生资源不足问题时回退到传统模式。
* ``upgrading`` ：指定时，表示升级期间回退到传统模式。
* ``timeout`` ：指定时，表示执行超时时回退到传统模式。
* ``unsupported`` ：指定时，表示遇到 MCQA 不支持的场景时回退到传统模式。

例如，下面的代码要求在 MCQA 不支持和资源不足时回退：

.. code-block:: python

    >>> o.execute_sql_interactive('select * from dual', fallback="noresource,unsupported")

你也可以使用 :meth:`~odps.ODPS.run_sql_interactive` 通过 MCQA 异步执行 SQL。类似 :meth:`~odps.ODPS.run_sql`，\
该方法会在提交任务后即返回 MCQA Instance，你需要自行等待 Instance 完成。需要注意的是，该方法不会自动回退。当执行失败时，\
你需要自行重试或执行 :meth:`~odps.ODPS.execute_sql`。

.. code-block:: python

   >>> instance = o.run_sql_interactive('select * from dual')  # 异步的方式执行
   >>> print(instance.get_logview_address())  # 获取logview地址
   >>> instance.wait_for_success()  # 阻塞直到完成

.. _sql_hints:

设置时区
---------
有时我们希望对于查询出来的时间数据显示为特定时区下的时间，可以通过 ``options.local_timezone`` 设置客户端的时区。

``options.local_timezone`` 可设置为以下三种类型：

* ``False``：使用 UTC 时间。
* ``True``：使用本地时区（默认设置）。
* 时区字符串：使用指定的时区，例如 ``Asia/Shanghai``。

例如，使用 UTC 时间：

.. code-block:: python

  >>> from odps import options
  >>> options.local_timezone = False

使用本地时区：

.. code-block:: python

  >>> from odps import options
  >>> options.local_timezone = True

使用 ``Asia/Shanghai``：

.. code-block:: python

  >>> from odps import options
  >>> options.local_timezone = "Asia/Shanghai"

.. note::

  设置 ``options.local_timezone`` 后，PyODPS 会根据它的值自动设置 ``odps.sql.timezone``。
  两者的值不同可能导致服务端和客户端时间不一致，因此不应再手动设置 ``odps.sql.timezone``。

设置运行参数
------------

有时，我们在运行时，需要设置运行时参数，我们可以通过设置 ``hints`` 参数，参数类型是 dict。该参数对
:meth:`~odps.ODPS.execute_sql` / :meth:`~odps.ODPS.execute_sql_interactive` /
:meth:`~odps.ODPS.run_sql` / :meth:`~odps.ODPS.run_sql_interactive` 均有效。

.. code-block:: python

   >>> hints = {'odps.stage.mapper.split.size': 16, 'odps.sql.reducer.instances': 1024}
   >>> o.execute_sql('select * from pyodps_iris', hints=hints)

我们可以对于全局配置设置sql.settings后，每次运行时则都会添加相关的运行时参数。

.. code-block:: python

   >>> from odps import options
   >>> options.sql.settings = {
   >>>     'odps.stage.mapper.split.size': 16,
   >>>     'odps.sql.reducer.instances': 1024,
   >>> }
   >>> o.execute_sql('select * from pyodps_iris')  # 会根据全局配置添加hints

.. _read_sql_exec_result:

读取 SQL 执行结果
---------------

运行 SQL 的 instance 能够直接执行 :meth:`~odps.models.Instance.open_reader` 的操作，一种情况是SQL返回了结构化的数据。

.. code-block:: python

   >>> with o.execute_sql('select * from dual').open_reader() as reader:
   >>>     for record in reader:
   >>>         # 处理每一个record

另一种情况是 SQL 可能执行的比如 ``desc``，这时通过 ``reader.raw`` 属性取到原始的SQL执行结果。

.. code-block:: python

   >>> with o.execute_sql('desc dual').open_reader() as reader:
   >>>     print(reader.raw)

如果 `options.tunnel.use_instance_tunnel == True`，在调用 open_reader 时，PyODPS 会默认调用 Instance Tunnel，
否则会调用旧的 Result 接口。如果你使用了版本较低的 MaxCompute 服务，或者调用 Instance Tunnel 出现了问题，PyODPS
会给出警告并自动降级到旧的 Result 接口，可根据警告信息判断导致降级的原因。如果 Instance Tunnel 的结果不合预期，
请将该选项设为 `False`。在调用 open_reader 时，也可以使用 ``tunnel`` 参数来指定使用何种结果接口，例如

.. code-block:: python

   >>> # 使用 Instance Tunnel
   >>> with o.execute_sql('select * from dual').open_reader(tunnel=True) as reader:
   >>>     for record in reader:
   >>>         # 处理每一个record
   >>> # 使用 Results 接口
   >>> with o.execute_sql('select * from dual').open_reader(tunnel=False) as reader:
   >>>     for record in reader:
   >>>         # 处理每一个record

PyODPS 默认不限制能够从 Instance 读取的数据规模，但 Project Owner 可能在 MaxCompute Project 上增加保护设置以限制对
Instance 结果的读取，此时只能使用受限读取模式读取数据，在此模式下可读取的行数受到 Project 配置限制，通常为 10000 行。如果
PyODPS 检测到读取 Instance 数据被限制，且 ``options.tunnel.limit_instance_tunnel`` 未设置，会自动启用受限读取模式。
如果你的 Project 被保护，想要手动启用受限读取模式，可以为 ``open_reader`` 方法增加 ``limit=True`` 选项，或者设置
``options.tunnel.limit_instance_tunnel = True``\ 。

在部分环境中，例如 DataWorks，``options.tunnel.limit_instance_tunnel`` 可能默认被置为 True。此时，如果需要读取\
所有数据，需要为 ``open_reader`` 增加参数 `tunnel=True, limit=False` 。需要注意的是，如果 Project 本身被保护，\
这两个参数\ **不能**\ 解除保护，MaxCompute 也\ **不提供**\ 绕开该权限限制读取更多数据的方法。此时应联系 Project Owner
开放相应的读权限。

如果你所使用的 MaxCompute 只能支持旧 Result 接口，同时你需要读取所有数据，可将 SQL 结果写入另一张表后用读表接口读取
（可能受到 Project 安全设置的限制）。

同时，PyODPS 支持直接将运行结果数据读成 pandas DataFrame。

.. code-block:: python

  >>> # 直接使用 reader 的 to_pandas 方法
  >>> with o.execute_sql('select * from dual').open_reader(tunnel=True) as reader:
  >>>     # pd_df 类型为 pandas DataFrame
  >>>     pd_df = reader.to_pandas()

.. _sql_to_pandas_mp:

如果需要使用多核加速读取速度，可以通过 ``n_process`` 指定使用进程数:

.. code-block:: python

  >>> import multiprocessing
  >>> n_process = multiprocessing.cpu_count()
  >>> with o.execute_sql('select * from dual').open_reader(tunnel=True) as reader:
  >>>     # n_process 指定成机器核数
  >>>     pd_df = reader.to_pandas(n_process=n_process)

.. note::

    从 2024 年年末开始，MaxCompute 服务将支持离线 SQL 任务 ``open_reader`` 使用与表类似的 Arrow
    接口，MCQA 作业暂不支持。在此之前，使用 ``Instance.open_reader(arrow=True)`` 读取数据将报错。

从 PyODPS 0.12.0 开始，你也可以直接调用 Instance 上的 :meth:`~odps.models.Instance.to_pandas`
方法直接将数据转换为 pandas。你可以指定转换为 pandas 的起始行号和行数，若不指定则读取所有数据。该方法也支持
``limit`` 参数，具体定义与 ``open_reader`` 方法相同。该方法默认会使用 Arrow 格式读取，并转换为
pandas。如果 Arrow 格式不被支持，将会回退到 Record 接口。

.. code-block:: python

  >>> inst = o.execute_sql('select * from dual')
  >>> pd_df = inst.to_pandas(start=10, count=20)

与表类似，从 PyODPS 0.12.0 开始，你也可以使用 Instance 上的 :meth:`~odps.models.Instance.iter_pandas`
方法按多个批次读取 pandas DataFrame，参数与 ``Table.iter_pandas`` 类似。

.. code-block:: python

  >>> inst = o.execute_sql('select * from dual')
  >>> for batch in inst.iter_pandas(start=0, count=1000, batch_size=100):
  >>>     print(batch)

设置 alias
------------

有时在运行时，比如某个UDF引用的资源是动态变化的，我们可以alias旧的资源名到新的资源，这样免去了重新删除并重新创建UDF的麻烦。

.. code-block:: python

    from odps.models import TableSchema

    myfunc = '''\
    from odps.udf import annotate
    from odps.distcache import get_cache_file

    @annotate('bigint->bigint')
    class Example(object):
        def __init__(self):
            self.n = int(get_cache_file('test_alias_res1').read())

        def evaluate(self, arg):
            return arg + self.n
    '''
    res1 = o.create_resource('test_alias_res1', 'file', file_obj='1')
    o.create_resource('test_alias.py', 'py', file_obj=myfunc)
    o.create_function('test_alias_func',
                      class_type='test_alias.Example',
                      resources=['test_alias.py', 'test_alias_res1'])

    table = o.create_table(
        'test_table',
        TableSchema.from_lists(['size'], ['bigint']),
        if_not_exists=True
    )

    data = [[1, ], ]
    # 写入一行数据，只包含一个值1
    o.write_table(table, 0, [table.new_record(it) for it in data])

    with o.execute_sql(
        'select test_alias_func(size) from test_table').open_reader() as reader:
        print(reader[0][0])

.. code-block:: python

    2

.. code-block:: python

    res2 = o.create_resource('test_alias_res2', 'file', file_obj='2')
    # 把内容为1的资源alias成内容为2的资源，我们不需要修改UDF或资源
    with o.execute_sql(
        'select test_alias_func(size) from test_table',
        aliases={'test_alias_res1': 'test_alias_res2'}).open_reader() as reader:
        print(reader[0][0])

.. code-block:: python

    3


在交互式环境执行 SQL
---------------------

在 ipython 和 jupyter 里支持 :ref:`使用 SQL 插件的方式运行 SQL <sqlinter>`，且支持 :ref:`参数化查询 <sqlparam>`，
详情参阅 :ref:`文档 <sqlinter>`。



设置 biz_id
------------

在少数情形下，可能在提交 SQL 时，需要同时提交 biz_id，否则执行会报错。此时，你可以设置全局 options 里的 biz_id。

.. code-block:: python

   from odps import options

   options.biz_id = 'my_biz_id'
   o.execute_sql('select * from pyodps_iris')
