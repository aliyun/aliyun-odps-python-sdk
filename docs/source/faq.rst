.. _faq:


常见问题
============

.. rubric:: 如何查看当前使用的 PyODPS 版本

.. code-block:: python

    import odps
    print(odps.__version__)

.. intinclude:: faq-int.rst
.. extinclude:: faq-ext.rst

.. rubric:: 怎么配置 SQL / DataFrame 的执行选项？
    :name: faq_options

ODPS SQL 的执行选项可在 `这里 <https://help.aliyun.com/apsara/enterprise/v_3_12_0_20200630/odps/enterprise-ascm-user-guide/common-maxcompute-sql-parameter-settings.html>`_ 找到。设置时，可将该选项设置到 ``options.sql.settings``，即

.. code-block:: python

    from odps import options
    # 将 <option_name> 和 <option_value> 替换为选项名和选项值
    options.sql.settings = {'<option_name>': '<option_value>'}

也可在每次调用执行时即席配置，该配置中的配置项会覆盖全局配置。

- 当使用 ``odps.execute_sql`` 时，可以使用

  .. code-block:: python

      from odps import options
      # 将 <option_name> 和 <option_value> 替换为选项名和选项值
      o.execute_sql('<sql_statement>', hints={'<option_name>': '<option_value>'})

- 当使用 ``dataframe.execute`` 或 ``dataframe.persist`` 时，可以使用

  .. code-block:: python

      from odps import options
      # 将 <option_name> 和 <option_value> 替换为选项名和选项值
      df.persist('<table_name>', hints={'<option_name>': '<option_value>'})

.. rubric:: 读取数据时报"project is protected"
    :name: faq_protected

Project 上的安全策略禁止读取表中的数据，此时，如果想使用全部数据，有以下选项可用：

- 联系 Project Owner 增加例外规则
- 使用 DataWorks 或其他脱敏工具先对数据进行脱敏，导出到非保护 Project，再进行读取

如果只想查看部分数据，有以下选项

- 改用 ``o.execute_sql('select * from <table_name>').open_reader()``
- 改用 :ref:`DataFrame <df>`，``o.get_table('<table_name>').to_df()``

.. rubric:: 出现 ImportError，并且在 ipython 或者 jupyter 下使用

如果 ``from odps import errors`` 也不行，则是缺少 ipython 组件，执行 ``pip install -U jupyter`` 解决。

.. rubric:: 执行 SQL 通过 open_reader 只能取到最多1万条记录，如何取多余1万条？

使用 ``create table as select ...`` 把SQL的结果保存成表，再使用 :ref:`table.open_reader <table_open_reader>` 来读取。

.. rubric:: 执行 SQL 很慢，如何排查？

PyODPS 提交 SQL 任务前，并没有进行重度操作。因此，绝大多数情形下，导致提交任务变慢的原因与 PyODPS 没有关系。
你可以考虑排查下面的原因。

1. 提交任务经过的代理服务器或者网络链路是否存在延迟；
2. 服务端是否存在任务排队延迟等情况；
3. 如果执行 SQL 的过程包括了拉取数据，是否数据规模过大或者数据分片过多导致拉取数据缓慢；
4. 如果是 DataWorks 作业，确认是否存在提交的任务（\ ``run_sql`` / ``execute_sql``）没有输出 Logview，尤其当 PyODPS \< 0.11.6 时。

如果你需要确认提交任务变慢是否由本地环境造成，可以尝试开启调试日志。PyODPS 将会把每个请求及返回都打印出来，可以根据\
请求和返回的日志确定延迟发生的位置。

.. code-block:: python

    import datetime
    import logging
    from odps import ODPS

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    o = ODPS(...)  #  此处填入账号，如果环境已提供 MaxCompute Entry 则忽略
    # 打印本地时间以确定本地操作发起的时间
    print("Check time:", datetime.datetime.now())
    # 提交任务
    inst = o.run_sql("select * from your_table")


此时，你的标准输出应当输出的内容应当形似

.. code-block:: text

    Check time: 2025-01-24 15:34:21.531330
    2025-01-24 15:34:21,532 - odps.rest - DEBUG - Start request.
    2025-01-24 15:34:21,532 - odps.rest - DEBUG - POST: http://service.<region>.maxcompute.aliyun.com/api/projects/<project>/instances
    2025-01-24 15:34:21,532 - odps.rest - DEBUG - data: b'<?xml version="1.0" encoding="utf-8"?>\n<Instance>\n  <Job>\n    <Priority>9</Priority>\n    <Tasks>\n      <SQL>\n        ....
    2025-01-24 15:34:21,532 - odps.rest - DEBUG - headers: {'Content-Type': 'application/xml'}
    2025-01-24 15:34:21,533 - odps.rest - DEBUG - request url + params /api/projects/<project>/instances?curr_project=<project>
    2025-01-24 15:34:21,533 - odps.accounts - DEBUG - headers before signing: {'Content-Type': 'application/xml', 'User-Agent': 'pyodps/0.12.2 CPython/3.7.12', 'Content-Length': '736'}
    2025-01-24 15:34:21,533 - odps.accounts - DEBUG - headers to sign: OrderedDict([('content-md5', ''), ('content-type', 'application/xml'), ('date', 'Fri, 24 Jan 2025 07:34:21 GMT')])
    2025-01-24 15:34:21,533 - odps.accounts - DEBUG - canonical string: POST

    application/xml
    Fri, 24 Jan 2025 07:34:21 GMT
    /projects/maxframe_ci_cd/instances?curr_project=maxframe_ci_cd
    2025-01-24 15:34:21,533 - odps.accounts - DEBUG - headers after signing: {'Content-Type': 'application/xml', 'User-Agent': 'pyodps/0.12.2 CPython/3.7.12', 'Content-Length': '736', ....
    2025-01-24 15:34:21,533 - urllib3.connectionpool - DEBUG - Resetting dropped connection: service.<region>.maxcompute.aliyun.com
    2025-01-24 15:34:22,027 - urllib3.connectionpool - DEBUG - http://service.<region>.maxcompute.aliyun.com:80 "POST /api/projects/<project>/instances?curr_project=<project> HTTP/1.1" 201 0
    2025-01-24 15:34:22,027 - odps.rest - DEBUG - response.status_code 201
    2025-01-24 15:34:22,027 - odps.rest - DEBUG - response.headers:
    {'Server': '<Server>', 'Date': 'Fri, 24 Jan 2025 07:34:22 GMT', 'Content-Type': 'text/plain;charset=utf-8', 'Content-Length': '0', 'Connection': 'close', 'Location': ....
    2025-01-24 15:34:22,027 - odps.rest - DEBUG - response.content: b''

从上面的输出中，可以知道代码启动任务的时间（2025-01-24 15:34:21,531）、请求发起时间（2025-01-24 15:34:21,533）以及\
服务端返回的时间（2025-01-24 15:34:22,027）获知各个阶段的时间开销。

如果你需要确认执行变慢是否是由拉取数据造成，可以先尝试将提交执行与拉取数据拆开，也就是说，使用 ``run_sql`` 提交任务，\
并使用 ``instance.wait_for_success`` 等待任务结束，再使用 ``instance.open_reader`` 读取数据。也就是说，\
将

.. code-block:: python

    with o.execute_sql('select * from your_table').open_reader() as reader:
        for row in reader:
            print(row)

改写为

.. code-block:: python

    inst = o.run_sql('select * from your_table')
    inst.wait_for_success()
    with inst.open_reader() as reader:
        for row in reader:
            print(row)

然后再确认各语句造成的延迟。

.. rubric:: 上传 pandas DataFrame 到 ODPS 时报错：ODPSError: ODPS entrance should be provided

原因是没有找到全局的ODPS入口，有三个方法：

- 使用 :ref:`room 机制 <cl>` ，``%enter`` 的时候，会配置全局入口
- 对odps入口调用 ``to_global`` 方法
- 使用odps参数，``DataFrame(pd_df).persist('your_table', odps=odps)``

.. rubric:: 在 DataFrame 中如何使用 max_pt ？

使用 ``odps.df.func`` 模块来调用 ODPS 内建函数

.. code-block:: python

    from odps.df import func
    df = o.get_table('your_table').to_df()
    df[df.ds == func.max_pt('your_project.your_table')]  # ds 是分区字段

.. rubric:: 通过 DataFrame 写表时报 table lifecycle is not specified in mandatory mode

Project 要求对每张表设置 lifecycle，因而需要在每次执行时设置

.. code-block:: python

    from odps import options
    options.lifecycle = 7  # 或者你期望的 lifecycle 整数值，单位为天

.. rubric:: 执行 SQL 时报 Please add put { "odps.sql.submit.mode" : "script"} for multi-statement query in settings

请参考 :ref:`SQL设置运行参数 <sql_hints>` 。

.. rubric:: 如何遍历 PyODPS DataFrame 中的每行数据
    :name: faq_enumerate_df

PyODPS DataFrame 不支持遍历每行数据。这样设计的原因是由于 PyODPS DataFrame 面向大规模数据设计，在这种场景下，
数据遍历是非常低效的做法。我们建议使用 DataFrame 提供的 ``apply`` 或 ``map_reduce`` 接口将原本串行的遍历操作并行化，
具体可参见 `这篇文章 <https://yq.aliyun.com/articles/138752>`_ 。如果确认你的场景必须要使用数据遍历，
而且遍历的代价可以接受，可以使用 ``to_pandas`` 方法将 DataFrame 转换为 Pandas DataFrame，或者将 DataFrame
存储为表后使用 ``read_table`` 或者 Tunnel 读取数据。

.. rubric:: 为何调用 to_pandas 后内存使用显著大于表的大小？
    :name: to_pandas_large

有两个原因可能导致这个现象发生。首先，MaxCompute 在存储数据时会对数据进行压缩，你看到的表大小应当是压缩后的大小。
其次，Python 中的值存在额外的存储开销。例如，对于字符串类型而言，每个 Python 字符串都会额外占用近 40 字节空间，
即便该字符串为空串，这可以通过调用 ``sys.getsizeof("")`` 发现。

需要注意的是，使用 Pandas 的 ``info`` 或者 ``memory_usage`` 方法获得的 Pandas DataFrame
内存使用可能是不准确的，因为这些方法默认不计算 string 或者其他 object 类型对象的实际内存占用。使用
``df.memory_usage(deep=True).sum()`` 获得的大小更接近实际内存使用，具体可参考
`这篇 Pandas 文档 <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.memory_usage.html>`_ 。

为减小读取数据时的内存开销，可以考虑使用 Arrow 格式，具体可以参考 :ref:`这里 <table_read>`。
