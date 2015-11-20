.. _sql:

****
SQL
****

PyOdps支持ODPS SQL的查询，并可以读取执行的结果。

执行SQL
=======

.. code-block:: python

   >>> odps.execute_sql('select * from dual')  #  同步的方式执行，会阻塞直到SQL执行完成
   >>> instance = odps.run_sql('select * from dual')  # 异步的方式执行
   >>> instance.wait_for_success()  # 阻塞直到完成

读取SQL执行结果
===============

运行SQL的instance能够直接执行 ``open_reader`` 的操作，一种情况是SQL返回了结构化的数据。

.. code-block:: python

   >>> with odps.execute_sql('select * from dual').open_reader() as reader:
   >>>     for record in reader:
   >>>         # 处理每一个record

另一种情况是SQL可能执行的比如 ``desc``，这时通过 ``reader.raw`` 属性取到原始的SQL执行结果。

.. code-block:: python

   >>> with odps.execute_sql('desc dual').open_reader() as reader:
   >>>     print(reader.raw)

