.. _sql:

SQL
=====

PyODPS支持ODPS SQL的查询，并可以读取执行的结果。
``execute_sql`` 或者 ``run_sql`` 方法的返回值是 `运行实例 <instances-zh.html>`_ 。

.. _execute_sql:

执行SQL
-------

.. code-block:: python

   >>> odps.execute_sql('select * from dual')  #  同步的方式执行，会阻塞直到SQL执行完成
   >>>
   >>> instance = odps.run_sql('select * from dual')  # 异步的方式执行
   >>> print(instance.get_logview_address())  # 获取logview地址
   >>> instance.wait_for_success()  # 阻塞直到完成

.. _sql_hints:

设置运行参数
------------

有时，我们在运行时，需要设置运行时参数，我们可以通过设置 ``hints`` 参数，参数类型是dict。

.. code-block:: python

   >>> odps.execute_sql('select * from pyodps_iris', hints={'odps.sql.mapper.split.size': 16})

我们可以对于全局配置设置sql.settings后，每次运行时则都会添加相关的运行时参数。

.. code-block:: python

   >>> from odps import options
   >>> options.sql.settings = {'odps.sql.mapper.split.size': 16}
   >>> odps.execute_sql('select * from pyodps_iris')  # 会根据全局配置添加hints


读取SQL执行结果
---------------

运行SQL的instance能够直接执行 ``open_reader`` 的操作，一种情况是SQL返回了结构化的数据。

.. code-block:: python

   >>> with odps.execute_sql('select * from dual').open_reader() as reader:
   >>>     for record in reader:
   >>>         # 处理每一个record

另一种情况是SQL可能执行的比如 ``desc``，这时通过 ``reader.raw`` 属性取到原始的SQL执行结果。

.. code-block:: python

   >>> with odps.execute_sql('desc dual').open_reader() as reader:
   >>>     print(reader.raw)

设置alias
------------

有时在运行时，比如某个UDF引用的资源是动态变化的，我们可以alias旧的资源名到新的资源，这样免去了重新删除并重新创建UDF的麻烦。

.. code-block:: python

    from odps.models import Schema

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
    res1 = odps.create_resource('test_alias_res1', 'file', file_obj='1')
    odps.create_resource('test_alias.py', 'py', file_obj=myfunc)
    odps.create_function('test_alias_func',
                         class_type='test_alias.Example',
                         resources=['test_alias.py', 'test_alias_res1'])

    table = odps.create_table(
        'test_table',
        schema=Schema.from_lists(['size'], ['bigint']),
        if_not_exists=True
    )

    data = [[1, ], ]
    # 写入一行数据，只有一行，一个值1
    odps.write_table(table, 0, [table.new_record(it) for it in data])

    with odps.execute_sql(
        'select test_alias_func(size) from test_table').open_reader() as reader:
        print(reader[0][0])

.. code-block:: python

    2

.. code-block:: python

    res2 = odps.create_resource('test_alias_res2', 'file', file_obj='2')
    # 把内容为1的资源alias成内容为2的资源，我们不需要修改UDF或资源
    with odps.execute_sql(
        'select test_alias_func(size) from test_table',
        aliases={'test_alias_res1': 'test_alias_res2'}).open_reader() as reader:
        print(reader[0][0])

.. code-block:: python

    3


设置biz_id
------------

需要设置全局options里的biz_id。

.. code-block:: python

   from odps import options

   options.biz_id = 'my_biz_id'
   odps.execute_sql('select * from pyodps_iris')
