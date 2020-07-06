.. _tables:

表
======

`表 <https://docs.aliyun.com/#/pub/odps/basic/definition&table>`_ 是ODPS的数据存储单元。

基本操作
--------

我们可以用 ODPS 入口对象的 ``list_tables`` 来列出项目空间下的所有表。

.. code-block:: python

   for table in o.list_tables():
       # 处理每个表

通过调用 ``exist_table`` 来判断表是否存在。

通过调用 ``get_table`` 来获取表。

.. code-block:: python

   >>> t = o.get_table('dual')
   >>> t.schema
   odps.Schema {
     c_int_a                 bigint
     c_int_b                 bigint
     c_double_a              double
     c_double_b              double
     c_string_a              string
     c_string_b              string
     c_bool_a                boolean
     c_bool_b                boolean
     c_datetime_a            datetime
     c_datetime_b            datetime
   }
   >>> t.lifecycle
   -1
   >>> print(t.creation_time)
   2014-05-15 14:58:43
   >>> t.is_virtual_view
   False
   >>> t.size
   1408
   >>> t.comment
   'Dual Table Comment'
   >>> t.schema.columns
   [<column c_int_a, type bigint>,
    <column c_int_b, type bigint>,
    <column c_double_a, type double>,
    <column c_double_b, type double>,
    <column c_string_a, type string>,
    <column c_string_b, type string>,
    <column c_bool_a, type boolean>,
    <column c_bool_b, type boolean>,
    <column c_datetime_a, type datetime>,
    <column c_datetime_b, type datetime>]
   >>> t.schema['c_int_a']
   <column c_int_a, type bigint>
   >>> t.schema['c_int_a'].comment
   'Comment of column c_int_a'


通过提供 ``project`` 参数，来跨project获取表。

.. code-block:: python

   >>> t = o.get_table('dual', project='other_project')


.. _table_schema:

创建表的Schema
---------------

有两种方法来初始化。第一种方式通过表的列、以及可选的分区来初始化。

.. code-block:: python

   >>> from odps.models import Schema, Column, Partition
   >>> columns = [Column(name='num', type='bigint', comment='the column'),
   >>>            Column(name='num2', type='double', comment='the column2')]
   >>> partitions = [Partition(name='pt', type='string', comment='the partition')]
   >>> schema = Schema(columns=columns, partitions=partitions)
   >>> schema.columns
   [<column num, type bigint>,
    <column num2, type double>,
    <partition pt, type string>]
   >>> schema.partitions
   [<partition pt, type string>]
   >>> schema.names  # 获取非分区字段的字段名
   ['num', 'num2']
   >>> schema.types  # 获取非分区字段的字段类型
   [bigint, double]


第二种方法是使用 ``Schema.from_lists``，这种方法更容易调用，但显然无法直接设置列和分区的注释了。

.. code-block:: python

   >>> schema = Schema.from_lists(['num', 'num2'], ['bigint', 'double'], ['pt'], ['string'])
   >>> schema.columns
   [<column num, type bigint>,
    <column num2, type double>,
    <partition pt, type string>]

创建表
------

可以使用表 schema 来创建表，方法如下：

.. code-block:: python

   >>> table = o.create_table('my_new_table', schema)
   >>> table = o.create_table('my_new_table', schema, if_not_exists=True)  # 只有不存在表时才创建
   >>> table = o.create_table('my_new_table', schema, lifecycle=7)  # 设置生命周期


更简单的方式是采用“字段名 字段类型”字符串来创建表，方法如下：

.. code-block:: python

   >>> table = o.create_table('my_new_table', 'num bigint, num2 double', if_not_exists=True)
   >>> # 创建分区表可传入 (表字段列表, 分区字段列表)
   >>> table = o.create_table('my_new_table', ('num bigint, num2 double', 'pt string'), if_not_exists=True)


在未经设置的情况下，创建表时，只允许使用 bigint、double、decimal、string、datetime、boolean、map 和 array 类型。
如果你使用的是位于公共云上的服务，或者支持 tinyint、struct 等新类型，可以设置 ``options.sql.use_odps2_extension = True``
打开这些类型的支持，示例如下：

.. code-block:: python

   >>> from odps import options
   >>> options.sql.use_odps2_extension = True
   >>> table = o.create_table('my_new_table', 'cat smallint, content struct<title:varchar(100), body string>')


同步表更新
-------------

有时候，一个表可能被别的程序做了更新，比如schema有了变化。此时可以调用 ``reload`` 方法来更新。

.. code-block:: python

   >>> table.reload()


行记录Record
-------------------

Record表示表的一行记录，我们在 Table 对象上调用 new_record 就可以创建一个新的 Record。

.. code-block:: python

   >>> t = o.get_table('mytable')
   >>> r = t.new_record(['val0', 'val1'])  # 值的个数必须等于表schema的字段数
   >>> r2 = t.new_record()  #  也可以不传入值
   >>> r2[0] = 'val0' # 可以通过偏移设置值
   >>> r2['field1'] = 'val1'  # 也可以通过字段名设置值
   >>> r2.field1 = 'val1'  # 通过属性设置值
   >>>
   >>> print(record[0])  # 取第0个位置的值
   >>> print(record['c_double_a'])  # 通过字段取值
   >>> print(record.c_double_a)  # 通过属性取值
   >>> print(record[0: 3])  # 切片操作
   >>> print(record[0, 2, 3])  # 取多个位置的值
   >>> print(record['c_int_a', 'c_double_a'])  # 通过多个字段取值


.. _table_read:

获取表数据
----------

有若干种方法能够获取表数据。首先，如果只是查看每个表的开始的小于1万条数据，则可以使用 ``head`` 方法。

.. code-block:: python

   >>> t = o.get_table('dual')
   >>> for record in t.head(3):
   >>>     # 处理每个Record对象


.. _table_open_reader:

其次，在table上可以执行 ``open_reader`` 操作来打一个reader来读取数据。

使用 with 表达式的写法：

.. code-block:: python

   >>> with t.open_reader(partition='pt=test') as reader:
   >>>     count = reader.count
   >>>     for record in reader[5:10]  # 可以执行多次，直到将count数量的record读完，这里可以改造成并行操作
   >>>         # 处理一条记录

不使用 with 表达式的写法：

.. code-block:: python

   >>> reader = t.open_reader(partition='pt=test')
   >>> count = reader.count
   >>> for record in reader[5:10]  # 可以执行多次，直到将count数量的record读完，这里可以改造成并行操作
   >>>     # 处理一条记录

更简单的调用方法是使用 ODPS 对象的 ``read_table`` 方法，例如

.. code-block:: python

   >>> for record in o.read_table('test_table', partition='pt=test'):
   >>>     # 处理一条记录


.. _table_write:


向表写数据
----------

类似于 ``open_reader``，table对象同样能执行 ``open_writer`` 来打开writer，并写数据。

使用 with 表达式的写法：

.. code-block:: python

   >>> with t.open_writer(partition='pt=test') as writer:
   >>>     records = [[111, 'aaa', True],                 # 这里可以是list
   >>>                [222, 'bbb', False],
   >>>                [333, 'ccc', True],
   >>>                [444, '中文', False]]
   >>>     writer.write(records)  # 这里records可以是可迭代对象
   >>>
   >>>     records = [t.new_record([111, 'aaa', True]),   # 也可以是Record对象
   >>>                t.new_record([222, 'bbb', False]),
   >>>                t.new_record([333, 'ccc', True]),
   >>>                t.new_record([444, '中文', False])]
   >>>     writer.write(records)
   >>>


如果分区不存在，可以使用 ``create_partition`` 参数指定创建分区，如

.. code-block:: python

   >>> with t.open_writer(partition='pt=test', create_partition=True) as writer:
   >>>     records = [[111, 'aaa', True],                 # 这里可以是list
   >>>                [222, 'bbb', False],
   >>>                [333, 'ccc', True],
   >>>                [444, '中文', False]]
   >>>     writer.write(records)  # 这里records可以是可迭代对象

更简单的写数据方法是使用 ODPS 对象的 write_table 方法，例如

.. code-block:: python

   >>> records = [[111, 'aaa', True],                 # 这里可以是list
   >>>            [222, 'bbb', False],
   >>>            [333, 'ccc', True],
   >>>            [444, '中文', False]]
   >>> o.write_table('test_table', records, partition='pt=test', create_partition=True)

.. note::

    **注意**\ ：每次调用 write_table，MaxCompute 都会在服务端生成一个文件。这一操作需要较大的时间开销，
    同时过多的文件会降低后续的查询效率。因此，我们建议在使用 write_table 方法时，一次性写入多组数据，
    或者传入一个 generator 对象。

    write_table 写表时会追加到原有数据。PyODPS 不提供覆盖数据的选项，如果需要覆盖数据，需要手动清除
    原有数据。对于非分区表，需要调用 table.truncate()，对于分区表，需要删除分区后再建立。

使用多进程并行写数据：

每个进程写数据时共享同一个 session_id，但是有不同的 block_id，每个 block 对应服务端的一个文件，
最后主进程执行 commit，完成数据上传。

.. code-block:: python

    import random
    from multiprocessing import Pool
    from odps.tunnel import TableTunnel

    def write_records(session_id, block_id):
        # 使用指定的 id 创建 session
        local_session = tunnel.create_upload_session(table.name, upload_id=session_id)
        # 创建 writer 时指定 block_id
        with local_session.open_record_writer(block_id) as writer:
            for i in range(5):
                # 生成数据并写入对应 block
                record = table.new_record([random.randint(1, 100), random.random()])
                writer.write(record)

    if __name__ == '__main__':
        N_WORKERS = 3

        table = o.create_table('my_new_table', 'num bigint, num2 double', if_not_exists=True)
        tunnel = TableTunnel(o)
        upload_session = tunnel.create_upload_session(table.name)

        # 每个进程使用同一个 session_id
        session_id = upload_session.id

        pool = Pool(processes=N_WORKERS)
        futures = []
        block_ids = []
        for i in range(N_WORKERS):
            futures.append(pool.apply_async(write_records, (session_id, i)))
            block_ids.append(i)
        [f.get() for f in futures]

        # 最后执行 commit，并指定所有 block
        upload_session.commit(block_ids)

删除表
-------

.. code-block:: python

   >>> o.delete_table('my_table_name', if_exists=True)  #  只有表存在时删除
   >>> t.drop()  # Table对象存在的时候可以直接执行drop函数


创建DataFrame
-----------------

PyODPS提供了 :ref:`DataFrame框架 <df>` ，支持更方便地方式来查询和操作ODPS数据。
使用 ``to_df`` 方法，即可转化为 DataFrame 对象。

.. code-block:: python

   >>> table = o.get_table('my_table_name')
   >>> df = table.to_df()

表分区
-------

基本操作
~~~~~~~~~~~

判断是否为分区表：

.. code-block:: python

   >>> if table.schema.partitions:
   >>>     print('Table %s is partitioned.' % table.name)

遍历表全部分区：

.. code-block:: python

   >>> for partition in table.partitions:
   >>>     print(partition.name)
   >>> for partition in table.iterate_partitions(spec='pt=test'):
   >>>     # 遍历二级分区

判断分区是否存在（该方法需要填写所有分区字段值）：

.. code-block:: python

   >>> table.exist_partition('pt=test,sub=2015')

判断给定前缀的分区是否存在：

.. code-block:: python

   >>> # 表 table 的分区字段依次为 pt, sub
   >>> table.exist_partitions('pt=test')

获取分区：

.. code-block:: python

   >>> partition = table.get_partition('pt=test')
   >>> print(partition.creation_time)
   2015-11-18 22:22:27
   >>> partition.size
   0

创建分区
~~~~~~~~

.. code-block:: python

   >>> t.create_partition('pt=test', if_not_exists=True)  # 不存在的时候才创建

删除分区
~~~~~~~~~

.. code-block:: python

   >>> t.delete_partition('pt=test', if_exists=True)  # 存在的时候才删除
   >>> partition.drop()  # Partition对象存在的时候直接drop

.. _tunnel:

数据上传下载通道
----------------


.. note::

    不推荐直接使用tunnel接口（难用且复杂），推荐直接使用表的 :ref:`写 <table_write>` 和 :ref:`读 <table_read>` 接口。



ODPS Tunnel是ODPS的数据通道，用户可以通过Tunnel向ODPS中上传或者下载数据。

**注意**，如果安装了 **Cython**，在安装pyodps时会编译C代码，加速Tunnel的上传和下载。

上传
~~~~~~

.. code-block:: python

   from odps.tunnel import TableTunnel

   table = o.get_table('my_table')

   tunnel = TableTunnel(odps)
   upload_session = tunnel.create_upload_session(table.name, partition_spec='pt=test')

   with upload_session.open_record_writer(0) as writer:
       record = table.new_record()
       record[0] = 'test1'
       record[1] = 'id1'
       writer.write(record)

       record = table.new_record(['test2', 'id2'])
       writer.write(record)

   upload_session.commit([0])

也可以使用流式上传的接口：

.. code-block:: python

   from odps.tunnel import TableTunnel

   table = o.get_table('my_table')

   tunnel = TableTunnel(odps)
   upload_session = tunnel.create_stream_upload_session(table.name, partition_spec='pt=test')

   with upload_session.open_record_writer() as writer:
       record = table.new_record()
       record[0] = 'test1'
       record[1] = 'id1'
       writer.write(record)

       record = table.new_record(['test2', 'id2'])
       writer.write(record)


下载
~~~~~~


.. code-block:: python

   from odps.tunnel import TableTunnel

   tunnel = TableTunnel(odps)
   download_session = tunnel.create_download_session('my_table', partition_spec='pt=test')

   with download_session.open_record_reader(0, download_session.count) as reader:
       for record in reader:
           # 处理每条记录
