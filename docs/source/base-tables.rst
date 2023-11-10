.. _tables:

表
======

`表 <https://help.aliyun.com/document_detail/27819.html>`_ 是ODPS的数据存储单元。

基本操作
--------

.. note::

    本文档中的代码对 PyODPS 0.11.3 及后续版本有效。对早于 0.11.3 版本的 PyODPS，请使用 ``odps.models.Schema`` 代替
    ``odps.models.TableSchema``，使用 ``schema`` 属性代替 ``table_schema`` 属性。

我们可以用 ODPS 入口对象的 ``list_tables`` 来列出项目空间下的所有表。

.. code-block:: python

   for table in o.list_tables():
       print(table.name)

通过该方法获取的 Table 对象不会自动加载表名以外的属性，此时获取这些属性（例如 ``table_schema`` 或者
``creation_time``）可能导致额外的请求并造成额外的时间开销。如果需要在列举表的同时读取这些属性，在
PyODPS 0.11.5 及后续版本中，可以为 ``list_tables`` 添加 ``extended=True`` 参数：

.. code-block:: python

   for table in o.list_tables(extended=True):
       print(table.name, table.creation_time)

通过调用 ``exist_table`` 来判断表是否存在。

.. code-block:: python

   o.exist_table('dual')

通过调用 ``get_table`` 来获取表。

.. code-block:: python

   >>> t = o.get_table('dual')
   >>> t.table_schema
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
   >>> t.table_schema.columns
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
   >>> t.table_schema['c_int_a']
   <column c_int_a, type bigint>
   >>> t.table_schema['c_int_a'].comment
   'Comment of column c_int_a'


通过提供 ``project`` 参数，来跨project获取表。

.. code-block:: python

   >>> t = o.get_table('dual', project='other_project')


.. _table_schema:

创建表的Schema
---------------

有两种方法来初始化。第一种方式通过表的列、以及可选的分区来初始化。

.. code-block:: python

   >>> from odps.models import TableSchema, Column, Partition
   >>> columns = [Column(name='num', type='bigint', comment='the column'),
   >>>            Column(name='num2', type='double', comment='the column2')]
   >>> partitions = [Partition(name='pt', type='string', comment='the partition')]
   >>> schema = TableSchema(columns=columns, partitions=partitions)
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

   >>> schema = TableSchema.from_lists(['num', 'num2'], ['bigint', 'double'], ['pt'], ['string'])
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

其次，在 table 实例上可以执行 ``open_reader`` 操作来打一个 reader 来读取数据。如果表为分区表，需要引入
``partition`` 参数指定需要读取的分区。

使用 with 表达式的写法：

.. code-block:: python

   >>> with t.open_reader(partition='pt=test,pt2=test2') as reader:
   >>>     count = reader.count
   >>>     for record in reader[5:10]:  # 可以执行多次，直到将count数量的record读完，这里可以改造成并行操作
   >>>         # 处理一条记录

不使用 with 表达式的写法：

.. code-block:: python

   >>> reader = t.open_reader(partition='pt=test,pt2=test2')
   >>> count = reader.count
   >>> for record in reader[5:10]:  # 可以执行多次，直到将count数量的record读完，这里可以改造成并行操作
   >>>     # 处理一条记录

更简单的调用方法是使用 ODPS 对象的 ``read_table`` 方法，例如

.. code-block:: python

   >>> for record in o.read_table('test_table', partition='pt=test,pt2=test2'):
   >>>     # 处理一条记录

直接读取成 Pandas DataFrame:

.. code-block:: python

   >>> with t.open_reader(partition='pt=test,pt2=test2') as reader:
   >>>     pd_df = reader.to_pandas()

.. _table_to_pandas_mp:

利用多进程加速读取:

.. code-block:: python

   >>> import multiprocessing
   >>> n_process = multiprocessing.cpu_count()
   >>> with t.open_reader(partition='pt=test,pt2=test2') as reader:
   >>>     pd_df = reader.to_pandas(n_process=n_process)

.. note::

    ``open_reader`` 或者 ``read_table`` 方法仅支持读取单个分区。如果需要读取多个分区的值，例如
    读取所有符合 ``dt>20230119`` 这样条件的分区，需要使用 ``iterate_partitions`` 方法，详见
    :ref:`遍历表分区 <iterate_partitions>` 章节。

.. _table_write:

向表写数据
----------

类似于 ``open_reader``，table对象同样能执行 ``open_writer`` 来打开writer，并写数据。如果表为分区表，需要引入
``partition`` 参数指定需要写入的分区。

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

    write_table 写表时会追加到原有数据。如果需要覆盖数据，可以为 write_table 增加一个参数 ``overwrite=True``
    （仅在 0.11.1以后支持），或者调用 table.truncate() / 删除分区后再建立分区。

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

.. _table_arrow_io:

使用 Arrow 格式读写数据
--------------------
`Apache Arrow <https://arrow.apache.org/>`_ 是一种跨语言的通用数据读写格式，支持在各种不同平台间进行数据交换。
自2021年起， MaxCompute 支持使用 Arrow 格式读取表数据，PyODPS 则从 0.11.2 版本开始支持该功能。具体地，如果在
Python 环境中安装 pyarrow 后，在调用 ``open_reader`` 或者 ``open_writer`` 时增加 ``arrow=True`` 参数，即可读写
`Arrow RecordBatch <https://arrow.apache.org/docs/python/data.html#record-batches>`_ 。

按 RecordBatch 读取表内容：

.. code-block:: python

   >>> reader = t.open_reader(partition='pt=test', arrow=True)
   >>> count = reader.count
   >>> for batch in reader:  # 可以执行多次，直到将所有 RecordBatch 读完
   >>>     # 处理一个 RecordBatch，例如转换为 Pandas
   >>>     print(batch.to_pandas())

写入 RecordBatch：

.. code-block:: python

   >>> import pandas as pd
   >>> import pyarrow as pa
   >>>
   >>> with t.open_writer(partition='pt=test', create_partition=True, arrow=True) as writer:
   >>>     records = [[111, 'aaa', True],
   >>>                [222, 'bbb', False],
   >>>                [333, 'ccc', True],
   >>>                [444, '中文', False]]
   >>>     df = pd.DataFrame(records, columns=["int_val", "str_val", "bool_val"])
   >>>     # 写入 RecordBatch
   >>>     batch = pa.RecordBatch.from_pandas(df)
   >>>     writer.write(batch)
   >>>     # 也可以直接写入 Pandas DataFrame
   >>>     writer.write(df)

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

判断表是否为分区表：

.. code:: python

   >>> if table.table_schema.partitions:
   >>>     print('Table %s is partitioned.' % table.name)

判断分区是否存在（该方法需要填写所有分区字段值）：

.. code:: python

   >>> table.exist_partition('pt=test,sub=2015')

判断给定前缀的分区是否存在：

.. code:: python

   >>> # 表 table 的分区字段依次为 pt, sub
   >>> table.exist_partitions('pt=test')

获取一个分区的相关信息：

.. code:: python

   >>> partition = table.get_partition('pt=test')
   >>> print(partition.creation_time)
   2015-11-18 22:22:27
   >>> partition.size
   0

.. note::

    这里的"分区"指的不是分区字段而是所有分区字段均确定的分区定义对应的子表。如果某个分区字段对应多个值，
    则相应地有多个子表，即多个分区。而 ``get_partition`` 只能获取一个分区的信息。因而，

    1. 如果某些分区未指定，那么这个分区定义可能对应多个子表，``get_partition`` 时则不被 PyODPS 支持。此时，需要使用
    ``iterate_partitions`` 分别处理每个分区。

    2. 如果某个分区字段被定义多次，或者使用类似 ``pt>20210302`` 这样的非确定逻辑表达式，则无法使用
    ``get_partition`` 获取分区。在此情况下，可以尝试使用 ``iterate_partitions`` 枚举每个分区。

创建分区
~~~~~~~~

下面的操作将创建一个分区，如果分区存在将报错：

.. code:: python

   >>> t.create_partition('pt=test')

下面的操作将创建一个分区，如果分区存在则跳过：

.. code:: python

   >>> t.create_partition('pt=test', if_not_exists=True)

.. _iterate_partitions:

遍历表分区
~~~~~~~~
下面的操作将遍历表全部分区：

.. code:: python

   >>> for partition in table.partitions:
   >>>     print(partition.name)

如果要遍历部分分区值确定的分区，可以使用 ``iterate_partitions`` 方法。

.. code:: python

   >>> for partition in table.iterate_partitions(spec='pt=test'):
   >>>     print(partition.name)

自 PyODPS 0.11.3 开始，支持为 ``iterate_partitions`` 指定简单的逻辑表达式及通过逗号连接，
每个子表达式均须满足的复合逻辑表达式。或运算符暂不支持。

.. code:: python

   >>> for partition in table.iterate_partitions(spec='dt>20230119'):
   >>>     print(partition.name)

.. note::

    在 0.11.3 之前的版本中，``iterate_partitions`` 仅支持枚举前若干个分区等于相应值的情形。例如，
    当表的分区字段按顺序分别为 pt1、pt2 和 pt3，那么 ``iterate_partitions`` 中的  ``spec``
    参数只能指定 ``pt1=xxx`` 或者 ``pt1=xxx,pt2=yyy`` 这样的形式。自 0.11.3 开始，
    ``iterate_partitions`` 支持更多枚举方式，但仍建议尽可能限定上一级分区以提高枚举的效率。

删除分区
~~~~~~~~~

下面的操作将删除一个分区：

.. code:: python

   >>> t.delete_partition('pt=test', if_exists=True)  # 存在的时候才删除
   >>> partition.drop()  # Partition对象存在的时候直接drop

获取值最大分区
~~~~~~~~~~~
很多时候你可能希望获取值最大的分区。例如，当以日期为分区值时，你可能希望获得日期最近的有数据的分区。PyODPS 自 0.11.3
开始支持此功能。

创建分区表并写入一些数据：

.. code-block:: python

    t = o.create_table("test_multi_pt_table", ("col string", "pt1 string, pt2 string"))
    for pt1, pt2 in (("a", "a"), ("a", "b"), ("b", "c"), ("b", "d")):
        o.write_table("test_multi_pt_table", [["value"]], partition="pt1=%s,pt2=%s" % (pt1, pt2))

如果想要获得值最大的分区，可以使用下面的代码：

.. code:: python

    >>> part = t.get_max_partition()
    >>> part
    <Partition cupid_test_release.`test_multi_pt_table`(pt1='b',pt2='d')>
    >>> part.partition_spec["pt1"]  # 获取某个分区字段的值
    b

如果只希望获得最新的分区而忽略分区内是否有数据，可以用

.. code:: python

    >>> t.get_max_partition(skip_empty=False)
    <Partition cupid_test_release.`test_multi_pt_table`(pt1='b',pt2='d')>

对于多级分区表，可以通过限定上级分区值来获得值最大的子分区，例如

.. code:: python

    >>> t.get_max_partition("pt1=a")
    <Partition cupid_test_release.`test_multi_pt_table`(pt1='a',pt2='b')>

.. _tunnel:

数据上传下载通道
----------------


.. note::

    不推荐直接使用 Tunnel 接口，该接口较为低级。推荐直接使用表的 :ref:`写 <table_write>` 和 :ref:`读 <table_read>` 接口，可靠性和易用性更高。


ODPS Tunnel是ODPS的数据通道，用户可以通过Tunnel向ODPS中上传或者下载数据。

**注意**，如果安装了 **Cython**，在安装pyodps时会编译C代码，加速Tunnel的上传和下载。

上传
~~~~~~

使用 Record 接口上传：

.. code-block:: python

   from odps.tunnel import TableTunnel

   table = o.get_table('my_table')

   tunnel = TableTunnel(o)
   upload_session = tunnel.create_upload_session(table.name, partition_spec='pt=test')

   with upload_session.open_record_writer(0) as writer:
       record = table.new_record()
       record[0] = 'test1'
       record[1] = 'id1'
       writer.write(record)

       record = table.new_record(['test2', 'id2'])
       writer.write(record)

   # 需要在 with 代码块外 commit，否则数据未写入即 commit，会导致报错
   upload_session.commit([0])

也可以使用流式上传的接口：

.. code-block:: python

   from odps.tunnel import TableTunnel

   table = o.get_table('my_table')

   tunnel = TableTunnel(o)
   upload_session = tunnel.create_stream_upload_session(table.name, partition_spec='pt=test')

   with upload_session.open_record_writer() as writer:
       record = table.new_record()
       record[0] = 'test1'
       record[1] = 'id1'
       writer.write(record)

       record = table.new_record(['test2', 'id2'])
       writer.write(record)

以及使用 Arrow 格式上传：

.. code-block:: python

   import pandas as pd
   import pyarrow as pa
   from odps.tunnel import TableTunnel

   table = o.get_table('my_table')

   tunnel = TableTunnel(o)
   upload_session = tunnel.create_upload_session(table.name, partition_spec='pt=test')

   with upload_session.open_arrow_writer(0) as writer:
       df = pd.DataFrame({"name": ["test1", "test2"], "id": ["id1", "id2"]})
       batch = pa.RecordBatch.from_pandas(df)
       writer.write(batch)

   # 需要在 with 代码块外 commit，否则数据未写入即 commit，会导致报错
   upload_session.commit([0])


下载
~~~~~~

使用 Record 接口读取：

.. code-block:: python

   from odps.tunnel import TableTunnel

   tunnel = TableTunnel(o)
   download_session = tunnel.create_download_session('my_table', partition_spec='pt=test')

   with download_session.open_record_reader(0, download_session.count) as reader:
       for record in reader:
           # 处理每条记录

使用 Arrow 接口读取：

.. code-block:: python

   from odps.tunnel import TableTunnel

   tunnel = TableTunnel(o)
   download_session = tunnel.create_download_session('my_table', partition_spec='pt=test')

   with download_session.open_arrow_reader(0, download_session.count) as reader:
       for batch in reader:
           # 处理每个 Arrow RecordBatch
