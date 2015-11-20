.. _tables:

****
表
****

`表 <https://docs.aliyun.com/#/pub/odps/basic/definition&table>`_ 是ODPS的数据存储单元。

基本操作
========

我们可以用 ``list_tables`` 来列出项目空间下的所有表。

.. code-block:: python

   for table in odps.list_tables():
       # 处理每个表

通过调用 ``exist_table`` 来判断表是否存在。

通过调用 ``get_table`` 来获取表。

.. code-block:: python

   >>> t = odps.get_table('dual')
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

创建表的Schema
==============

有两种方法来初始化。第一种方式通过表的列、以及可选的分区来初始化。

.. code-block:: python

   >>> from odps.models import Schema, Column, Partition
   >>> columns = [Column(name='num', type='bigint', comment='the column')]
   >>> partitions = [Partition(name='pt', type='string', comment='the partition')]
   >>> schema = Schema(columns=columns, partitions=partitions)
   >>> schema.columns
   [<column num, type bigint>, <partition pt, type string>]


第二种方法是使用 ``Schema.from_lists``，这种方法更容易调用，但显然无法直接设置列和分区的注释了。

.. code-block:: python

   >>> schema = Schema.from_lists(['num'], ['bigint'], ['pt'], ['string'])
   >>> schema.columns
   [<column num, type bigint>, <partition pt, type string>]

创建表
======

现在知道怎么创建表的Schema，创建一个表就很容易了。

.. code-block:: python

   >>> table = odps.create_table('my_new_table', schema)
   >>> table = odps.create_table('my_new_table', schema, if_not_exists=True)  # 只有不存在表时才创建

其他还可以设置lifecycle等参数。

获取表数据
==========

有若干种方法能够获取表数据。首先，如果只是查看每个表的开始的小于1万条数据，则可以使用 ``head`` 方法。

.. code-block:: python

   >>> t = odps.get_table('dual')
   >>> for record in t.head(3):
   >>>     print(record[0])  # 取第0个位置的值
   >>>     print(record['c_double_a'])  # 通过字段取值
   >>>     print(record[0: 3])  # 切片操作
   >>>     print(record[0, 2, 3])  # 取多个位置的值
   >>>     print(record['c_int_a', 'c_double_a'])  # 通过多个字段取值

其次，在table上可以执行 ``open_reader`` 操作来打一个reader来读取数据。记住这里需要使用 **with表达式**。

.. code-block:: python

   >>> with t.open_reader(partition='pt=test') as reader:
   >>>     count = reader.count
   >>>     for record in reader[5:10]  # 可以执行多次，直到将count数量的record读完，这里可以改造成并行操作
   >>>         # 处理一条记录

最后，可以使用Tunnel API来进行读取操作，``open_reader`` 操作其实也是对Tunnel API的封装。
详细参考 `数据上传下载通道 <tunnel-zh.html>`_ 。

向表写数据
==========

类似于 ``open_reader``，table对象同样能执行 ``open_writer`` 来打开writer，并写数据。同样记住使用 **with表达式**。

.. code-block:: python

   >>> with t.open_writer(partition='pt=test') as writer:
   >>>     t.write(records)  # 这里records可以是任意可迭代的records，默认写到block 0
   >>>
   >>> with t.open_writer(partition='pt=test', blocks=[0, 1]) as writer:  # 这里同是打开两个block
   >>>     t.write(0, gen_records(block=0))
   >>>     t.write(1, gen_records(block=1))  # 这里两个写操作可以多线程并行，各个block间是独立的

同样，向表写数据也是对Tunnel API的封装，详细参考 `数据上传下载通道 <tunnel-zh.html>`_ 。

删除表
=======

.. code-block:: python

   >>> odps.delete_table('my_table_name', if_exists=True)  #  只有表存在时删除
   >>> t.drop()  # Table对象存在的时候可以直接执行drop函数

表分区
=======

基本操作
~~~~~~~~~~~

遍历表全部分区：

.. code-block:: python

   >>> for partition in table.partitions:
   >>>     print(partition.name)
   >>> for partition in table.iterate_partitions(spec='pt=test'):
   >>>     # 遍历二级分区

判断分区是否存在：

.. code-block:: python

   >>> table.exist_partition('pt=test,sub=2015')

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