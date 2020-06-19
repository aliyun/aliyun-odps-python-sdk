.. _resource:

资源
=======

`资源 <https://docs.aliyun.com/#/pub/odps/basic/definition&resource>`_ 在ODPS上常用在UDF和MapReduce中。

列出所有资源还是可以使用 ``list_resources``，判断资源是否存在使用 ``exist_resource``。
删除资源时，可以调用 ``delete_resource``，或者直接对于Resource对象调用 ``drop`` 方法。

在PyODPS中，主要支持两种资源类型，一种是文件，另一种是表。下面分别说明。

文件资源
---------

文件资源包括基础的 ``file`` 类型、以及 ``py``、``jar``、``archive``。

创建文件资源
~~~~~~~~~~~~~~~

创建文件资源可以通过给定资源名、文件类型、以及一个file-like的对象（或者是字符串对象）来创建，比如

.. code-block:: python

   resource = o.create_resource('test_file_resource', 'file', file_obj=open('/to/path/file'))  # 使用file-like的对象
   resource = o.create_resource('test_py_resource', 'py', file_obj='import this')  # 使用字符串


可以通过 ``temp=True`` 创建一个临时资源。

.. code-block:: python

   resource = o.create_resource('test_file_resource', 'file', file_obj=open('/to/path/file'), temp=True)


读取和修改文件资源
~~~~~~~~~~~~~~~~~~

对文件资源调用 ``open`` 方法，或者在odps入口调用 ``open_resource`` 都能打开一个资源，
打开后的对象会是file-like的对象。
类似于Python内置的 ``open`` 方法，文件资源也支持打开的模式。我们看例子：

.. code-block:: python

   >>> with resource.open('r') as fp:  # 以读模式打开
   >>>     content = fp.read()  # 读取全部的内容
   >>>     fp.seek(0)  # 回到资源开头
   >>>     lines = fp.readlines()  # 读成多行
   >>>     fp.write('Hello World')  # 报错，读模式下无法写资源
   >>>
   >>> with o.open_resource('test_file_resource', mode='r+') as fp:  # 读写模式打开
   >>>     fp.read()
   >>>     fp.tell()  # 当前位置
   >>>     fp.seek(10)
   >>>     fp.truncate()  # 截断后面的内容
   >>>     fp.writelines(['Hello\n', 'World\n'])  # 写入多行
   >>>     fp.write('Hello World')
   >>>     fp.flush()  # 手动调用会将更新提交到ODPS

所有支持的打开类型包括：

* ``r``，读模式，只能打开不能写
* ``w``，写模式，只能写入而不能读文件，注意用写模式打开，文件内容会被先清空
* ``a``，追加模式，只能写入内容到文件末尾
* ``r+``，读写模式，能任意读写内容
* ``w+``，类似于 ``r+``，但会先清空文件内容
* ``a+``，类似于 ``r+``，但写入时只能写入文件末尾

同时，PyODPS中，文件资源支持以二进制模式打开，打开如说一些压缩文件等等就需要以这种模式，
因此 ``rb`` 就是指以二进制读模式打开文件，``r+b`` 是指以二进制读写模式打开。

表资源
-------

创建表资源
~~~~~~~~~~~~

.. code-block:: python

   >>> o.create_resource('test_table_resource', 'table', table_name='my_table', partition='pt=test')

更新表资源
~~~~~~~~~~~

.. code-block:: python

   >>> table_resource = o.get_resource('test_table_resource')
   >>> table_resource.update(partition='pt=test2', project_name='my_project2')

获取表及分区
~~~~~~~~~~~~~

.. code-block:: python

   >>> table_resource = o.get_resource('test_table_resource')
   >>> table = table_resource.table
   >>> print(table.name)
   >>> partition = table_resource.partition
   >>> print(partition.spec)

读写内容
~~~~~~~~

.. code-block:: python

   >>> table_resource = o.get_resource('test_table_resource')
   >>> with table_resource.open_writer() as writer:
   >>>     writer.write([0, 'aaaa'])
   >>>     writer.write([1, 'bbbbb'])
   >>> with table_resource.open_reader() as reader:
   >>>     for rec in reader:
   >>>         print(rec)

