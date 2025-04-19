.. _resource:

资源
=======

`资源 <https://help.aliyun.com/document_detail/27822.html>`_ 在ODPS上常用在UDF和MapReduce中。

在PyODPS中，主要支持两种资源类型，一种是文件，另一种是表。它们的基本操作（列举和删除）相同，但创建和修改方法略有差异，下面分别说明。

基本操作
-------

列出所有资源还是可以使用 :meth:`~odps.ODPS.list_resources`，判断资源是否存在使用 :meth:`~odps.ODPS.exist_resource`。\
删除资源时，可以调用 :meth:`~odps.ODPS.delete_resource`，或者直接对于Resource对象调用 :meth:`~odps.models.Resource.drop` 方法。

例如，要列举 Project 下的所有资源，可以使用下面的方法：

.. code-block:: python

    for res in o.list_resources():
        print(res.name)

要列举资源名包含给定前缀的资源，可以使用下面的方法：

.. code-block:: python

    for res in o.list_resources(prefix="prefix"):
        print(res.name)

判断给定名字的资源是否存在，可以使用下面的方法：

.. code-block:: python

    o.exist_resource("resource_name.tar.gz")

删除给定资源，可以使用 ODPS 入口对象的 :meth:`~odps.models.Resource.delete_resource` 方法，也可以使用
:class:`~odps.models.Resource` 对象自己的 :meth:`~odps.models.Resource.drop` 方法。

.. code-block:: python

    # 使用 ODPS.delete_resource 方法
    o.delete_resource("resource_name.tar.gz")
    # 使用 Resource.drop 方法
    o.get_resource("resource_name.tar.gz").drop()

文件资源
---------

文件资源包括基础的 ``file`` 类型、以及 ``py``、``jar``、``archive``。

创建文件资源
~~~~~~~~~~~~~~~

创建文件资源可以通过给定资源名、文件类型、以及一个file-like的对象（或者是字符串对象）来创建，比如

.. code-block:: python

   # 使用 file-like 的对象创建文件资源，注意压缩包等文件需要用二进制模式读取
   resource = o.create_resource('test_file_resource', 'file', fileobj=open('/to/path/file', 'rb'))
   # 使用字符串
   resource = o.create_resource('test_py_resource', 'py', fileobj='import this')


可以通过 ``temp=True`` 创建一个临时资源。

.. code-block:: python

   resource = o.create_resource('test_file_resource', 'file', fileobj=open('/to/path/file'), temp=True)

.. note::

    在 fileobj 参数中传入字符串，创建的资源内容为 **字符串本身** 而非字符串代表的路径指向的文件。

    如果文件过大（例如大小超过 64MB），PyODPS 可能会使用分块上传模式，而这不被旧版 MaxCompute 部署所支持。
    如需在旧版 MaxCompute 中上传大文件，请配置 ``options.upload_resource_in_chunks = False`` 。

读取和修改文件资源
~~~~~~~~~~~~~~
对文件资源调用 ``open`` 方法，或者在 MaxCompute 入口调用 ``open_resource`` 都能打开一个资源，
打开后的对象会是 file-like 的对象。
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

对于较大的文件资源，可以使用流式方式读写文件，使用方法为在调用 :meth:`~odps.ODPS.open_resource` 时增加一个
``stream=True`` 选项：

.. code-block:: python

   >>> with o.open_resource('test_file_resource', mode='w') as fp:  # 写模式打开
   >>>     fp.writelines(['Hello\n', 'World\n'])  # 写入多行
   >>>     fp.write('Hello World')
   >>>     fp.flush()  # 手动调用会将更新提交到 MaxCompute
   >>>
   >>> with resource.open('r', stream=True) as fp:  # 以读模式打开
   >>>     content = fp.read()  # 读取全部的内容
   >>>     line = fp.readline()  # 回到资源开头
   >>>     lines = fp.readlines()  # 读成多行

当 ``stream=True`` 时，只支持 ``r`` ， ``rb`` ， ``w`` ， ``wb`` 四种模式。

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

