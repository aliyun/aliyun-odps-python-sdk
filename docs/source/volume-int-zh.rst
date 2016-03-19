.. _volume:

**********
Volume
**********

Volume 是 ODPS 提供的文件访问对象，每个 Project 中可以有多个 Volume，可以把每个 Volume 看作文件系统的一级目录。

在 PyODPS 中，用户可以通过 ``list_volumes`` 列出一个 Project 下的所有 Volume，通过 ``create_volume`` 创建新的 Volume，
通过 ``delete_volume`` 删除 Volume。

对于 Volume 下的分区，可以采用 ``list_volume_partitions`` 列出一个 Volume 下的所有分区，采用 ``delete_volume`` 删除分区。

Volume 上传和下载文件可以使用 ``open_reader`` 和 ``open_writer`` 方法。

Volume 操作
===============

通过 create_volume 创建一个 Volume 的方法如下：

.. code-block:: python

   volume = odps.create_volume('my_volume')

列举 Project 下所有 Volume 的方法如下：

.. code-block:: python

   volumes = odps.list_volumes()

删除 Volume 的方法如下：

.. code-block:: python

   odps.delete_volume('my_volume')

分区操作
===============

列举一个 Volume 下所有分区的方法如下：

.. code-block:: python

   partitions = odps.list_volume_partitions('my_volume')


列举一个 Volume 一个分区下所有文件的方法如下：

.. code-block:: python

   files = odps.list_volume_files('my_volume', 'my_partition')

删除一个分区及其下所有文件的方法如下：

.. code-block::  python

   odps.delete_volume_partition('my_volume', 'my_partition')

分区无法手动创建，可通过上传文件自动创建。

文件操作
===============

上传文件的方法如下，需要使用 with 表达式：

.. code-block:: python

   with odps.open_volume_writer('my_volume', 'my_partition') as writer:
       # 第一种方法：先打开再写入
       writer.open('file_name1').write('content')
       # 第二种方法：直接写入
       writer.write('file_name2', 'content')

其中，`writer.open` 方法会打开一个包含 write 方法的对象，可向其中写入内容。`writer.write` 方法会直接向指定文件写入内容，反复
调用时，指定的文件只会被打开一次。

下载文件的方法如下，需要使用 with 表达式：

.. code-block:: python

    with odps.open_volume_reader('my_volume', 'my_partition', 'my_file') as reader:
        reader.read(size)

其中，reader 为一个类似文件的对象，支持 `read`、`readline`、`readlines` 操作，可直接读取文本。