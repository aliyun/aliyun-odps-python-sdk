.. _tunnel:

*******************
数据上传下载通道
*******************


ODPS Tunnel是ODPS的数据通道，用户可以通过Tunnel向ODPS中上传或者下载数据。

**注意**，如果安装了 **cython**，在安装pyodps时会编译C代码，加速Tunnel的上传和下载。

上传
=====

.. code-block:: python

   from odps.tunnel import TableTunnel

   table = odps.get_table('my_table')

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

下载
======


.. code-block:: python

   from odps.tunnel import TableTunnel

   tunnel = TableTunnel(odps)
   download_session = tunnel.create_download_session('my_table', partition_spec='pt=test')

   with download_session.open_record_reader(0, download_session.count) as reader:
       for record in reader:
           # 处理每条记录