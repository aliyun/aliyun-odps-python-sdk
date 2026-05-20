.. _storage_api_v2:

Storage API V2
==============

MaxCompute Storage API V2 是 MaxCompute 提供的高吞吐数据读写接口。相比于基于 Tunnel 的数据通道，
Storage API V2 提供了更细粒度的会话管理、支持 Arrow 格式和 Blob 格式读写、支持增量读取以及表预览等功能，
适用于大规模并行数据读写场景。

.. note::

    Storage API V2 需要服务端支持，请确保 MaxCompute 集群已开启相关功能。

基本概念
--------

Storage API V2 的核心概念包括：

- **会话（Session）**：读写操作的事务上下文。读会话管理数据分片，写会话保证数据原子性。
- **分片（Split）**：读会话将数据按大小、并行度等方式分为多个分片，每个分片可独立读取，支持并行处理。
- **流（Stream）**：写会话中的数据上传通道。一个写会话可创建多个流以支持并行写入。
- **压缩（Compression）**：支持 UNCOMPRESSED（默认）、LZ4 和 ZSTD 压缩算法，减少网络传输数据量。
- **路由令牌（Route Token）**：服务端返回的路由标识，用于会话亲和性，确保后续请求路由到同一节点。
- **Exactly-Once 模式**：写流支持精确一次语义，通过 access_token 和 row_offset 实现幂等写入。

客户端初始化
------------

使用 Storage API V2 需要创建 ``StorageApiClient`` 或 ``StorageApiArrowClient`` 实例。
两者均需要传入 ODPS 入口对象和表对象（或 Instance 对象）。

.. code-block:: python

   from odps import ODPS
   from odps.apis.storage_api_v2 import StorageApiClient, StorageApiArrowClient

   # 初始化 ODPS 入口对象
   odps = ODPS(
       access_id="your_access_id",
       secret_access_key="your_secret_access_key",
       project="your_project",
       endpoint="your_endpoint",
   )

   # 获取表对象
   table = odps.get_table("your_table")

   # 创建基础客户端（返回原始字节流）
   client = StorageApiClient(odps, table)

   # 创建 Arrow 客户端（返回 Arrow RecordBatch，推荐）
   arrow_client = StorageApiArrowClient(odps, table)

``StorageApiArrowClient`` 继承自 ``StorageApiClient``，额外提供了 ``read_rows_arrow``、
``write_rows_arrow`` 和 ``preview_table_arrow`` 方法，能够直接读写 PyArrow RecordBatch 对象，
便于与 pandas 等数据处理框架配合使用。大多数场景下推荐使用 ``StorageApiArrowClient``。

也可以通过指定 ``quota_name`` 和 ``rest_endpoint`` 参数来使用指定配额和自定义端点：

.. code-block:: python

   client = StorageApiArrowClient(
       odps, table,
       quota_name="your_quota",
       rest_endpoint="https://your-custom-endpoint",
   )

还可以通过 ``tags`` 参数设置请求标签：

.. code-block:: python

   client = StorageApiArrowClient(
       odps, table,
       tags="tag1,tag2",
   )

读取数据
--------

完整读取流程
~~~~~~~~~~~~

使用 Storage API V2 读取数据需要以下步骤：

1. 创建读会话
2. 按分片读取数据
3. 关闭读会话（会话会自动过期，无需手动关闭）

创建读会话
~~~~~~~~~~

通过 :meth:`~odps.apis.storage_api_v2.StorageApiClient.create_read_session` 方法创建读会话。
读会话确定了数据的分片方式、返回的列和分区等。

.. code-block:: python

   from odps.apis.storage_api_v2 import StorageApiArrowClient

   arrow_client = StorageApiArrowClient(odps, table)

   # 创建读会话，使用默认分片选项
   read_resp = arrow_client.create_read_session()
   print(f"Session ID: {read_resp.session_id}")
   print(f"数据分片数: {read_resp.splits_count}")
   print(f"总记录数: {read_resp.record_count}")
   print(f"会话状态: {read_resp.session_status}")
   print(f"过期时间: {read_resp.expiration_time}")

创建读会话时可以指定需要读取的列、分区和分片选项：

.. code-block:: python

   from odps.apis.storage_api_v2 import SplitOptions

   # 仅读取指定列和分区
   read_resp = arrow_client.create_read_session(
       required_data_columns=["id", "name", "value"],
       required_partitions=["pt=20230101"],
   )

   # 按并行度分片，创建 10 个分片
   split_opts = SplitOptions()
   split_opts.split_mode = SplitOptions.SplitMode.PARALLELISM
   split_opts.split_number = 10
   read_resp = arrow_client.create_read_session(
       split_options=split_opts,
   )

   # 按行偏移分片
   split_opts = SplitOptions()
   split_opts.split_mode = SplitOptions.SplitMode.ROW_OFFSET
   split_opts.split_number = 1000000  # 每个分片包含 100 万行
   read_resp = arrow_client.create_read_session(
       split_options=split_opts,
   )

创建读会话时还可以指定分区列和 Bucket ID：

.. code-block:: python

   # 仅读取指定分区列
   read_resp = arrow_client.create_read_session(
       required_partition_columns=["pt"],
   )

   # 读取指定 Bucket（适用于聚簇表）
   read_resp = arrow_client.create_read_session(
       required_bucket_ids=["0", "1"],
   )

使用 Arrow 格式读取数据
~~~~~~~~~~~~~~~~~~~~~~~

使用 ``StorageApiArrowClient`` 的 ``read_rows_arrow`` 方法可以直接读取为 Arrow RecordBatch：

.. code-block:: python

   import pyarrow as pa

   read_resp = arrow_client.create_read_session()

   # 遍历所有分片读取数据
   for split_index in range(read_resp.splits_count):
       reader = arrow_client.read_rows_arrow(
           read_resp.session_id, split_index=split_index
       )
       while True:
           batch = reader.read()
           if batch is None:
               break
           df = batch.to_pandas()
           # 处理 DataFrame

使用原始字节流读取数据
~~~~~~~~~~~~~~~~~~~~~~~

使用 ``StorageApiClient`` 的 ``read_rows_stream`` 方法返回 ``StreamReader``，
可以读取原始字节流：

.. code-block:: python

   from odps.apis.storage_api_v2 import StorageApiClient, ArrowReader

   client = StorageApiClient(odps, table)
   read_resp = client.create_read_session()

   reader = client.read_rows_stream(read_resp.session_id, split_index=0)
   arrow_reader = ArrowReader(reader)
   while True:
       batch = arrow_reader.read()
       if batch is None:
           break
       df = batch.to_pandas()
       # 处理 DataFrame

读取指定范围的数据
~~~~~~~~~~~~~~~~~~

可以通过 ``row_offset`` 和 ``row_count`` 参数读取指定范围的数据：

.. code-block:: python

   # 从第 1000 行开始读取 500 行
   reader = arrow_client.read_rows_arrow(
       read_resp.session_id,
       split_index=0,
       row_offset=1000,
       row_count=500,
   )

还可以通过 ``max_batch_rows`` 控制每个批次的行数，以管理内存使用：

.. code-block:: python

   # 每个批次最多 1024 行，减少内存占用
   reader = arrow_client.read_rows_arrow(
       read_resp.session_id,
       split_index=0,
       max_batch_rows=1024,
   )

通过 ``max_batch_raw_size`` 可以控制每个批次的原始字节大小：

.. code-block:: python

   # 每个批次原始大小不超过 8MB
   reader = arrow_client.read_rows_arrow(
       read_resp.session_id,
       split_index=0,
       max_batch_raw_size=8 * 1024 * 1024,
   )

读取时还可以通过 ``skip_row_num`` 跳过指定行数、通过 ``data_columns`` 指定返回列、
通过 ``data_format`` 指定数据格式：

.. code-block:: python

   # 跳过前 100 行
   reader = arrow_client.read_rows_arrow(
       read_resp.session_id,
       split_index=0,
       skip_row_num=100,
   )

   # 仅读取指定列
   reader = arrow_client.read_rows_arrow(
       read_resp.session_id,
       split_index=0,
       data_columns=["id", "name"],
   )

并行读取数据
~~~~~~~~~~~~

Storage API V2 的分片机制天然支持并行读取。每个分片可以独立读取，适合多线程或多进程场景：

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   read_resp = arrow_client.create_read_session()

   def read_split(split_index):
       reader = arrow_client.read_rows_arrow(
           read_resp.session_id, split_index=split_index
       )
       batches = []
       while True:
           batch = reader.read()
           if batch is None:
               break
           batches.append(batch)
       return pa.concat_batches(batches) if batches else None

   # 使用线程池并行读取所有分片
   with ThreadPoolExecutor(max_workers=read_resp.splits_count) as pool:
       futures = [
           pool.submit(read_split, i)
           for i in range(read_resp.splits_count)
       ]
       results = [f.result() for f in futures]

刷新读会话
~~~~~~~~~~

读会话有过期时间。如果在长时间读取过程中会话过期，可以使用 ``get_read_session`` 的
``refresh`` 参数延长会话有效期：

.. code-block:: python

   from odps.apis.storage_api_v2 import SessionStatus

   # 检查会话状态
   status = arrow_client.get_read_session(read_resp.session_id)
   if status.session_status == SessionStatus.EXPIRED:
       # 刷新会话
       status = arrow_client.get_read_session(read_resp.session_id, refresh=True)

增量读取
~~~~~~~~

Storage API V2 支持增量读取模式，可以捕获表的数据变更：

.. code-block:: python

   from odps.apis.storage_api_v2 import IncrementalReadOptions

   incr_opts = IncrementalReadOptions()
   incr_opts.start_version = 100
   read_resp = arrow_client.create_read_session(
       incremental_read=True,
       incremental_read_options=incr_opts,
   )
   print(f"当前最新版本: {read_resp.latest_version}")

也可以按时间戳范围进行增量读取：

.. code-block:: python

   incr_opts = IncrementalReadOptions()
   incr_opts.start_time_stamp = "2023-01-01 00:00:00"
   incr_opts.end_time_stamp = "2023-01-02 00:00:00"
   read_resp = arrow_client.create_read_session(
       incremental_read=True,
       incremental_read_options=incr_opts,
   )

从 SQL Instance 读取数据
~~~~~~~~~~~~~~~~~~~~~~~~~~

Storage API V2 支持从 SQL 执行结果（Instance）中读取数据，此时客户端为只读模式：

.. code-block:: python

   # 执行 SQL 获取 Instance
   instance = odps.execute_sql("SELECT * FROM your_table LIMIT 1000")

   # 使用 Instance 创建客户端
   instance_client = StorageApiArrowClient(odps, instance)
   read_resp = instance_client.create_read_session()

   # 读取结果
   for split_index in range(read_resp.splits_count):
       reader = instance_client.read_rows_arrow(
           read_resp.session_id, split_index=split_index
       )
       while True:
           batch = reader.read()
           if batch is None:
               break
           df = batch.to_pandas()

.. note::

    基于 Instance 的客户端不支持写操作。调用写操作方法会抛出 ``ValueError``。

预览表数据
----------

预览功能提供了一种轻量级的数据浏览方式，无需创建会话即可快速查看表数据：

.. code-block:: python

   from odps.apis.storage_api_v2 import ArrowReader

   # 使用 Arrow 客户端预览
   reader = arrow_client.preview_table_arrow(limit=10)
   batch = reader.read()
   if batch is not None:
       df = batch.to_pandas()
       print(df)

   # 使用基础客户端预览
   stream_reader = client.preview_table(limit=10)
   arrow_reader = ArrowReader(stream_reader)
   batch = arrow_reader.read()
   if batch is not None:
       df = batch.to_pandas()
       print(df)

可以指定分区和列进行预览：

.. code-block:: python

   # 预览指定分区
   reader = arrow_client.preview_table_arrow(limit=10, partition="pt=20230101")

   # 预览指定列
   reader = arrow_client.preview_table_arrow(limit=10, columns=["id", "name"])

.. note::

    预览功能仅适用于表，不支持 Instance 客户端。预览返回的行数可能不精确，
    生产环境建议使用 ``create_read_session`` + ``read_rows_arrow`` 进行精确读取。

写入数据
--------

完整写入流程
~~~~~~~~~~~~

使用 Storage API V2 写入数据需要以下步骤：

1. 创建写会话
2. 创建写流
3. 写入数据
4. 关闭写流
5. 提交写会话

.. code-block:: python

   import pyarrow as pa
   from odps.apis.storage_api_v2 import StorageApiArrowClient

   arrow_client = StorageApiArrowClient(odps, table)

   # 1. 创建写会话
   write_resp = arrow_client.create_write_session()
   session_id = write_resp.session_id

   # 2. 创建写流
   stream_resp = arrow_client.create_write_stream(session_id, stream_id=0)

   # 3. 写入数据
   schema = pa.schema([
       pa.field("id", pa.int64()),
       pa.field("name", pa.string()),
       pa.field("value", pa.float64()),
   ])
   batch = pa.record_batch([
       pa.array([1, 2, 3]),
       pa.array(["Alice", "Bob", "Carol"]),
       pa.array([100.0, 200.0, 150.0]),
   ], schema=schema)

   writer = arrow_client.write_rows_arrow(
       session_id, stream_id=0, record_count=3,
   )
   writer.write(batch)
   commit_msg, success = writer.finish()

   # 4. 关闭写流
   arrow_client.close_write_stream(session_id, stream_id=0)

   # 5. 提交写会话
   arrow_client.commit_write_session(session_id)

写入分区表
~~~~~~~~~~

写入分区表时需要在创建写会话时指定分区：

.. code-block:: python

   # 写入指定分区
   write_resp = arrow_client.create_write_session(
       partial_partition_spec="pt=20230101"
   )

如果需要覆盖已有分区数据，可以使用 ``flags`` 参数：

.. code-block:: python

   # 覆盖写入分区
   write_resp = arrow_client.create_write_session(
       partial_partition_spec="pt=20230101",
       flags={"overwrite": True},
   )

并行写入数据
~~~~~~~~~~~~

一个写会话可以创建多个写流，支持并行写入：

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   write_resp = arrow_client.create_write_session()
   session_id = write_resp.session_id

   def write_stream(stream_id, data_batch):
       # 创建写流
       arrow_client.create_write_stream(session_id, stream_id=stream_id)
       # 写入数据
       writer = arrow_client.write_rows_arrow(
           session_id, stream_id=stream_id, record_count=len(data_batch),
       )
       writer.write(data_batch)
       writer.finish()
       # 关闭写流
       arrow_client.close_write_stream(session_id, stream_id=stream_id)

   # 多线程并行写入
   with ThreadPoolExecutor(max_workers=3) as pool:
       futures = [
           pool.submit(write_stream, i, batch_data[i])
           for i in range(3)
       ]
       [f.result() for f in futures]

   # 提交写会话
   arrow_client.commit_write_session(session_id)

使用原始字节流写入数据
~~~~~~~~~~~~~~~~~~~~~~~

使用 ``StorageApiClient`` 的 ``write_rows_stream`` 方法可以获取 ``StreamWriter``，
然后使用 ``ArrowWriter`` 包装进行写入：

.. code-block:: python

   from odps.apis.storage_api_v2 import StorageApiClient, ArrowWriter, Compression

   client = StorageApiClient(odps, table)

   write_resp = client.create_write_session()
   session_id = write_resp.session_id
   client.create_write_stream(session_id, stream_id=0)

   stream_writer = client.write_rows_stream(
       session_id, stream_id=0, record_count=3,
   )
   arrow_writer = ArrowWriter(stream_writer, Compression.UNCOMPRESSED)
   arrow_writer.write(batch)
   commit_msg, success = arrow_writer.finish()

   client.close_write_stream(session_id, stream_id=0)
   client.commit_write_session(session_id)

Exactly-Once 写入模式
~~~~~~~~~~~~~~~~~~~~~

Storage API V2 支持 Exactly-Once 语义，确保写入操作的幂等性。启用后，服务端会返回
access_token 和 row_offset，用于重试时保证数据不重复：

.. code-block:: python

   # 创建写会话
   write_resp = arrow_client.create_write_session()
   session_id = write_resp.session_id

   # 创建写流，启用 Exactly-Once 模式
   stream_resp = arrow_client.create_write_stream(session_id, stream_id=0)
   access_token = stream_resp.access_token

   # 写入数据时传入 access_token
   writer = arrow_client.write_rows_arrow(
       session_id, stream_id=0, record_count=3,
       access_token=access_token,
   )
   writer.write(batch)
   commit_msg, success = writer.finish()

   # 获取 Exactly-Once 模式的写入偏移量
   write_stream_resp = writer.get_write_stream_response()
   if write_stream_resp is not None:
       print(f"写入偏移量: {write_stream_resp.exactly_once_row_offset}")

   # 查询写流状态也可获取 access_token 和 row_offset
   stream_status = arrow_client.get_write_stream(
       session_id, stream_id=0, stream_version=0,
   )
   print(f"最新 schema 版本: {stream_status.latest_schema_version}")
   print(f"写入偏移量: {stream_status.row_offset}")

.. note::

    Exactly-Once 模式下，写流重试时需要使用上次成功写入返回的 ``row_offset`` 和
    ``access_token`` 来保证幂等性。

中止写会话
~~~~~~~~~~

如果写入过程中发生错误，可以中止写会话以丢弃所有已上传的数据：

.. code-block:: python

   write_resp = arrow_client.create_write_session()
   session_id = write_resp.session_id

   try:
       # 写入数据...
       pass
   except Exception as e:
       # 发生错误时中止会话
       arrow_client.abort_write_session(session_id)
       raise

查询写会话状态
~~~~~~~~~~~~~~

可以使用 ``get_write_session`` 方法查看写会话中各流的状态：

.. code-block:: python

   status = arrow_client.get_write_session(session_id)
   if status.streams:
       for stream_info in status.streams:
           print(f"Stream: {stream_info}")

也可以使用 ``get_write_stream`` 方法查询单个写流的详细状态：

.. code-block:: python

   stream_status = arrow_client.get_write_stream(
       session_id, stream_id=0, stream_version=0,
   )
   print(f"流状态: {stream_status.status}")
   print(f"已写入记录数: {stream_status.record_count}")
   print(f"已写入字节数: {stream_status.byte_size}")

对于事务表（Delta 表），提交写会话时可以指定需要提交的流 ID 和版本：

.. code-block:: python

   arrow_client.commit_write_session(
       session_id,
       stream_ids=["stream-1", "stream-2"],
       stream_versions=[1, 1],
   )

路由令牌与会话亲和性
~~~~~~~~~~~~~~~~~~~~

服务端在创建会话和写流时会返回路由令牌（route token），客户端会自动存储并在后续请求中
携带该令牌，确保请求路由到同一节点，提高性能：

.. code-block:: python

   # 路由令牌会自动从响应中提取并存储
   write_resp = arrow_client.create_write_session()
   print(f"当前路由令牌: {arrow_client.route_token}")

   # 也可以手动指定 route_token 参数覆盖自动值
   arrow_client.commit_write_session(
       session_id, route_token="your-route-token",
   )

Blob 数据读写
-------------

Storage API V2 支持 Blob 类型数据的读写。Blob 读写适用于包含 BLOB 列的表
（表格式需为 V2 且开启事务性）。

写入 Blob 数据
~~~~~~~~~~~~~~

使用 ``write_blob_stream`` 方法以流式方式上传单个 Blob：

.. code-block:: python

   # 写入 Blob 数据（默认不压缩，启用 MD5 校验）
   blob_writer = arrow_client.write_blob_stream(
       session_id, stream_id=0, partition_values=["pt=20230101"],
   )
   blob_writer.write(b"your blob data here")
   response = blob_writer.finish()
   print(f"Blob reference: {response.blob_reference}")

写入时可以指定压缩算法：

.. code-block:: python

   from odps.apis.storage_api_v2 import Compression

   # 使用 ZSTD 压缩写入 Blob
   blob_writer = arrow_client.write_blob_stream(
       session_id, stream_id=0, partition_values=["pt=20230101"],
       compression=Compression.ZSTD,
   )
   blob_writer.write(b"your blob data here")
   response = blob_writer.finish()

使用 ``write_blob_batch`` 方法批量上传多个 Blob：

.. code-block:: python

   from odps.apis.storage_api_v2 import BlobWriteItem, ChecksumType

   # 创建 BlobWriteItem 列表
   items = [
       BlobWriteItem(
           data=b"blob data 1",
           partition_values=["pt=20230101"],
           column_index=0,
           mime_type="image/png",
           checksum_type=ChecksumType.MD5,
           distribution_key="key1",
       ),
       BlobWriteItem(
           data=b"blob data 2",
           partition_values=["pt=20230101"],
           column_index=0,
       ),
   ]

   # 批量写入
   response = arrow_client.write_blob_batch(
       items, session_id=session_id, stream_id=0,
   )
   print(f"Blob references: {response.blob_references}")

``BlobWriteItem`` 支持 ``distribution_key`` 参数用于指定分布键，
``checksum_type`` 参数支持以下校验类型：

- ``ChecksumType.NONE`` -- 不校验（默认）
- ``ChecksumType.CRC32`` -- CRC32 校验
- ``ChecksumType.MD5`` -- MD5 校验

读取 Blob 数据
~~~~~~~~~~~~~~

使用 ``read_blobs`` 方法通过 Blob 引用下载数据。返回 ``BlobDataIterator`` 迭代器，
每次迭代返回 ``(data_bytes, mime_type)`` 元组：

.. code-block:: python

   # 批量读取 Blob
   iterator = arrow_client.read_blobs(blob_references=["ref1", "ref2"])
   for data, mime_type in iterator:
       print(f"Blob size: {len(data)}, MIME type: {mime_type}")

也可以使用 ``stream=True`` 参数获取 ``BlobStreamReader`` 进行流式读取：

.. code-block:: python

   # 流式读取 Blob
   stream_reader = arrow_client.read_blobs(
       blob_references=["ref1", "ref2"], stream=True,
   )
   while stream_reader is not None:
       chunk = stream_reader.read(4096)  # 每次读取 4KB
       if not chunk:
           # 当前 Blob 读取完毕，切换到下一个
           stream_reader = stream_reader.next()
       else:
           # 处理数据块
           process_chunk(chunk)
           print(f"Current blob MIME type: {stream_reader.mime_type}")

读取时可以指定压缩算法：

.. code-block:: python

   # 读取使用 ZSTD 压缩的 Blob
   iterator = arrow_client.read_blobs(
       blob_references=["ref1", "ref2"],
       compression=Compression.ZSTD,
   )

完整的 Blob 读写示例
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from odps.apis.storage_api_v2 import (
       StorageApiArrowClient, BlobWriteItem, ChecksumType,
   )

   arrow_client = StorageApiArrowClient(odps, blob_table)

   # 创建写会话和写流
   write_resp = arrow_client.create_write_session()
   session_id = write_resp.session_id
   arrow_client.create_write_stream(session_id, stream_id=0)

   # 写入 Blob 数据
   items = [
       BlobWriteItem(data=b"hello world", partition_values=["pt=test"]),
       BlobWriteItem(data=b"foo bar", partition_values=["pt=test"]),
   ]
   write_resp = arrow_client.write_blob_batch(
       items, session_id=session_id, stream_id=0,
   )
   blob_refs = write_resp.blob_references

   # 关闭写流并提交会话
   arrow_client.close_write_stream(session_id, stream_id=0)
   arrow_client.commit_write_session(session_id)

   # 读取 Blob 数据
   iterator = arrow_client.read_blobs(blob_references=blob_refs)
   for data, mime_type in iterator:
       print(f"Data: {data!r}, MIME type: {mime_type}")

压缩选项
--------

Storage API V2 内置支持以下压缩算法：

- ``Compression.UNCOMPRESSED`` -- 不压缩（默认）
- ``Compression.ZSTD`` -- ZSTD 压缩（需要安装 ``zstandard`` 库）
- ``Compression.LZ4_FRAME`` -- LZ4 帧格式压缩（需要安装 ``lz4`` 库）

默认情况下，读写操作不使用压缩（``Compression.UNCOMPRESSED``）。可以在读取和写入时指定压缩算法：

.. code-block:: python

   from odps.apis.storage_api_v2 import Compression

   # 写入时使用 ZSTD 压缩
   writer = arrow_client.write_rows_arrow(
       session_id, stream_id=0, record_count=100,
       compression=Compression.ZSTD,
   )

   # 读取时使用 LZ4 帧格式解压
   reader = arrow_client.read_rows_arrow(
       session_id, split_index=0,
       compression=Compression.LZ4_FRAME,
   )

数据过滤
--------

Storage API V2 支持在创建读会话时通过 ``filter_predicate`` 参数指定过滤条件，减少网络传输数据量：

.. code-block:: python

   # 使用过滤谓词
   read_resp = arrow_client.create_read_session(
       filter_predicate="id > 100 AND name = 'test'",
   )

如果谓词下推失败，可以设置 ``filter_predicate_fallback=True`` 回退到服务端过滤：

.. code-block:: python

   read_resp = arrow_client.create_read_session(
       filter_predicate="id > 100",
       filter_predicate_fallback=True,
   )

创建读会话时还可以通过 ``split_max_file_num`` 参数限制分片中的最大文件数：

.. code-block:: python

   read_resp = arrow_client.create_read_session(
       split_max_file_num=1000,
   )

Arrow 格式选项
--------------

创建读会话时可以通过 ``arrow_options`` 参数控制 Arrow 格式的精度：

.. code-block:: python

   from odps.apis.storage_api_v2 import ArrowOptions

   # 设置时间戳精度为微秒
   arrow_opts = ArrowOptions()
   arrow_opts.timestamp_unit = ArrowOptions.TimestampUnit.MICRO
   arrow_opts.date_time_unit = ArrowOptions.TimestampUnit.MILLI

   read_resp = arrow_client.create_read_session(
       arrow_options=arrow_opts,
   )

ArrowOptions 支持的精度选项：

- ``TimestampUnit.SECOND`` -- 秒
- ``TimestampUnit.MILLI`` -- 毫秒
- ``TimestampUnit.MICRO`` -- 微秒
- ``TimestampUnit.NANO`` -- 纳秒（默认）

SplitOptions 分片选项
---------------------

创建读会话时可以通过 ``split_options`` 参数控制数据的分片方式：

.. code-block:: python

   from odps.apis.storage_api_v2 import SplitOptions

   # 按大小分片（默认），每个分片 256MB
   split_opts = SplitOptions()

   # 按并行度分片，创建指定数量的分片
   split_opts = SplitOptions()
   split_opts.split_mode = SplitOptions.SplitMode.PARALLELISM
   split_opts.split_number = 10

   # 按行偏移分片
   split_opts = SplitOptions()
   split_opts.split_mode = SplitOptions.SplitMode.ROW_OFFSET
   split_opts.split_number = 100000  # 每个分片 10 万行

   # 按 Bucket ID 分片（适用于聚簇表）
   split_opts = SplitOptions()
   split_opts.split_mode = SplitOptions.SplitMode.BUCKET

   # 是否跨分区分片（默认为 True）
   split_opts = SplitOptions()
   split_opts.cross_partition = False

SplitOptions.SplitMode 支持的模式：

- ``SIZE`` -- 按数据大小分片（默认），``split_number`` 指定每个分片的大小（字节）
- ``PARALLELISM`` -- 按并行度分片，``split_number`` 指定分片数量
- ``ROW_OFFSET`` -- 按行偏移分片，``split_number`` 指定每个分片的行数
- ``BUCKET`` -- 按 Bucket ID 分片

常见场景
--------

将整张表读取为 pandas DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pyarrow as pa
   from odps.apis.storage_api_v2 import StorageApiArrowClient

   arrow_client = StorageApiArrowClient(odps, table)
   read_resp = arrow_client.create_read_session()

   batches = []
   for split_index in range(read_resp.splits_count):
       reader = arrow_client.read_rows_arrow(
           read_resp.session_id, split_index=split_index
       )
       while True:
           batch = reader.read()
           if batch is None:
               break
           batches.append(batch)

   df = pa.concat_batches(batches).to_pandas() if batches else None

从分区表读取特定分区
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   read_resp = arrow_client.create_read_session(
       required_partitions=["pt=20230101"],
   )

   for split_index in range(read_resp.splits_count):
       reader = arrow_client.read_rows_arrow(
           read_resp.session_id, split_index=split_index
       )
       while True:
           batch = reader.read()
           if batch is None:
               break
           # 处理 batch

写入 pandas DataFrame 到表
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import pyarrow as pa
   from odps.apis.storage_api_v2 import StorageApiArrowClient

   df = pd.DataFrame({
       "id": [1, 2, 3],
       "name": ["Alice", "Bob", "Carol"],
       "value": [100.0, 200.0, 150.0],
   })
   batch = pa.RecordBatch.from_pandas(df)

   arrow_client = StorageApiArrowClient(odps, table)

   # 创建写会话
   write_resp = arrow_client.create_write_session()
   session_id = write_resp.session_id

   # 创建写流
   arrow_client.create_write_stream(session_id, stream_id=0)

   # 写入数据
   writer = arrow_client.write_rows_arrow(
       session_id, stream_id=0, record_count=len(df),
   )
   writer.write(batch)
   writer.finish()

   # 关闭写流并提交会话
   arrow_client.close_write_stream(session_id, stream_id=0)
   arrow_client.commit_write_session(session_id)

读取 SQL 查询结果
~~~~~~~~~~~~~~~~~

.. code-block:: python

   instance = odps.execute_sql("SELECT * FROM your_table WHERE id > 100")
   instance_client = StorageApiArrowClient(odps, instance)

   read_resp = instance_client.create_read_session()
   for split_index in range(read_resp.splits_count):
       reader = instance_client.read_rows_arrow(
           read_resp.session_id, split_index=split_index
       )
       while True:
           batch = reader.read()
           if batch is None:
               break
           df = batch.to_pandas()
           # 处理查询结果

多进程并行读取
~~~~~~~~~~~~~~

.. code-block:: python

   from multiprocessing import Pool

   read_resp = arrow_client.create_read_session()

   def read_split(split_index):
       # 每个进程创建独立的客户端
       from odps import ODPS
       from odps.apis.storage_api_v2 import StorageApiArrowClient
       local_odps = ODPS(...)  # 初始化 ODPS
       local_client = StorageApiArrowClient(local_odps, table)
       reader = local_client.read_rows_arrow(
           read_resp.session_id, split_index=split_index
       )
       batches = []
       while True:
           batch = reader.read()
           if batch is None:
               break
           batches.append(batch.to_pandas())
       return batches

   if __name__ == "__main__":
       with Pool(processes=read_resp.splits_count) as pool:
           results = pool.map(read_split, range(read_resp.splits_count))

会话状态说明
------------

读会话状态（SessionStatus）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

========== ===========================================================
状态       说明
========== ===========================================================
INIT       会话初始化中
NORMAL     会话正常，可以读取数据
CRITICAL   会话出现严重错误
EXPIRED    会话已过期，需要刷新或重新创建
COMMITTING 写会话正在提交中
COMMITTED  写会话已提交完成
========== ===========================================================

流状态（Status）
~~~~~~~~~~~~~~~~

======== ===========================================================
状态     说明
======== ===========================================================
INIT     流初始化中
OK       流操作完成
WAIT     等待数据中
RUNNING  流正在运行中
======== ===========================================================

参考
----

完整的 API 参考文档请参见 :ref:`api_storage_v2`。
