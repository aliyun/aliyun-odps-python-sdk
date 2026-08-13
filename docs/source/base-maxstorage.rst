.. _maxstorage:

MaxStorage
==========

MaxCompute MaxStorage 是 MaxCompute 提供的高吞吐数据读写接口。相比于基于 Tunnel 的数据通道，
MaxStorage 提供了更细粒度的会话管理、支持 Arrow 格式和 Blob 格式读写、支持增量读取以及表预览等功能，
适用于大规模并行数据读写场景。

.. note::

    MaxStorage 需要服务端支持，请确保 MaxCompute 集群已开启相关功能。

基本概念
--------

MaxStorage 的核心概念包括：

- **会话（Session）**：读写操作的事务上下文。读会话管理数据分片，写会话保证数据原子性。
- **分片（Split）**：读会话将数据按大小、并行度等方式分为多个分片，每个分片可独立读取，支持并行处理。
- **流（Stream）**：写会话中的数据上传通道。一个写会话可创建多个流以支持并行写入。
- **压缩（Compression）**：读路径支持 UNCOMPRESSED（默认）、LZ4 和 ZSTD 压缩算法，写路径通过 Arrow IPC 内置压缩实现，减少网络传输数据量。
- **路由令牌（Route Token）**：服务端返回的路由标识，用于会话亲和性，确保后续请求路由到同一节点。
- **Exactly-Once 模式**：写流支持精确一次语义，通过 access_token 和 row_offset 实现幂等写入。
- **API 版本**：``api_version`` 参数（默认 ``"2"``）选择 URL 路径 ``api/storage/v2`` 或 ``api/storage/v3``，v3 特性通过 ``_supports_v3()`` 门控。

客户端初始化
------------

使用 MaxStorage 需要创建 ``MaxStorageClient`` 实例，传入 ODPS 入口对象。

.. code-block:: python

   from odps import ODPS
   from odps.maxstorage import MaxStorageClient

   # 初始化 ODPS 入口对象
   odps = ODPS(
       access_id="your_access_id",
       secret_access_key="your_secret_access_key",
       project="your_project",
       endpoint="your_endpoint",
   )

   # 创建 MaxStorage 客户端，默认 api_version="2"
   client = MaxStorageClient(odps)

   # 使用 API v3（开启 WriteMode、嵌套 Blob 等高级特性）
   client_v3 = MaxStorageClient(odps, api_version="3")

也可以通过 ``tunnel_endpoint`` 参数手动指定 Tunnel 端点，绕过自动发现：

.. code-block:: python

   client = MaxStorageClient(odps, tunnel_endpoint="http://your-tunnel-endpoint")

还可以通过 ``quota_name`` 参数指定资源配额：

.. code-block:: python

   client = MaxStorageClient(odps, quota_name="your_quota")

读取数据
--------

完整读取流程
~~~~~~~~~~~~

使用 MaxStorage 读取数据需要以下步骤：

1. 创建读会话
2. 按分片读取数据
3. 关闭读会话（会话会自动过期，无需手动关闭）

创建读会话
~~~~~~~~~~

通过 :meth:`~odps.maxstorage.MaxStorageClient.create_table_read_session` 方法创建读会话。
读会话确定了数据的分片方式、返回的列和分区等。创建后会轮询直到会话状态变为 ``NORMAL``。

.. code-block:: python

   from odps.maxstorage import MaxStorageClient, SplitOptions, SplitMode

   client = MaxStorageClient(odps)

   # 创建读会话，使用默认分片选项（按大小分片）
   read_session = client.create_table_read_session("your_table")
   print(f"Session ID: {read_session.id}")
   print(f"数据分片数: {len(read_session.splits)}")
   print(f"总记录数: {read_session.record_count}")
   print(f"过期时间: {read_session.expiration_time}")

创建读会话时可以指定需要读取的列、分区和分片选项：

.. code-block:: python

   # 仅读取指定列和分区
   read_session = client.create_table_read_session(
       "your_table",
       columns=["id", "name", "value"],
       partitions=["pt=20230101"],
   )

   # 按行偏移分片，每个分片 100 万行
   split_opts = SplitOptions(
       split_mode=SplitMode.ROW_OFFSET,
       split_number=1000000,
   )
   read_session = client.create_table_read_session(
       "your_table",
       split_options=split_opts,
   )

.. note::

   ``read_session.splits`` 返回的分片类型取决于 ``SplitMode``：
   ``SplitMode.SIZE`` 下为 :class:`~odps.maxstorage.IndexedInputSplit`（按索引寻址），
   ``SplitMode.ROW_OFFSET`` 下为 :class:`~odps.maxstorage.RowRangeInputSplit`（按行偏移区间寻址）。
   两者均可直接传给 :meth:`~odps.maxstorage.TableReadSession.open_arrow_reader`。

.. note::

    对于 append2.0 / 事务表，服务端仅支持 ``SplitMode.ROW_OFFSET`` 分片模式。

使用 Arrow 格式读取数据
~~~~~~~~~~~~~~~~~~~~~~~

通过 :meth:`~odps.maxstorage.TableReadSession.open_arrow_reader` 方法可以直接读取为 Arrow RecordBatch：

.. code-block:: python

   import pyarrow as pa

   read_session = client.create_table_read_session("your_table")

   # 遍历所有分片读取数据
   for split in read_session.splits:
       reader = read_session.open_arrow_reader(split)
       while True:
           batch = reader.read()
           if batch is None:
               break
           df = batch.to_pandas()
           # 处理 DataFrame
       reader.close()

读取时可以通过 ``max_batch_rows`` 控制每个批次的行数，通过 ``skip_row_num`` 跳过指定行数：

.. code-block:: python

   reader = read_session.open_arrow_reader(
       read_session.splits[0],
       max_batch_rows=1024,
       skip_row_num=100,
   )

也可以通过 ``get_as_record_reader()`` 获取行形式的 :class:`~odps.models.Record` 迭代器。
返回的 :class:`~odps.maxstorage.ArrowRecordReader` 除了迭代外，还暴露以下属性：

- ``schema``：返回 :class:`~odps.types.OdpsSchema`，包含列名与类型信息。
- ``count``：返回已读取的记录数，迭代结束后等于该分片的总行数。

.. code-block:: python

   reader = read_session.open_arrow_reader(read_session.splits[0])
   rr = reader.get_as_record_reader()
   print(rr.schema.columns)  # 列信息
   for record in rr:
       print(record[0], record[1])
   print(rr.count)           # 已读取的记录数

并行读取数据
~~~~~~~~~~~~

MaxStorage 的分片机制天然支持并行读取。每个分片可以独立读取，适合多线程场景：

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   read_session = client.create_table_read_session("your_table")

   def read_split(split):
       reader = read_session.open_arrow_reader(split)
       batches = []
       while True:
           batch = reader.read()
           if batch is None:
               break
           batches.append(batch)
       reader.close()
       return pa.concat_batches(batches) if batches else None

   # 使用线程池并行读取所有分片
   with ThreadPoolExecutor(max_workers=len(read_session.splits)) as pool:
       futures = [pool.submit(read_split, s) for s in read_session.splits]
       results = [f.result() for f in futures]

增量读取
~~~~~~~~

增量读取用于只读取自某个版本以来新增或变更的数据：

.. code-block:: python

   from odps.maxstorage import IncrementalReadOptions

   incr_opts = IncrementalReadOptions(
       version="v1",
       from_=100,
       to=200,
   )
   read_session = client.create_table_read_session(
       "your_table",
       incremental_read_enabled=True,
       incremental_read_options=incr_opts,
   )

压缩读取
~~~~~~~~

读路径压缩通过 ``compress_option`` 参数启用，默认为不压缩。支持 ZSTD 和 LZ4_FRAME。

.. note::

   使用压缩前需安装对应的库：

   - **ZSTD**：``pip install zstandard``
   - **LZ4**：``pip install lz4``

.. code-block:: python

   from odps.tunnel import CompressOption

   compress_option = CompressOption(
       CompressOption.CompressAlgorithm.ODPS_ZSTD,
   )
   reader = read_session.open_arrow_reader(
       read_session.splits[0],
       compress_option=compress_option,
   )

也可以使用 ``compress_algo`` 简写：

.. code-block:: python

   reader = read_session.open_arrow_reader(
       read_session.splits[0],
       compress_algo="zstd",
   )

读取 Instance 结果
~~~~~~~~~~~~~~~~~~

通过 :meth:`~odps.maxstorage.MaxStorageClient.create_instance_read_session` 可以读取 SQL Instance 的结果：

.. code-block:: python

   instance = odps.execute_sql("SELECT * FROM your_table LIMIT 100")
   instance_session = client.create_instance_read_session(instance)

   reader = instance_session.open_arrow_reader(offset=0, count=100)
   while True:
       batch = reader.read()
       if batch is None:
           break
       print(batch.to_pandas())
   reader.close()

表预览
~~~~~~

通过 :meth:`~odps.maxstorage.MaxStorageClient.preview_table` 可以快速预览表的前若干行，无需创建读会话：

.. code-block:: python

   reader = client.preview_table("your_table", limit=10)
   while True:
       batch = reader.read()
       if batch is None:
           break
       print(batch.to_pandas())
   reader.close()

写入数据
--------

完整写入流程
~~~~~~~~~~~~

使用 MaxStorage 写入数据需要以下步骤：

1. 创建写会话
2. 创建写流
3. 写入数据（批量或记录形式）
4. 提交会话

创建写会话
~~~~~~~~~~

通过 :meth:`~odps.maxstorage.MaxStorageClient.create_table_write_session` 方法创建写会话：

.. code-block:: python

   from odps.maxstorage import MaxStorageClient, WriteMode

   client = MaxStorageClient(odps)

   # 创建批处理写会话（默认）
   write_session = client.create_table_write_session("your_table")

   # 创建流式写会话
   write_session = client.create_table_write_session(
       "your_table",
       write_mode=WriteMode.STREAMING,
   )

   # 写入指定分区
   write_session = client.create_table_write_session(
       "your_table",
       partition_spec="pt=20230101",
   )

批量写入 Arrow RecordBatch
~~~~~~~~~~~~~~~~~~~~~~~~~~

通过 :meth:`~odps.maxstorage.TableWriteSession.open_arrow_writer` 创建写流，再调用 ``write_batch`` 写入：

.. code-block:: python

   import pyarrow as pa

   write_session = client.create_table_write_session("your_table")
   writer = write_session.open_arrow_writer(stream_id="0")

   schema = pa.schema([
       ("id", pa.int64()),
       ("name", pa.string()),
   ])
   batch = pa.RecordBatch.from_arrays(
       [pa.array([1, 2, 3], type=pa.int64()),
        pa.array(["a", "b", "c"], type=pa.string())],
       schema=schema,
   )
   writer.write_batch(batch)
   writer.close()
   write_session.commit()

异步刷新
~~~~~~~~

通过 :meth:`~odps.maxstorage.TableArrowWriter.flush_async` 可以异步刷新缓冲区，配合 ``max_pending_buffers`` 控制背压：

.. code-block:: python

   writer = write_session.open_arrow_writer(stream_id="0", max_pending_buffers=4)
   for batch in batches:
       writer.write_batch(batch)
       writer.flush_async()
   writer.close()
   write_session.commit()

记录形式写入
~~~~~~~~~~~~

通过 ``get_as_record_writer()`` 可以使用行形式的 :class:`~odps.models.Record` 接口写入，
无需手动构造 Arrow 批次。该方法可用于任意 :class:`TableArrowWriter`（包括普通表和
BLOB 表）。

**普通表写入**（无 BLOB 列）：

.. code-block:: python

   from odps.models import Record

   write_session = client.create_table_write_session("your_table")
   writer = write_session.open_arrow_writer(stream_id="0")
   record_writer = writer.get_as_record_writer()

   record_writer.write(Record(columns=["id", "name"], values=[1, "alice"]))
   record_writer.write(Record(columns=["id", "name"], values=[2, "bob"]))
   record_writer.close()
   write_session.commit()

**Delta 表写入**（含 ``__operation`` 列，支持 UPSERT/DELETE）：

.. code-block:: python

   from odps.models import Record

   write_session = client.create_table_write_session("delta_table")
   writer = write_session.open_arrow_writer(stream_id="0")
   record_writer = writer.get_as_record_writer()  # 返回 DeltaTableRecordWriter

   record_writer.write(Record(columns=["id", "name"], values=[1, "alice"]))  # UPSERT
   record_writer.write(Record(columns=["id", "name"], values=[2, "bob"]))    # UPSERT
   record_writer.delete(Record(columns=["id", "name"], values=[2, None]))    # DELETE
   record_writer.close()
   write_session.commit()

.. note::

    ``get_as_record_writer()`` 根据 schema 自动选择 writer 类型：若表包含
    ``__operation`` 列，返回 ``DeltaTableRecordWriter``，其 ``write()`` 方法
    标记 UPSERT、``delete()`` 方法标记 DELETE，两种操作可在同一 writer 实例上
    自由交替使用。否则返回 ``AppendTableRecordWriter``。
    BLOB 表的 Record API 写入见 :ref:`blob-auto-upload`。

压缩写入
~~~~~~~~

写路径通过 Arrow IPC 内置压缩实现，通过 ``compress_option`` 参数启用：

.. code-block:: python

   from odps.tunnel import CompressOption

   compress_option = CompressOption(
       CompressOption.CompressAlgorithm.ODPS_ZSTD,
   )
   writer = write_session.open_arrow_writer(
       stream_id="0",
       compress_option=compress_option,
   )

提交与终止
~~~~~~~~~~

写入完成后调用 :meth:`~odps.maxstorage.TableWriteSession.commit` 提交数据。
若需要丢弃数据，调用 :meth:`~odps.maxstorage.TableWriteSession.abort`：

.. code-block:: python

   # 提交
   write_session.commit()

   # 终止（丢弃所有已上传数据）
   write_session.abort()

.. note::

    ``close()`` 不会自动提交：若未显式提交或终止，``close()`` 会自动终止会话。

跨进程读写
~~~~~~~~~~

MaxStorage 的会话 ID 是服务端分配的全局标识，可以在进程间传递。这使得
"一个进程创建会话、多个进程并行读写" 的架构成为可能。

跨进程读取
^^^^^^^^^^

主进程创建读会话，将 ``session_id`` 和分片信息分发给多个工作进程。每个工作
进程通过 ``session_id`` 重载会话，独立读取各自负责的分片：

.. code-block:: python

   # ---- 主进程：创建读会话，分发分片 ----
   from odps.maxstorage import MaxStorageClient

   client = MaxStorageClient(odps)
   read_session = client.create_table_read_session("your_table")

   session_id = read_session.id          # 传递给工作进程
   splits = read_session.splits          # 分片列表，每个分片可独立读取

   # 将 session_id 和 splits 分发给工作进程（例如通过 multiprocessing.Queue、
   # Redis、文件等）


.. code-block:: python

   # ---- 工作进程：重载会话，读取指定分片 ----
   from odps.maxstorage import MaxStorageClient

   client = MaxStorageClient(odps)

   # 通过 session_id 重载已有会话（不会创建新会话）
   read_session = client.create_table_read_session("your_table", session_id=session_id)

   # 每个工作进程只读取自己负责的分片
   for split in my_splits:
       reader = read_session.open_arrow_reader(split)
       while True:
           batch = reader.read()
           if batch is None:
               break
           process(batch)
       reader.close()

跨进程写入
^^^^^^^^^^

主进程创建写会话，将 ``session_id`` 分发给多个工作进程。每个工作进程通过
``session_id`` 重载会话并以 ``resume=True`` 创建写流，独立写入数据。所有
工作进程完成后，主进程提交会话：

.. code-block:: python

   # ---- 主进程：创建写会话 ----
   from odps.maxstorage import MaxStorageClient, WriteMode

   client = MaxStorageClient(odps)
   write_session = client.create_table_write_session("your_table")
   session_id = write_session.id          # 传递给工作进程

   # 等待所有工作进程完成写入...

   # 主进程提交会话（只需 session_id 即可重载并提交）
   write_session = client.create_table_write_session("your_table", session_id=session_id)
   write_session.commit()


.. code-block:: python

   # ---- 工作进程：重载会话，resume 写入 ----
   from odps.maxstorage import MaxStorageClient

   client = MaxStorageClient(odps)
   write_session = client.create_table_write_session("your_table", session_id=session_id)

   # resume=True 调用 getWriteStream 获取已有写流的状态
   writer = write_session.open_arrow_writer(stream_id=my_stream_id, resume=True)

   writer.write_batch(my_batch)
   writer.close()   # 关闭写流（不会提交会话）

   # 工作进程只需关闭写流，提交由主进程统一完成

.. note::

- 每个工作进程使用不同的 ``stream_id``（"0"、"1"、"2"……）以并行写入同一会话。
   - 工作进程的 ``writer.close()`` 仅关闭写流，不会提交数据。必须由主进程调用
     ``write_session.commit()`` 才能使数据可见。
   - 若工作进程异常退出，主进程可调用 ``write_session.abort()`` 丢弃所有已上传数据。
   - 写会话的 ``session_id`` 重载时会自动获取最新的 ``route_token``，确保请求路由
     到正确的服务端节点。

Exactly-Once 模式
~~~~~~~~~~~~~~~~~

通过 ``exactly_once_mode=True`` 启用精确一次语义：

.. code-block:: python

   writer = write_session.open_arrow_writer(
       stream_id="0",
       exactly_once_mode=True,
   )
   # 写入时每次 flush 携带当前 row_offset，服务端返回新的 ExactlyOnceRowOffset
   writer.write_batch(batch)
   writer.flush()

   # 恢复时通过 get_row_offset() 获取最新偏移
   writer = write_session.open_arrow_writer(
       stream_id="0",
       exactly_once_mode=True,
       resume=True,
   )
   offset = writer.get_row_offset()

Blob 读写
---------

什么是 Blob
~~~~~~~~~~~

**Blob**（Binary Large Object）是 MaxCompute 中用于存储二进制数据的列类型（``BLOB``），
适用于存储图片、音频、视频、文档等非结构化数据。在 MaxStorage 中，BLOB 列的值在
Arrow 层面以 ``bytes``（或 ``pa.binary()``）表示。

什么是 Blob 引用
~~~~~~~~~~~~~~~~~

由于单条 Blob 数据可能很大，MaxStorage 不会把原始二进制内容直接内嵌在 Arrow 批次中
写入存储层。而是采用**两阶段写入**：

1. **上传 Blob 数据**：通过 ``write_blob_batch`` / ``write_blob_stream`` 将原始二进制
   数据上传到服务端的 Blob 存储区，服务端返回一个 **blob 引用**（blob reference）。
2. **写入引用**：将该 blob 引用作为 ``bytes`` 填入 Arrow 批次中 BLOB 列对应的位置，
   随普通数据一起 ``write_batch`` 提交。服务端根据引用还原出实际的二进制内容。

**blob 引用**是一个不透明的 ``bytes``（或 ``str``）值，由服务端在上传时生成，其内部
编码对客户端不可见。客户端只需将其原样写入 BLOB 列即可，**不要**尝试解码、修改或
拼接引用。读取时，BLOB 列返回的同样是 blob 引用；需要调用 ``BlobManager.read_blobs``
传入引用才能取回真正的二进制数据。

简而言之：**BLOB 列里存的是引用，不是原始二进制内容；引用是上传后拿到的 ``bytes``，
用它换回原始数据靠 ``read_blobs``。**

MaxStorage 不提供 ``Blob`` 包装类——blob 引用就是原始的 ``bytes``/``str``，
这使得它与 Arrow / pandas 等生态无缝衔接。

MaxStorage 提供两种 BLOB 写入模式：

- **手动上传**（见下文）：显式调用 ``write_blob_batch`` / ``write_blob_stream``
  上传，拿到引用后自行填入 BLOB 列。适用于需要精确控制上传时机或元数据的场景。
- **自动上传**（见 :ref:`blob-auto-upload`）：创建 writer 时指定
  ``auto_upload_blobs=True``，直接在 BLOB 列传入 ``bytes`` / 文件对象，
  writer 自动上传并替换为引用。适用于无需手动管理引用的场景。

手动上传
~~~~~~~~

在手动上传模式下，你显式调用 ``write_blob_batch`` / ``write_blob_stream`` 上传
BLOB 数据，拿到引用后自行填入 Arrow 批次的 BLOB 列。此模式适用于需要精确
控制上传时机、批量大小或元数据的场景。创建 writer 时无需 ``auto_upload_blobs``
（默认即为普通 writer，BLOB 列接受引用 ``bytes``）。

``build_blob_write_item`` 的 ``data`` 参数接受 ``bytes``、``bytearray`` 或任意可读的
文件对象（file-like，需实现 ``read``/``seek``/``tell``）。传入文件对象时支持
流式读取，无需预先将大 Blob 全量加载到内存：

.. code-block:: python

   from io import BytesIO

   # bytes / bytearray —— 直接传入
   item1 = writer.build_blob_write_item(b"image-bytes", column_name="img")
   item2 = writer.build_blob_write_item(bytearray(b"more-bytes"), column_name="img")

   # 文件对象 —— 支持流式读取，避免大 Blob 全量加载
   item3 = writer.build_blob_write_item(open("large.jpg", "rb"), column_name="img")
   item4 = writer.build_blob_write_item(BytesIO(b"in-memory-stream"), column_name="img")

.. code-block:: python

   import pyarrow as pa

   write_session = client.create_table_write_session("your_table")
   writer = write_session.open_arrow_writer(stream_id="0")

   # 1. 批量上传 Blob，拿到引用（resp.blob_references 已是 list[bytes]）
   items = [
       writer.build_blob_write_item(b"image1-bytes", column_name="img"),
       writer.build_blob_write_item(b"image2-bytes", column_name="img"),
   ]
   resp = writer.write_blob_batch(items)
   refs = resp.blob_references  # list[bytes]

   # 2. 将引用写入 BLOB 列，随普通数据一起提交
   batch = pa.RecordBatch.from_arrays(
       [pa.array([0, 1], pa.int64()), pa.array(refs, pa.binary())],
       schema=pa.schema([("id", pa.int64()), ("img", pa.binary())]),
   )
   writer.write_batch(batch)

   writer.close()
   write_session.commit()

流式上传单个 Blob：

.. code-block:: python

   import pyarrow as pa

   # 1. 流式上传单个 Blob，拿到引用（resp.blob_reference 已是 bytes）
   blob_writer = writer.write_blob_stream(column_name="img")
   with open("image.jpg", "rb") as f:
       while True:
           chunk = f.read(65536)
           if not chunk:
               break
           blob_writer.write(chunk)
   resp = blob_writer.finish()
   ref = resp.blob_reference  # bytes

   # 2. 将引用写入 BLOB 列
   batch = pa.RecordBatch.from_arrays(
       [pa.array([0], pa.int64()), pa.array([ref], pa.binary())],
       schema=pa.schema([("id", pa.int64()), ("img", pa.binary())]),
   )
   writer.write_batch(batch)

   writer.close()
   write_session.commit()

.. _blob-auto-upload:

自动上传
~~~~~~~~

在自动上传模式下，创建 writer 时指定 ``auto_upload_blobs=True``，得到
:class:`TableArrowBlobUploadWriter`。此时 BLOB 列可直接传入 ``bytes`` / 文件对象，
writer 在 ``write_batch`` 或 ``rw.write()`` 时自动批量上传每个 BLOB 单元格并
替换为引用。此模式适用于无需手动管理引用的场景。

**Arrow API 写入**（顶层 BLOB）：

.. code-block:: python

   import pyarrow as pa

   write_session = client.create_table_write_session("blob_table")
   writer = write_session.open_arrow_writer(stream_id="0", auto_upload_blobs=True)

   # BLOB 列直接传 bytes，writer 自动上传
   batch = pa.RecordBatch.from_arrays(
       [pa.array([0, 1], pa.int64()), pa.array([b"img0", b"img1"], pa.binary())],
       schema=pa.schema([("id", pa.int64()), ("img", pa.binary())]),
   )
   writer.write_batch(batch)
   writer.close()
   write_session.commit()

**Record API 写入**（无需手动构造 Arrow 批次）：

.. code-block:: python

   from io import BytesIO
   from odps.models import Record

   write_session = client.create_table_write_session("blob_table")
   writer = write_session.open_arrow_writer(stream_id="0", auto_upload_blobs=True)
   rw = writer.get_as_record_writer()

   # BLOB 列传 bytes / BytesIO / 文件对象，writer 自动批量上传
   rw.write(Record(columns=["id", "img"], values=[1, b"raw-bytes"]))
   rw.write(Record(columns=["id", "img"], values=[2, BytesIO(b"file-like")]))
   rw.close()
   write_session.commit()

传入文件对象时，writer 会读取其全部内容并**立即关闭**该文件句柄，因此调用方无需
手动关闭。

**嵌套 BLOB 写入**（``ARRAY<BLOB>``，需 API v3）：

.. code-block:: python

   from io import BytesIO

   # 嵌套 ARRAY<BLOB> 表：a BIGINT, b ARRAY<BLOB>
   client_v3 = MaxStorageClient(odps, api_version="3")
   write_session = client_v3.create_table_write_session("nested_blob_table")
   writer = write_session.open_arrow_writer(stream_id="0", auto_upload_blobs=True)
   rw = writer.get_as_record_writer()

   # b 列传 list[bytes]，每个元素自动上传
   rw.write([0, [b"blob_0_a", b"blob_0_b"]])
   rw.write([1, [BytesIO(b"blob_1_a"), b"blob_1_b", BytesIO(b"blob_1_c")]])
   rw.close()
   write_session.commit()


.. note::

   ``auto_upload_blobs=True`` 时返回 :class:`TableArrowBlobUploadWriter`，
   否则返回普通 :class:`TableArrowWriter`（BLOB 列必须已包含引用 ``bytes``）。
   两种 writer 都支持 ``get_as_record_writer()`` 和手动上传辅助方法
   （``build_blob_write_item`` / ``write_blob_stream`` / ``write_blob_batch``）。
   传入文件对象（``file-like``）写入 BLOB 时，writer 默认**不会**关闭这些文件句柄；
   若需自动关闭，创建 writer 时指定 ``auto_close_files=True``。该选项对 Arrow
   API（``write_batch``）、手动批量上传（``write_blob_batch``）与 Record
   API（``get_as_record_writer``）均生效。


读取 Blob
~~~~~~~~~

通过 :meth:`~odps.maxstorage.MaxStorageClient.open_blob_manager` 创建 ``BlobManager``，
再调用 ``read_blobs`` / ``read_blob`` 下载：

.. code-block:: python

   blob_manager = client.open_blob_manager("your_table")

   # 读取多个 Blob，返回 BlobDataIterator
   iterator = blob_manager.read_blobs([ref1, ref2])
   for record in iterator:
       print(len(record.data))  # record.data 为 bytes

   # 流式读取（避免缓冲整个 Blob）
   stream_reader = blob_manager.read_blobs([ref1], stream=True)
   while True:
       chunk = stream_reader.read(4096)
       if not chunk:
           break
       # 处理 chunk

   # 读取单个 Blob，返回 file-like 对象
   fp = blob_manager.read_blob(ref1)
   data = fp.read()

通过 Record API 读取时，BLOB 列返回引用，需 ``BlobManager`` 下载：

.. code-block:: python

   # 顶层 BLOB
   read_session = client.create_table_read_session("blob_table")
   reader = read_session.open_arrow_reader(read_session.splits[0])
   blob_manager = client.open_blob_manager("blob_table")

   for record in reader.get_as_record_reader():
       ref = record[1]              # bytes —— blob 引用
       data = next(blob_manager.read_blobs([ref])).data
       print(record[0], len(data))  # a, len(b)

   # 嵌套 ARRAY<BLOB>
   read_session = client_v3.create_table_read_session("nested_blob_table")
   reader = read_session.open_arrow_reader(read_session.splits[0])
   blob_manager = client_v3.open_blob_manager("nested_blob_table")

   for record in reader.get_as_record_reader():
       ref_list = record[1]         # list[bytes] —— 引用列表
       blobs = [b.data for b in blob_manager.read_blobs(ref_list)]
       print(record[0], [len(b) for b in blobs])


嵌套 Blob
~~~~~~~~~

当 BLOB 列嵌套在复杂类型（ARRAY、STRUCT、MAP）中时，需要通过
``find_all_blob_column_ids()`` 解析出嵌套列的服务端列 ID（dot-path 形式如 ``b.element``）。
该特性仅支持 API 版本 3 及以上（创建客户端时指定 ``api_version="3"``）。

BLOB 列在 Python / Arrow 中的表示取决于所在复杂类型。下表列出常见组合：

================================ =============================== ====================== ====================================
ODPS 类型                        Python 值                        Arrow 类型             dot-path
================================ =============================== ====================== ====================================
``BLOB``                         ``bytes``                        ``binary``             ``b``
``ARRAY<BLOB>``                  ``list[bytes]``                  ``list(binary)``       ``b.element``
``STRUCT<f: BLOB>``              ``{"f": bytes}``                 ``struct<f: binary>``  ``s.f``
``MAP<string, BLOB>``            ``{str: bytes}``                 ``map<string, binary>`` ``m.value``
``ARRAY<STRUCT<f: BLOB>>``       ``list[{"f": bytes}]``           ``list(struct<f: binary>)`` ``a.element.f``
``STRUCT<items: ARRAY<BLOB>>``   ``{"items": [bytes, ...]}``      ``struct<items: list(binary)>`` ``s.items.element``
================================ =============================== ====================== ====================================

.. note::

   dot-path 中 ``.element`` 表示 ARRAY 的元素层，``.value`` 表示 MAP 的值层，
   ``.<field_name>`` 表示 STRUCT 的字段层。多层嵌套时按从外到内的顺序拼接。

ARRAY<BLOB> 示例
^^^^^^^^^^^^^^^^

表结构：``a BIGINT, b ARRAY<BLOB>``。列 ``b`` 的每个元素是一个独立的 Blob。

**手动上传**（先批量上传，再将引用写入 Arrow）：

.. code-block:: python

   client_v3 = MaxStorageClient(odps, api_version="3")
   write_session = client_v3.create_table_write_session("nested_blob_table")
   writer = write_session.open_arrow_writer(stream_id="0")

   # 嵌套列名：array<blob> 列 b → "b.element"
   # 批量上传两行各 2~3 个 blob（resp.blob_references 已是 list[bytes]）
   items = [
       writer.build_blob_write_item(b"row0_blob0", column_name="b.element"),
       writer.build_blob_write_item(b"row0_blob1", column_name="b.element"),
       writer.build_blob_write_item(b"row1_blob0", column_name="b.element"),
   ]
   resp = writer.write_blob_batch(items)
   refs = resp.blob_references  # list[bytes]

   # 将引用按行组织成 list[bytes]，写入 Arrow list(binary) 列
   row_refs = [refs[0:2], refs[2:3]]
   batch = pa.RecordBatch.from_arrays(
       [pa.array([0, 1], pa.int64()), pa.array(row_refs, pa.list_(pa.binary()))],
       schema=pa.schema([("a", pa.int64()), ("b", pa.list_(pa.binary()))]),
   )
   writer.write_batch(batch)
   writer.close()
   write_session.commit()

读取时，``b`` 列返回 ``list[bytes]``（每个元素是一个引用），逐个下载即可：

.. code-block:: python

   read_session = client_v3.create_table_read_session("nested_blob_table")
   reader = read_session.open_arrow_reader(read_session.splits[0])
   blob_manager = client_v3.open_blob_manager("nested_blob_table")

   for batch in reader:
       for a_val, blob_ref_list in zip(batch.column(0).to_pylist(),
                                       batch.column(1).to_pylist()):
           # blob_ref_list 是 list[bytes]（引用列表）
           blobs = [b.data for b in blob_manager.read_blobs(blob_ref_list)]
           print(a_val, [len(b) for b in blobs])

STRUCT<f: BLOB> 示例
^^^^^^^^^^^^^^^^^^^^

表结构：``id BIGINT, s STRUCT<f: BLOB>``。列 ``s`` 是一个结构体，字段 ``f`` 是 Blob。

.. code-block:: python

   items = [writer.build_blob_write_item(b"struct_blob_0", column_name="s.f")]
   resp = writer.write_blob_batch(items)
   ref = resp.blob_references[0]  # bytes

   # Arrow 行：{"f": <ref bytes>}
   batch = pa.RecordBatch.from_arrays(
       [pa.array([0], pa.int64()),
        pa.array([{"f": ref}], pa.struct([("f", pa.binary())]))],
       schema=pa.schema([("id", pa.int64()),
                         ("s", pa.struct([("f", pa.binary())]))]),
   )
   writer.write_batch(batch)
   writer.close()
   write_session.commit()

读取时，``s`` 列返回 ``dict``，其 ``"f"`` 字段是引用：

.. code-block:: python

   for batch in reader:
       for id_val, s_val in zip(batch.column(0).to_pylist(),
                                batch.column(1).to_pylist()):
           ref = s_val["f"]
           data = next(blob_manager.read_blobs([ref])).data
           print(id_val, len(data))


Blob 元数据
~~~~~~~~~~~

每个 Blob 可携带两条元数据：**MIME 类型**（``mime_type``）和**自定义文件名**
（``custom_file_name``，仅 API v3）。写入时通过 ``BlobWriteItem`` 设置，
读取时从 ``BlobRecord`` / ``BlobStreamReader`` 取回。

完整流程：上传带元数据的 Blob → 写入引用 → 读取引用 → 下载并验证元数据。

.. code-block:: python

   from odps.maxstorage import MaxStorageClient

   client_v3 = MaxStorageClient(odps, api_version="3")

   # --- 写入：上传带元数据的 Blob ---
   write_session = client_v3.create_table_write_session("blob_table")
   writer = write_session.open_arrow_writer(stream_id="0")

   items = [
       writer.build_blob_write_item(
           b"\x89PNG image data",
           column_name="b",
           mime_type="image/png",
           custom_file_name="photo.png",
       ),
       writer.build_blob_write_item(
           b'{"key": "value"}',
           column_name="b",
           mime_type="application/json",
           custom_file_name="config.json",
       ),
   ]
   resp = writer.write_blob_batch(items)
   refs = resp.blob_references  # list[bytes]

   # 将引用写入 Arrow 批次
   batch = pa.RecordBatch.from_arrays(
       [pa.array([0, 1], pa.int64()), pa.array(refs, pa.binary())],
       schema=pa.schema([("a", pa.int64()), ("b", pa.binary())]),
   )
   writer.write_batch(batch)
   writer.close()
   write_session.commit()

   # --- 读取：下载 Blob 并取回元数据 ---
   read_session = client_v3.create_table_read_session("blob_table")
   reader = read_session.open_arrow_reader(read_session.splits[0])
   blob_manager = client_v3.open_blob_manager("blob_table")

   for batch in reader:
       for a_val, ref in zip(batch.column(0).to_pylist(),
                             batch.column(1).to_pylist()):
           # 方式一：BlobRecord（一次性读取，含元数据）
           record = next(blob_manager.read_blobs([ref]))
           print(record.data)              # bytes —— 原始内容
           print(record.mime_type)         # "image/png" / "application/json"
           print(record.custom_file_name)  # "photo.png" / "config.json"

           # 方式二：BlobStreamReader（流式读取，含元数据）
           stream_reader = blob_manager.read_blobs([ref], stream=True)
           print(stream_reader.mime_type)        # 元数据在读取前即可获取
           print(stream_reader.custom_file_name)
           while True:
               chunk = stream_reader.read(4096)
               if not chunk:
                   break
               # 处理 chunk
           stream_reader.next()  # 前进到下一个 Blob（若有）

.. note::

   ``custom_file_name`` 仅在 API v3 及以上可用；v2 客户端上传时该字段被静默
   忽略，下载时 ``BlobRecord.custom_file_name`` 始终为 ``None``。
   ``mime_type`` 在 v2 / v3 均可用。


逐 Blob 元数据回调
^^^^^^^^^^^^^^^^^^

上面的 "手动上传" 需要手动调用 ``write_blob_batch`` 上传 Blob 并在
``BlobWriteItem`` 上逐个设置元数据。而在**自动上传模式**下（即直接把 BLOB 原始
数据放入 Arrow 批次 ``write_batch``，或通过 Record API ``rw.write()`` 写入），
writer 自动批量上传每个 BLOB 单元格。此时可通过 ``blob_metadata_callback``
回调为**每一个 BLOB 单元格单独指定** ``mime_type`` 和 ``custom_file_name``。

回调签名：``callback(row_index, column_name, blob_data) -> (mime_type, custom_file_name) | None``

- ``row_index``：当前行索引（从 0 开始）。
- ``column_name``：BLOB 列的 dot-path（顶层列为列名如 ``"b"``，嵌套列为 ``"b.element"``、``"s.f"``）。
- ``blob_data``：原始 BLOB 数据。Record API 路径下为用户传入的原始值（``bytes`` / 文件对象），
  回调在文件被读取为 bytes 之前触发；Arrow API 路径下为已反序列化的 ``bytes``。

.. note::

   - 回调**对每个 BLOB 单元格恰好调用一次**。
   - ``None`` 值和已有的 blob 引用（``str``）不会触发回调——它们不会被上传。
   - 回调返回 ``None`` 时，回退到 ``open_arrow_writer`` 传入的 ``blob_mime_type`` /
     ``blob_custom_file_name`` 会话级默认值（也可单独使用，无需回调）；若默认值也为
     ``None``，则该 Blob 不携带元数据。
   - 流式上传（``write_blob_stream``）的 wire 协议不包含 framing header，
     因此流式路径**不支持**元数据。

示例（从文件名推导 ``custom_file_name``）：

.. code-block:: python

   import os
   from odps.models import Record

   write_session = client.create_table_write_session("blob_table")

   def metadata_fn(row_index, column_name, blob_data):
       # blob_data 是用户传入的原始文件对象，可用其 .name 推导文件名
       name = os.path.basename(getattr(blob_data, "name", f"blob_{row_index}"))
       return None, name

   writer = write_session.open_arrow_writer(
       stream_id="0",
       auto_upload_blobs=True,
       blob_metadata_callback=metadata_fn,
   )
   rw = writer.get_as_record_writer()

   rw.write(Record(columns=["a", "b"], values=[1, open("photo.jpg", "rb")]))
   rw.write(Record(columns=["a", "b"], values=[2, open("doc.pdf", "rb")]))
   rw.close()
   write_session.commit()
   # 两个 Blob 的 custom_file_name 分别为 "photo.jpg"、"doc.pdf"
