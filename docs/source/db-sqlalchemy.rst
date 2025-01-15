.. _sqlalchemy_odps:

集成 SQLAlchemy
=================

.. Note:: 在 PyODPS 0.10.0 中开始支持

PyODPS 支持集成 SQLAlchemy，可以使用 SQLAlchemy 查询 MaxCompute 数据。

创建连接
-----------

创建连接可以在连接字符串中指定 ``access_id``、``access_key`` 和 ``project`` 等。

.. code-block:: python

    import os
    from sqlalchemy import create_engine

    # 确保 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量设置为用户 Access Key ID，
    # ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量设置为用户 Access Key Secret，
    # 不建议直接使用 Access Key ID / Access Key Secret 字符串，下同
    conn_string = 'odps://%s:%s@<project>' % (
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    )
    engine = create_engine(conn_string)

要在连接字符串中指定 ``endpoint``，可以按如下方式：

.. code-block:: python

    import os
    from sqlalchemy import create_engine

    conn_string = 'odps://%s:%s@<project>/?endpoint=<endpoint>' % (
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    )
    engine = create_engine(conn_string)

这里把 ``<access_id>`` 等替换成相应的账号。

对于已有的 ODPS 对象 ``o`` ，调用 ``o.to_global()`` 设为全局账号后，在连接字符串中就不需要指定了。

.. code-block:: python

    from sqlalchemy import create_engine
    o.to_global()  # set ODPS object as global one
    engine = create_engine('odps://')

接着创建连接。

.. code-block:: python

   conn = engine.connect()

如果需要为 SQL 作业配置执行选项，可以使用 PyODPS 提供的 ``options`` 对象：

.. code-block:: python

    import os
    from odps import options
    from sqlalchemy import create_engine

    conn_string = 'odps://%s:%s@<project>/?endpoint=<endpoint>' % (
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    )
    options.sql.settings = {'odps.sql.hive.compatible': 'true'}
    engine = create_engine(conn_string)

也可以直接配置在连接字符串中：

.. code-block:: python

    import os
    from sqlalchemy import create_engine

    conn_string = 'odps://%s:%s@<project>/?endpoint=<endpoint>&odps.sql.hive.compatible=true' % (
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    )
    engine = create_engine(conn_string)

使用上述方式时，每个 engine 对象都会拥有不同的选项。

部分商业智能引擎（例如 Apache Superset）可能会频繁列举 MaxCompute 表等对象，这可能会带来较大的延迟。\
如果你在数据分析过程中对新增的 MaxCompute 对象不敏感，在 PyODPS 0.12.0 及以上版本中可以考虑为连接字符串\
增加 ``cache_names=true`` 选项以启用对象名缓存，并可指定缓存超时的时间 ``cache_seconds=<超时秒数>``
（默认为 24 * 3600）。下面的例子开启缓存并将缓存超时时间设定为 1200 秒。

.. code-block:: python

    import os
    from sqlalchemy import create_engine

    conn_string = 'odps://%s:%s@<project>/?endpoint=<endpoint>&cache_names=true&cache_seconds=1200' % (
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    )
    engine = create_engine(conn_string)

调用 SQLAlchemy 接口
----------------------

创建了连接之后，就可以正常调用 SQLAlchemy 接口。以下对建表、写入数据、查询分别举例说明。

建表
~~~~~~~

.. code-block:: python

   from sqlalchemy import Table, Column, Integer, String, MetaData
   metadata = MetaData()

   users = Table('users', metadata,
       Column('id', Integer),
       Column('name', String),
       Column('fullname', String),
   )

   metadata.create_all(engine)


写入数据
~~~~~~~~~

.. code-block:: python

   ins = users.insert().values(id=1, name='jack', fullname='Jack Jones')
   conn.execute(ins)


查询数据
~~~~~~~~~

.. code-block:: python

   >>> from sqlalchemy.sql import select
   >>> s = select([users])
   >>> result = conn.execute(s)
   >>> for row in result:
   >>>     print(row)
   (1, 'jack', 'Jack Jones')
