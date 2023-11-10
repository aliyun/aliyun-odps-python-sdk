.. _sqlalchemy_odps:

集成 SQLAlchemy
=================

.. Note:: 在 PyODPS 0.10.0 中开始支持

PyODPS 支持集成 SQLAlchemy，可以使用 SQLAlchemy 查询 MaxCompute 数据。

创建连接
-----------

创建连接可以在连接串中指定 ``access_id``、``access_key`` 和 ``project`` 等。

.. code-block:: python

    import os
    from sqlalchemy import create_engine

    # 确保 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量设置为用户 Access Key ID，
    # ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量设置为用户 Access Key Secret，
    # 不建议直接使用 Access Key ID / Access Key Secret 字符串
    conn_string = 'odps://%s:%s@<project>' % (
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    )
    engine = create_engine(conn_string)

要在连接串中指定 ``endpoint``，可以按如下方式：

.. code-block:: python

    import os
    from sqlalchemy import create_engine

    # 确保 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量设置为用户 Access Key ID，
    # ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量设置为用户 Access Key Secret，
    # 不建议直接使用 Access Key ID / Access Key Secret 字符串
    conn_string = 'odps://%s:%s@<project>/?endpoint=<endpoint>' % (
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    )
    engine = create_engine(conn_string)

这里把 ``<access_id>`` 等替换成相应的账号。

对于已有的 ODPS 对象 ``o`` ，调用 ``o.to_global()`` 设为全局账号后，在连接串中就不需要指定了。

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

    # 确保 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量设置为用户 Access Key ID，
    # ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量设置为用户 Access Key Secret，
    # 不建议直接使用 Access Key ID / Access Key Secret 字符串
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

    # 确保 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量设置为用户 Access Key ID，
    # ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量设置为用户 Access Key Secret，
    # 不建议直接使用 Access Key ID / Access Key Secret 字符串
    conn_string = 'odps://%s:%s@<project>/?endpoint=<endpoint>&odps.sql.hive.compatible=true' % (
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
       os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    )
    engine = create_engine(conn_string)

使用上述方式时，每个 engine 对象都会拥有不同的选项。

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
