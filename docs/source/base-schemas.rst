.. _schema:

Schema
=======

.. note::

    Schema 属于 MaxCompute 的公测功能，需要通过 `新功能测试申请 <https://help.aliyun.com/document_detail/128366.htm>`_
    开通。使用 Schema 需要 PyODPS 0.11.3 以上版本。

`Schema <https://help.aliyun.com/document_detail/437084.html>`_ 是 MaxCompute
介于项目和表 / 资源 / 函数之间的概念，对表 / 资源 / 函数进行进一步归类。

Schema 基本操作
----------------
你可以使用 ``exist_schema`` 判断 Schema 对象是否存在：

.. code-block:: python

    print(o.exist_schema("test_schema"))

使用 ``create_schema`` 创建一个 Schema 对象：

.. code-block:: python

    schema = o.create_schema("test_schema")
    print(schema)

使用 ``delete_schema`` 删除一个 Schema 对象：

.. code-block:: python

    schema = o.delete_schema("test_schema")

使用 ``get_schema`` 获得一个 Schema 对象并打印 Schema Owner：

.. code-block:: python

    schema = o.get_schema("test_schema")
    print(schema.owner)

使用 ``list_schema`` 列举所有 Schema 对象并打印名称：

.. code-block:: python

    for schema in o.list_schema():
        print(schema.name)

操作 Schema 中的对象
-------------------
在开启 Schema 后，MaxCompute 入口对象默认操作的 MaxCompute 对象都位于名为 ``DEFAULT``
的 Schema 下。为操作其他 Schema 下的对象，需要在创建入口对象时指定 Schema，例如：

.. code-block:: python

    import os
    from odps import ODPS
    # 保证 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量设置为用户 Access Key ID，
    # ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量设置为用户 Access Key Secret
    # 不建议直接使用 Access Key ID / Access Key Secret 字符串
    o = ODPS(
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        project='**your-project**',
        endpoint='**your-endpoint**',
        schema='**your-schema-name**',
    )

也可以为不同对象的操作方法指定 ``schema`` 参数。例如，下面的方法列举了 ``test_schema``
下所有的表：

.. code-block:: python

    for table in o.list_tables(schema='test_schema'):
        print(table)

下列方法给出了如何从 ``test_schema`` 获取表 ``dual`` 并输出表结构：

.. code-block:: python

    table = o.get_table('dual', schema='test_schema')
    print(table.table_schema)

在执行 SQL 时，可以指定默认 Schema：

.. code-block:: python

    o.execute_sql("SELECT * FROM dual", default_schema="test_schema")

对于表而言，如果项目空间没有启用 Schema，``get_table`` 方法对于 ``x.y`` 形式的表名，默认按照
``project.table`` 处理。如果当前租户开启了\ `租户级语法开关 <https://help.aliyun.com/zh/maxcompute/user-guide/tenant-information>`_\ ，\
``get_table`` 会将 ``x.y`` 作为 ``schema.table`` 处理，否则依然按照 ``project.table``
处理。如果租户上没有配置该选项，可以配置 ``options.enable_schema = True``，此后所有 ``x.y``
都将被作为 ``schema.table`` 处理：

.. code-block:: python

    from odps import options
    options.enable_schema = True
    print(o.get_table("myschema.mytable"))

.. note::

   ``options.enable_schema`` 自 PyODPS 0.12.0 开始支持，低版本 PyODPS 需要使用
   ``options.always_enable_schema``。
