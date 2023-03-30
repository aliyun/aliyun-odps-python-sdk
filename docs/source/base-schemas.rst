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
你可以使用 ``create_schema`` 创建一个 Schema 对象：

.. code-block:: python

    schema = o.create_schema("test_schema")
    print(schema)

使用 ``delete_schema`` 删除一个 Schema 对象：

.. code-block:: python

    schema = o.delete_schema("test_schema")

使用 ``list_schema`` 列举所有 Schema 对象：

.. code-block:: python

    for schema in o.list_schema():
        print(schema)

操作 Schema 中的对象
-------------------
在开启 Schema 后，MaxCompute 入口对象默认操作的 MaxCompute 对象都位于名为 ``DEFAULT``
的 Schema 下。为操作其他 Schema 下的对象，需要在创建入口对象时指定 Schema，例如：

.. code-block:: python

    o = ODPS('**your-access-id**', '**your-secret-access-key**', '**your-default-project**',
             endpoint='**your-end-point**', schema='**your-schema-name**')

也可以为不同对象的操作方法指定 ``schema`` 参数。例如，下面的方法列举了 ``test_schema``
下所有的表：

.. code-block:: python

    for table in o.list_tables(schema='test_schema'):
        print(table)

在执行 SQL 时，可以指定默认 Schema：

.. code-block:: python

    o.execute_sql("SELECT * FROM dual", default_schema="test_schema")

对于表而言，如果项目空间没有启用 Schema，``get_table`` 方法对于 ``x.y`` 形式的表名，默认按照
``project.table`` 处理。如果当前租户开启了 ``odps.namespace.schema`` 配置，``get_table``
会将 ``x.y`` 作为 ``schema.table`` 处理，否则依然按照 ``project.table`` 处理。如果租户上
没有配置该选项，可以配置 ``options.always_enable_schema = True``，此后所有 ``x.y``
都将被作为 ``schema.table`` 处理：

.. code-block:: python

    from odps import options
    options.always_enable_schema = True
    print(o.get_table("myschema.mytable"))
