.. _dbapi_odps:

DBAPI 接口
==========

.. Note:: 在 PyODPS 0.10.0 中开始支持。事务操作不被 MaxCompute 支持，因而未实现相关接口。

PyODPS 支持使用 `Python DBAPI <https://peps.python.org/pep-0249/>`_
兼容的数据库访问接口访问 MaxCompute。

创建连接
-----------
可以通过指定 ``access_id``、``access_key``、``project``和``endpoint``
来建立连接：

.. code-block:: python

    >>> import odps.dbapi
    >>> conn = odps.dbapi.connect('<access_id>', '<access_key>', '<project>', '<endpoint>')

也可以使用现有的 ODPS 入口对象来建立连接：

.. code-block:: python

    >>> import odps.dbapi
    >>> conn = odps.dbapi.connect(o)  # type(o) is ODPS

执行 SQL
----------
创建游标并在游标上执行 SQL：

.. code-block:: python

    >>> cursor = conn.cursor()
    >>> cursor.execute("SELECT * FROM pyodps_iris")
    >>> print(cursor.description)
    [('sepal_length', 'double', None, None, None, None, True),
     ('sepal_width', 'double', None, None, None, None, True),
     ('petal_length', 'double', None, None, None, None, True),
     ('petal_width', 'double', None, None, None, None, True),
     ('category', 'string', None, None, None, None, True)]

读取结果
----------
使用和标准的 DBAPI 一样使用迭代的方式读取结果：

.. code-block:: python

    >>> for rec in cursor:
    >>>     print(rec)

也可以一次性读取所有结果：

.. code-block:: python

    >>> print(cursor.fetchall())
