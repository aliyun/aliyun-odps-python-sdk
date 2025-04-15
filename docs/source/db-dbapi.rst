.. _dbapi_odps:

DBAPI 接口
==========

.. Note:: 在 PyODPS 0.10.0 中开始支持。事务操作不被 MaxCompute 支持，因而未实现相关接口。

PyODPS 支持使用 `Python DBAPI <https://peps.python.org/pep-0249/>`_
兼容的数据库访问接口访问 MaxCompute。

创建连接
-----------
可以通过指定 ``access_id``、\ ``access_key``、\ ``project``\ 和\ ``endpoint``
来建立连接：

.. code-block:: python

    >>> import odps.dbapi
    >>> conn = odps.dbapi.connect('<access_id>', '<access_key>', '<project>', '<endpoint>')

也可以使用现有的 ODPS 入口对象来建立连接：

.. code-block:: python

    >>> import odps.dbapi
    >>> conn = odps.dbapi.connect(o)  # type(o) is ODPS

如果要使用 MaxQA（MCQA 2.0）查询加速，可以指定 ``use_sqa='v2'`` 并指定 ``quota_name`` 参数：

.. code-block:: python

    >>> conn = odps.dbapi.connect(o, use_sqa='v2', quota_name='my_quota')

执行 SQL
----------
你可以创建游标并在游标上执行 SQL：

.. code-block:: python

    >>> cursor = conn.cursor()
    >>> cursor.execute("SELECT * FROM pyodps_iris")
    >>> print(cursor.description)
    [('sepal_length', 'double', None, None, None, None, True),
     ('sepal_width', 'double', None, None, None, None, True),
     ('petal_length', 'double', None, None, None, None, True),
     ('petal_width', 'double', None, None, None, None, True),
     ('category', 'string', None, None, None, None, True)]

PyODPS 使用与标准库 ``sqlite3`` 一样的参数指定方式指定查询参数。你可以使用 ``?`` 指定非命名参数，
使用 ``:name`` 指定命名参数，并使用 tuple 替换非命名参数，使用 dict 替换命名参数。

.. code-block:: python

    >>> # 使用非命名参数
    >>> cursor.execute("SELECT * FROM pyodps_iris WHERE petal_length > ?", (1.5,))
    >>> print(cursor.fetchone())
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']
    >>> # 使用命名参数
    >>> cursor.execute("SELECT * FROM pyodps_iris WHERE petal_length > :length", {'length': 1.5})
    >>> print(cursor.fetchone())
    [5.4, 3.9, 1.7, 0.4, 'Iris-setosa']

读取结果
----------
你可以使用和标准的 DBAPI 一样使用迭代的方式读取结果：

.. code-block:: python

    >>> for rec in cursor:
    >>>     print(rec)

也可以一次性读取所有结果：

.. code-block:: python

    >>> print(cursor.fetchall())
