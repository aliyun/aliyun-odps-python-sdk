.. _functions:

函数
========

ODPS用户可以编写自定义 `函数 <https://docs.aliyun.com/#/pub/odps/basic/definition&function>`_ 用在ODPS SQL中。

基本操作
---------

可以调用 ODPS 入口对象的 ``list_functions`` 来获取项目空间下的所有函数，``exist_function`` 能判断是否存在函数，
``get_function`` 获取函数对象。

创建函数
---------

.. code-block:: python

   >>> resource = o.get_resource('my_udf.py')
   >>> function = o.create_function('test_function', class_type='my_udf.Test', resources=[resource, ])


.. note::

    注意，公共云由于安全原因，使用 Python UDF 需要申请。

删除函数
---------

.. code-block:: python

   >>> o.delete_function('test_function')
   >>> function.drop()  # Function对象存在时直接调用drop

更新函数
---------

只需对函数调用 ``update`` 方法即可。

.. code-block:: python

   >>> function = o.get_function('test_function')
   >>> new_resource = o.get_resource('my_udf2.py')
   >>> function.class_type = 'my_udf2.Test'
   >>> function.resources = [new_resource, ]
   >>> function.update()  # 更新函数

