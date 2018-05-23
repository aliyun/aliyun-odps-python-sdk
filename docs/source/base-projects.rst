.. _projects:

项目空间
=========

`项目空间 <https://docs.aliyun.com/#/pub/odps/basic/definition&project>`_ 是ODPS的基本组织单元，
有点类似于Database的概念。

我们通过 ODPS 入口对象的 ``get_project`` 来取到某个项目空间。

.. code-block:: python

   project = o.get_project('my_project')  # 取到某个项目
   project = o.get_project()              # 取到默认项目

如果不提供参数，则取到默认项目空间。

``exist_project`` 方法能检验某个项目空间是否存在。
