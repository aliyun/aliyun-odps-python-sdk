.. _instances:

任务实例
========

Task如SQLTask是ODPS的基本计算单元，当一个Task在执行时会被实例化，
以 `ODPS实例 <https://help.aliyun.com/document_detail/27825.html>`_ 的形式存在。

基本操作
--------

可以调用 ``list_instances`` 来获取项目空间下的所有instance，``exist_instance`` 能判断是否存在某instance，
``get_instance`` 能获取实例。

.. code-block:: python

   >>> for instance in o.list_instances():
   >>>     print(instance.id)
   >>> o.exist_instance('my_instance_id')


停止一个instance可以在odps入口使用 ``stop_instance``，或者对instance对象调用 ``stop`` 方法。

.. _logview:

获取 LogView 地址
---------------

对于 SQL 等任务，通过调用 ``get_logview_address`` 方法即可。

.. code-block:: python

   >>> # 从已有的 instance 对象
   >>> instance = o.run_sql('desc pyodps_iris')
   >>> print(instance.get_logview_address())
   >>> # 从 instance id
   >>> instance = o.get_instance('2016042605520945g9k5pvyi2')
   >>> print(instance.get_logview_address())

对于 XFlow 任务，需要枚举其子任务，再获取子任务的 LogView：

.. code-block:: python

    >>> instance = o.run_xflow('AppendID', 'algo_public',
                               {'inputTableName': 'input_table', 'outputTableName': 'output_table'})
    >>> for sub_inst_name, sub_inst in o.get_xflow_sub_instances(instance).items():
    >>>     print('%s: %s' % (sub_inst_name, sub_inst.get_logview_address()))

任务实例状态
-------------

一个instance的状态可以是 ``Running``、``Suspended`` 或者 ``Terminated``，用户可以通过 ``status`` 属性来获取状态。
``is_terminated`` 方法返回当前instance是否已经执行完成，``is_successful`` 方法返回当前instance是否正确完成执行，
任务处于运行中或者执行失败都会返回False。

.. code-block:: python

   >>> instance = o.get_instance('2016042605520945g9k5pvyi2')
   >>> instance.status
   <Status.TERMINATED: 'Terminated'>
   >>> from odps.models import Instance
   >>> instance.status == Instance.Status.TERMINATED
   True
   >>> instance.status.value
   'Terminated'


调用 ``wait_for_completion`` 方法会阻塞直到instance执行完成，``wait_for_success`` 方法同样会阻塞，不同的是，
如果最终任务执行失败，则会抛出相关异常。

子任务操作
-----------

一个Instance真正运行时，可能包含一个或者多个子任务，我们称为Task，要注意这个Task不同于ODPS的计算单元。

我们可以通过 ``get_task_names`` 来获取所有的Task任务，它返回一个所有子任务的名称列表。

.. code-block:: python

   >>> instance.get_task_names()
   ['SQLDropTableTask']

拿到Task的名称，我们就可以通过 ``get_task_result`` 来获取这个Task的执行结果。
``get_task_results`` 以字典的形式返回每个Task的执行结果

.. code-block:: python

   >>> instance = o.execute_sql('select * from pyodps_iris limit 1')
   >>> instance.get_task_names()
   ['AnonymousSQLTask']
   >>> instance.get_task_result('AnonymousSQLTask')
   '"sepallength","sepalwidth","petallength","petalwidth","name"\n5.1,3.5,1.4,0.2,"Iris-setosa"\n'
   >>> instance.get_task_results()
   OrderedDict([('AnonymousSQLTask',
              '"sepallength","sepalwidth","petallength","petalwidth","name"\n5.1,3.5,1.4,0.2,"Iris-setosa"\n')])

有时候我们需要在任务实例运行时显示所有子任务的运行进程。使用 ``get_task_progress`` 能获得Task当前的运行进度。

.. code-block:: python

   >>> while not instance.is_terminated():
   >>>     for task_name in instance.get_task_names():
   >>>         print(instance.id, instance.get_task_progress(task_name).get_stage_progress_formatted_string())
   >>>     time.sleep(10)
   20160519101349613gzbzufck2 2016-05-19 18:14:03 M1_Stg1_job0:0/1/1[100%]

