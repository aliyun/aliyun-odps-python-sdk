.. _models:

模型
========

ODPS 提供了两种模型。一种模型为离线模型（OfflineModel），另一种模型为在线模型（OnlineModel）。离线模型为分类 / 回归算法使用
训练数据得到的模型，而在线模型是部署离线模型或自定义流程从而形成的在线服务。

离线模型
---------

离线模型是 XFlow 分类 / 回归算法输出的模型。用户可以使用 PyODPS ML 或直接使用 odps.run_xflow 创建一个离线模型，例如下面使用
run_xflow 的例子：

.. code:: python

    >>> o.run_xflow('LogisticRegression', 'algo_public', dict(modelName='logistic_regression_model_name',
    >>>                regularizedLevel='1', maxIter='100', regularizedType='l1', epsilon='0.000001', labelColName='y',
    >>>                featureColNames='pdays,emp_var_rate', goodValue='1', inputTableName='bank_data'))

在模型创建后，用户可以列出当前 Project 下的模型：

.. code:: python

    >>> models = o.list_offline_models(prefix='prefix')

也可以通过模型名获取模型并读取模型 PMML（如果支持）：

.. code:: python

    >>> model = o.get_offline_model('logistic_regression_model_name')
    >>> pmml = model.get_model()

删除模型可使用下列语句：

.. code:: python

    >>> o.delete_offline_model('logistic_regression_model_name')

在线模型
---------

在线模型是 ODPS 提供的模型在线部署能力。用户可以通过 Pipeline 部署自己的模型。详细信息请参考“机器学习平台——在线服务”章节。

需要注意的是，在线模型的服务使用的是独立的 Endpoint，需要配置 Predict Endpoint。通过

.. code:: python

    >>> o = ODPS('your-access-id', 'your-secret-access-key', 'your-default-project',
    >>>          endpoint='your-end-point', predict_endpoint='predict_endpoint')

即可在 ODPS 对象上添加相关配置。Predict Endpoint 的地址请参考相关说明或咨询管理员。

部署离线模型上线
~~~~~~~~~~~~~~~~

PyODPS 提供了离线模型的部署功能。部署方法为

.. code:: python

    >>> model = o.create_online_model('online_model_name', 'offline_model_name')


部署自定义 Pipeline 上线
~~~~~~~~~~~~~~~~~~~~~~~~
含有自定义 Pipeline 的在线模型可自行构造 ModelPredictor 对象，例子如下：

.. code:: python

    >>> from odps.models.ml import ModelPredictor, ModelProcessor, BuiltinProcessor, PmmlProcessor, PmmlRunMode
    >>> predictor = ModelPredictor(target_name='label')
    >>> predictor.pipeline.append(BuiltinProcessor(offline_model_name='sample_offlinemodel',
    >>>                                            offline_model_project='online_test'))
    >>> predictor.pipeline.append(PmmlProcessor(pmml='data_preprocess.xml',
    >>>                                         resources='online_test/resources/data_preprocess.xml',
    >>>                                         run_mode=PmmlRunMode.Converter))
    >>> predictor.pipeline.append(CustomProcessor(class_name='SampleProcessor',
    >>>                                           lib='libsample_processor.so',
    >>>                                           resources='online_test/resources/sample_processor.tar.gz'))
    >>> model = o.create_online_model('online_model_name', predictor)

其中，BuiltinProcessor、PmmlProcessor 和 CustomProcessor 分别指 ODPS OfflineModel 形成的 Pipeline 节点、PMML
模型文件形成的 Pipeline 节点和用户自行开发的 Pipeline 节点。

在线模型操作
~~~~~~~~~~~~

与其他 ODPS 对象类似，创建后，可列举、获取和删除在线模型：

.. code:: python

    >>> models = o.list_online_models(prefix='prefix')
    >>> model = o.get_online_model('online_model_name')
    >>> o.delete_online_model('online_model_name')

可使用模型名和数据进行在线预测，输入数据可以是 Record，也可以是字典或数组和 Schema 的组合：

.. code:: python

    >>> data = [[4, 3, 2, 1], [1, 2, 3, 4]]
    >>> result = o.predict_online_model('online_model_name', data,
    >>>                                 schema=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

也可为模型设置 ABTest。参数中的 modelx 可以是在线模型名，也可以是 get_online_model 获得的模型对象本身，而 percentagex 表示
modelx 在 ABTest 中所占的百分比，范围为 0 至 100：

.. code:: python

    >>> result = o.config_online_model_ab_test('online_model_name', model1, percentage1, model2, percentage2)

修改模型参数可以通过修改 OnlineModel 对象的属性，再调用 update 方法实现，如

.. code:: python

    >>> model = o.get_online_model('online_model_name')
    >>> model.cpu = 200
    >>> model.update()

与其他对象不同的是，在线模型的创建和删除较为耗时。PyODPS 默认 create_online_model 和 delete_online_model 以及
OnlineModel 的 update 方法在整个操作完成后才返回。用户可以通过 async 选项控制是否要在模型创建请求提交后立即返回，
然后自己控制等待。例如，下列语句

.. code:: python

    >>> model = o.create_online_model('online_model_name', 'offline_model_name')

等价于

.. code:: python

    >>> model = o.create_online_model('online_model_name', 'offline_model_name', async=True)
    >>> model.wait_for_service()

而

.. code:: python

    >>> o.delete_online_model('online_model_name')

等价于

.. code:: python

    >>> o.delete_online_model('* online_model_name *', async=True)
    >>> model.wait_for_deletion()
