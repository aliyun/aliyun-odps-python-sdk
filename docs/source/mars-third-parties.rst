.. _mars-contrib:

****************
Mars 三方库集成
****************


使用 xgboost 与 lightgbm
------------------------

目前 Mars 的基础镜像中是不含 XGBoost 与 LightGBM 这两个库，所以如果想使用到 Mars 集成 `XGBoost <https://docs.pymars.org/zh_CN/latest/user_guide/learn/xgboost.html>`__ 与
`LightGBM <https://docs.pymars.org/zh_CN/latest/user_guide/learn/lightgbm.html>`__\ 的能力，
我们需要在创建集群时指定镜像类型为 ``extended``：

.. code:: python

    client = o.create_mars_cluster(1, 8, 32, mars_image='extended')

接下来就可以使用到 XGBoost 与 LightGBM 的能力，这里我们以 LightGBM 为例：

.. code:: python

    def light_gbm():
        import lightgbm
        import mars.tensor as mt
        from mars.learn.contrib.lightgbm import LGBMClassifier

        n_rows = 1000
        n_columns = 10
        chunk_size = 50
        rs = mt.random.RandomState(0)
        X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        y = rs.rand(n_rows, chunk_size=chunk_size)
        y = (y * 10).astype(mt.int32)
        classifier = LGBMClassifier(n_estimators=2)
        classifier.fit(X, y, eval_set=[(X, y)])
        prediction = classifier.predict(X)

    light_gbm()


对于 Dataworks 这种端上没有安装 LightGBM 的环境，可以使用 :ref:`Job 模式 <job_mode>` 提交 Mars 作业：

.. code:: python

    o.run_mars_job(light_gbm, mars_image='extended')


使用 faiss 加速 KNN
---------------------

在 extended 镜像中，我们也安装了 `Faiss <https://github.com/facebookresearch/faiss>`__，当数据规模比较大时，可以使用 Faiss 加速 KNN 的计算。
