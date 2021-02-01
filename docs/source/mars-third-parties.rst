.. _mars-contrib:

****************
Mars 三方库集成
****************


基础镜像
--------

Mars 的基础镜像中安装了一些常见的三方库，可以直接在 remote 函数中使用。

.. code:: shell

    #pip list
    Package                   Version
    ------------------------- -------------------
    aliyun-python-sdk-core    2.13.25
    aliyun-python-sdk-core-v3 2.13.11
    aliyun-python-sdk-kms     2.11.0
    attrs                     19.3.0
    backcall                  0.2.0
    bleach                    3.1.5
    bokeh                     2.1.1
    cachetools                4.1.1
    certifi                   2020.6.20
    cffi                      1.14.0
    chardet                   3.0.4
    cloudpickle               1.5.0
    conda                     4.8.3
    conda-package-handling    1.7.0
    crcmod                    1.7
    cryptography              2.9.2
    cycler                    0.10.0
    Cython                    0.29.21
    decorator                 4.4.2
    defusedxml                0.6.0
    entrypoints               0.3
    gevent                    20.6.2
    google-auth               1.20.0
    greenlet                  0.4.16
    idna                      2.9
    importlib-metadata        1.7.0
    ipykernel                 5.3.4
    ipython                   7.16.1
    ipython-genutils          0.2.0
    ipywidgets                7.5.1
    jedi                      0.17.2
    Jinja2                    2.11.2
    jmespath                  0.10.0
    joblib                    0.16.0
    jsonschema                3.2.0
    jupyter                   1.0.0
    jupyter-client            6.1.6
    jupyter-console           6.1.0
    jupyter-core              4.6.3
    kiwisolver                1.2.0
    kubernetes                11.0.0
    llvmlite                  0.33.0+1.g022ab0f
    lz4                       3.1.0
    MarkupSafe                1.1.1
    matplotlib                3.2.2
    mistune                   0.8.4
    mkl-fft                   1.1.0
    mkl-random                1.1.1
    mkl-service               2.3.0
    nbconvert                 5.6.1
    nbformat                  5.0.7
    notebook                  6.0.3
    numba                     0.50.1
    numexpr                   2.7.1
    numpy                     1.19.1
    oauthlib                  3.1.0
    olefile                   0.46
    oss2                      2.12.1
    packaging                 20.4
    pandas                    1.1.4
    pandocfilters             1.4.2
    parso                     0.7.1
    pexpect                   4.8.0
    pickleshare               0.7.5
    Pillow                    7.2.0
    pip                       20.0.2
    prometheus-client         0.8.0
    prompt-toolkit            3.0.5
    protobuf                  3.12.3
    psutil                    5.7.0
    ptyprocess                0.6.0
    pyasn1                    0.4.8
    pyasn1-modules            0.2.8
    pycosat                   0.6.3
    pycparser                 2.20
    pycryptodome              3.9.8
    Pygments                  2.6.1
    pyOpenSSL                 19.1.0
    pyparsing                 2.4.7
    pyrsistent                0.16.0
    PySocks                   1.7.1
    python-dateutil           2.8.1
    pytz                      2020.1
    PyYAML                    5.3.1
    pyzmq                     19.0.1
    qtconsole                 4.7.5
    QtPy                      1.9.0
    requests                  2.23.0
    requests-oauthlib         1.3.0
    rsa                       4.6
    ruamel-yaml               0.15.87
    scikit-learn              0.23.1
    scipy                     1.5.0
    Send2Trash                1.5.0
    setuptools                46.4.0.post20200518
    six                       1.14.0
    SQLAlchemy                1.3.18
    terminado                 0.8.3
    testpath                  0.4.4
    threadpoolctl             2.1.0
    tornado                   6.0.4
    tqdm                      4.46.0
    traitlets                 4.3.3
    typing-extensions         3.7.4.2
    urllib3                   1.25.8
    wcwidth                   0.2.5
    webencodings              0.5.1
    websocket-client          0.57.0
    wheel                     0.34.2
    widgetsnbextension        3.5.1
    zipp                      3.1.0
    zope.event                4.4
    zope.interface            5.1.0


使用 extended 镜像
------------------

目前 Mars 的基础镜像中是不含 XGBoost 与 LightGBM 等这些库，所以如果想使用到 Mars 集成 `XGBoost <https://docs.pymars.org/zh_CN/latest/user_guide/learn/xgboost.html>`__ 与
`LightGBM <https://docs.pymars.org/zh_CN/latest/user_guide/learn/lightgbm.html>`__\ 的能力，
我们需要在创建集群时指定镜像类型为 ``extended``：

.. code:: python

    client = o.create_mars_cluster(1, 8, 32, image='extended')

extended 镜像中的三方库以及版本如下：

.. code:: shell

    #pip list
    Package                   Version
    ------------------------- -------------------
    aliyun-python-sdk-core    2.13.25
    aliyun-python-sdk-core-v3 2.13.11
    aliyun-python-sdk-kms     2.11.0
    attrs                     19.3.0
    backcall                  0.2.0
    bleach                    3.1.5
    bokeh                     2.1.1
    cachetools                4.1.1
    certifi                   2020.11.8
    cffi                      1.14.0
    chardet                   3.0.4
    cloudpickle               1.5.0
    conda                     4.9.2
    conda-package-handling    1.7.0
    crcmod                    1.7
    cryptography              2.9.2
    cycler                    0.10.0
    Cython                    0.29.21
    decorator                 4.4.2
    defusedxml                0.6.0
    entrypoints               0.3
    faiss                     1.6.3
    gevent                    20.6.2
    google-auth               1.20.0
    greenlet                  0.4.16
    idna                      2.9
    importlib-metadata        1.7.0
    ipykernel                 5.3.4
    ipython                   7.16.1
    ipython-genutils          0.2.0
    ipywidgets                7.5.1
    jedi                      0.17.2
    Jinja2                    2.11.2
    jmespath                  0.10.0
    joblib                    0.16.0
    jsonschema                3.2.0
    jupyter                   1.0.0
    jupyter-client            6.1.6
    jupyter-console           6.1.0
    jupyter-core              4.6.3
    kiwisolver                1.2.0
    kubernetes                11.0.0
    lightgbm                  2.3.0
    llvmlite                  0.33.0+1.g022ab0f
    lz4                       3.1.0
    MarkupSafe                1.1.1
    matplotlib                3.2.2
    mistune                   0.8.4
    mkl-fft                   1.1.0
    mkl-random                1.1.1
    mkl-service               2.3.0
    nbconvert                 5.6.1
    nbformat                  5.0.7
    notebook                  6.0.3
    numba                     0.50.1
    numexpr                   2.7.1
    numpy                     1.19.1
    oauthlib                  3.1.0
    olefile                   0.46
    oss2                      2.12.1
    packaging                 20.4
    pandas                    1.1.4
    pandocfilters             1.4.2
    parso                     0.7.1
    patsy                     0.5.1
    pexpect                   4.8.0
    pickleshare               0.7.5
    Pillow                    7.2.0
    pip                       20.0.2
    prometheus-client         0.8.0
    prompt-toolkit            3.0.5
    protobuf                  3.12.3
    psutil                    5.7.0
    ptyprocess                0.6.0
    pyasn1                    0.4.8
    pyasn1-modules            0.2.8
    pycosat                   0.6.3
    pycparser                 2.20
    pycryptodome              3.9.8
    Pygments                  2.6.1
    pyOpenSSL                 19.1.0
    pyparsing                 2.4.7
    pyrsistent                0.16.0
    PySocks                   1.7.1
    python-dateutil           2.8.1
    pytz                      2020.1
    PyYAML                    5.3.1
    pyzmq                     19.0.1
    qtconsole                 4.7.5
    QtPy                      1.9.0
    requests                  2.23.0
    requests-oauthlib         1.3.0
    rsa                       4.6
    ruamel-yaml               0.15.87
    scikit-learn              0.23.1
    scipy                     1.5.0
    Send2Trash                1.5.0
    setuptools                46.4.0.post20200518
    shap                      0.37.0
    six                       1.14.0
    slicer                    0.0.3
    SQLAlchemy                1.3.18
    statsmodels               0.12.1
    terminado                 0.8.3
    testpath                  0.4.4
    threadpoolctl             2.1.0
    tornado                   6.0.4
    tqdm                      4.46.0
    traitlets                 4.3.3
    typing-extensions         3.7.4.2
    urllib3                   1.25.8
    wcwidth                   0.2.5
    webencodings              0.5.1
    websocket-client          0.57.0
    wheel                     0.34.2
    widgetsnbextension        3.5.1
    xgboost                   1.2.0
    zipp                      3.1.0
    zope.event                4.4
    zope.interface            5.1.0

使用 xgboost 与 lightgbm
~~~~~~~~~~~~~~~~~~~~~~~~~

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

    o.run_mars_job(light_gbm, image='extended')


使用 faiss 加速 KNN
~~~~~~~~~~~~~~~~~~~

在 extended 镜像中，我们也安装了 `Faiss <https://github.com/facebookresearch/faiss>`__，当数据规模比较大时，可以使用 Faiss 加速 KNN 的计算。


使用 tensorflow 镜像
--------------------

除了 extended 镜像，我们也提供了 tensorflow 镜像，可以使用到 `Mars 集成 TensorFlow 的能力 <https://docs.pymars.org/en/latest/user_guide/learn/tensorflow.html>`__

我们需要在创建集群时指定镜像类型为 ``tensorflow``：

.. code:: python

    client = o.create_mars_cluster(4, 8, 32, image='tensorflow')

tensorflow 镜像中的三方库以及版本如下：

.. code:: shell

    #pip list
    Package                   Version
    ------------------------- -------------------
    absl-py                   0.11.0
    aliyun-python-sdk-core    2.13.25
    aliyun-python-sdk-core-v3 2.13.11
    aliyun-python-sdk-kms     2.11.0
    astor                     0.8.1
    attrs                     19.3.0
    backcall                  0.2.0
    bleach                    3.1.5
    bokeh                     2.1.1
    cachetools                4.1.1
    certifi                   2020.6.20
    cffi                      1.14.0
    chardet                   3.0.4
    cloudpickle               1.5.0
    conda                     4.9.2
    conda-package-handling    1.7.0
    crcmod                    1.7
    cryptography              2.9.2
    cycler                    0.10.0
    Cython                    0.29.21
    decorator                 4.4.2
    deepctr                   0.8.2
    deepmatch                 0.2.0
    defusedxml                0.6.0
    entrypoints               0.3
    gast                      0.4.0
    gensim                    3.8.3
    gevent                    20.6.2
    google-auth               1.20.0
    greenlet                  0.4.16
    grpcio                    1.31.0
    h5py                      2.10.0
    idna                      2.9
    importlib-metadata        2.0.0
    ipykernel                 5.3.4
    ipython                   7.16.1
    ipython-genutils          0.2.0
    ipywidgets                7.5.1
    jedi                      0.17.2
    jieba                     0.42.1
    Jinja2                    2.11.2
    jmespath                  0.10.0
    joblib                    0.16.0
    jsonschema                3.2.0
    jupyter                   1.0.0
    jupyter-client            6.1.6
    jupyter-console           6.1.0
    jupyter-core              4.6.3
    Keras                     2.2.4
    Keras-Applications        1.0.8
    Keras-Preprocessing       1.1.0
    kiwisolver                1.2.0
    kubernetes                11.0.0
    llvmlite                  0.33.0+1.g022ab0f
    lz4                       3.1.0
    Markdown                  3.3.3
    MarkupSafe                1.1.1
    matplotlib                3.2.2
    mistune                   0.8.4
    mkl-fft                   1.1.0
    mkl-random                1.1.1
    mkl-service               2.3.0
    mock                      4.0.2
    nbconvert                 5.6.1
    nbformat                  5.0.7
    notebook                  6.0.3
    numba                     0.50.1
    numexpr                   2.7.1
    numpy                     1.19.1
    oauthlib                  3.1.0
    olefile                   0.46
    oss2                      2.12.1
    packaging                 20.4
    pandas                    1.1.4
    pandocfilters             1.4.2
    parso                     0.7.1
    pexpect                   4.8.0
    pickleshare               0.7.5
    Pillow                    7.2.0
    pip                       20.0.2
    prometheus-client         0.8.0
    prompt-toolkit            3.0.5
    protobuf                  3.12.3
    psutil                    5.7.0
    ptyprocess                0.6.0
    pyasn1                    0.4.8
    pyasn1-modules            0.2.8
    pycosat                   0.6.3
    pycparser                 2.20
    pycryptodome              3.9.8
    Pygments                  2.6.1
    pyOpenSSL                 19.1.0
    pyparsing                 2.4.7
    pyrsistent                0.16.0
    PySocks                   1.7.1
    python-dateutil           2.8.1
    pytz                      2020.1
    PyYAML                    5.3.1
    pyzmq                     19.0.1
    qtconsole                 4.7.5
    QtPy                      1.9.0
    requests                  2.23.0
    requests-oauthlib         1.3.0
    rsa                       4.6
    ruamel-yaml               0.15.87
    scikit-learn              0.23.1
    scipy                     1.5.0
    Send2Trash                1.5.0
    setuptools                46.4.0.post20200518
    six                       1.14.0
    smart-open                3.0.0
    SQLAlchemy                1.3.18
    tensorboard               1.13.1
    tensorflow                1.13.1
    tensorflow-estimator      1.13.0
    termcolor                 1.1.0
    terminado                 0.8.3
    testpath                  0.4.4
    threadpoolctl             2.1.0
    tornado                   6.0.4
    tqdm                      4.46.0
    traitlets                 4.3.3
    typing-extensions         3.7.4.2
    urllib3                   1.25.8
    wcwidth                   0.2.5
    webencodings              0.5.1
    websocket-client          0.57.0
    Werkzeug                  1.0.1
    wheel                     0.34.2
    widgetsnbextension        3.5.1
    zipp                      3.4.0
    zope.event                4.4
    zope.interface            5.1.0


