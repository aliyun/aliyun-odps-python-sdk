.. _mars-introduction:

************
Mars 介绍
************

Mars 能利用并行和分布式技术，加速 Python 数据科学栈，包括 `numpy <https://numpy.org/>`__\ 、\ `pandas <https://pandas.pydata.org/>`__ 和 `scikit-learn <https://scikit-learn.org/>`__\ 。
新的 Remote API 能轻松并行执行 Python 函数。此外，也能轻松与 TensorFlow、PyTorch 和 XGBoost 集成。

`Mars tensor <https://docs.pymars.org/zh_CN/latest/getting_started/tensor.html>`__ 的接口和 numpy 保持一致，但支持大规模高维数组。样例代码如下。

.. code:: python

    import mars.tensor as mt

    a = mt.random.rand(10000, 50)
    b = mt.random.rand(50, 5000)
    a.dot(b).execute()

`Mars DataFrame <https://docs.pymars.org/zh_CN/latest/getting_started/dataframe.html>`__ 接口和 pandas 保持一致，但可以支撑大规模数据处理和分析。样例代码如下。

.. code:: python

    import mars.dataframe as md

    ratings = md.read_csv('Downloads/ml-20m/ratings.csv')
    movies = md.read_csv('Downloads/ml-20m/movies.csv')
    movie_rating = ratings.groupby('movieId', as_index=False).agg({'rating': 'mean'})
    result = movie_rating.merge(movies[['movieId', 'title']], on='movieId')
    result.sort_values(by='rating', ascending=False).execute()

`Mars learn <https://docs.pymars.org/zh_CN/latest/getting_started/learn.html>`__ 保持和 scikit-learn 接口一致。样例代码如下。

.. code:: python

    import mars.dataframe as md
    from mars.learn.neighbors import NearestNeighbors

    df = md.read_csv('data.csv')
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(df)
    neighbors = nn.kneighbors(df).fetch()

Mars learn 可以很方便地与 `TensorFlow <https://docs.pymars.org/zh_CN/latest/user_guide/learn/tensorflow.html>`__\ 、PyTorch 、 `XGBoost <https://docs.pymars.org/zh_CN/latest/user_guide/learn/xgboost.html>`__ 和
`LightGBM <https://docs.pymars.org/zh_CN/latest/user_guide/learn/lightgbm.html>`__\  集成，点击链接查看文档。

`Mars Remote <https://docs.pymars.org/zh_CN/latest/getting_started/remote.html>`__ 可以轻松并行和分布式执行一系列 Python 函数。样例代码如下：

.. code:: python

    import numpy as np
    import mars.remote as mr

    def calc_chunk(n, i):
        rs = np.random.RandomState(i)
        a = rs.uniform(-1, 1, size=(n, 2))
        d = np.linalg.norm(a, axis=1)
        return (d < 1).sum()

    def calc_pi(fs, N):
        return sum(fs) * 4 / N

    N = 200_000_000
    n = 10_000_000

    fs = [mr.spawn(calc_chunk, args=(n, i))
          for i in range(N // n)]
    pi = mr.spawn(calc_pi, args=(fs, N))
    print(pi.execute().fetch())


这里 ``calc_chunk`` 被 ``spawn`` 了 20次，这20次调用可以分布到多核，或者到集群里并行执行。最后 Mars 会自动等待这20次调用结束后触发 ``calc_pi`` 的执行。
