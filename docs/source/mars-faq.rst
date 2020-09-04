.. _mars-faq:

********************
Mars 使用场景和 FAQ
********************

Mars 和 PyODPS DataFrame 对比
-----------------------------

有同学会问，Mars 和 PyODPS DataFrame 有什么区别呢？

API
~~~

Mars DataFrame 的接口完全兼容 pandas。除了 DataFrame，Mars tensor 兼容 numpy，Mars learn 兼容 scikit-learn。

而 PyODPS 只有 DataFrame 接口，和 pandas 的接口存在着很多不同。

索引
~~~~

Mars DataFrame 有 pandas 索引的概念。

.. code:: ipython

    In [1]: import mars.dataframe as md

    In [5]: import mars.tensor as mt

    In [7]: df = md.DataFrame(mt.random.rand(10, 3), index=md.date_range('2020-5-1', periods=10))

    In [9]: df.loc['2020-5'].execute()
    Out[9]:
                       0         1         2
    2020-05-01  0.061912  0.507101  0.372242
    2020-05-02  0.833663  0.818519  0.943887
    2020-05-03  0.579214  0.573056  0.319786
    2020-05-04  0.476143  0.245831  0.434038
    2020-05-05  0.444866  0.465851  0.445263
    2020-05-06  0.654311  0.972639  0.443985
    2020-05-07  0.276574  0.096421  0.264799
    2020-05-08  0.106188  0.921479  0.202131
    2020-05-09  0.281736  0.465473  0.003585
    2020-05-10  0.400000  0.451150  0.956905

PyODPS 里没有索引的概念，因此跟索引有关的操作全部都不支持。

数据顺序
~~~~~~~~

Mars DataFrame 一旦创建，保证顺序，因此一些时序操作比如 ``shift``\ ，以及向前向后填空值如\ ``ffill``\ 、\ ``bfill``\ ，只有 Mars DataFrame 支持。

.. code:: ipython

    In [3]: df = md.DataFrame([[1, None], [None, 1]])

    In [4]: df.execute()
    Out[4]:
         0    1
    0  1.0  NaN
    1  NaN  1.0

    In [5]: df.ffill().execute() # 空值用上一行的值
    Out[5]:
         0    1
    0  1.0  NaN
    1  1.0  1.0

PyODPS 由于背后使用 MaxCompute 计算和存储数据，而 MaxCompute 并不保证数据顺序，所以这些操作再 MaxCompute 上都无法支持。

执行层
~~~~~~

**PyODPS 本身只是个客户端，不包含任何服务端部分。**\ PyODPS DataFrame 在真正执行时，会将计算编译到 MaxCompute SQL 执行。因此，PyODPS DataFrame 支持的操作，取决于 MaxCompute SQL 本身。此外，每一次调用 ``execute`` 方法时，会提交一次 MaxCompute 作业，需要在集群内调度。

**Mars 本身包含客户端和分布式执行层。**\ 通过调用 ``o.create_mars_cluster`` ，会在 MaxCompute 内部拉起 Mars 集群，一旦 Mars 集群拉起，后续的交互就直接和 Mars 集群进行。计算会直接提交到这个集群，调度开销极小。在数据规模不是特别大的时候，Mars 应更有优势。



使用场景指引
------------

有同学会关心，何时使用 Mars，何时使用 PyODPS DataFrame？我们分别阐述。

适合 Mars 的使用场景。
~~~~~~~~~~~~~~~~~~~~~~

-  如果你经常使用 PyODPS DataFrame 的 ``to_pandas()`` 方法，将 PyODPS DataFrame 转成 pandas DataFrame，推荐使用 Mars DataFrame。
-  Mars DataFrame 目标是完全兼容 pandas 的接口以及行为，如果你熟悉 pandas 的接口，而不愿意学习 PyODPS DataFrame 的接口，那么使用 Mars。
-  Mars DataFrame 因为兼容 pandas 的行为，因此如下的特性如果你需要用到，那么使用 Mars。

    -  Mars DataFrame 包含行和列索引，如果需要使用索引，使用 Mars。
    -  Mars DataFrame 创建后会保证顺序，通过 iloc 等接口可以获取某个偏移的数据。如 ``df.iloc[10]`` 可以获取第10行数据。此外，如 ``df.shift()`` 、\ ``df.ffill()`` 等需要有保证顺序特性的接口也在 Mars DataFrame 里得到了实现，有这方面的需求可以使用 Mars。
-  Mars 还包含 `Mars tensor <https://docs.pymars.org/zh_CN/latest/tensor/index.html>`__ 来并行和分布式化 Numpy，以及 `Mars learn <https://docs.pymars.org/zh_CN/latest/learn/index.html>`__ 来并行和分布式化 scikit-learn、以及支持在 Mars 集群里分布式运行 TensorFlow、PyTorch 和 XGBoost。有这方面的需求使用 Mars。

-  Mars 集群一旦创建，后续不再需要通过 MaxCompute 调度，任务可以直接提交到 Mars 集群执行；此外，Mars 对于中小型任务（数据量 T 级别以下），会有较好的性能。这些情况可以使用 Mars。

适合 PyODPS DataFrame 的使用场景
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  PyODPS DataFrame 会把 DataFrame 任务编译成 MaxCompute SQL 执行，如果希望依托 MaxCompute 调度任务，使用 PyODPS DataFrame。
-  PyODPS DataFrame 会编译任务到 MaxCompute 执行，由于 MaxCompute 相当稳定，而 Mars 相对比较新，如果对稳定性有很高要求，那么使用 PyODPS DataFrame。
-  数据量特别大（T 级别以上），使用 PyODPS DataFrame。

Mars 参考文档
-------------

-  Mars 开源地址：https://github.com/mars-project/mars
-  Mars 文档：https://docs.pymars.org/zh\_CN/latest/
-  Mars 团队专栏：https://zhuanlan.zhihu.com/mars-project

FAQ
---

Q：一个用户创建的 Mars 集群，别人能不能用。

A：可以，参考 :ref:`使用已经创建的 Mars 集群 <exist_cluster>`。





