.. _mars:

*****
Mars
*****


.. rubric:: 申请试用
    :class: rubric-h2

公共云用户我们采用申请开通的方式，需要使用请提工单申请。

.. rubric:: 环境准备
    :class: rubric-h2

要在 MaxCompute 上运行 Mars，需要有相应的运行环境。这可以分为如下几种情况。

1. 开箱即用的环境，如 dataworks，会包含所需要的依赖。
2. 其他环境，需要自己安装相关依赖。

我们分别展开。

.. rubric:: 开箱即用的环境
    :class: rubric-h3

开箱即用的环境，如 dataworks 的 **pyodps3 节点**\ ，已经包含了 PyODPS 和 Mars。

在新建的 pyodps3 节点里运行如下命令检查版本，确保满足要求。

.. code:: python

    from odps import __version__ as odps_version
    from mars import __version__ as mars_version

    print(odps_version)
    print(mars_version)

输出的第一个为 PyODPS 版本，第二个为 Mars 版本。要求 **PyODPS 至少是 0.9.0**\ 。


.. rubric:: 其他环境
    :class: rubric-h3

这个环境就要求通过 pip 安装 PyODPS 和 Mars。\ **Python 版本推荐使用 3.7 版本，至少需要是 3.5 版本。**

通过如下命令安装：

.. code:: bash

    pip install -U pip  # 可选，确保 pip 版本足够新
    pip install pyarrow==0.12.1  # 目前 pyarrow 版本固定到 0.12.1
    pip install pyodps>=0.9.2  # pyodps 需要至少 0.9.2
    pip install pymars>=0.4.0rc1  # mars 版本需要至少是 0.4.0rc1

.. _odps_entry:

.. rubric:: 准备 ODPS 入口
    :class: rubric-h2

ODPS 入口是 MaxCompute 所有操作的基础：

-  对于开箱即用的环境，如 dataworks，我们会自动创建 ``o`` 即 ODPS 入口实例，因此可以不需要创建。
-  对于其他环境，需要通过 ``access_id``\ 、\ ``access_key`` 等参数创建，详细参考 :ref:`快速开始 <quick_start>` 。

在 MaxCompute 上使用 Mars，我们提供了简单易用的接口来拉起 Mars 集群，用户不需要关心安装和维护集群。同时，通过 MaxCompute 拉起的 Mars，也支持直接读写 MaxCompute 表。


.. toctree::
   :maxdepth: 1

   mars-introduction
   mars-quickstart
   mars-user-guide
