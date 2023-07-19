.. PyODPS documentation master file, created by
   sphinx-quickstart on Wed Nov 18 09:47:14 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyODPS: ODPS Python SDK and data analysis framework
===================================================

`PyODPS <https://github.com/aliyun/aliyun-odps-python-sdk>`_ 是ODPS的Python版本的SDK，
它提供了对ODPS对象的基本操作；并提供了DataFrame框架，能轻松在ODPS上进行数据分析。

.. intinclude:: index-platform-int.rst

.. rubric:: 安装
    :class: rubric-h2

PyODPS 支持Python 2.7 以上的 Python 版本，包括Python 3。系统安装了 pip 后，只需运行：

::

  pip install pyodps

PyODPS 的相关依赖会自动安装。

**注意**，对于Linux和Mac用户，先安装Cython，再运行安装pyodps命令，能加速Tunnel的上传和下载。

安装有 `合适版本 <https://wiki.python.org/moin/WindowsCompilers>`_ Visual C++和Cython的Windows用户也可使用Tunnel加速功能。

.. _quick_start:

.. rubric:: 快速开始
    :class: rubric-h2

首先，我们需要阿里云帐号来初始化一个 ODPS 的入口（参数值请自行替换，不包含星号）：

.. code-block:: python

    import os
    from odps import ODPS
    # 确保 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量设置为用户 Access Key ID，
    # ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量设置为用户 Access Key Secret，
    # 不建议直接使用 Access Key ID / Access Key Secret 字符串
    o = ODPS(
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        project='**your-project**',
        endpoint='**your-endpoint**',
    )

这样就已经初始化，就可以对表、资源、函数等进行操作了。

如果你使用 `STS Token <https://help.aliyun.com/document_detail/112449.html>`_ 访问
ODPS，可以使用下面的语句初始化 ODPS 入口对象：

.. code-block:: python

    import os
    from odps import ODPS
    from odps.accounts import StsAccount
    # 确保 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量设置为 Access Key ID，
    # ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量设置为 Access Key Secret，
    # ALIBABA_CLOUD_STS_TOKEN 环境变量设置为 STS Token，
    # 不建议直接使用 Access Key ID / Access Key Secret 字符串
    account = StsAccount(
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        os.getenv('ALIBABA_CLOUD_STS_TOKEN'),
    )
    o = ODPS(
        account=account,
        project='**your-default-project**',
        endpoint='**your-end-point**',
    )

在主入口，我们对于主要的ODPS对象都提供了最基本的几个操作，包括 ``list``、``get``、``exist``、``create``、``delete``。

我们会对这几部分来分别展开说明。后文中的 o 对象如无说明均指的是 ODPS 入口对象。

.. toctree::
   :maxdepth: 1

   installation-int
   installation-ext
   platform
   base
   df
   interactive
   options
   pyodps-pack
   faq
   mars
   api
