.. PyOdps documentation master file, created by
   sphinx-quickstart on Wed Nov 18 09:47:14 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyOdps: ODPS Python SDK
=======================

`PyOdps <https://github.com/aliyun/aliyun-odps-python-sdk>`_ 是ODPS的Python版本的SDK，
它提供了对ODPS对象的基本操作。

安装
-------

PyOdps支持Python 2.6以上包括Python 3。系统安装了pip后，只需运行：

::

  pip install pyodps

PyOdps的相关依赖会自动安装。

快速开始
----------

首先，我们需要阿里云的帐号来初始化一个ODPS的入口：

.. code-block:: python

   from odps import ODPS

   odps = ODPS('**your-access-id**', '**your-secret-access-key**', '**your-default-project**',
               endpoint='**your-end-point**')

这样就已经初始化，就可以对表、资源、函数等进行操作了。

在主入口，我们对于主要的ODPS对象都提供了最基本的几个操作，包括 ``list``、``get``、``exist``、``create``、``delete``。

我们会对这几部分来分别展开说明。

.. toctree::
   :maxdepth: 1

   projects-zh
   tables-zh
   sql-zh
   resources-zh
   functions-zh
   tunnel-zh
   api




