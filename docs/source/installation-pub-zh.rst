.. _install:

**************
安装指南
**************


首先确保pip安装，pip安装可以参考 `地址 <https://pip.pypa.io/en/stable/installing/>`_ 。

接着确保setuptools和requests的版本，对于非windows（windows也可以，但要确保编译器配置正确）可以安装cython加速tunnel上传下载。

.. code-block:: sh

    pip install setuptools>=3.0
    pip install requests>=2.4.0
    pip install cython>=0.19.0  # 可选


接着就可以安装PyODPS：

.. code-block:: sh

    pip install pyodps


检查安装完成：

.. code-block:: sh

    python -c "from odps import ODPS"


如果使用的python不是系统默认的python版本，安装完pip则可以：

.. code-block:: sh

    /home/tops/bin/python2.7 -m pip install setuptools>=3.0


其余类似。