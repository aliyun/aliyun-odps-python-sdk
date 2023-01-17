.. _pyodps_pack:

制作和使用三方包
=====================

制作三方包
--------
PyODPS 提供了 ``pyodps-pack`` 命令行工具，用于制作符合 PyODPS 及 DataWorks PyODPS 节点标准的三方包，使用方法类似
``pip`` 命令。你可以使用该工具将所有依赖项目制作成一个 ``.tar.gz`` 压缩包，其中包含所有依照 MaxCompute / DataWorks
环境编译并打包的项目依赖。如果你的项目有自行创建的 Python 包，也可以使用该工具进行打包。

准备工作
~~~~~~
你需要安装 Docker 以顺利运行 ``pyodps-pack``。对于 Linux 环境，可以参考 `Docker 官方文档 <https://docs.docker.com/engine/install/>`_
安装 Docker。对于 MacOS / Windows，个人开发者可以使用 `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ 。
对于没有购买过授权的企业用户，推荐使用开源的 `Rancher Desktop <https://rancherdesktop.io/>`_ 。我们没有在包括 ``minikube``
在内的其他 Docker 环境中测试 ``pyodps-pack`` ，不保证在这些环境中的可用性。

对于期望在版本较老的专有云中的 MaxCompute / DataWorks 使用 ``--legacy-image`` 选项打包的用户，在 Windows / MacOS
或者部分内核的 Linux 系统中可能出现无法打包的错误，请参考
`本文 <https://mail.python.org/pipermail/wheel-builders/2016-December/000239.html>`_
配置合适的打包环境。

对于 Windows 用户，可能你的 Docker 服务需要依赖 Windows 系统的 Server 服务才能启动，而 Server 服务由于安全问题在很多企业被禁止启动。
在遇到问题时，请改用 Linux 打包或者设法启用 Server 服务。Rancher Desktop 在 Windows 10 下可能无法使用 ``containerd``
作为容器引擎，可以尝试改用 ``dockerd`` 。

打包所有依赖
~~~~~~~~~
.. note::

    MaxCompute 建议除非不得已，新项目请尽量使用 Python 3。我们不保证下面的打包步骤对 Python 2 的可用性。
    旧项目在可能的情况下请尽量迁移到 Python 3 以减少后续维护的难度。

    在 Linux 中使用下列命令时，请使用 ``sudo`` 调用 ``pyodps-pack`` 以保证 Docker 正常运行。

安装完 PyODPS 后，你可以使用下面的命令为 Python 3 打包 pandas 及所有 pandas 的依赖项：

.. code-block:: bash

    pyodps-pack pandas

需要指定版本时，可以使用

.. code-block:: bash

    pyodps-pack pandas==1.2.5

经过一系列的打包步骤，工具会显示包中的所有依赖版本

.. code-block::

    Package         Version
    --------------- -------
    numpy           1.21.6
    pandas          1.2.5
    python-dateutil 2.8.2
    pytz            2022.6
    six             1.16.0

并在当前目录中生成一个 ``packages.tar.gz`` 文件，其中包括上面列出的所有依赖项目。

如果你希望为 Python 2.7 打包，请确定你的包要在 MaxCompute 还是 DataWorks 中使用。如果你不确定你的包将在哪个环境中使用，
请参考 `这篇文章 <https://developer.aliyun.com/article/704713>`_ 。如果要在 MaxCompute 中使用 Python 2.7
包，可以使用下面的打包命令：

.. code-block:: bash

    pyodps-pack --mcpy27 pandas

如果生成的 Python 2.7 包要在 DataWorks 中使用，可以使用下面的打包命令：

.. code-block:: bash

    pyodps-pack --dwpy27 pandas

打包自定义代码
~~~~~~~~~~~
``pyodps-pack`` 支持打包使用 ``setup.py`` 或者 ``pyproject.toml`` 组织的用户自定义 Python project。如果你之前从未
接触过相关知识，可以参考 `这个链接 <https://pip.pypa.io/en/stable/reference/build-system/>`_ 获取更多信息。

下面用基于 ``pyproject.toml`` 组织的项目举例介绍一下如何使用 ``pyodps-pack`` 打包。假定项目的目录结构如下：

.. code-block::

    test_package_root
    ├── test_package
    │   ├── __init__.py
    │   ├── mod1.py
    │   └── subpackage
    │       ├── __init__.py
    │       └── mod2.py
    └── pyproject.toml

其中 ``pyproject.toml`` 内容可能为

.. code-block:: toml

    [project]
    name = "test_package"
    description = "pyodps-pack example package"
    version = "0.1.0"
    dependencies = [
        "pandas>=1.0.5"
    ]

完成包的开发后，使用下面的命令可以将此包和所有依赖打包进 ``packages.tar.gz`` 文件中（ ``path_to_package``
为 ``test_package_root`` 的上级路径）：

.. code-block:: bash

    pyodps-pack /<path_to_package>/test_package_root

打包 Git Repo 中的代码
~~~~~~~~~~~~~~~~~~~
``pyodps-pack`` 支持打包远程 Git 代码仓库（例如 Github）中的代码。以 PyODPS 本身为例，可以使用下面的命令执行打包：

.. code-block:: bash

    pyodps-pack git+https://github.com/aliyun/aliyun-odps-python-sdk.git

如果想要打包某个分支或者 Tag，可以使用

.. code-block:: bash

    pyodps-pack git+https://github.com/aliyun/aliyun-odps-python-sdk.git@v0.11.2.2

如果打包前需要安装一些打包依赖（例如 ``cython``），可以使用 ``--install-requires`` 参数增加安装时依赖。
也可以编写一个格式与 ``requirements.txt`` 相同的 ``install-requires.txt`` 文件，并使用
``--install-requires-file`` 选项指定。例如，如果需要先安装 ``Cython`` 再打包 PyODPS，可以写

.. code-block:: bash

    pyodps-pack \
        --install-requires cython \
        git+https://github.com/aliyun/aliyun-odps-python-sdk.git@v0.11.2.2

也可以创建一个 ``install-requires.txt`` 文件并编写：

.. code-block::

    cython>0.29

打包命令可以写成

.. code-block:: bash

    pyodps-pack \
        --install-requires-file install-requires.txt \
        git+https://github.com/aliyun/aliyun-odps-python-sdk.git@v0.11.2.2

更复杂的例子：二进制依赖
~~~~~~~~~~~~~~~~~~
一部分包包含额外的二进制依赖，例如需要编译 / 安装的外部动态链接库。``pyodps-pack`` 提供了
``--run-before`` 参数用以指定打包前需要执行的步骤，该步骤中可以安装所需的二进制依赖。
我们用地理信息库 `GDAL <https://gdal.org/>`_ 来说明如何打包。

首先确定打包时需要安装的二进制依赖。根据 GDAL 3.6.0.1 在 `PyPI 上的文档 <https://pypi.org/project/GDAL/>`_
，我们需要安装 3.6.0 以上版本的 libgdal。 `libgdal 的编译说明 <https://gdal.org/build_hints.html#build-hints>`_
则指出，该包依赖 6.0 以上的 PROJ 包，这两个二进制包均使用 CMake 打包。据此，编写二进制包安装文件并保存为 ``install-gdal.sh``：

.. code-block:: bash

    #!/bin/bash
    set -e

    cd /tmp
    curl -o proj-6.3.2.tar.gz https://download.osgeo.org/proj/proj-6.3.2.tar.gz
    tar xzf proj-6.3.2.tar.gz
    cd proj-6.3.2
    mkdir build && cd build
    cmake ..
    cmake --build .
    cmake --build . --target install

    cd /tmp
    curl -o gdal-3.6.0.tar.gz http://download.osgeo.org/gdal/3.6.0/gdal-3.6.0.tar.gz
    tar xzf gdal-3.6.0.tar.gz
    cd gdal-3.6.0
    mkdir build && cd build
    cmake ..
    cmake --build .
    cmake --build . --target install

此后，使用 ``pyodps-pack`` 进行打包：

.. code-block:: bash

    pyodps-pack --install-requires oldest-supported-numpy --run-before install-gdal.sh gdal==3.6.0.1

命令详情
~~~~~~
下面给出 ``pyodps-pack`` 命令的可用参数，可用于控制打包过程：

.. code-block::

    -r, --requirement <file>

- 根据给定的依赖文件打包。该选项可被指定多次。

.. code-block::

    -o, --output <file>

- 指定打包生成目标文件名，默认为 ``packages.tar.gz``。

.. code-block::

    --install-requires <item>

- 指定打包时所需的 PyPI 依赖，可指定多个。这些依赖 **不一定** 会包含在最终的包中。

.. code-block::

    --install-requires-file <file>

- 指定打包时所需的 PyPI 依赖定义文件，可指定多个。这些依赖 **不一定** 会包含在最终的包中。

.. code-block::

    --run-before <script-file>

- 指定打包前需要执行的 Bash 脚本，通常可用于安装二进制依赖。

.. code-block::

    -X, --exclude <dependency>

- 指定打包时需要从最终包删除的 PyPI 依赖。该选项可被指定多次。

.. code-block::

    --no-deps

- 指定打包时不包含指定项目的依赖项。

.. code-block::

    -i, --index-url <index-url>

- 指定打包时所需的 PyPI URL。如果缺省，会使用 ``pip config list`` 命令返回的 ``global.index-url``
  值，该值通常配置在 ``pip.conf`` 配置文件中。

.. code-block::

    --trusted-host <host>

- 指定打包时需要忽略证书问题的 HTTPS 域名。

.. code-block::

    -l, --legacy-image

- 指定后，将使用 CentOS 5 镜像进行打包，这使得包可以被用在旧版专有云等环境中。

.. code-block::

    --mcpy27

- 指定后，将为 MaxCompute 内的 Python 2.7 制作三方包。如果启用，将默认 ``--legacy-image`` 选项开启。

.. code-block::

    --dwpy27

- 指定后，将为 DataWorks 内的 Python 2.7 制作三方包。如果启用，将默认 ``--legacy-image`` 选项开启。

.. code-block::

    --prefer-binary

- 指定后，将倾向于选择 PyPI 中包含二进制编译的旧版而不是仅有源码包的新版。

.. code-block::

    --debug

- 指定后，将输出命令运行的详细信息，用于排查问题。

除此之外，还有若干环境变量可供配置：

.. code-block::

    BEFORE_BUILD="command before build"

指定打包前需要执行的命令。

.. code-block::

    AFTER_BUILD="command after build"

指定编译后生成 Tar 包前需要执行的命令。

.. code-block::

    DOCKER_IMAGE="quay.io/pypa/manylinux2010_x86_64"

自定义需要使用的 Docker Image。建议基于 ``pypa/manylinux`` 系列镜像定制自定义打包用 Docker Image。

使用三方包
--------

上传三方包
~~~~~~~~
使用三方包前，请确保你生成的包被上传到 MaxCompute Archive 资源。可以使用下面的代码上传资源。
需要注意的是，你需要将 packages.tar.gz 替换成你刚生成的包所在的路径和文件名：

.. code-block:: python

    from odps import ODPS

    o = ODPS("<access_id>", "<secret_access_key>", "<project_name>", "<endpoint>")
    o.create_resource("test_packed.tar.gz", "archive", fileobj=open("packages.tar.gz", "rb"))

也可以使用 DataWorks 上传。具体步骤为：

1. 进入数据开发页面。

   a. 登录 DataWorks 控制台。
   b. 在左侧导航栏，单击工作空间列表。
   c. 选择工作空间所在地域后，单击相应工作空间后的进入数据开发。

2. 鼠标悬停至新建图标，单击MaxCompute \> 资源 \> Archive

   也可以展开业务流程目录下的目标业务流程，右键单击 MaxCompute，选择新建 \> 资源 \> Archive

3. 在新建资源对话框中，输入资源名称，并选择目标文件夹。
4. 单击点击上传，选择相应的文件进行上传。
5. 单击确定。
6. 单击工具栏中的提交图标，提交资源至调度开发服务器端。

更详细的细节请参考 `这篇文章 <https://help.aliyun.com/document_detail/136928.html>`_ 。

在 Python UDF 中使用三方包
~~~~~~~~~~~~~~~~~~~~~~~
你需要对你的 UDF 进行修改以使用上传的三方包。具体地，你需要在 UDF 类的 ``__init__`` 方法中添加对三方包的引用，
然后再在UDF代码中（例如 evaluate / process 方法）调用三方包。

我们以实现 scipy 中的 psi 函数为例展示如何在 Python UDF 中使用三方包。首先使用下面的命令打包：

.. code-block:: bash

    pyodps-pack -o scipy-bundle.tar.gz scipy

随后编写下面的代码，并保存为 ``test_psi_udf.py``：

.. code-block:: python

    import sys
    from odps.udf import annotate


    @annotate("double->double")
    class MyPsi(object):
        def __init__(self):
            # 将路径增加到引用路径
            sys.path.insert(0, "work/scipy-bundle.tar.gz/packages")

        def evaluate(self, arg0):
            # 将 import 语句保持在 evaluate 函数内部
            from scipy.special import psi

            return float(psi(arg0))

对上面的代码做一些解释。``__init__`` 函数中将 ``work/scipy-bundle.tar.gz/packages`` 添加到 ``sys.path``，
因为 MaxCompute 会将所有 UDF 引用的 Archive 资源以资源名称为目录解压到 ``work`` 目录下，而 ``packages`` 则是
``pyodps-pack`` 生成包的子目录。而将对 scipy 的 import 放在 evaluate 函数体内部的原因是三方包仅在执行时可用，当
UDF 在 MaxCompute 服务端被解析时，解析环境不包含三方包，函数体外的三方包 import 会导致报错。

随后需要将 ``test_psi_udf.py`` 上传为 MaxCompute Python 资源，以及将 ``scipy-bundle.tar.gz`` 上传为 Archive
资源。此后，创建 UDF 名为 ``test_psi_udf``，引用上面两个资源文件，并指定类名为 ``test_psi_udf.MyPsi``。

利用 PyODPS 完成上述步骤的代码为

.. code-block:: python

    from odps import ODPS

    o = ODPS("<access_id>", "<secret_access_key>", "<project_name>", "<endpoint>")
    bundle_res = o.create_resource(
        "scipy-bundle.tar.gz", "archive", fileobj=open("scipy-bundle.tar.gz", "rb")
    )
    udf_res = o.create_resource(
        "test_psi_udf.py", "py", fileobj=open("test_psi_udf.py", "rb")
    )
    o.create_function(
        "test_psi_udf", class_type="test_psi_udf.MyPsi", resources=[bundle_res, udf_res]
    )

使用 MaxCompute Console 上传的方法为

.. code-block:: sql

    add archive scipy-bundle.tar.gz;
    add py test_psi_udf.py;
    create function test_psi_udf as test_psi_udf.MyPsi using test_psi_udf.py,scipy-bundle.tar.gz;

完成上述步骤后，即可使用 UDF 执行 SQL：

.. code-block:: sql

    set odps.pypy.enabled=false;
    set odps.isolation.session.enable=true;
    select test_psi_udf(sepal_length) from iris;

在 PyODPS DataFrame 中使用三方包
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PyODPS DataFrame 支持在 execute / persist 时使用 libraries 参数使用上面的第三方库。
下面以 map 方法为例，apply / map_reduce 方法的过程类似。

首先，用下面的命令打包 scipy：

.. code-block:: bash

    pyodps-pack -o scipy-bundle.tar.gz scipy

假定我们的表名为 ``test_float_col`` ，内容只包含一列 float 值：

.. code-block::

       col1
    0  3.75
    1  2.51

计算 psi(col1) 的值，可以编写下面的代码：

.. code-block:: python

    from odps import ODPS, options

    def psi(v):
        from scipy.special import psi

        return float(psi(v))

    # 如果 Project 开启了 Isolation，下面的选项不是必需的
    options.sql.settings = {"odps.isolation.session.enable": True}

    o = ODPS("<access_id>", "<secret_access_key>", "<project_name>", "<endpoint>")
    df = o.get_table("test_float_col").to_df()
    # 直接执行并取得结果
    df.col1.map(psi).execute(libraries=["scipy-bundle.tar.gz"])
    # 保存到另一张表
    df.col1.map(psi).persist("result_table", libraries=["scipy-bundle.tar.gz"])

如果希望在整个代码执行过程中使用相同的三方包，可以设置全局选项：

.. code-block:: python

    from odps import options
    options.df.libraries = ["scipy-bundle.tar.gz"]

此后即可在 DataFrame 执行时用到相关的三方包。

在 DataWorks 中使用三方包
~~~~~~~~~~~~~~~~~~~~~~
DataWorks PyODPS 节点预装了若干三方包，同时提供了 ``load_resource_package`` 方法用以引用其他的包，
具体使用方式可参考 :ref:`这里 <dw_3rdparty_lib>`。

手动上传和使用三方包
----------------
.. note::

    以下内容仅作为维护旧项目或者旧环境的参考，新项目建议直接使用 ``pyodps-pack`` 打包。

部分旧项目可能使用了之前的方式使用三方包，即手动上传所有依赖的 Wheel 包并在代码中引用，或者使用了不支持二进制包的旧版 MaxCompute
环境，本章节为这部分场景准备。下面以在 map 中使用 python_dateutil 为例说明使用三方包的步骤。

首先，我们可以使用pip download命令，下载包以及其依赖到某个路径。
这里下载后会出现两个包：six-1.10.0-py2.py3-none-any.whl和python_dateutil-2.5.3-py2.py3-none-any.whl
（这里注意需要下载支持linux环境的包）

.. code-block:: shell

    $ pip download python-dateutil -d /to/path/

然后我们分别把两个文件上传到ODPS资源

.. code-block:: python

    # 这里要确保资源名的后缀是正确的文件类型
    odps.create_resource('six.whl', 'file', file_obj=open('six-1.10.0-py2.py3-none-any.whl', 'rb'))
    odps.create_resource('python_dateutil.whl', 'file', file_obj=open('python_dateutil-2.5.3-py2.py3-none-any.whl', 'rb'))

现在我们有个DataFrame，只有一个string类型字段。

.. code:: python

    >>> df
                   datestr
    0  2016-08-26 14:03:29
    1  2015-08-26 14:03:29

全局配置使用到的三方库：

.. code:: python

    >>> from odps import options
    >>>
    >>> def get_year(t):
    >>>     from dateutil.parser import parse
    >>>     return parse(t).strftime('%Y')
    >>>
    >>> options.df.libraries = ['six.whl', 'python_dateutil.whl']
    >>> df.datestr.map(get_year)
       datestr
    0     2016
    1     2015

或者，通过立即运行方法的 ``libraries`` 参数指定：


.. code:: python

    >>> def get_year(t):
    >>>     from dateutil.parser import parse
    >>>     return parse(t).strftime('%Y')
    >>>
    >>> df.datestr.map(get_year).execute(libraries=['six.whl', 'python_dateutil.whl'])
       datestr
    0     2016
    1     2015

PyODPS 默认支持执行纯 Python 且不含文件操作的第三方库。在较新版本的 MaxCompute 服务下，PyODPS
也支持执行带有二进制代码或带有文件操作的 Python 库。这些库的后缀必须是 cp27-cp27m-manylinux1_x86_64，
以 archive 格式上传，whl 后缀的包需要重命名为 zip。同时，作业需要开启 ``odps.isolation.session.enable``
选项，或者在 Project 级别开启 Isolation。下面的例子展示了如何上传并使用 scipy 中的特殊函数：

.. code-block:: python

    # 对于含有二进制代码的包，必须使用 Archive 方式上传资源，whl 后缀需要改为 zip
    odps.create_resource('scipy.zip', 'archive', file_obj=open('scipy-0.19.0-cp27-cp27m-manylinux1_x86_64.whl', 'rb'))

    # 如果 Project 开启了 Isolation，下面的选项不是必需的
    options.sql.settings = { 'odps.isolation.session.enable': True }

    def psi(value):
        # 建议在函数内部 import 第三方库，以防止不同操作系统下二进制包结构差异造成执行错误
        from scipy.special import psi
        return float(psi(value))

    df.float_col.map(psi).execute(libraries=['scipy.zip'])


对于只提供源码的二进制包，可以在 Linux Shell 中打包成 Wheel 再上传，Mac 和 Windows 中生成的 Wheel
包无法在 MaxCompute 中使用：

.. code-block:: shell

    python setup.py bdist_wheel