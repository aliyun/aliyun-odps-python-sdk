.. _pack_minikube:

使用 Minikube 作为 Docker 环境
_______________________________
`Minikube <https://minikube.sigs.k8s.io/docs/>`_ 是一种常用的 Docker Desktop 替代环境。与 Docker Desktop 或者
Rancher Desktop 直接通过图形界面启动不同，Minikube 需要通过命令行启动并手动配置环境。

依照 `这篇文档 <https://minikube.sigs.k8s.io/docs/start/>`_ 完成安装 Minikube 后，启动 Minikube：

.. code-block:: bash

    minikube start

此后，需要手动设置 Minikube 需要的环境变量。MacOS 用户可以使用

.. code-block:: bash

    eval $(minikube -p minikube docker-env)

此后，即可在当前 Shell 会话中使用 pyodps-pack 进行打包。如果启动新的 Shell 会话，你可能需要重新配置环境变量。

对于 Windows 用户，可能需要使用 HyperV 作为默认 VM 驱动，参见 `这篇文档 <https://minikube.sigs.k8s.io/docs/drivers/hyperv/>`_：

.. code-block:: batch

    minikube start --driver=hyperv

此后为当前 Shell 配置环境变量。如果你使用的是 CMD，可以使用下面的命令：

.. code-block:: batch

    @FOR /f "tokens=*" %i IN ('minikube -p minikube docker-env --shell cmd') DO @%i

如果你使用的是 Powershell，可以使用下面的命令：

.. code-block:: powershell

    & minikube -p minikube docker-env --shell powershell | Invoke-Expression

关于如何使用 Minikube 的进一步信息请参考 `Minikube 文档 <https://minikube.sigs.k8s.io/docs/>`_ 。
