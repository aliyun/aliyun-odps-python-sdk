从平台到自行部署
----------------
如果你需要在本地调试 PyODPS，或者平台中没有提供你所需的包，或者平台的资源限制不能满足你的要求，此时你可能需要
从平台迁移到自己部署的 PyODPS 环境。

安装 PyODPS 的步骤可以参考 :ref:`安装指南 <install>`。你需要手动创建先前平台为你创建的 ODPS 入口对象。可以在先前的平台
使用下列语句生成创建 ODPS 入口对象所需要的语句模板，然后手动修改为可用的代码：

.. code-block:: python

    print("\nfrom odps import ODPS\no = ODPS(%r, '<access-key>', %r, '<endpoint>')\n" % (o.account.access_id, o.project))

其中，\<access-key> 和 \<endpoint> 需要手动替换成可用的值。access-key 可在 DataWorks 中点击右上角图标 -> 用户信息，
再点击“点击查看”获得。Endpoint 可在 `MaxCompute开通Region和服务连接对照表 <https://help.aliyun.com/document_detail/34951.html#h2-maxcompute-region-3>`_ 获得，
或者联系项目管理员。

修改完成后，将上述代码放置到所有代码的最顶端即可。
