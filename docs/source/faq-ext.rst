.. rubric:: 安装失败 / 出现问题

请参考 `PyODPS 安装常见问题解决 <https://yq.aliyun.com/articles/277333>`_ 。

.. rubric:: 提示 Project not found

Endpoint配置不对，详细配置参考
`MaxCompute 开通 Region 和服务连接对照表 <https://help.aliyun.com/document_detail/34951.html#h2-maxcompute-region-3>`_ 。
此外还需要注意 ODPS 入口对象参数位置是否填写正确。

.. rubric:: 如何手动指定 Tunnel Endpoint
    :name: faq_tunnel_endpoint

可以使用下面的方法创建带有 Tunnel Endpoint 的 ODPS 入口（参数值请自行替换，不包含星号）：

.. code-block:: python

   from odps import ODPS

   o = ODPS('**your-access-id**', '**your-secret-access-key**', '**your-default-project**',
            endpoint='**your-end-point**', tunnel_endpoint='**your-tunnel-endpoint**')
