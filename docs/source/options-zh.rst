.. _options:

==============
配置选项
==============


PyODPS 提供了一系列的配置选项，可通过 ``odps.options`` 获得。下面列出了可配的 ODPS 选项。

通用配置
===============

================ ========================= =======
选项             说明	                   默认值
================ ========================= =======
access_id        ODPS Access ID            None
access_key       ODPS Access Key           None
end_point        ODPS Endpoint             None
default_project  默认 Project              None
log_view_host    LogView 主机名            None
log_view_hours   LogView 保持时间（小时）  24
tunnel_endpoint  Tunnel Endpoint           None
lifecycle        所有表生命周期            None
temp_lifecycle   临时表生命周期            1
biz_id           用户 ID                   None
chunk_size       写入缓冲区大小            1496
retry_times      请求重试次数              4
connect_timeout  连接超时                  5
read_timeout     读取超时                  120
================ ========================= =======


DataFrame 配置
==================

================ ========================= =======
选项             说明	                   默认值
================ ========================= =======
verbose          是否打印日志              False
verbose_log      日志接收器                None
df.analyze       是否启用非 ODPS 内置函数  True
================ ========================= =======


PAI 配置
==================

================= ========================= ===========
选项              说明	                   默认值
================= ========================= ===========
pai.xflow_project 默认 Xflow 工程名         algo_public
pai.parallel_num  最多并行执行作业数        5
pai.retry_times   最大重试次数              3
================= ========================= ===========
