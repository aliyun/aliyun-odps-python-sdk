.. _options:

==============
配置选项
==============


PyODPS 提供了一系列的配置选项，可通过 ``odps.options`` 获得，如下面的例子：

.. code-block:: python

    from odps import options
    # 设置所有输出表的生命周期（lifecycle 选项）
    options.lifecycle = 30
    # 使用 Tunnel 下载 string 类型时使用 bytes（tunnel.string_as_binary 选项）
    options.tunnel.string_as_binary = True
    # PyODPS DataFrame 用 ODPS 执行时，参照下面 dataframe 相关配置，sort 时设置 limit 到一个比较大的值
    options.df.odps.sort.limit = 100000000

下面列出了可配的 ODPS 选项。

通用配置
===============

.. csv-table::
   :header-rows: 1

   "选项", "说明", "默认值"
   "endpoint", "ODPS Endpoint", "None"
   "default_project", "默认 Project", "None"
   "logview_host", "LogView 主机名", "None"
   "logview_hours", "LogView 保持时间（小时）", "24"
   "use_legacy_logview", "使用旧版 LogView 地址，None 表示自动根据 Endpoint 处理", "None"
   "quota_name", "提交任务时使用的计算 Quota 名称", "None"
   "local_timezone", "使用的时区，None 表示不处理，True 表示本地时区，False 表示 UTC，也可用 pytz 的时区", "None"
   "lifecycle", "所有表生命周期", "None"
   "verify_ssl", "验证服务端 SSL 证书", "True"
   "temp_lifecycle", "临时表生命周期", "1"
   "biz_id", "用户 ID", "None"
   "verbose", "是否打印日志", "False"
   "verbose_log", "日志接收器", "None "
   "chunk_size", "写入缓冲区大小", "65536"
   "retry_times", "请求重试次数", "4"
   "pool_connections", "缓存在连接池的连接数", "10"
   "pool_maxsize", "连接池最大容量", "10"
   "connect_timeout", "连接超时", "120"
   "read_timeout", "读取超时", "120"
   "api_proxy", "API 代理服务器", "None"
   "data_proxy", "数据代理服务器", "None"
   "completion_size", "对象补全列举条数限制", "10"
   "table_auto_flush_time", "使用 ``table.open_writer`` 时的定时提交间隔", "150"
   "display.notebook_widget", "使用交互式插件", "True"
   "sql.settings", "ODPS SQL运行全局hints", "None"
   "sql.use_odps2_extension", "启用 MaxCompute 2.0 语言扩展", "None"
   "sql.enable_schema", "在任何情形下启用 MaxCompute Schema", "None"

数据上传/下载配置
==================

.. csv-table::
   :header-rows: 1

   "选项", "说明", "默认值"
   "tunnel.endpoint", "Tunnel Endpoint", "None"
   "tunnel.use_instance_tunnel", "使用 Instance Tunnel 获取执行结果", "True"
   "tunnel.limit_instance_tunnel", "是否限制 Instance Tunnel 获取结果的条数", "None"
   "tunnel.string_as_binary", "在 string 类型中使用 bytes 而非 unicode", "False"
   "tunnel.quota_name", "配置 Tunnel Quota 的名称", "False"
   "tunnel.block_buffer_size", "配置缓存 Block Writer 的缓存大小", "20 * 1024 ** 2"
   "tunnel.tags", "配置使用 Tunnel 所需的标签", "None"

DataFrame 配置
==================

.. csv-table::
   :header-rows: 1

   "选项", "说明", "默认值"
   "interactive", "是否在交互式环境", "根据检测值"
   "df.analyze", "是否启用非 ODPS 内置函数", "True"
   "df.optimize", "是否开启DataFrame全部优化", "True"
   "df.optimizes.pp", "是否开启DataFrame谓词下推优化", "True"
   "df.optimizes.cp", "是否开启DataFrame列剪裁优化", "True"
   "df.optimizes.tunnel", "是否开启DataFrame使用tunnel优化执行", "True"
   "df.quote", "ODPS SQL后端是否用``来标记字段和表名", "True"
   "df.image", "DataFrame运行使用的镜像名", "None"
   "df.libraries", "DataFrame运行使用的第三方库（资源名）", "None"
   "df.supersede_libraries", "使用自行上传的包替换服务中的版本", "True"
   "df.odps.sort.limit", "DataFrame有排序操作时，默认添加的limit条数", "10000"
