# -*- coding: utf-8 -*-
# Copyright 1999-2026 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .core import I18NMessage

DEV_PROJECT_MSG = I18NMessage(
    """
    Your script is executed with a development project <dw_project_name>_dev.
    If MaxCompute objects you obtain differs from your expectations, please
    reference your projects explicitly. For instance,

    o.get_table("table_name", project="<dw_project_name>")
    """,
    cn="""
    你的脚本正在名为 <dw_project_name>_dev 的开发项目下执行。如果你获取的 MaxCompute
    对象与预期不同，请显示指定你需要使用的项目名。例如

    o.get_table("table_name", project="<dw_project_name>")
    """,
)
INSTALLATION_NOT_CHANGED_MSG = I18NMessage(
    """
    The last change time of your PyODPS installation is <last_change_time> and
    unlikely to be the cause of the error. Please check recent changes of your
    environments, for instance, code, third-party Python packages or relevant
    table schemas.
    """,
    cn="""
    你的 PyODPS 相关包变更时间为 <last_change_time>，因而该变更不太可能是本次报错的问题所在。
    请检查你最近的环境变更，例如代码、第三方 Python 包或者所使用的表结构是否有变化。
    """,
)
OOM_KILL_MSG = I18NMessage(
    """
    Once you see the message `Got killed` with -9 exit code, it is quite possible
    that you've run out of memory. Please try your best avoiding downloading massive
    data from MaxCompute. Related methods are `open_reader` of tables or instances,
    and `to_pandas` of DataFrames. It is recommended to use PyODPS DataFrame created
    from MaxCompute tables, or MaxCompute SQL directly. Find more details at
    https://www.alibabacloud.com/help/en/maxcompute/user-guide/overview-3 .
    """,
    cn="""
    看到 Got killed (-9) 有可能因为 Out of memory。请确保不要使用从 MaxCompute 下载数据来处理。
    下载数据操作常包括 Table/Instance 的 open_reader 以及 DataFrame 的 to_pandas 方法。
    推荐使用 PyODPS DataFrame（从 MaxCompute 表创建）和 MaxCompute SQL 来处理数据。更详细
    的内容可以参考 https://help.aliyun.com/zh/maxcompute/user-guide/overview-3 。
    """,
)
OTHER_KILL_MSG = I18NMessage(
    """
    Execution of your code crashed accidentally. Sometimes it might be caused by PyODPS
    package or possibly caused by crash of a third-party library. You might add print
    statements to check the location of crash for your code. If it crashes when calling
    PyODPS methods, please contact our supporting stuff for help. Otherwise you might
    retry or test your code locally with libraries specified in
    https://pyodps.readthedocs.io/en/stable/platform-d2.html#dw-3rdparty-lib .
    Note that we do not provide supports on third-party libraries. Please contact
    supporting groups of these libraries for help.
    """,
    cn="""
    你的代码在执行中遇到未知原因而崩溃。有时这是由于 PyODPS 包本身导致的，但也有可能是某个三方包造成的。
    你可以为你的代码增加 print 语句来确定到底哪行代码导致执行崩溃。如果发生崩溃时，你正在调用 PyODPS
    方法，请联系我们的支持团队。否则你可能需要在本地安装文档
    https://pyodps.readthedocs.io/zh_CN/stable/platform-d2.html#dw-3rdparty-lib
    中列举的三方包版本来测试是否代码可以正常工作。需要注意的是，我们不提供对三方包本身的支持，请联系
    三方包的维护者以寻求帮助。
    """,
)
TUNNEL_TIMEOUT_MSG = I18NMessage(
    """
    It often means tunnel servers are busy when you see errors like ReadTimeoutError.
    Consider increase value of options.read_timeout (which is 120 seconds by default).
    Note that downloading massive data from MaxCompute is not recommended. It is
    recommended to use PyODPS DataFrame created from MaxCompute tables, or
    MaxCompute SQL directly. Find more details at
    https://www.alibabacloud.com/help/en/maxcompute/user-guide/overview-3 .
    """,
    cn="""
    看到 ReadTimeoutError 很可能因为 Tunnel 服务端繁忙，可以考虑增大 options.read_timeout
    的值（默认为120，单位为秒）。需要注意的是从 ODPS 下载大量数据并不是推荐的行为，
    推荐使用 PyODPS DataFrame（从 MaxCompute 表创建）和 MaxCompute SQL 来处理数据。更详细的
    内容可以参考 https://help.aliyun.com/zh/maxcompute/user-guide/overview-3 。
    """,
)
TUNNEL_SESSION_CREATE_TIMEOUT_MSG = I18NMessage(
    """
    ReadTimeout error here is often caused by too many small files in the source
    table. You might try reducing count of small files by merging them. Details can
    be seen at https://www.alibabacloud.com/help/en/maxcompute/use-cases/merge-small-files .
    If information above does not resolve your issues, you can also try increasing
    value of options.read_timeout by code below.

    from odps import options
    options.read_timeout = 1200  # update read timeout, 120 seconds by default
    """,
    cn="""
    导致此处 ReadTimeout 的原因通常是源表小文件过多，可以尝试通过合并减少源表小文件数量。详细信息可见
    https://help.aliyun.com/zh/maxcompute/use-cases/merge-small-files 。
    如果该信息不能解决你的问题，你也可以尝试按下面的例子增大 options.read_timeout 的值：

    from odps import options
    options.read_timeout = 1200  # 修改读数据超时，默认为 120 秒
    """,
)
NO_LOCAL_IMPORT_MSG = I18NMessage(
    """
    Note that the "directory structure" shown in DataStudio DOES NOT reflect true
    file system organization, hence you CANNOT simply import file names shown in
    DataStudio even if it has been uploaded as a MaxCompute resource. If you need to
    import a third-party file, please see references at
    https://www.alibabacloud.com/help/en/dataworks/use-cases/use-a-pyodps-node-to-reference-a-third-party-package .
    If you have more questions please consult DataWorks.
    """,
    cn="""
    注意：DataStudio 中的“目录结构”并非文件系统中真实存在的目录结构，直接 import 或者打开 DataStudio
    中显示的文件路径会导致执行失败，即便该文件已被上传为 MaxCompute 资源。如果需要使用三方文件，
    请参考 https://help.aliyun.com/zh/dataworks/use-cases/use-a-pyodps-node-to-reference-a-third-party-package
    ，相关问题请咨询 DataWorks。
    """,
)
DATAFRAME_NO_PERSIST_MSG = I18NMessage(
    """
    The DataFrame you need to persist might be a pandas DataFrame which does not
    have a `persist` method. Please use code below to store the DataFrame into
    MaxCompute (assuming that `df` is the DataFrame to store):

    from odps import DataFrame
    DataFrame(df, odps=o).persist("table_name")
    """,
    cn="""
    你需要 persist 的 DataFrame 可能是 pandas DataFrame，其上并没有 persist 方法。请使用
    下面的方式保存（假定 df 为需要保存的 DataFrame）：

    from odps import DataFrame
    DataFrame(df, odps=o).persist("table_name")
    """,
)
PROJECT_PROTECT_PRIVILEGE_MSG = I18NMessage(
    """
    When a MaxCompute project is under protection, you may receive the message `You have
    NO privilege 'odps:Select'` even if you already have the privilege over certain objects.
    Please consult project owners to set exceptional rules.
    """,
    cn="""
    当一个 Project 被保护，即使用户具备 Select 权限也有可能收到 You have NO privilege 'odps:Select'
    的消息。请联系 Project Owner 设置例外规则。
    """,
)
SCHEDULE_ARGS_MSG = I18NMessage(
    """
    To avoid invading user code, PyODPS nodes will `never` replace strings like ${param_name}.
    To provide DataWorks arguments, we added a global variable `args` containing scheduling
    variables before your code is executed. For instance, if you've configured `ds=${yyyymmdd}`
    in node property tab, you may get the value of `ds` with code below:

    print('ds=' + args['ds'])

    Details can be seen at https://pyodps.readthedocs.io/en/latest/platform-d2.html#id4 。
    """,
    cn="""
    为避免侵入代码，PyODPS 节点 不会 在代码中替换 ${param_name} 这样的字符串，而是在执行代码前，
    在全局变量中增加一个名为 args 的 dict，调度参数可以在此获取。例如，在节点基本属性 -> 参数中设置
    ds=${yyyymmdd} ，则可以通过下面的方式在代码中获取此参数

    print('ds=' + args['ds'])

    详细信息可以参考 https://pyodps.readthedocs.io/zh_CN/latest/platform-d2.html#id4 。
    """,
)
REQUESTS_TIMEOUT_MSG = I18NMessage(
    """
    When encountering timeout errors when accessing MaxCompute or MaxCompute tunnel endpoints,
    lines below might be tried.

    from odps import options
    options.connect_timeout = 1200  # update connection timeout, 120 seconds by default
    options.read_timeout = 1200  # update read timeout, 120 seconds by default

    If you are encountering timeout when accessing other addresses, please contact DataWorks
    for support.
    """,
    cn="""
    遇到 MaxCompute / MaxCompute Tunnel 访问请求超时，可以尝试下面的方法：

    from odps import options
    options.connect_timeout = 1200  # 修改连接超时，默认为 120 秒
    options.read_timeout = 1200  # 修改读数据超时，默认为 120 秒

    如果访问其他地址超时，请联系 DataWorks 支持。
    """,
)
ENCODING_ERROR_MSG = I18NMessage(
    """
    When encountering encoding issues when reading data from MaxCompute tables with PyODPS
    methods, it is possible that your MaxCompute table stores non-visible binaries in
    String-typed fields. You may try adding the option below in front of all your code
    and rerun.

    from odps import options
    options.tunnel.string_as_binary = True

    If errors does not occur when you are reading data, please check your own code first.
    """,
    cn="""
    如果你在使用 PyODPS 相关方法读取 MaxCompute 表数据时遇到编码问题时，可能是你的 MaxCompute 表在
    String 类型字段存储了非字符串二进制值，可以尝试在代码最前面添加下面代码后重试。

    from odps import options
    options.tunnel.string_as_binary = True

    如果报错不在读取数据时发生，请先检查自己的代码逻辑。
    """,
)
SCRIPT_MODULE_NOT_FOUND_ERROR_MSG = I18NMessage(
    """
    When encountering ImportError or ModuleNotFoundError in your user-defined function,
    it is often caused by a missing third-party library. Note that MaxCompute and DataWorks
    use different sources of libraries. See
    https://pyodps.readthedocs.io/en/stable/pyodps-pack.html for more details.
    """,
    cn="""
    如果在自定义函数中遇到 ImportError 或者 ModuleNotFoundError，这通常由三方包缺失导致。注意 MaxCompute
    和 DataWorks 使用的三方包来源不同。参考 https://pyodps.readthedocs.io/zh_CN/stable/pyodps-pack.html
    以获得更多信息。
    """,
)
RESOURCE_REFERENCE_USED_MSG = I18NMessage(
    """
    It seems that you are using @resource_reference provided by DataWorks.
    If the module you need to import is registered by this annotation,
    you need to consult DataWorks for help. Note that modules referenced with
    this method CANNOT be used inside server-side user-defined functions.
    """,
    cn="""
    似乎你正在使用 DataWorks 提供的 @resource_reference。如果你需要导入的模块通过此注解注册，
    请咨询 DataWorks。需要注意的是，通过这种方式注册的模块不能在服务端自定义函数中使用。
    """,
)
SCRIPT_GENERATOR_NOT_CALLABLE_ERROR_MSG = I18NMessage(
    """
    When encountering TypeError('generator' object is not callable) when execution your
    user-defined function with PyODPS DataFrame, it often mean that you supply some resource
    to your UDF but fail to write your UDF in a nested function. Correct UDF function with
    resources should be

    def ufunc(resources):  # `resources` are resources supplied to your function
        def handle(rows):  # `rows` are data to be processed in your function
            for row in rows:
                yield row
        return handle

    See https://pyodps.readthedocs.io/en/stable/df-element.html#function-resource,
    https://pyodps.readthedocs.io/en/stable/df-sort-distinct-apply.html#id10 or
    https://pyodps.readthedocs.io/en/stable/df-sort-distinct-apply.html#id12 for more details.
    """,
    cn="""
    如果为 PyODPS DataFrame 调用自定义函数时遇到 TypeError('generator' object is not callable)，
    这通常意味着你为该函数指定了资源，但并未将函数修改为需要的形式。PyODPS DataFrame 要求带资源的自定义函数
    写成如下形式：

    def ufunc(resources):  # `resources` 为该 UDF 的输入资源
        def handle(rows):  # `rows` 为该 UDF 的输入数据
            for row in rows:
                yield row
        return handle

    参考 https://pyodps.readthedocs.io/zh_CN/stable/df-element.html#function-resource ，
    https://pyodps.readthedocs.io/zh_CN/stable/df-sort-distinct-apply.html#id10 或者
    https://pyodps.readthedocs.io/zh_CN/stable/df-sort-distinct-apply.html#id12 以获取更多信息。
    """,
)
SCRIPT_ERROR_MSG = I18NMessage(
    """
    When encountering ScriptError, it is possible that there is something wrong
    with your UDFs. You may open LogView url, look for failed instances and diagnose
    StdErr for more information. Documents can be seen at
    https://www.alibabacloud.com/help/en/maxcompute/user-guide/use-logview-v2-0-to-view-job-information 。
    """,
    cn="""
    遇到脚本错误时，可能是你的 UDF 编写有问题，可以打开 LogView 后寻找出错的 Instance，然后
    查看 StdErr 以获得进一步信息，参考文档
    https://help.aliyun.com/zh/maxcompute/user-guide/use-logview-v2-0-to-view-job-information 。
    """,
)
ODPS_OBJ_CLOSURE_ERROR_MSG = I18NMessage(
    """
    Please do not reference PyODPS objects in custom functions or custom aggregations
    in DataFrame. This will lead to errors. When you need to reference results from
    other DataFrames, you may consider calling `apply` over the whole DataFrame
    or rewrite your logic with `join`. Usage of `apply` function can be seen at
    https://pyodps.readthedocs.io/en/latest/df-sort-distinct-apply.html#dfudtfapp
    """,
    cn="""
    不要在自定义函数 / 聚合中引用 PyODPS 对象，这将导致报错。如果需要引用其他 DataFrame 对象的结果，
    可以考虑对所有列同时使用 apply，然后将不需要修改的列原样输出，或者改写成 Join。Apply 的用法见
    https://pyodps.readthedocs.io/zh_CN/latest/df-sort-distinct-apply.html#dfudtfapp
    """,
)
INTERNAL_SERVER_ERROR_MSG = I18NMessage(
    "Internal error occurred in MaxCompute, please submit a support ticket "
    "or consult our supporting stuffs for help.",
    cn="MaxCompute 发生内部错误，请提交工单或联系售后解决。",
)
RESULT_READER_CREATE_MSG = I18NMessage(
    """
    When using Instance Reader in your code, it is not utilizing Tunnel. Usually
    it does not make difference. When you happen to meet encoding errors or difficulty
    in reading complex data types, you may try `instance.open_reader(tunnel=True)`.
    While this option may limit records you get, you may use
    `instance.open_reader(tunnel=True, limit=False) ` once you have read privilege over
    instances.
    """,
    cn="""
    你使用的 Instance Reader 没有利用 Tunnel，通常情况下下面的信息可以忽略。如果遇到编码错误或者
    复杂数据类型等情形，可以尝试使用 instance.open_reader(tunnel=True) 。该 Reader 限制了
    读取数据的规模，如果你需要读取全部数据，可以使用 instance.open_reader(tunnel=True, limit=False) 。
    解除该限制前，请确保你拥有 Instance 数据的读取权限。
    """,
)
INSTANCE_TUNNEL_LIMIT_MSG = I18NMessage(
    """
    You are using Instance Tunnel to read data which limits number of records
    you can retrieve by default. If you want to read all records from the instance,
    you may use `instance.open_reader(tunnel=True, limit=False)`. Make sure you
    have privilege to read from the instance before removing this limitation.
    """,
    cn="""
    你使用的 Instance Tunnel 默认限制读取数据的规模。如果你需要读取全部数据，可以使用
    instance.open_reader(tunnel=True, limit=False) 。解除该限制前，请确保你拥有 Instance
    数据的读取权限。
    """,
)
DATAFRAME_UNKNOWN_DTYPE_MSG = I18NMessage(
    """
    `object` type exists in DataFrame you need to save into MaxCompute. Please convert
    it into some type MaxCompute can handle. If you want to convert these values as
    string, you may use parameter `unknown_as_string=True`. Detals can be found at
    http://pyodps.readthedocs.io/en/latest/df-basic.html?highlight=unknown_as_string#dataframe
    """,
    cn="""
    需要输出的 DataFrame 数据类型包括 object，请将其转换为 MaxCompute 可以处理的类型。如果需要
    转换为字符串，可以使用 unknown_as_string=True 参数，详见
    http://pyodps.readthedocs.io/zh_CN/latest/df-basic.html?highlight=unknown_as_string#dataframe
    """,
)
SYNTAX_ERROR_MSG = I18NMessage(
    """
    Python syntax of your code seems incorrect. Please fix them given messages above.
    This error has nothing to do with MaxCompute or PyODPS package.
    """,
    cn="""
    你的 Python 代码存在语法错误，请根据上方提示修复。该问题与 MaxCompute / PyODPS 无关。
    """,
)
PURE_SQL_PARSE_ERROR = I18NMessage(
    """
    Syntax of your SQL statement submitted with PyODPS seems incorrect.
    Please fix them given messages above. This error has nothing to do with PyODPS package.
    """,
    cn="""
    你通过 PyODPS 提交的 SQL 语句存在语法错误，请根据上方提示修复。该问题与 PyODPS 无关。
    """,
)
NAME_ERROR_MSG = I18NMessage(
    """
    You have undefined variables in your code. Please fix it given the variable name above.
    This error usually has nothing to do with MaxCompute or PyODPS package.
    """,
    cn="""
    你的 Python 代码使用了未赋值的变量，请根据上方提示的变量名修复。该问题通常与 MaxCompute / PyODPS 无关。
    """,
)
TRACEBACK_NO_PYODPS_MSG = I18NMessage(
    """
    Seems error messages have nothing to do with PyODPS. Please check the correctness of your
    own logic. Network connectivity issues or other problems can be addressed in DataWorks
    documentations, or you can also request DataWorks for assistance.
    """,
    cn="""
    错误信息中似乎不包含 PyODPS 相关的代码，请检查自己的代码逻辑是否正确。网络联通性等问题请查看 DataWorks
    相关文档或寻求 DataWorks 帮助。
    """,
)
BUILTIN_OBJECT_REPLACED_MSG = I18NMessage(
    """
    Seems that you've replaced objects supplied by the node or Python builtins.
    Changed variables are
        {REPLACED_VARS}
    This could produce unexpected results. If you didn't get results as expected,
    please check if these changes cause the error. Note that if you want to change
    the default account provided by the node, try using `o.as_account` instead of
    creating a new `ODPS` instance yourself.
    """,
    cn="""
    你似乎替换了部分节点提供的对象或者 Python 内置对象，替换的对象为
        {REPLACED_VARS}
    替换这些对象可能会导致难以预期的结果。如果你没有获得符合预期的执行结果，请检查是否这些更改导致了错误。
    注意，如果你需要使用自己定义的账号而非系统提供的账号，可以使用 `o.as_account` 而不是自己重新创建一个
    `ODPS` 实例。
    """,
)
RELOAD_SYS_WARN_MSG = I18NMessage(
    """
    The code behavior after calling reload(sys) is often unpredictable. Please try
    avoid using it. When encountering encoding issues, try utilizing encoding
    capabilities with Python 3.
    """,
    cn="""
    使用 reload(sys) 后的代码行为在很多时候难以预测，请避免使用 reload(sys)。
    如果你需要解决编码问题，请使用 Python 3 处理编码问题。
    """,
)
PYTHON2_DEPRECATE_MSG = I18NMessage(
    """
    Support of Python 2.7 is discontinued since January 1, 2020. Though the package PyODPS
    still supports it for now, supports for new functionalities might be dropped at some time.
    Please consider migrating to Python 3 instead.
    """,
    cn="""
    Python 2.7 已经于2020年1月1日停止后续支持。尽管目前 PyODPS 仍然支持在 Python 2.7 上运行，
    未来某一时刻我们可能会停止支持为 Python 2 增加新功能。请考虑将你的代码迁移到 Python 3。
    """,
)
