.. _odps_types:

基本类型
================

.. _data_types:

数据类型
----------

PyODPS 对 MaxCompute 中的类型支持实现于 ``odps.types`` 包中。所有的数据类型均表示为
:class:`odps.types.DataType` 类的子类生成的实例。例如，64位整数类型 ``bigint`` 使用
:class:`odps.types.Bigint` 的实例表示，而32位整数数组类型 ``array<int>`` 使用
:class:`odps.types.Array` 的实例表示，且其 ``value_type`` 属性为 :class:`odps.types.Int`
类型。

.. note::

    PyODPS 默认不开放对 ``bigint``、\ ``string``、\ ``double``、\ ``boolean``、\ ``datetime``、\
    ``decimal`` 类型外其他类型的完整支持。需要完整使用除这些类型外的其他类型，需要设置选项
    ``options.sql.use_odps2_extension = True``\ 。关于设置选项可参考\ :ref:`这份文档 <options>` 。

通过字符串指定类型实例
~~~~~~~~~~~~~~~~~~~~~~

通常情况下，在 PyODPS 中，你都可以直接用 MaxCompute DDL 中表示类型的字符串来表示类型，这可以避免\
了解类型的实现细节。例如，当我们创建一个列实例，可以直接传入 ``array<int>`` 代表一个32位整数数组，\
而不需要关心使用哪个类去实现：

.. code-block:: python

    >>> import odps.types as odps_types
    >>>
    >>> column = odps_types.Column("col", "array<int>")
    >>> print(type(column.type))
    <class 'odps.types.Array'>
    >>> print(type(column.type.value_type))
    <class 'odps.types.Int'>

如果需要，你可以使用 :func:`~odps.types.validate_data_type` 函数获取字符串表示的 MaxCompute 类型实例。

.. code-block:: python

    >>> from odps.types import validate_data_type
    >>>
    >>> array_type = validate_data_type("array<bigint>")
    >>> print(array_type.value_type)
    bigint

可定义大小的类型
~~~~~~~~~~~~~~~
MaxCompute 部分类型可定义类型的大小，例如 ``char`` / ``varchar`` 可以定义最大长度，Decimal
可以定义精度（precision）和小数位数（scale）。定义这些类型时，可以构造对应类型描述类的实例，例如

.. code-block:: python

    >>> from odps.types import validate_data_type
    >>>
    >>> # 定义 char / varchar 类型实例，长度为 10
    >>> char_type = validate_data_type('char(10)')
    >>> varchar_type = validate_data_type('varchar(10)')
    >>> # 定义 decimal 类型实例，精度为 10，小数位数为 2
    >>> decimal_type = validate_data_type('decimal(10, 2)')

``char`` / ``varchar`` 类型实例的大小可通过 ``size_limit`` 属性获取，而 ``decimal``
类型实例的精度和小数位数可通过 ``precision`` 和 ``scale`` 属性获取。

.. code-block:: python

    >>> from odps.types import validate_data_type
    >>>
    >>> # 获取 char / varchar 类型长度
    >>> char_type = validate_data_type('char(10)')
    >>> print("size_limit:", char_type.size_limit)
    size_limit: 10
    >>> # 获取 decimal 类型精度和小数位数
    >>> decimal_type = validate_data_type('decimal(10, 2)')
    >>> print("precision:", decimal_type.precision, "scale:", decimal_type.scale)
    precision: 10 scale: 2

复合类型
~~~~~~~~
MaxCompute 支持的复合类型有 Array、Map 和 Struct，可通过构造函数或者类型字符串获取\
对应的类型描述类实例。下面的例子展示了如何创建 Array 和 Map 类型描述实例。

.. code-block:: python

    >>> import odps.types as odps_types
    >>>
    >>> # 创建值类型为 bigint 的 Array 类型描述实例
    >>> array_type = odps_types.Array(odps_types.bigint)
    >>> # 创建关键字类型为 string，值类型为 array<bigint> 的 Map 类型描述实例
    >>> map_type = odps_types.Map(odps_types.string, odps_types.Array(odps_types.bigint))

使用字符串生成相同的类型：

.. code-block:: python

    >>> from odps.types import validate_data_type
    >>>
    >>> # 创建值类型为 bigint 的 Array 类型描述实例
    >>> array_type = validate_data_type("array<bigint>")
    >>> # 创建关键字类型为 string，值类型为 array<bigint> 的 Map 类型描述实例
    >>> map_type = validate_data_type("map<string, array<bigint>>")

:class:`~odps.types.Array` 类型描述实例的元素类型可通过 ``value_type`` 属性获取。\
:class:`~odps.types.Map` 类型描述实例的关键字类型可通过 ``key_type`` 属性获取，\
而值类型可通过 ``value_type`` 属性获取。

.. code-block:: python

    >>> from odps.types import validate_data_type
    >>>
    >>> # 获取 Array 类型元素类型
    >>> array_type = validate_data_type("array<bigint>")
    >>> print("value_type:", array_type.value_type)
    value_type: bigint
    >>> # 获取 Map 类型关键字类型和值类型
    >>> map_type = validate_data_type("map<string, array<bigint>>")
    >>> print("key_type:", map_type.key_type, "value_type:", map_type.value_type)
    key_type: string value_type: array<bigint>

你可以通过 ``dict[str, DataType]`` 或者 ``list[tuple[str, DataType]]`` 创建 Struct 类型描述实例。\
对于 ``dict`` 类型，需要注意在 Python 3.6 及之前版本，Python 不保证 ``dict`` 的顺序，这可能导致\
定义的字段类型与预期不符。下面的例子展示了如何创建 Struct 类型描述实例。

.. code-block:: python

    >>> import odps.types as odps_types
    >>>
    >>> # 通过 tuple 列表创建一个 Struct 类型描述实例，其中包含两个字段，
    >>> # 分别名为 a 和 b，类型分别为 bigint 和 string
    >>> struct_type = odps_types.Struct(
    >>>     [("a", odps_types.bigint), ("b", odps_types.string)]
    >>> )
    >>> # 通过 dict 创建一个相同的 Struct 类型描述实例
    >>> struct_type = odps_types.Struct(
    >>>     {"a": odps_types.bigint, "b": odps_types.string}
    >>> )

使用字符串生成相同的类型：

.. code-block:: python

    >>> from odps.types import validate_data_type
    >>>
    >>> struct_type = validate_data_type("struct<a:bigint, b:string>")

:class:`~odps.types.Struct` 类型描述实例的各个字段类型可通过 ``field_types`` 属性获取，\
该属性为一个由字段名和字段类型组成的 ``OrderedDict`` 实例。

.. code-block:: python

    >>> from odps.types import validate_data_type
    >>>
    >>> # 获取 Struct 类型各个字段类型
    >>> struct_type = validate_data_type("struct<a:bigint, b:string>")
    >>> for field_name, field_type in struct_type.field_types.items():
    >>>     print("field_name:", field_name, "field_type:", field_type)
    field_name: a field_type: bigint
    field_name: b field_type: string

.. _table_schema:

表结构及相关类
--------------

.. note::

    本章节中的代码对 PyODPS 0.11.3 及后续版本有效。对早于 0.11.3 版本的 PyODPS，请使用 ``odps.models.Schema`` 代替
    ``odps.models.TableSchema``。

:class:`~odps.models.TableSchema` 类型用于表示表的结构，其中包含字段名称和类型。你可以使用表的列以及\
（可选的）分区来初始化。

.. code-block:: python

   >>> from odps.models import TableSchema, Column, Partition
   >>>
   >>> columns = [
   >>>     Column(name='num', type='bigint', comment='the column'),
   >>>     Column(name='num2', type='double', comment='the column2'),
   >>>     Column(name='arr', type='array<int>', comment='the column3'),
   >>> ]
   >>> partitions = [Partition(name='pt', type='string', comment='the partition')]
   >>> schema = TableSchema(columns=columns, partitions=partitions)
   >>> print(schema)
   odps.Schema {
     num     bigint      # the column
     num2    double      # the column2
     arr     array<int>  # the column3
   }
   Partitions {
     pt      string      # the partition
   }

第二种方法是使用 :meth:`TableSchema.from_lists() <odps.models.TableSchema.from_lists>`
方法。这种方法更容易调用，但无法直接设置列和分区的注释。

.. code-block:: python

   >>> from odps.models import TableSchema, Column, Partition
   >>>
   >>> schema = TableSchema.from_lists(
   >>>    ['num', 'num2', 'arr'], ['bigint', 'double', 'array<int>'], ['pt'], ['string']
   >>> )
   >>> print(schema)
   odps.Schema {
     num     bigint
     num2    double
     arr     array<int>
   }
   Partitions {
     pt      string
   }

你可以从 :class:`~odps.models.TableSchema` 实例中获取表的一般字段和分区字段。\ :attr:`~odps.models.TableSchema.simple_columns`
和 :attr:`~odps.models.TableSchema.partitions` 属性分别指代一般列和分区列，而 :attr:`~odps.models.TableSchema.columns`
属性则指代所有字段。这三个属性的返回值均为 :class:`~odps.types.Column` 或 :class:`~odps.types.Partition` 类型组成的列表。\
你也可以通过 ``names`` 和 ``types`` 属性分别获取非分区字段的名称和类型。

.. code-block:: python

   >>> from odps.models import TableSchema, Column, Partition
   >>>
   >>> schema = TableSchema.from_lists(
   >>>    ['num', 'num2', 'arr'], ['bigint', 'double', 'array<int>'], ['pt'], ['string']
   >>> )
   >>> print(schema.columns)  # 类型为 Column 的列表
   [<column num, type bigint>,
    <column num2, type double>,
    <column arr, type array<int>>,
    <partition pt, type string>]
   >>> print(schema.simple_columns)  # 类型为 Column 的列表
   [<column num, type bigint>,
    <column num2, type double>,
    <column arr, type array<int>>]
   >>> print(schema.partitions)  # 类型为 Partition 的列表
   [<partition pt, type string>]
   >>> print(schema.simple_columns[-1].type.value_type)  # 获取最后一列数组的值类型
   int
   >>> print(schema.names)  # 获取非分区字段的字段名
   ['num', 'num2']
   >>> print(schema.types)  # 获取非分区字段的字段类型
   [bigint, double]

在使用 :class:`~odps.models.TableSchema` 时，:class:`~odps.types.Column` 和 :class:`~odps.types.Partition`
类型分别用于表示表的字段和分区。你可以通过字段名和类型创建 :class:`~odps.types.Column` 实例，也可以同时指定列注释以及字段是否可以为空。\
你也可以通过相应的字段获取字段的名称、类型等属性，其中类型为:ref:`数据类型 <data_types>`中的类型实例。

.. code-block:: python

    >>> from odps.models import Column
    >>>
    >>> col = Column(name='num_col', type='array<int>', comment='comment of the col', nullable=False)
    >>> print(col)
    <column num_col, type array<int>, not null>
    >>> print(col.name)
    num_col
    >>> print(col.type)
    array<int>
    >>> print(col.type.value_type)
    int
    >>> print(col.comment)
    comment of the col
    >>> print(col.nullable)
    False

相比 :class:`~odps.types.Column` 类型，\ :class:`~odps.types.Partition` 类型仅仅是类名有差异，此处不再介绍。

.. _record-type:

行记录（Record）
----------------
:class:`~odps.models.Record` 类型表示表的一行记录，为
:meth:`Table.open_reader() <odps.models.Table.open_reader>` /
:meth:`Table.open_reader() <odps.models.Table.open_writer>` 当 ``arrow=False``
时所使用的数据结构，也用于
:meth:`TableDownloadSession.open_record_reader() <odps.tunnel.TableDownloadSession.open_record_reader>` /
:meth:`TableUploadSession.open_record_writer() <odps.tunnel.TableUploadSession.open_record_writer>` 。\
我们在 Table 对象上调用 new_record 就可以创建一个新的 Record。

下面的例子中，假定表结构为

.. code-block::

   odps.Schema {
     c_int_a                 bigint
     c_string_a              string
     c_bool_a                boolean
     c_datetime_a            datetime
     c_array_a               array<string>
     c_map_a                 map<bigint,string>
     c_struct_a              struct<a:bigint,b:string>
   }

该表对应 record 的修改和读取示例为

.. code-block:: python

   >>> import datetime
   >>> t = o.get_table('mytable')
   >>> r = t.new_record([1024, 'val1', False, datetime.datetime.now(), None, None])  # 值的个数必须等于表schema的字段数
   >>> r2 = t.new_record()  # 初始化时也可以不传入值
   >>> r2[0] = 1024  # 可以通过偏移设置值
   >>> r2['c_string_a'] = 'val1'  # 也可以通过字段名设置值
   >>> r2.c_string_a = 'val1'  # 通过属性设置值
   >>> r2.c_array_a = ['val1', 'val2']  # 设置 array 类型的值
   >>> r2.c_map_a = {1: 'val1'}  # 设置 map 类型的值
   >>> r2.c_struct_a = (1, 'val1')  # 使用 tuple 设置 struct 类型的值，当 PyODPS >= 0.11.5
   >>> r2.c_struct_a = {"a": 1, "b": 'val1'}  # 也可以使用 dict 设置 struct 类型的值
   >>>
   >>> print(record[0])  # 取第0个位置的值
   >>> print(record['c_string_a'])  # 通过字段取值
   >>> print(record.c_string_a)  # 通过属性取值
   >>> print(record[0: 3])  # 切片操作
   >>> print(record[0, 2, 3])  # 取多个位置的值
   >>> print(record['c_int_a', 'c_double_a'])  # 通过多个字段取值

MaxCompute 不同数据类型在 Record 中对应 Python 类型的关系如下：

.. csv-table::
   :header-rows: 1

   "MaxCompute 类型", "Python 类型", "说明"
   "``tinyint``, ``smallint``, ``int``, ``bigint``", "``int``", ""
   "``float``, ``double``", "``float``", ""
   "``string``", "``str``", "见说明1"
   "``binary``", "``bytes``", ""
   "``datetime``", "``datetime.datetime``", "见说明2"
   "``date``", "``datetime.date``", ""
   "``boolean``", "``bool``", ""
   "``decimal``", "``decimal.Decimal``", "见说明3"
   "``map``", "``dict``", ""
   "``array``", "``list``", ""
   "``struct``", "``tuple`` / ``namedtuple``", "见说明4"
   "``timestamp``", "``pandas.Timestamp``", "见说明2，需要安装 pandas"
   "``timestamp_ntz``", "``pandas.Timestamp``", "结果不受时区影响，需要安装 pandas"
   "``interval_day_time``", "``pandas.Timedelta``", "需要安装 pandas"
   "``interval_year_month``", "``odps.Monthdelta``", "见说明5"

对部分类型的说明如下。

1. PyODPS 默认 string 类型对应 Unicode 字符串，在 Python 3 中为 str，在 Python 2 中为
   unicode。对于部分在 string 中存储 binary 的情形，可能需要设置 ``options.tunnel.string_as_binary = True``
   以避免可能的编码问题。
2. PyODPS 默认使用 Local Time 作为时区，如果要使用 UTC 则需要设置 ``options.local_timezone = False``。
   如果要使用其他时区，需要设置该选项为指定时区，例如 ``Asia/Shanghai``。MaxCompute
   不会存储时区值，因而在写入数据时，会将该时间转换为 Unix Timestamp 进行存储。
3. 对于 Python 2，当安装 cdecimal 包时，会使用 ``cdecimal.Decimal``。
4. 对于 PyODPS \< 0.11.5，MaxCompute struct 对应 Python dict 类型。PyODPS \>= 0.11.5
   则默认对应 namedtuple 类型。如果要使用旧版行为则需要设置选项 ``options.struct_as_dict = True``。\
   DataWorks 环境下，为保持历史兼容性，该值默认为 False。为 Record 设置 struct 类型的字段值时，\
   PyODPS \>= 0.11.5 可同时接受 dict 和 tuple 类型，旧版则只接受 dict 类型。
5. Monthdelta 可使用年 / 月进行初始化，使用示例如下：

   .. code-block:: python

        >>> from odps import Monthdelta
        >>>
        >>> md = Monthdelta(years=1, months=2)
        >>> print(md.years)
        1
        >>> print(md.months)
        1
        >>> print(md.total_months)
        14

6. 关于如何设置 ``options.xxx``，请参考文档\ :ref:`配置选项 <options>`。
