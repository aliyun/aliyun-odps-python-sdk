.. _api_types:

Data types
===========

.. autoclass:: odps.types.Boolean

.. autoclass:: odps.types.Tinyint

.. autoclass:: odps.types.Smallint

.. autoclass:: odps.types.Int

.. autoclass:: odps.types.Bigint

.. autoclass:: odps.types.Decimal

    .. autoattribute:: precision

        Precision (or total digits) of the decimal type.

    .. autoattribute:: scale

        Decimal scale (or decimal digits) of the decimal type.

.. autoclass:: odps.types.Float

.. autoclass:: odps.types.Double

.. autoclass:: odps.types.Binary

.. autoclass:: odps.types.Char

    .. autoattribute:: size_limit

        Size limit of the varchar type.

.. autoclass:: odps.types.String

.. autoclass:: odps.types.Varchar

    .. autoattribute:: size_limit

        Size limit of the varchar type.

.. autoclass:: odps.types.Json

.. autoclass:: odps.types.Date

.. autoclass:: odps.types.Datetime

.. autoclass:: odps.types.Timestamp

.. autoclass:: odps.types.TimestampNTZ

.. autoclass:: odps.types.Array
    :members:

    .. autoattribute:: value_type

        Type of elements in the array.

.. autoclass:: odps.types.Map
    :members:

    .. autoattribute:: key_type

        Type of keys in the map.

    .. autoattribute:: value_type

        Type of values in the map.

.. autoclass:: odps.types.Struct
    :members:

    .. autoattribute:: field_types

        Types of fields in the struct, as an OrderedDict.

        :Example:

        The example below extracts field types of a struct.

        .. code-block:: python

            import odps.types as odps_types

            # obtain field types of the Struct instance
            struct_type = odps_types.Struct(
                {"a": odps_types.bigint, "b": odps_types.string}
            )
            for field_name, field_type in struct_type.field_types.items():
                print("field_name:", field_name, "field_type:", field_type)

.. autofunction:: odps.types.validate_data_type

.. autoclass:: odps.types.Column
    :members:

    .. autoattribute:: name

        Name of the column.

    .. autoattribute:: type

        Type of the column.

    .. autoattribute:: nullable

        True if the column is nullable.

.. autoclass:: odps.types.Partition
    :members:

    .. autoattribute:: name

        Name of the column.

    .. autoattribute:: type

        Type of the column.

    .. autoattribute:: nullable

        True if the column is nullable.

.. autoclass:: odps.models.Record
    :members:

.. autoclass:: odps.models.TableSchema
    :members:
    :exclude-members: TableColumn, TablePartition

    .. autoproperty:: columns
    .. autoproperty:: partitions
    .. autoproperty:: simple_columns
    .. automethod:: from_lists
