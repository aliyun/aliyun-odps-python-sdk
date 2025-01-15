.. _api_models:

Model objects
==============

.. autoclass:: odps.models.Project
    :members:

.. autoclass:: odps.models.Table
    :members:

    .. autoattribute:: name

        Name of the table

    .. autoattribute:: comment

        Comment of the table

    .. autoattribute:: owner

        Owner of the table

    .. autoattribute:: creation_time

        Creation time of the table in local time.

    .. autoattribute:: last_data_modified_time

        Last data modified time of the table in local time.

    .. autoattribute:: table_schema

        Schema of the table, in :class:`~odps.models.TableSchema` type.

    .. autoattribute:: type

        Type of the table, can be managed_table, external_table, view or materialized_view.

    .. autoattribute:: size

        Logical size of the table.

    .. autoattribute:: lifecycle

        Lifecycle of the table in days.

.. autoclass:: odps.models.partition.Partition
    :members:

.. autoclass:: odps.models.Instance
    :members:

.. autoclass:: odps.models.Resource
    :members:

.. autoclass:: odps.models.FileResource
    :members:

.. autoclass:: odps.models.PyResource
    :members:

.. autoclass:: odps.models.JarResource
    :members:

.. autoclass:: odps.models.ArchiveResource
    :members:

.. autoclass:: odps.models.TableResource
    :members:

.. autoclass:: odps.models.Function
    :members:

.. autoclass:: odps.models.Worker
    :members:

.. autoclass:: odps.models.ml.OfflineModel
    :members:

.. autoclass:: odps.models.security.User
    :members:

.. intinclude:: api-models-int.rst
