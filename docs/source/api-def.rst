.. _apidef:

Definitions
===========

ODPS Entry Object
~~~~~~~~~~~~~~~~~

.. autoclass:: odps.ODPS
    :members:
    :exclude-members: attach_session, create_session, default_session

ODPS Model Objects
~~~~~~~~~~~~~~~~~

.. autoclass:: odps.models.Project
    :members:

.. autoclass:: odps.models.Table
    :members:

.. autoclass:: odps.models.TableSchema
    :members:

.. autoclass:: odps.models.table.TableSchema
    :members:

.. autoclass:: odps.models.partition.Partition
    :members:

.. autoclass:: odps.models.Record
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

.. intinclude:: api-def-int.rst

ODPS Tunnel
~~~~~~~~~~~

.. autoclass:: odps.tunnel.TableTunnel
    :members:

.. autoclass:: odps.tunnel.TableDownloadSession
    :members:
    :exclude-members: Status

    .. autoattribute:: id

        ID of the download session which can be passed to :meth:`~odps.tunnel.TableTunnel.create_download_session`
        for session reuse.

    .. autoattribute:: count

        Count of records in the table.

    .. autoattribute:: schema

        Schema of the table.

.. autoclass:: odps.tunnel.TableStreamUploadSession
    :members:
    :inherited-members:
    :exclude-members: Status

    .. autoattribute:: id

        ID of the stream upload session which can be passed to :meth:`~odps.tunnel.TableTunnel.create_stream_upload_session`
        for session reuse.

.. autoclass:: odps.tunnel.TableUploadSession
    :members:
    :inherited-members:
    :exclude-members: Status

    .. autoattribute:: id

        ID of the upload session which can be passed to :meth:`~odps.tunnel.TableTunnel.create_upload_session`
        for session reuse.

.. autoclass:: odps.tunnel.TableUpsertSession
    :members:
    :inherited-members:
    :exclude-members: Status

    .. autoattribute:: id

        ID of the upsert session which can be passed to :meth:`~odps.tunnel.TableTunnel.create_upsert_session`
        for session reuse.

.. autoclass:: odps.tunnel.InstanceTunnel
    :members:

.. autoclass:: odps.tunnel.InstanceDownloadSession
    :members:
    :exclude-members: Status

    .. autoattribute:: id

        ID of the download session which can be passed to :meth:`~odps.tunnel.InstanceTunnel.create_download_session`
        for session reuse.

    .. autoattribute:: count

        Count of records in the instance result.

    .. autoattribute:: schema

        Schema of the instance result.

.. autoclass:: odps.tunnel.ArrowWriter
    :members:
    :inherited-members:

.. autoclass:: odps.tunnel.RecordWriter
    :members:
    :inherited-members:

.. autoclass:: odps.tunnel.BufferedArrowWriter
    :members:
    :inherited-members:

.. autoclass:: odps.tunnel.BufferedRecordWriter
    :members:
    :inherited-members:

.. autoclass:: odps.tunnel.TunnelArrowReader
    :members:
    :inherited-members:

.. autoclass:: odps.tunnel.TunnelRecordReader
    :members:
    :inherited-members:

.. autoclass:: odps.tunnel.Upsert
    :members:
    :inherited-members:
    :exclude-members: Operation, Status

.. autoclass:: odps.tunnel.VolumeTunnel
    :members:

.. autoclass:: odps.tunnel.VolumeDownloadSession
    :members:
    :exclude-members: Status

.. autoclass:: odps.tunnel.VolumeUploadSession
    :members:
    :exclude-members: Status
