.. _api_tunnel:

Tunnel
=======

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