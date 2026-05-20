.. _api_storage_v2:

Storage API V2
==============

.. autoclass:: odps.apis.storage_api_v2.StorageApiClient
    :members:
    :exclude-members: read_blobs

    .. automethod:: read_blobs

.. autoclass:: odps.apis.storage_api_v2.StorageApiArrowClient
    :members:
    :inherited-members:
    :exclude-members: read_blobs

    .. automethod:: read_blobs

Read / Write Sessions
---------------------

.. autoclass:: odps.apis.storage_api_v2.CreateReadSessionRequest
    :members:

.. autoclass:: odps.apis.storage_api_v2.CreateReadSessionResponse
    :members:

.. autoclass:: odps.apis.storage_api_v2.CreateWriteSessionRequest
    :members:

.. autoclass:: odps.apis.storage_api_v2.CreateWriteSessionResponse
    :members:

.. autoclass:: odps.apis.storage_api_v2.GetWriteSessionResponse
    :members:

.. autoclass:: odps.apis.storage_api_v2.SessionStatus
    :members:

.. autoclass:: odps.apis.storage_api_v2.SessionStats
    :members:

Read / Write Streams
--------------------

.. autoclass:: odps.apis.storage_api_v2.CreateWriteStreamRequest
    :members:

.. autoclass:: odps.apis.storage_api_v2.CreateWriteStreamResponse
    :members:

.. autoclass:: odps.apis.storage_api_v2.GetWriteStreamResponse
    :members:

.. autoclass:: odps.apis.storage_api_v2.ReadStreamRequest
    :members:

.. autoclass:: odps.apis.storage_api_v2.WriteStreamRequest
    :members:

.. autoclass:: odps.apis.storage_api_v2.CloseWriteStreamRequest
    :members:

.. autoclass:: odps.apis.storage_api_v2.CloseWriteStreamResponse
    :members:

.. autoclass:: odps.apis.storage_api_v2.StreamReader
    :members:

.. autoclass:: odps.apis.storage_api_v2.StreamWriter
    :members:

.. autoclass:: odps.apis.storage_api_v2.ArrowReader
    :members:

.. autoclass:: odps.apis.storage_api_v2.ArrowWriter
    :members:

Blob I/O
--------

.. autoclass:: odps.apis.storage_api_v2.BlobDataIterator
    :members:

.. autoclass:: odps.apis.storage_api_v2.BlobStreamReader
    :members:

.. autoclass:: odps.apis.storage_api_v2.BlobStreamWriter
    :members:

.. autoclass:: odps.apis.storage_api_v2.BlobWriteItem
    :members:

.. autoclass:: odps.apis.storage_api_v2.ReadBlobRequest
    :members:

.. autoclass:: odps.apis.storage_api_v2.WriteBlobRequest
    :members:

.. autoclass:: odps.apis.storage_api_v2.WriteBlobResponse
    :members:

.. autoclass:: odps.apis.storage_api_v2.ChecksumType
    :members:

Preview
-------

.. autoclass:: odps.apis.storage_api_v2.PreviewTableRequest
    :members:

Schema and Options
------------------

.. autoclass:: odps.apis.storage_api_v2.DataSchema
    :members:

.. autoclass:: odps.apis.storage_api_v2.Column
    :members:

.. autoclass:: odps.apis.storage_api_v2.SplitOptions
    :members:

.. autoclass:: odps.apis.storage_api_v2.ArrowOptions
    :members:

.. autoclass:: odps.apis.storage_api_v2.Compression
    :members:

.. autoclass:: odps.apis.storage_api_v2.DataFormat
    :members:

.. autoclass:: odps.apis.storage_api_v2.IncrementalReadOptions
    :members:

.. autoclass:: odps.apis.storage_api_v2.Status
    :members:
