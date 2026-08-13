.. _api_maxstorage:

MaxStorage
==========

Client
------

.. autoclass:: odps.maxstorage.MaxStorageClient
    :members:

Read Sessions
-------------

.. autoclass:: odps.maxstorage.TableReadSession
    :members:

.. autoclass:: odps.maxstorage.InstanceReadSession
    :members:

.. autoclass:: odps.maxstorage.IndexedInputSplit
    :members:

.. autoclass:: odps.maxstorage.RowRangeInputSplit
    :members:

Readers
-------

.. autoclass:: odps.maxstorage.ArrowReader
    :members:

.. autoclass:: odps.maxstorage.ArrowRecordReader
    :members:

Write Sessions
--------------

.. autoclass:: odps.maxstorage.TableWriteSession
    :members:

Writers
-------

.. autoclass:: odps.maxstorage.TableArrowWriter
    :members:

.. autoclass:: odps.maxstorage.TableArrowBlobUploadWriter
    :members:

.. autoclass:: odps.maxstorage.AppendTableRecordWriter
    :members:

.. autoclass:: odps.maxstorage.DeltaTableRecordWriter
    :members:

Blob I/O
--------

.. autoclass:: odps.maxstorage.BlobManager
    :members:

.. autoclass:: odps.maxstorage.BlobRecord
    :members:

.. autoclass:: odps.maxstorage.BlobDataIterator
    :members:

.. autoclass:: odps.maxstorage.BlobStreamReader
    :members:

.. autoclass:: odps.maxstorage.BlobStreamWriter
    :members:

.. autoclass:: odps.maxstorage.BlobWriteItem
    :members:
    :exclude-members: write_frame_to, ChecksumType

Schema and Options
------------------

.. autoclass:: odps.maxstorage.WriteMode
    :members:

.. autoclass:: odps.maxstorage.SplitMode
    :members:

.. autoclass:: odps.maxstorage.SplitOptions
    :members:

.. autoclass:: odps.maxstorage.IncrementalReadOptions
    :members:

.. autoclass:: odps.maxstorage.DataFormat
    :members:

.. autoclass:: odps.maxstorage.SessionStatus
    :members:

.. autoclass:: odps.maxstorage.Status
    :members:

.. autoclass:: odps.maxstorage.TimestampUnit
    :members:

Errors
------

.. autoclass:: odps.maxstorage.MaxStorageError
    :members:

.. autoclass:: odps.maxstorage.StorageServiceError
    :members:

.. autoclass:: odps.maxstorage.StorageClientError
    :members:

.. autoclass:: odps.maxstorage.BlobDownloadError
    :members:
