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

"""Record reader for :mod:`odps.maxstorage`; see :class:`ArrowRecordReader`."""

from ...models import Record
from ...tunnel.io.reader import ArrowRecordFieldConverter
from ...tunnel.io.reader import ArrowRecordReader as _TunnelArrowRecordReader
from ...types import Blob


class ArrowRecordReader(_TunnelArrowRecordReader):
    """Record reader over a maxstorage :class:`ArrowReader`.

    Subclasses the tunnel :class:`~odps.tunnel.io.reader.ArrowRecordReader` so
    that type conversion is identical to the tunnel record reader.  The only
    divergence is BLOB columns: the tunnel never supports them, and
    :func:`odps.types.validate_value` rejects the ``Blob`` type.  Here the
    Arrow ``binary`` value (the server-side blob reference) is stored directly
    as ``bytes``, bypassing validation.
    """

    def __init__(self, arrow_reader):
        super().__init__(arrow_reader, make_compat=True)
        schema = arrow_reader.schema
        self._blob_col_indices = set()
        self._lower_to_arrow = None
        if schema is not None:
            for idx, col in enumerate(schema.columns):
                if isinstance(col.type, Blob):
                    self._blob_col_indices.add(idx)

    @property
    def count(self) -> int:
        """Total rows available — delegates to the underlying ArrowReader."""
        return self._arrow_reader.count

    def read(self):
        if not self._blob_col_indices:
            return super().read()

        if self._cur_batch is None or self._batch_pos >= self._cur_batch.num_rows:
            self._cur_batch = self._arrow_reader.read_next_batch()
            self._batch_pos = 0
            self._lower_to_arrow = None
            if self._cur_batch is None or self._cur_batch.num_rows == 0:
                return None

        if self._field_converters is None:
            table_schema = self._arrow_reader.schema
            self._field_converters = [
                ArrowRecordFieldConverter(table_schema[col_name].type, arrow_type)
                for col_name, arrow_type in zip(
                    self._cur_batch.schema.names, self._cur_batch.schema.types
                )
            ]

        columns = self.schema.columns
        if self._lower_to_arrow is None:
            self._lower_to_arrow = {}
            for idx, name in enumerate(self._cur_batch.schema.names):
                self._lower_to_arrow.setdefault(name.lower(), idx)
        lower_to_arrow = self._lower_to_arrow

        rec = Record(schema=self.schema)
        for col_idx, col in enumerate(columns):
            arrow_idx = lower_to_arrow.get(col.name.lower())
            if arrow_idx is None:
                continue
            if col_idx in self._blob_col_indices:
                # BLOB: store the reference bytes directly, bypassing
                # validate_value which does not understand the Blob type.
                rec._values[col_idx] = self._cur_batch.column(arrow_idx)[
                    self._batch_pos
                ].as_py()
            else:
                value = self._cur_batch.column(arrow_idx)[self._batch_pos].as_py()
                converter = self._field_converters[arrow_idx]
                rec._set(col_idx, converter(value))

        self._batch_pos += 1
        self._total_pos += 1
        return rec
