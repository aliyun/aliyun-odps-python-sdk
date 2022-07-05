# -*- coding: utf-8 -*-
# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

from ..io.stream import CompressOption, SnappyOutputStream, DeflateOutputStream, RequestsIO
from ... import errors, options
from ...compat import six

try:
    from .pdwriter_c import BasePandasWriter
except ImportError:
    BasePandasWriter = None


if BasePandasWriter:
    class TunnelPandasWriter(BasePandasWriter):
        def __init__(self, schema, request_callback, compress_option=None):
            self._req_io = RequestsIO(request_callback, chunk_size=options.chunk_size)

            if compress_option is None:
                out = self._req_io
            elif compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_RAW:
                out = self._req_io
            elif compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_ZLIB:
                out = DeflateOutputStream(self._req_io)
            elif compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_SNAPPY:
                out = SnappyOutputStream(self._req_io)
            else:
                raise errors.InvalidArgument('Invalid compression algorithm.')
            super(TunnelPandasWriter, self).__init__(schema, out)
            self._req_io.start()

        def write(self, data, columns=None, limit=-1, dim_offsets=None):
            if self._req_io._async_err:
                ex_type, ex_value, tb = self._req_io._async_err
                six.reraise(ex_type, ex_value, tb)
            super(TunnelPandasWriter, self).write(data, columns=columns, limit=limit,
                                                  dim_offsets=dim_offsets)

        write.__doc__ = BasePandasWriter.write.__doc__

        def close(self):
            super(TunnelPandasWriter, self).close()
            self._req_io.finish()
