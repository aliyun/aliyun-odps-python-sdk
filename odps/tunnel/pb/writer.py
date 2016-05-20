#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from . import Encoder


class ProtobufWriter(object):
    """ProtobufWriter is a stream-interface wrapper around pywirebuf_encoder.Encoder(cpp)
    and encoder.Encoder(py)
    """

    DEFAULT_BUFFER_SIZE = 4096

    def __init__(self, output, buffer_size=None):
        self._encoder = Encoder()
        self._output = output
        self._buffer_size = buffer_size or self.DEFAULT_BUFFER_SIZE
        self._n_total = 0

    def _mode(self):
        try:
            from .encoder_c import Encoder

            return 'c' if isinstance(self._encoder, Encoder) else 'py'
        except ImportError:
            return 'py'

    def flush(self):
        if len(self._encoder) > 0:
            data = self._encoder.tostring()
            self._output.write(data)
            self._n_total += len(self._encoder)
            self._encoder = Encoder()

    def close(self):
        self.flush()
        self._output.flush()

    def _refresh_buffer(self):
        """Control the buffer size of _encoder. Flush if necessary"""
        if len(self._encoder) > self._buffer_size:
            self.flush()

    @property
    def n_bytes(self):
        return self._n_total + len(self._encoder)

    def __len__(self):
        return self.n_bytes

    def write_tag(self, field_num, wire_type):
        self._encoder.append_tag(field_num, wire_type)
        self._refresh_buffer()

    def write_long(self, val):
        self._encoder.append_sint64(val)
        self._refresh_buffer()

    def write_uint(self, val):
        self._encoder.append_uint32(val)
        self._refresh_buffer()

    def write_bool(self, val):
        self._encoder.append_bool(val)
        self._refresh_buffer()

    def write_double(self, val):
        self._encoder.append_double(val)
        self._refresh_buffer()

    def write_string(self, val):
        self._encoder.append_string(val)
        self._refresh_buffer()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
