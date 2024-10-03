#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import io

import pytest

from ...utils import to_binary, to_text
from ..pb import wire_format
from ..pb.decoder import Decoder as PyDecoder
from ..pb.encoder import Encoder as PyEncoder

try:
    from ..pb.decoder_c import Decoder as CDecoder
    from ..pb.encoder_c import Encoder as CEncoder
except ImportError:
    CEncoder = CDecoder = None


@pytest.mark.parametrize(
    "encoder_cls, decoder_cls",
    [
        pytest.param(PyEncoder, PyDecoder, id="py"),
        pytest.param(CEncoder, CDecoder, id="c"),
    ],
)
def test_encode_and_decode(encoder_cls, decoder_cls):
    if encoder_cls is None or decoder_cls is None:
        pytest.skip("No Encoder or Decoder built by cython found")

    encoder = encoder_cls()
    encoder.append_tag(0, wire_format.WIRETYPE_VARINT)
    encoder.append_tag(1, wire_format.WIRETYPE_VARINT)
    encoder.append_sint64(-(2**40))
    encoder.append_tag(2, wire_format.WIRETYPE_LENGTH_DELIMITED)
    encoder.append_string(to_binary("hello"))
    encoder.append_tag(3, wire_format.WIRETYPE_VARINT)
    encoder.append_bool(True)
    encoder.append_tag(4, wire_format.WIRETYPE_FIXED64)
    encoder.append_float(3.14)
    encoder.append_double(0.31415926)
    encoder.append_tag(5, wire_format.WIRETYPE_VARINT)
    encoder.append_uint32(2**30)
    encoder.append_tag(6, wire_format.WIRETYPE_VARINT)
    encoder.append_uint64(2**40)
    buffer_size = len(encoder)

    tube = io.BytesIO(encoder.tostring())
    decoder = decoder_cls(tube)
    assert (0, wire_format.WIRETYPE_VARINT) == decoder.read_field_number_and_wire_type()
    assert (1, wire_format.WIRETYPE_VARINT) == decoder.read_field_number_and_wire_type()
    assert -(2**40) == decoder.read_sint64()
    assert (
        2,
        wire_format.WIRETYPE_LENGTH_DELIMITED,
    ) == decoder.read_field_number_and_wire_type()
    assert to_text("hello") == to_text(decoder.read_string())
    assert (3, wire_format.WIRETYPE_VARINT) == decoder.read_field_number_and_wire_type()
    assert decoder.read_bool()
    assert (
        4,
        wire_format.WIRETYPE_FIXED64,
    ) == decoder.read_field_number_and_wire_type()
    assert pytest.approx(3.14) == decoder.read_float()
    assert 0.31415926 == decoder.read_double()
    assert (5, wire_format.WIRETYPE_VARINT) == decoder.read_field_number_and_wire_type()
    assert 2**30 == decoder.read_uint32()
    assert (6, wire_format.WIRETYPE_VARINT) == decoder.read_field_number_and_wire_type()
    assert 2**40 == decoder.read_uint64()
    assert buffer_size == len(decoder)
