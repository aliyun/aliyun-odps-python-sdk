#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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
import warnings

from odps.tests.core import TestBase, to_str
from odps.compat import unittest
from odps.tunnel.pb.wire_format import *
from odps.utils import to_binary


class Test(TestBase):

    def testPyEncodeAndDecode(self):
        from odps.tunnel.pb.encoder import Encoder
        from odps.tunnel.pb.decoder import Decoder

        encoder = Encoder()
        encoder.append_tag(0, WIRETYPE_VARINT)
        encoder.append_int32(2 ** 20)
        encoder.append_tag(1, WIRETYPE_VARINT)
        encoder.append_sint64(-2 ** 40)
        encoder.append_tag(2, WIRETYPE_LENGTH_DELIMITED)
        encoder.append_string(to_binary("hello"))
        encoder.append_tag(3, WIRETYPE_VARINT)
        encoder.append_bool(True)
        encoder.append_tag(4, WIRETYPE_FIXED64)
        encoder.append_float(3.14)
        encoder.append_double(0.31415926)
        encoder.append_tag(5, WIRETYPE_VARINT)
        encoder.append_uint32(2 ** 30)
        encoder.append_tag(6, WIRETYPE_VARINT)
        encoder.append_uint64(2 ** 40)
        buffer_size = len(encoder)

        tube = io.BytesIO(encoder.tostring())
        decoder = Decoder(tube)
        self.assertEqual((0, WIRETYPE_VARINT), decoder.read_field_number_and_wire_type())
        self.assertEqual(2**20, decoder.read_int32())
        self.assertEqual((1, WIRETYPE_VARINT), decoder.read_field_number_and_wire_type())
        self.assertEqual(-2**40, decoder.read_sint64())
        self.assertEqual((2, WIRETYPE_LENGTH_DELIMITED), decoder.read_field_number_and_wire_type())
        self.assertEqual(to_str("hello"), to_str(decoder.read_string()))
        self.assertEqual((3, WIRETYPE_VARINT), decoder.read_field_number_and_wire_type())
        self.assertEqual(True, decoder.read_bool())
        self.assertEqual((4, WIRETYPE_FIXED64), decoder.read_field_number_and_wire_type())
        self.assertAlmostEqual(3.14, decoder.read_float(), delta=0.001)
        self.assertEqual(0.31415926, decoder.read_double())
        self.assertEqual((5, WIRETYPE_VARINT), decoder.read_field_number_and_wire_type())
        self.assertEqual(2**30, decoder.read_uint32())
        self.assertEqual((6, WIRETYPE_VARINT), decoder.read_field_number_and_wire_type())
        self.assertEqual(2**40, decoder.read_uint64())
        self.assertEqual(buffer_size, decoder.position())

    def testCEncodeAndDecode(self):
        try:
            from odps.tunnel.pb.encoder_c import Encoder
            from odps.tunnel.pb.decoder_c import Decoder

            encoder = Encoder()
            encoder.append_tag(0, WIRETYPE_VARINT)
            encoder.append_tag(1, WIRETYPE_VARINT)
            encoder.append_sint64(-2 ** 40)
            encoder.append_tag(2, WIRETYPE_LENGTH_DELIMITED)
            encoder.append_string(to_binary("hello"))
            encoder.append_tag(3, WIRETYPE_VARINT)
            encoder.append_bool(True)
            encoder.append_tag(4, WIRETYPE_FIXED64)
            encoder.append_float(3.14)
            encoder.append_double(0.31415926)
            encoder.append_tag(5, WIRETYPE_VARINT)
            encoder.append_uint32(2 ** 30)
            encoder.append_tag(6, WIRETYPE_VARINT)
            encoder.append_uint64(2 ** 40)
            buffer_size = len(encoder)

            tube = io.BytesIO(encoder.tostring())
            decoder = Decoder(tube)
            self.assertEqual((0, WIRETYPE_VARINT), decoder.read_field_number_and_wire_type())
            self.assertEqual((1, WIRETYPE_VARINT), decoder.read_field_number_and_wire_type())
            self.assertEqual(-2 ** 40, decoder.read_sint64())
            self.assertEqual((2, WIRETYPE_LENGTH_DELIMITED), decoder.read_field_number_and_wire_type())
            self.assertEqual(to_str("hello"), to_str(decoder.read_string()))
            self.assertEqual((3, WIRETYPE_VARINT), decoder.read_field_number_and_wire_type())
            self.assertEqual(True, decoder.read_bool())
            self.assertEqual((4, WIRETYPE_FIXED64), decoder.read_field_number_and_wire_type())
            self.assertAlmostEqual(3.14, decoder.read_float(), delta=0.001)
            self.assertEqual(0.31415926, decoder.read_double())
            self.assertEqual((5, WIRETYPE_VARINT), decoder.read_field_number_and_wire_type())
            self.assertEqual(2 ** 30, decoder.read_uint32())
            self.assertEqual((6, WIRETYPE_VARINT), decoder.read_field_number_and_wire_type())
            self.assertEqual(2 ** 40, decoder.read_uint64())
            self.assertEqual(buffer_size, decoder.position())
        except ImportError:
            warnings.warn('No Encoder or Decoder built by cython found')


if __name__ == '__main__':
    unittest.main()
