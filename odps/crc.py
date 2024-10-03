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

import zlib

from .compat import six
from .config import options


class Crc32(object):
    def __init__(self):
        self.crc = None

    def update(self, buf, off=None, length=None):
        assert isinstance(buf, (six.binary_type, bytearray))
        off = off or 0
        length = length or len(buf)
        to_crc = buf[off : off + length]
        if isinstance(to_crc, bytearray):
            to_crc = bytes(to_crc)
        if self.crc:
            self.crc = zlib.crc32(to_crc, self.crc)
        else:
            self.crc = zlib.crc32(to_crc)

    def reset(self):
        self.crc = None

    def getvalue(self):
        return self.crc


try:
    if not options.force_py:
        from .src.crc32c_c import Crc32c
    else:
        Crc32c = None
except ImportError as e:
    if options.force_c:
        raise e
    Crc32c = None


if Crc32c is None:
    # fmt: off
    _CRC_TABLE = (
        0X00000000, 0XF26B8303, 0XE13B70F7, 0X1350F3F4,
        0XC79A971F, 0X35F1141C, 0X26A1E7E8, 0XD4CA64EB,
        0X8AD958CF, 0X78B2DBCC, 0X6BE22838, 0X9989AB3B,
        0X4D43CFD0, 0XBF284CD3, 0XAC78BF27, 0X5E133C24,
        0X105EC76F, 0XE235446C, 0XF165B798, 0X030E349B,
        0XD7C45070, 0X25AFD373, 0X36FF2087, 0XC494A384,
        0X9A879FA0, 0X68EC1CA3, 0X7BBCEF57, 0X89D76C54,
        0X5D1D08BF, 0XAF768BBC, 0XBC267848, 0X4E4DFB4B,
        0X20BD8EDE, 0XD2D60DDD, 0XC186FE29, 0X33ED7D2A,
        0XE72719C1, 0X154C9AC2, 0X061C6936, 0XF477EA35,
        0XAA64D611, 0X580F5512, 0X4B5FA6E6, 0XB93425E5,
        0X6DFE410E, 0X9F95C20D, 0X8CC531F9, 0X7EAEB2FA,
        0X30E349B1, 0XC288CAB2, 0XD1D83946, 0X23B3BA45,
        0XF779DEAE, 0X05125DAD, 0X1642AE59, 0XE4292D5A,
        0XBA3A117E, 0X4851927D, 0X5B016189, 0XA96AE28A,
        0X7DA08661, 0X8FCB0562, 0X9C9BF696, 0X6EF07595,
        0X417B1DBC, 0XB3109EBF, 0XA0406D4B, 0X522BEE48,
        0X86E18AA3, 0X748A09A0, 0X67DAFA54, 0X95B17957,
        0XCBA24573, 0X39C9C670, 0X2A993584, 0XD8F2B687,
        0X0C38D26C, 0XFE53516F, 0XED03A29B, 0X1F682198,
        0X5125DAD3, 0XA34E59D0, 0XB01EAA24, 0X42752927,
        0X96BF4DCC, 0X64D4CECF, 0X77843D3B, 0X85EFBE38,
        0XDBFC821C, 0X2997011F, 0X3AC7F2EB, 0XC8AC71E8,
        0X1C661503, 0XEE0D9600, 0XFD5D65F4, 0X0F36E6F7,
        0X61C69362, 0X93AD1061, 0X80FDE395, 0X72966096,
        0XA65C047D, 0X5437877E, 0X4767748A, 0XB50CF789,
        0XEB1FCBAD, 0X197448AE, 0X0A24BB5A, 0XF84F3859,
        0X2C855CB2, 0XDEEEDFB1, 0XCDBE2C45, 0X3FD5AF46,
        0X7198540D, 0X83F3D70E, 0X90A324FA, 0X62C8A7F9,
        0XB602C312, 0X44694011, 0X5739B3E5, 0XA55230E6,
        0XFB410CC2, 0X092A8FC1, 0X1A7A7C35, 0XE811FF36,
        0X3CDB9BDD, 0XCEB018DE, 0XDDE0EB2A, 0X2F8B6829,
        0X82F63B78, 0X709DB87B, 0X63CD4B8F, 0X91A6C88C,
        0X456CAC67, 0XB7072F64, 0XA457DC90, 0X563C5F93,
        0X082F63B7, 0XFA44E0B4, 0XE9141340, 0X1B7F9043,
        0XCFB5F4A8, 0X3DDE77AB, 0X2E8E845F, 0XDCE5075C,
        0X92A8FC17, 0X60C37F14, 0X73938CE0, 0X81F80FE3,
        0X55326B08, 0XA759E80B, 0XB4091BFF, 0X466298FC,
        0X1871A4D8, 0XEA1A27DB, 0XF94AD42F, 0X0B21572C,
        0XDFEB33C7, 0X2D80B0C4, 0X3ED04330, 0XCCBBC033,
        0XA24BB5A6, 0X502036A5, 0X4370C551, 0XB11B4652,
        0X65D122B9, 0X97BAA1BA, 0X84EA524E, 0X7681D14D,
        0X2892ED69, 0XDAF96E6A, 0XC9A99D9E, 0X3BC21E9D,
        0XEF087A76, 0X1D63F975, 0X0E330A81, 0XFC588982,
        0XB21572C9, 0X407EF1CA, 0X532E023E, 0XA145813D,
        0X758FE5D6, 0X87E466D5, 0X94B49521, 0X66DF1622,
        0X38CC2A06, 0XCAA7A905, 0XD9F75AF1, 0X2B9CD9F2,
        0XFF56BD19, 0X0D3D3E1A, 0X1E6DCDEE, 0XEC064EED,
        0XC38D26C4, 0X31E6A5C7, 0X22B65633, 0XD0DDD530,
        0X0417B1DB, 0XF67C32D8, 0XE52CC12C, 0X1747422F,
        0X49547E0B, 0XBB3FFD08, 0XA86F0EFC, 0X5A048DFF,
        0X8ECEE914, 0X7CA56A17, 0X6FF599E3, 0X9D9E1AE0,
        0XD3D3E1AB, 0X21B862A8, 0X32E8915C, 0XC083125F,
        0X144976B4, 0XE622F5B7, 0XF5720643, 0X07198540,
        0X590AB964, 0XAB613A67, 0XB831C993, 0X4A5A4A90,
        0X9E902E7B, 0X6CFBAD78, 0X7FAB5E8C, 0X8DC0DD8F,
        0XE330A81A, 0X115B2B19, 0X020BD8ED, 0XF0605BEE,
        0X24AA3F05, 0XD6C1BC06, 0XC5914FF2, 0X37FACCF1,
        0X69E9F0D5, 0X9B8273D6, 0X88D28022, 0X7AB90321,
        0XAE7367CA, 0X5C18E4C9, 0X4F48173D, 0XBD23943E,
        0XF36E6F75, 0X0105EC76, 0X12551F82, 0XE03E9C81,
        0X34F4F86A, 0XC69F7B69, 0XD5CF889D, 0X27A40B9E,
        0X79B737BA, 0X8BDCB4B9, 0X988C474D, 0X6AE7C44E,
        0XBE2DA0A5, 0X4C4623A6, 0X5F16D052, 0XAD7D5351,
    )
    # fmt: on

    _CRC_INIT = 0xFFFFFFFF

    class Crc32c(object):
        _method = "py"

        def __init__(self):
            self.crc = _CRC_INIT

        def update(self, buf, off=None, length=None):
            """
            :param buf: buf to update
            :type buf: bytearray
            :param off: offset
            :param length: length
            """

            off = off or 0
            length = length or len(buf)

            to_crc = buf[off : off + length]

            crc = self.crc
            for b in to_crc:
                table_index = (crc ^ b) & 0xFF
                crc = (_CRC_TABLE[table_index] ^ (crc >> 8)) & 0xFFFFFFFF
            self.crc = crc & 0xFFFFFFFF

        def reset(self):
            self.crc = _CRC_INIT

        @classmethod
        def _crc_finalize(cls, crc):
            return crc ^ 0xFFFFFFFF

        def getvalue(self):
            return Crc32c._crc_finalize(self.crc)
