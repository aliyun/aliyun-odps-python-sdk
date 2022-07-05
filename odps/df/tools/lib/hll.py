#!/usr/bin/env python
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

import sys
from math import ceil, log, pow


if sys.version < '3':
    PY3 = False
else:
    PY3 = True


def get_SHA1_bin(word):
    """
    Return SHA1 hash of any string

    :param word:
    :return:
    """
    from hashlib import sha1
    if PY3 and isinstance(word, str):
        word = word.encode('utf-8')

    hash_s = sha1()
    hash_s.update(word)
    return bin(int(hash_s.hexdigest(), 16))[2:].zfill(160)


def get_index(binstr, end_index=160):
    """
    Return the position of the first 1 bit
    from the left in the word until end_index

    :param binstr:
    :param end_index:
    :return:
    """
    res = -1
    try:
        res = binstr.index('1') + 1
    except ValueError:
        res = end_index
    return res


class HyperLogLog(object):
    """
    Implements a HyperLogLog
    """

    __ALPHA16 = 0.673
    __ALPHA32 = 0.697
    __ALPHA64 = 0.709

    def __init__(self, error_rate, splitter=None):
        if not (0 < error_rate < 1):
            raise ValueError('error_rate must be between 0 and 1.')

        self._k = int(ceil(log(pow(1.04 / error_rate, 2), 2)))
        self._bucket_number = 1 << self._k
        self._alpha = self.__get_alpha(self._bucket_number)

        self._splitter = splitter

    def buffer(self):
        return [0] * self._bucket_number

    def __get_alpha(self, m):
        if m <= 16:
            return self.__ALPHA16
        elif m <= 32:
            return self.__ALPHA32
        elif m <= 64:
            return self.__ALPHA64
        else:
            return 0.7213 / (1 + 1.079 / float(m))

    def __call__(self, buffer, item):
        """
        Add the items to the HyperLogLog.

        :param buffer:
        :param item:
        :return:
        """

        items = [str(item), ]
        if self._splitter is not None:
            items = str(item).split(self._splitter)

        for item in items:
            binword = get_SHA1_bin(item)
            pos = int(binword[:self._k], 2)
            # retrieve the position of leftmost 1
            aux = get_index(binword[self._k:], 160 - self._k)
            # set its own register value to maximum value seen so far
            buffer[pos] = max(aux, buffer[pos])

    def _estimate(self, buffer):
        """
        Return the estimate of the cardinality

        :return: esitimate of the cardinality
        """
        m = self._bucket_number
        raw_e = self._alpha * pow(m, 2) / sum([pow(2, -x) for x in buffer])
        if raw_e <= 5 / 2.0 * m:
            v = buffer.count(0)
            if v != 0:
                return m * log(m / float(v), 2)
            else:
                return raw_e
        elif raw_e <= 1 / 30.0 * 2 ** 160:
            return raw_e
        else:
            return -2 ** 160 * log(1 - raw_e / 2.0 ** 160, 2)

    def getvalue(self, buffer):
        return int(round(self._estimate(buffer)))

    def merge(self, buffer, other_hyper_log_log):
        """
        Merge the HyperLogLog

        :param other_hyper_log_log:
        :return:
        """

        for i in range(len(buffer)):
            buffer[i] = max(buffer[i], other_hyper_log_log[i])

    def _word_size_calculator(self, n_max):
        """
        Estimate the size of the memory units, using the maximum cardinality as an argument
        :param n_max: maximum cardinality
        :return: size of the memory units
        """

        return int(ceil(log(log(n_max, 2), 2)))
