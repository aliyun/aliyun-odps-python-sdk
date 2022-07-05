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
# Modified from part of python-hashes by sangelone.


import math
import hashlib
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle

if sys.version < '3':
    PY3 = False
else:
    PY3 = True


default_hashbits = 96


class HashType(object):
    def __init__(self, value='', hashbits=default_hashbits, hash_=None):
        "Relies on create_hash() provided by subclass"
        self.hashbits = hashbits
        if hash_:
            self.hash = hash_
        else:
            self.create_hash(value)

    def __trunc__(self):
        return self.hash

    def __str__(self):
        return str(self.hash)

    def __int__(self):
        return int(self.hash)

    def __float__(self):
        return float(self.hash)

    def __cmp__(self, other):
        if self.hash < int(other): return -1
        if self.hash > int(other): return 1
        return 0

    def hex(self):
        return hex(self.hash)

    def hamming_distance(self, other_hash):
        x = (self.hash ^ other_hash.hash) & ((1 << self.hashbits) - 1)
        tot = 0
        while x:
            tot += 1
            x &= x-1
        return tot


class BloomFilter(HashType):
    def __init__(self, capacity=3000, false_positive_rate=0.01, value=''):
        """
        'value' is the initial string or list of strings to hash,
        'capacity' is the expected upper limit on items inserted, and
        'false_positive_rate' is self-explanatory but the smaller it is, the larger your hashes!
        """
        self.capacity = capacity
        self.error_rate = false_positive_rate
        self.create_hash(value, capacity, false_positive_rate)

    def create_hash(self, initial, capacity, error):
        """
        Calculates a Bloom filter with the specified parameters.
        Initalizes with a string or list/set/tuple of strings. No output.

        Reference material: http://bitworking.org/news/380/bloom-filter-resources
        """
        self.hash = 0
        self.hashbits, self.num_hashes = self._optimal_size(capacity, error)

        if len(initial):
            if type(initial) == str:
                self.add(initial)
            else:
                for t in initial:
                    self.add(t)

    def _binary(self, item):
        if PY3 and isinstance(item, str):
            item = item.encode('utf-8')

        return item

    def _hashes(self, item):
        """
        To create the hash functions we use the SHA-1 hash of the
        string and chop that up into 20 bit values and then
        mod down to the length of the Bloom filter.
        """
        item = self._binary(item)

        m = hashlib.sha1()
        m.update(item)
        digits = m.hexdigest()

        # Add another 160 bits for every 8 (20-bit long) hashes we need
        for i in range(int(self.num_hashes // 8)):
            m.update(self._binary(str(i)))
            digits += m.hexdigest()

        hashes = [int(digits[i*5:i*5+5], 16) % self.hashbits for i in range(self.num_hashes)]
        return hashes

    def _optimal_size(self, capacity, error):
        """Calculates minimum number of bits in filter array and
        number of hash functions given a number of enteries (maximum)
        and the desired error rate (falese positives).

        Example:
            m, k = self._optimal_size(3000, 0.01)   # m=28756, k=7
        """
        m = math.ceil((capacity * math.log(error)) / math.log(1.0 / (math.pow(2.0, math.log(2.0)))))
        k = math.ceil(math.log(2.0) * m / capacity)
        return int(m), int(k)

    def add(self, item):
        "Add an item (string) to the filter. Cannot be removed later!"
        for pos in self._hashes(item):
            self.hash |= (2 ** pos)

    def union(self, other):
        if self.capacity != other.capacity or self.error_rate != other.error_rate:
            raise ValueError('Unioning filters requires both filters to have both '
                             'the same capacity and error rate')

        new_bloom = BloomFilter(capacity=self.capacity,
                                false_positive_rate=self.error_rate)
        new_bloom.hash = self.hash
        new_bloom.hash |= other.hash
        return new_bloom

    def intersection(self, other):
        if self.capacity != other.capacity or self.error_rate != other.error_rate:
            raise ValueError('Intersecting filters requires both filters to have both '
                             'the same capacity and error rate')

        new_bloom = BloomFilter(capacity=self.capacity,
                                false_positive_rate=self.error_rate)
        new_bloom.hash = self.hash
        new_bloom.hash &= other.hash
        return new_bloom

    def __contains__(self, name):
        "This function is used by the 'in' keyword"
        retval = True
        for pos in self._hashes(name):
            retval = retval and bool(self.hash & (2 ** pos))
        return retval
