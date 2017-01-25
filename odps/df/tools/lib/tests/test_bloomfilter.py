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

from odps.compat import unittest, irange as xrange
from odps.df.tools.lib import BloomFilter
from odps.tests.core import TestBase


class Test(TestBase):
    def test_union(self):
        bloom_one = BloomFilter(100, 0.001)
        bloom_two = BloomFilter(100, 0.001)
        chars = [chr(i) for i in range(97, 123)]
        for char in chars[int(len(chars) / 2):]:
            bloom_one.add(char)
        for char in chars[:int(len(chars) / 2)]:
            bloom_two.add(char)
        new_bloom = bloom_one.union(bloom_two)
        for char in chars:
            self.assertTrue(char in new_bloom)

    def test_intersection(self):
        bloom_one = BloomFilter(100, 0.001)
        bloom_two = BloomFilter(100, 0.001)
        chars = [chr(i) for i in xrange(97, 123)]
        for char in chars:
            bloom_one.add(char)
        for char in chars[:int(len(chars) / 2)]:
            bloom_two.add(char)
        new_bloom = bloom_one.intersection(bloom_two)
        for char in chars[:int(len(chars) / 2)]:
            self.assertTrue(char in new_bloom)
        for char in chars[int(len(chars) / 2):]:
            self.assertTrue(char not in new_bloom)

    def test_intersection_capacity_fail(self):
        bloom_one = BloomFilter(1000, 0.001)
        bloom_two = BloomFilter(100, 0.001)

        def _run():
            new_bloom = bloom_one.intersection(bloom_two)

        self.assertRaises(ValueError, _run)

    def test_union_capacity_fail(self):
        bloom_one = BloomFilter(1000, 0.001)
        bloom_two = BloomFilter(100, 0.001)

        def _run():
            new_bloom = bloom_one.union(bloom_two)

        self.assertRaises(ValueError, _run)

    def test_intersection_k_fail(self):
        bloom_one = BloomFilter(100, 0.001)
        bloom_two = BloomFilter(100, 0.01)

        def _run():
            new_bloom = bloom_one.intersection(bloom_two)

        self.assertRaises(ValueError, _run)

    def test_union_k_fail(self):
        bloom_one = BloomFilter(100, 0.01)
        bloom_two = BloomFilter(100, 0.001)

        def _run():
            new_bloom = bloom_one.union(bloom_two)

        self.assertRaises(ValueError, _run)

if __name__ == '__main__':
    unittest.main()