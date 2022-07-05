# encoding: utf-8
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

from ._customize import (
    ReplaceNull,
    ReplaceEmpty,
    ReplaceNullEmpty,
    ReplaceCustom,
    ReplacePercentile,
    ReplaceZScore,
    ReplaceConfidence,
)

from ..algolib.loader import load_defined_algorithms

# required by autodoc of sphinx
__all__ = [
    "ReplaceNull",
    "ReplaceEmpty",
    "ReplaceNullEmpty",
    "ReplaceCustom",
    "ReplacePercentile",
    "ReplaceZScore",
    "ReplaceConfidence",
]

load_defined_algorithms(sys.modules[__name__], "preprocess")
