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

from libc.stdint cimport *
from libc.string cimport *

cimport numpy as np

ctypedef union ValueUnion:
    int64_t *pl_array
    double *pdbl_array
    float *pflt_array
    np.npy_bool *pbool_array

ctypedef struct ArrayVariantPtrs:
    uint32_t wire_type
    ValueUnion v
