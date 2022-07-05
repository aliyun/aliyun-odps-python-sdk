# !/usr/bin/env python
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

import logging
import json
import itertools

import numpy as np
from mars.tensor.datasource.core import TensorNoInput
from mars.tensor.utils import decide_chunk_sizes, normalize_shape
from mars.serialize import ValueType, ListField, StringField
from mars import opcodes as OperandDef

from ..io import glob
from ....compat import izip, BytesIO

logger = logging.getLogger(__name__)


class TensorTableCOO(TensorNoInput):
    _op_type_ = OperandDef.TABLE_COO

    _paths = ListField("paths", ValueType.string)
    _dim_cols = ListField("dim_cols", ValueType.string)
    _value_col = StringField("value_col")
    _storage_options = StringField("storage_options")

    def __init__(
        self,
        dtype=None,
        paths=None,
        dim_cols=None,
        value_col=None,
        storage_options=None,
        sparse=True,
        **kw
    ):
        super(TensorTableCOO, self).__init__(
            _paths=paths,
            _dim_cols=dim_cols,
            _value_col=value_col,
            _dtype=dtype,
            _storage_options=storage_options,
            _sparse=sparse,
            **kw
        )

    @property
    def paths(self):
        return self._paths

    @property
    def dim_cols(self):
        return self._dim_cols

    @property
    def value_col(self):
        return self._value_col

    @property
    def storage_options(self):
        return self._storage_options

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]

        storage_opts = json.loads(op.storage_options)

        logger.debug("Start scanning data files in %s", op.paths[0])
        chunk_files = dict()
        for key in glob(op.paths[0], **storage_opts):
            file_name, _ = key.rsplit(".", 1)
            _, fn_suffix = file_name.rsplit("/", 1)
            dim_suffix = fn_suffix.rsplit("@", 1)[-1]
            dim_indices = tuple(int(pt) for pt in dim_suffix.split(","))
            if dim_indices not in chunk_files:
                chunk_files[dim_indices] = []
            chunk_files[dim_indices].append(key)
        logger.debug("Finish scanning data files in %s", op.paths[0])

        try:
            target_chunk_size = tensor.params.raw_chunk_size
        except AttributeError:
            target_chunk_size = tensor.extra_params.raw_chunk_size
        chunk_size = decide_chunk_sizes(
            tensor.shape, target_chunk_size, tensor.dtype.itemsize
        )
        chunk_size_idxes = (range(len(size)) for size in chunk_size)

        out_chunks = []
        for chunk_shape, chunk_idx in izip(
            itertools.product(*chunk_size), itertools.product(*chunk_size_idxes)
        ):
            chunk_op = op.copy().reset_key()
            chunk_op._paths = chunk_files.get(chunk_idx, [])
            out_chunk = chunk_op.new_chunk(None, shape=chunk_shape, index=chunk_idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(
            op.inputs, tensor.shape, nsplits=chunk_size, chunks=out_chunks
        )

    @classmethod
    def execute(cls, ctx, op):
        import pyarrow.parquet as pq
        import pandas as pd
        import scipy.sparse as sps
        from mars.lib.sparse import SparseNDArray
        from ..io import open as fs_open

        dfs = []
        storage_opts = json.loads(op.storage_options)
        for p in op.paths:
            with fs_open(p, "rb", **storage_opts) as inp_file:
                f = inp_file.read()
                dfs.append(pq.read_table(BytesIO(f)).to_pandas())

        chunk = op.outputs[0]
        if op.sparse and len(dfs) == 0:
            if len(chunk.shape) == 1:
                csr_array = sps.csr_matrix((chunk.shape[0], 1))
                ctx[chunk.key] = SparseNDArray(csr_array, shape=chunk.shape)
            else:
                csr_array = sps.csr_matrix(chunk.shape)
                ctx[chunk.key] = SparseNDArray(csr_array)
            return

        df_merged = pd.concat(dfs, ignore_index=True)
        dim_arrays = [df_merged[col] for col in op.dim_cols]
        value_array = df_merged[op.value_col].astype(chunk.dtype)
        del df_merged

        if op.sparse:
            if len(chunk.shape) == 1:
                dim_arrays.append(np.zeros((len(dim_arrays[0]))))
                csr_array = sps.csr_matrix(
                    (value_array, tuple(dim_arrays)), shape=(chunk.shape[0], 1)
                )
            else:
                csr_array = sps.csr_matrix(
                    (value_array, tuple(dim_arrays)), shape=chunk.shape
                )
            del dim_arrays, value_array
            ctx[chunk.key] = SparseNDArray(csr_array, shape=chunk.shape)
        else:
            arr = np.empty(chunk.shape, dtype=value_array.dtype)
            arr[tuple(dim_arrays)] = value_array
            ctx[chunk.key] = arr


def read_coo(
    path_pattern,
    dim_cols,
    value_col,
    dtype=float,
    shape=None,
    chunk_size=None,
    sparse=False,
    **storage_opts
):
    if sparse and len(dim_cols) > 2:
        raise ValueError("Can only support reading 1-d or 2-d data if sparse")

    dtype = np.dtype(dtype)
    op = TensorTableCOO(
        dtype=dtype,
        paths=[path_pattern],
        dim_cols=dim_cols,
        value_col=value_col,
        storage_options=json.dumps(storage_opts),
        sparse=sparse,
    )
    return op(normalize_shape(shape), chunk_size=chunk_size)
