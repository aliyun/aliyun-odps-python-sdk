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

import json
from io import BytesIO

import numpy as np
from mars import opcodes as OperandDef
from mars.serialize import (
    ValueType,
    ListField,
    TupleField,
    StringField,
    KeyField,
    BoolField,
)
from mars.tensor.array_utils import as_np_array
from mars.tensor.datastore.core import TensorDataStore
from mars.lib.sparse import SparseNDArray


class TensorStoreCOO(TensorDataStore):
    _op_type_ = OperandDef.STORE_COO

    _input = KeyField("input")
    _path = StringField("path")
    _dim_cols = ListField("dim_cols", ValueType.string)
    _value_col = StringField("value_col")
    _storage_options = StringField("storage_options")
    _global_index = BoolField("global_index", default=False)
    _axis_offsets = TupleField("axis_offsets")

    def __init__(
        self,
        dtype=None,
        path=None,
        dim_cols=None,
        value_col=None,
        storage_options=None,
        sparse=True,
        global_index=False,
        **kw
    ):
        super(TensorStoreCOO, self).__init__(
            _path=path,
            _dim_cols=dim_cols,
            _value_col=value_col,
            _dtype=dtype,
            _storage_options=storage_options,
            _global_index=global_index,
            _sparse=sparse,
            **kw
        )

    @property
    def input(self):
        return self._input

    @property
    def path(self):
        return self._path

    @property
    def dim_cols(self):
        return self._dim_cols

    @property
    def value_col(self):
        return self._value_col

    @property
    def storage_options(self):
        return self._storage_options

    @property
    def global_index(self):
        return self._global_index

    @property
    def axis_offsets(self):
        return self._axis_offsets

    def _set_inputs(self, inputs):
        super(TensorStoreCOO, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def calc_shape(self, *inputs_shape):
        return (0,) * len(inputs_shape[0])

    @classmethod
    def tile(cls, op):
        in_tensor = op.input

        out_chunks = []
        out_chunk_shape = (0,) * in_tensor.ndim
        axis_offsets = [[0] + np.cumsum(ns)[:-1].tolist() for ns in in_tensor.nsplits]
        for chunk in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            chunk_path = "%s/%s.parquet" % (
                chunk_op.path,
                ",".join(str(j) for j in chunk.index),
            )
            chunk_op._path = chunk_path
            chunk_op._axis_offsets = tuple(
                axis_offsets[axis][idx] for axis, idx in enumerate(chunk.index)
            )
            out_chunk = chunk_op.new_chunk(
                [chunk], shape=out_chunk_shape, index=chunk.index
            )
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(
            op.inputs,
            op.outputs[0].shape,
            chunks=out_chunks,
            nsplits=((0,) * len(ns) for ns in in_tensor.nsplits),
        )

    @classmethod
    def execute(cls, ctx, op):
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ..io import open as fs_open

        to_store_data = ctx[op.inputs[0].key]
        storage_opts = json.loads(op.storage_options)
        axis_offsets = op.axis_offsets
        store_global_index = op.global_index
        dim_cols = op.dim_cols
        col_to_array = {}

        if isinstance(to_store_data, SparseNDArray):
            # sparse, convert to coo matrix
            matrix = to_store_data.raw.tocoo(copy=False)
            ndim = matrix.ndim

            if len(dim_cols) > 1:
                col_to_array[dim_cols[0]] = matrix.row
                if store_global_index:
                    # global index
                    col_to_array["global_" + dim_cols[0]] = matrix.row + axis_offsets[0]
                col_to_array[dim_cols[1]] = matrix.col
                if store_global_index:
                    col_to_array["global_" + dim_cols[1]] = matrix.col + axis_offsets[1]
            else:
                col_to_array[dim_cols[0]] = matrix.col
                if store_global_index:
                    col_to_array["global_" + dim_cols[0]] = matrix.col + axis_offsets[0]

            col_to_array[op.value_col] = matrix.data
        else:
            # dense, convert to numpy array
            arr = as_np_array(to_store_data)
            ndim = arr.ndim

            index = (
                np.array(np.meshgrid(*[np.arange(s) for s in arr.shape]))
                .T.reshape(-1, arr.ndim)
                .T
            )
            for j, col, ind in zip(range(len(dim_cols)), dim_cols, index):
                col_to_array[col] = ind
                if store_global_index:
                    col_to_array["global_" + col] = ind + axis_offsets[j]
            col_to_array[op.value_col] = arr.ravel()

        df = pd.DataFrame(col_to_array)
        if len(op.dim_cols) > ndim:
            for col in op.dim_cols[ndim:]:
                df[col] = None
        table = pa.Table.from_pandas(df)
        bio = BytesIO()
        pq.write_table(table, bio)
        bio.seek(0)

        # write oss
        with fs_open(op.path, "wb", **storage_opts) as out_file:
            out_file.write(bio.read())

        ctx[op.outputs[0].key] = np.empty((0,) * to_store_data.ndim)


def write_coo(a, path, dim_cols, value_col, **storage_opts):
    if a.ndim > len(dim_cols):
        raise ValueError(
            "dim_cols must have the same size as or greater than tensor's ndim"
        )

    global_index = storage_opts.pop("global_index", True)
    op = TensorStoreCOO(
        dtype=a.dtype,
        path=path,
        dim_cols=dim_cols,
        value_col=value_col,
        storage_options=json.dumps(storage_opts),
        sparse=a.issparse(),
        global_index=global_index,
    )
    return op(a)
