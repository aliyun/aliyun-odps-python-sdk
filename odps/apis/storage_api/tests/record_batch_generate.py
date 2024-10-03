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

try:
    import pyarrow as pa
except ImportError:
    pa = None
import logging

RANDOM_STRING_LENGTH = 10
logger = logging.getLogger(__name__)


def generate_bigint_value(item, column_index, row_index):
    return (item.val + row_index) * (column_index + 1)


def generate_bigint_list(item, column_index):
    pylist = []
    for i in range(0, item.batch_size):
        pylist.append(generate_bigint_value(item, column_index, i))

    return pylist


def generate_string_value(item, column_index, row_index):
    str = ""
    for idx in range(0, RANDOM_STRING_LENGTH):
        c = chr(ord("a") + (item.val + row_index) * (column_index + idx) % 26)
        str += c

    return str


def generate_string_list(item, column_index):
    pylist = []
    for i in range(0, item.batch_size):
        pylist.append(generate_string_value(item, column_index, i))

    return pylist


def build_list_from_schema(item, type, column_index):
    if type == "bigint":
        return generate_bigint_list(item, column_index)
    elif type == "string":
        return generate_string_list(item, column_index)
    else:
        raise ValueError("Type " + type + " not supported yet")


def check_array_based_on_type_info(array, column_index, row_index, item, offset):
    if type(array) == pa.Int64Array:
        if array[row_index - offset].as_py() != generate_bigint_value(
            item, column_index, row_index % item.batch_size
        ):
            return False
    elif type(array) == pa.StringArray:
        if array[row_index - offset].as_py() != generate_string_value(
            item, column_index, row_index % item.batch_size
        ):
            return False
    else:
        raise ValueError("Type " + str(type(array)) + " not supported yet")

    return True


def verify_data(record_batch, item, total_line):
    offset = total_line % item.batch_size
    for i in range(0, len(item.data_columns)):
        for j in range(
            total_line % item.batch_size,
            total_line % item.batch_size + record_batch.num_rows,
        ):
            if not check_array_based_on_type_info(
                record_batch.column(i), i, j, item, offset
            ):
                logger.info("Row value is not correct")
                return False

    return True


def generate_data_based_on_odps_schema(item):
    build_arrays = []
    generate_data_columns_count = len(item.data_columns)

    if not item.has_partition:
        generate_data_columns_count -= 1

    name_list = []
    for i in range(0, generate_data_columns_count):
        array = build_list_from_schema(item, item.data_columns[i].type, i)
        name_list.append(item.data_columns[i].name)
        build_arrays.append(pa.array(array))

    if not item.has_partition:
        name_list.append(item.data_columns[generate_data_columns_count].name)
        partition_array = []
        for i in range(0, item.batch_size):
            partition_array.append("test_write_1")

        build_arrays.append(pa.array(partition_array))

    record_batch = pa.RecordBatch.from_arrays(build_arrays, name_list)

    return record_batch


def generate_base_table(item):
    return generate_data_based_on_odps_schema(item)
