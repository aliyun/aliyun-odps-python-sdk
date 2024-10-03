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

from ..compat import six
from ..config import options
from ..errors import ODPSClientError
from ..lib.tblib import pickling_support
from ..readers import AbstractRecordReader
from ..utils import call_with_retry

pickling_support.install()


class TunnelReaderMixin(object):
    @property
    def count(self):
        raise NotImplementedError

    def _to_pandas_with_processes(
        self, start=None, count=None, columns=None, append_partitions=None, n_process=1
    ):
        import multiprocessing
        from multiprocessing import Pipe

        import pandas as pd

        session_id = self._download_session.id
        start = start or 0
        count = count or self._download_session.count
        count = min(count, self._download_session.count - start)
        try:
            _mp_context = multiprocessing.get_context("fork")
        except ValueError:
            _mp_context = multiprocessing.get_context("spawn")
        except AttributeError:
            # for py27 compatibility
            _mp_context = multiprocessing

        n_process = min(count, n_process)
        split_count = count // n_process + (count % n_process != 0)
        conns = []
        for i in range(n_process):
            parent_conn, child_conn = Pipe()

            p = _mp_context.Process(
                target=self._get_process_split_reader(
                    columns=columns, append_partitions=append_partitions
                ),
                args=(child_conn, session_id, start, split_count, i),
            )
            p.start()
            start += split_count
            conns.append(parent_conn)

        try:
            results = [c.recv() for c in conns]
        except EOFError:
            six.raise_from(
                ODPSClientError(
                    "Read process ended unexpectedly. Try finding errors outputed above."
                ),
                None,
            )
        splits = sorted(results, key=lambda x: x[0])
        if any(not d[2] for d in splits):
            exc_info = next(d[1] for d in splits if not d[2])
            six.reraise(*exc_info)
        return pd.concat([d[1] for d in splits]).reset_index(drop=True)

    def _get_process_split_reader(self, columns=None, append_partitions=None):
        raise NotImplementedError

    def _open_and_iter_reader(
        self,
        start,
        record_count,
        step=None,
        compress=False,
        columns=None,
        append_partitions=None,
        counter=None,
    ):
        raise NotImplementedError

    def iter_pandas(
        self, batch_size=None, start=None, count=None, columns=None, **kwargs
    ):
        batch_size = batch_size or options.tunnel.read_row_batch_size
        start = start or 0
        count = count or self.count
        for st in range(start, start + count, batch_size):
            cur_batch_size = min(batch_size, count - (st - start))
            yield self.to_pandas(
                start=st, count=cur_batch_size, columns=columns, **kwargs
            )


class TunnelRecordReader(TunnelReaderMixin, AbstractRecordReader):
    def __init__(self, parent, download_session, columns=None, append_partitions=None):
        self._it = iter(self)
        self._parent = parent
        self._download_session = download_session
        self._column_names = columns
        self._append_partitions = append_partitions

    @property
    def download_id(self):
        return self._download_session.id

    @property
    def count(self):
        return self._download_session.count

    @property
    def status(self):
        return self._download_session.status

    def __iter__(self):
        for record in self.read():
            yield record

    def __next__(self):
        return next(self._it)

    next = __next__

    def _iter(
        self,
        start=None,
        end=None,
        step=None,
        compress=False,
        columns=None,
        append_partitions=None,
    ):
        count = self._calc_count(start, end, step)
        return self.read(
            start=start,
            count=count,
            step=step,
            compress=compress,
            columns=columns,
            append_partitions=append_partitions,
        )

    def _open_and_iter_reader(
        self,
        start,
        record_count,
        step=None,
        compress=False,
        columns=None,
        append_partitions=None,
        counter=None,
    ):
        counter = counter or [0]
        with call_with_retry(
            self._download_session.open_record_reader,
            start,
            record_count,
            compress=compress,
            columns=columns,
            append_partitions=append_partitions,
        ) as reader:
            for record in reader[::step]:
                counter[0] += step
                yield record

    def read(
        self,
        start=None,
        count=None,
        step=None,
        compress=False,
        append_partitions=None,
        columns=None,
    ):
        start = start or 0
        step = step or 1
        max_rec_count = self.count - start
        rec_count = (
            min(max_rec_count, count * step) if count is not None else max_rec_count
        )
        columns = columns or self._column_names
        append_partitions = (
            append_partitions
            if append_partitions is not None
            else self._append_partitions
        )

        if rec_count == 0:
            return

        for record in self._open_and_iter_reader(
            start,
            rec_count,
            step=step,
            compress=compress,
            append_partitions=append_partitions,
            columns=columns,
        ):
            yield record

    def to_pandas(
        self, start=None, count=None, columns=None, append_partitions=None, n_process=1
    ):
        columns = columns or self._column_names
        append_partitions = (
            append_partitions
            if append_partitions is not None
            else self._append_partitions
        )
        if not append_partitions and columns is None:
            columns = [c.name for c in self.schema.simple_columns]
        if n_process == 1 or self._download_session.count == 0:
            return super(TunnelRecordReader, self).to_pandas(
                start=start,
                count=count,
                columns=columns,
                append_partitions=append_partitions,
            )
        else:
            return self._to_pandas_with_processes(
                start=start,
                count=count,
                columns=columns,
                append_partitions=append_partitions,
                n_process=n_process,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TunnelArrowReader(TunnelReaderMixin):
    def __init__(self, parent, download_session, columns=None, append_partitions=False):
        self._it = iter(self)
        self._parent = parent
        self._download_session = download_session
        self._column_names = columns
        self._append_partitions = append_partitions

    @property
    def download_id(self):
        return self._download_session.id

    @property
    def count(self):
        return self._download_session.count

    @property
    def status(self):
        return self._download_session.status

    def __iter__(self):
        for batch in self.read():
            yield batch

    def __next__(self):
        return next(self._it)

    next = __next__

    def _open_and_iter_reader(
        self,
        start,
        record_count,
        step=None,
        compress=False,
        columns=None,
        append_partitions=None,
        counter=None,
    ):
        counter = counter or [0]
        with call_with_retry(
            self._download_session.open_arrow_reader,
            start,
            record_count,
            compress=compress,
            columns=columns,
            append_partitions=append_partitions,
        ) as reader:
            while True:
                batch = reader.read_next_batch()
                if batch is not None:
                    counter[0] += batch.num_rows
                    yield batch
                else:
                    break

    def read(
        self,
        start=None,
        count=None,
        compress=False,
        columns=None,
        append_partitions=None,
    ):
        start = start or 0
        max_rec_count = self.count - start
        rec_count = min(max_rec_count, count) if count is not None else max_rec_count
        columns = columns or self._column_names
        append_partitions = (
            append_partitions
            if append_partitions is not None
            else self._append_partitions
        )

        if rec_count == 0:
            return

        for batch in self._open_and_iter_reader(
            start,
            rec_count,
            compress=compress,
            columns=columns,
            append_partitions=append_partitions,
        ):
            yield batch

    def read_all(self, start=None, count=None, columns=None, append_partitions=None):
        start = start or 0
        count = count if count is not None else self.count - start
        columns = columns or self._column_names
        append_partitions = (
            append_partitions
            if append_partitions is not None
            else self._append_partitions
        )

        if count == 0:
            from ..tunnel.io.types import odps_schema_to_arrow_schema

            arrow_schema = odps_schema_to_arrow_schema(self.schema)
            return arrow_schema.empty_table()

        with self._download_session.open_arrow_reader(
            start, count, columns=columns, append_partitions=append_partitions
        ) as reader:
            return reader.read()

    def to_pandas(
        self, start=None, count=None, columns=None, append_partitions=None, n_process=1
    ):
        start = start or 0
        count = count if count is not None else self.count - start
        columns = columns or self._column_names
        append_partitions = (
            append_partitions
            if append_partitions is not None
            else self._append_partitions
        )

        if n_process == 1:
            if count == 0:
                from ..tunnel.io.types import odps_schema_to_arrow_schema

                arrow_schema = odps_schema_to_arrow_schema(self.schema)
                return arrow_schema.empty_table().to_pandas()

            with self._download_session.open_arrow_reader(
                start, count, columns=columns, append_partitions=append_partitions
            ) as reader:
                return reader.to_pandas()
        else:
            return self._to_pandas_with_processes(
                start=start,
                count=count,
                columns=columns,
                append_partitions=append_partitions,
                n_process=n_process,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
