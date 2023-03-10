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

from ..compat import six
from ..lib.tblib import pickling_support
from ..readers import AbstractRecordReader

pickling_support.install()


class SpawnedTunnelReaderMixin(object):
    def _to_pandas_with_processes(self, start=None, count=None, columns=None, n_process=1):
        import pandas as pd
        import multiprocessing
        from multiprocessing import Pipe

        session_id = self._download_session.id
        start = start or 0
        count = count or self._download_session.count
        count = min(count, self._download_session.count - start)
        try:
            _mp_context = multiprocessing.get_context('fork')
        except ValueError:
            _mp_context = multiprocessing.get_context('spawn')
        except AttributeError:
            # for py27 compatibility
            _mp_context = multiprocessing

        n_process = min(count, n_process)
        split_count = count // n_process + (count % n_process != 0)
        conns = []
        for i in range(n_process):
            parent_conn, child_conn = Pipe()

            p = _mp_context.Process(
                target=self._get_process_split_reader(columns=columns),
                args=(child_conn, session_id, start, split_count, i),
            )
            p.start()
            start += split_count
            conns.append(parent_conn)

        results = [c.recv() for c in conns]
        splits = sorted(results, key=lambda x: x[0])
        if any(not d[2] for d in splits):
            exc_info = next(d[1] for d in splits if not d[2])
            six.reraise(*exc_info)
        return pd.concat([d[1] for d in splits]).reset_index(drop=True)

    def _get_process_split_reader(self, columns=None):
        raise NotImplementedError


class TunnelRecordReader(SpawnedTunnelReaderMixin, AbstractRecordReader):
    def __init__(self, parent, download_session, columns=None):
        self._it = iter(self)
        self._parent = parent
        self._download_session = download_session
        self._column_names = columns

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

    def _iter(self, start=None, end=None, step=None, compress=False, columns=None):
        count = self._calc_count(start, end, step)
        return self.read(
            start=start, count=count, step=step, compress=compress, columns=columns
        )

    def read(self, start=None, count=None, step=None,
             compress=False, columns=None):
        start = start or 0
        step = step or 1
        count = count * step if count is not None else self.count - start
        columns = columns or self._column_names

        if count == 0:
            return

        with self._download_session.open_record_reader(
            start, count, compress=compress, columns=columns
        ) as reader:
            for record in reader[::step]:
                yield record

    def to_pandas(self, start=None, count=None, columns=None, n_process=1):
        columns = columns or self._column_names
        if n_process == 1 or self._download_session.count == 0:
            return super(TunnelRecordReader, self).to_pandas(
                start=start, count=count, columns=columns
            )
        else:
            return self._to_pandas_with_processes(
                start=start, count=count, columns=columns, n_process=n_process
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TunnelArrowReader(SpawnedTunnelReaderMixin):
    def __init__(self, parent, download_session, columns=None):
        self._it = iter(self)
        self._parent = parent
        self._download_session = download_session
        self._column_names = columns

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

    def read(self, start=None, count=None, columns=None):
        start = start or 0
        count = count if count is not None else self.count - start
        columns = columns or self._column_names

        if count == 0:
            return

        with self._download_session.open_arrow_reader(
            start, count, columns=columns
        ) as reader:
            while True:
                batch = reader.read_next_batch()
                if batch is not None:
                    yield batch
                else:
                    break

    def read_all(self, start=None, count=None, columns=None):
        start = start or 0
        count = count if count is not None else self.count - start
        columns = columns or self._column_names

        if count == 0:
            return

        with self._download_session.open_arrow_reader(
            start, count, columns=columns
        ) as reader:
            return reader.read()

    def to_pandas(self, start=None, count=None, columns=None, n_process=1):
        columns = columns or self._column_names
        if n_process == 1:
            return self.read_all(start=start, count=count, columns=columns).to_pandas()
        else:
            return self._to_pandas_with_processes(
                start=start, count=count, columns=columns, n_process=n_process
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
