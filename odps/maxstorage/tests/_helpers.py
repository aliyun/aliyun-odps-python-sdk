# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

"""Shared utilities for maxstorage unit tests."""

import io

__all__ = ["TrackedStream", "CloseTrackingStream"]


class TrackedStream(io.BytesIO):
    """``BytesIO`` subclass that records every ``read`` call signature.

    If the payload were materialized, ``read()`` (no-arg) would be called
    once.  Chunked streaming calls ``read(chunk_size)`` repeatedly with
    an explicit positive size — never ``read()`` with no args.
    """

    def __init__(self, data):
        super().__init__(data)
        self.read_calls = []

    def read(self, size=-1):
        self.read_calls.append(size)
        return super().read(size)


class CloseTrackingStream(io.BytesIO):
    """``BytesIO`` that records how many times ``close()`` was called."""

    def __init__(self, data):
        super().__init__(data)
        self.close_count = 0

    def close(self):
        self.close_count += 1
        super().close()
