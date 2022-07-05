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

from odps.udf import (annotate, BaseUDAF, BaseUDTF)


@annotate(' bigint, bigint -> bigint ')
class Plus(object):

    def evaluate(self, a, b):
        if None in (a, b):
            return None
        return a + b


@annotate('bigint->double')
class Avg(BaseUDAF):

    def new_buffer(self):
        return [0, 0]

    def iterate(self, buffer, number):
        if number is not None:
            buffer[0] += number
            buffer[1] += 1

    def merge(self, buffer, pbuffer):
        buffer[0] += pbuffer[0]
        buffer[1] += pbuffer[1]

    def terminate(self, buffer):
        if buffer[1] == 0:
            return 0
        return float(buffer[0]) / buffer[1]


@annotate('string -> string')
class Explode(BaseUDTF):

    def process(self, arg):
        if arg is None:
            return
        props = arg.split('|')
        for p in props:
            self.forward(p)

    def close(self):
        self.forward('ok')


@annotate('*-> string')
class Star(BaseUDTF):

    def process(self, *args):
        [self.forward(arg) for arg in args]

    def close(self):
        self.forward('ok')


@annotate('-> string')
class Empty(BaseUDTF):

    def process(self):
        self.forward("empty")

    def close(self):
        self.forward('ok')