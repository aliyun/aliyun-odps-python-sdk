# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""UDF runners implementing the local run framework.
"""

import sys
from datetime import datetime

from odps import udf
from odps import distcache
from . import utils


__all__ = ['get_default_runner']


def get_default_runner(udf_class, input_col_delim=',', null_indicator='NULL'):
    """Create a default runner with specified udf class.
    """
    proto = udf.get_annotation(udf_class)
    in_types, out_types = parse_proto(proto)
    arg_parser = ArgParser(in_types, input_col_delim, null_indicator)
    raw_feed = make_file_raw_feed(sys.stdin)
    stdin_feed = make_feed(raw_feed, arg_parser)
    collector = StdoutCollector(out_types)
    ctor = _get_runner_class(udf_class)
    return ctor(udf_class, stdin_feed, collector)


def simple_run(udf_class, args):
    """
    """
    proto = udf.get_annotation(udf_class)
    in_types, out_types = parse_proto(proto)
    feed = direct_feed(args)
    collector = DirectCollector(out_types)
    ctor = _get_runner_class(udf_class)
    runner = ctor(udf_class, feed, collector)
    runner.run()
    return collector.results


def initialize():
    """Initialize the local run environment.
    """
    distcache.get_cache_table = utils.get_cache_table

    
def _get_types(types):
    entries = []
    for t in types.split(','):
        t = t.strip()
        if t not in _allowed_data_types:
            raise Exception('type not in '+ ','.join(_allowed_data_types))
        entries.append(_type_registry[t])
    return entries


def _get_in_types(types):
    if types == '':
        return []
    return _get_types(types) if types != '*' else [_type_registry[types], ]


def _get_runner_class(udf_class):
    if udf.BaseUDAF in udf_class.__mro__:
        ctor = UDAFRunner
    elif udf.BaseUDTF in udf_class.__mro__:
        ctor = UDTFRunner
    else:
        ctor = UDFRunner
    return ctor


def parse_proto(proto):
    tokens = proto.lower().split('->')
    if len(tokens) != 2:
        raise Exception('@annotate(' + proto + ') error')
    return _get_in_types(tokens[0].strip()), _get_types(tokens[1].strip())


def make_file_raw_feed(fp):
    for line in fp:
        line = line.rstrip('\n') # discard \n
        yield line
        

def make_feed(raw_feed, arg_parser):
    for line in raw_feed:
        yield arg_parser.parse(line)


def direct_feed(args):
    for a in args:
        yield a


class ArgParser(object):

    NULL_INDICATOR = 'NULL'

    def __init__(self, types, delim=',', null_indicator='NULL'):
        self.types = types
        self.delim = delim
        self.null_indicator = null_indicator

    def parse(self, line):
        tokens = []
        for token in map(lambda s: s.strip(), line.split(self.delim)):
            if token == self.null_indicator:
                tokens.append(None)
            else:
                tokens.append(token)
        if len(self.types) == 1 and self.types[0].typestr == '*':
            return tokens
        if len(self.types) == 0 and line.strip() == '':
            return ''

        if len(tokens) != len(self.types):
            raise Exception('Schema error: ' + line)
        return map(lambda tp, data: tp.converter(data), self.types, tokens)


class ArgFormater(object):

    DELIM = '\t'
    NULL_INDICATOR = 'NULL'
    
    def __init__(self, types):
        self.types = types

    def format(self, *args):
        ret = self.DELIM.join([str(a) for a in args])
        return ret


class BaseCollector(object):
    """Basic common logic of collector.
    """
    def __init__(self, schema):
        self.schema = schema
        
    def _validate_records(self, *args):
        if len(args) != len(self.schema):
            raise Exception('Schema error: ' + repr(args))
        for i, a in enumerate(args):
            if a is None:
                continue
            elif not isinstance(a, self.schema[i].type):
                raise Exception('Schema error: ' + repr(args))

    def collect(self, *args):
        self._validate_records(*args)
        self.handle_collect(*args)
    
    
class StdoutCollector(BaseCollector):
    """Collect the results to stdout.
    """

    def __init__(self, schema):
        super(StdoutCollector, self).__init__(schema)
        self.formater = ArgFormater(schema)

    def handle_collect(self, *args):
        print(self.formater.format(*args))


class DirectCollector(BaseCollector):
    """Collect results which can be fetched via self.results into memory.
    """

    def __init__(self, schema):
        super(DirectCollector, self).__init__(schema)
        self.results = []

    def handle_collect(self, *args):
        if len(self.schema) == 1:
            self.results.append(args[0])
        else:
            self.results.append(args)


class BaseRunner(object):
    
    def __init__(self, udf_class, feed, collector):
        self.udf_class = udf_class
        self.feed = feed
        self.collector = collector
        # check signature
        self.obj = udf_class()


class UDFRunner(BaseRunner):

    def run(self):
        obj = self.obj
        collector = self.collector
        for args in self.feed:
            r = obj.evaluate(*args)
            collector.collect(r)


class UDTFRunner(BaseRunner):

    def run(self):
        obj = self.obj
        collector = self.collector
        def local_forward(*r):
            collector.collect(*r)
        obj.forward = local_forward
        for args in self.feed:
            obj.process(*args)
        obj.close()


class UDAFRunner(BaseRunner):

    def run(self):
        obj = self.obj
        collector = self.collector
        buf0 = obj.new_buffer()
        buf1 = obj.new_buffer()
        turn = True
        for args in self.feed:
            if turn:
                buf = buf0
                turn = False
            else:
                buf = buf1
                turn = True
            obj.iterate(buf, *args)
        merge_buf = obj.new_buffer()
        obj.merge(merge_buf, buf0)
        obj.merge(merge_buf, buf1)
        collector.collect(obj.terminate(merge_buf))



###########################################
###         Static type registry        ###

def register_type(enum, typestr, tp):
    type_obj = TypeEntry(typestr, tp)
    globals()[enum] = type_obj
    _type_registry[typestr] = type_obj


_type_registry = {
}


def _gen_converter(tp):
    def f(v):
        if v == "NULL" or v is None:
            return None
        return tp(v)
    return f


class TypeEntry(object):

    def __init__(self, typestr, tp):
        self.typestr = typestr
        self.type = tp
        self.converter = _gen_converter(tp)

register_type('TP_BIGINT',   'bigint',   int)
register_type('TP_STRING',   'string',   str)
register_type('TP_DATETIME', 'datetime', int)
register_type('TP_DOUBLE',   'double',   float)
register_type('TP_BOOLEAN',  'boolean',  bool)
register_type('TP_STAR',     '*',     lambda x: x)

_allowed_data_types = [k for k in _type_registry.keys() if k != '*']

###########################################


initialize()

