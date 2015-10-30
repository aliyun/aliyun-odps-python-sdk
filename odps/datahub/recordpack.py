try :
    from cStringIO import StringIO
except ImportError :
    from StringIO import StringIO
from odps.tunnel.conf import CompressOption
from odps.tunnel.writer import ProtobufOutputWriter

class RecordPack(object):
    def __init__(self, schema):
        self.schema = schema
        self.record_count = 0L
        self.block_threshold = 1024 * 1024 * 2;
        self.pack_sealed = False;
        self.fp = StringIO()

        self.stream_writer = ProtobufOutputWriter(self.schema, self.fp, None, CompressOption())

    def append(self, record):
        if self.stream_writer is None:
            self.stream_writer = ProtobufOutputWriter(self.schema, self.fp, None, CompressOption())

        if self.stream_writer.get_total_bytes() >= self.block_threshold or self.pack_sealed is True:
            return False
        self.stream_writer.write(record);
        self.record_count = self.record_count + 1
        return True

    def clear(self):
        if self.stream_writer is not None:
            self.stream_writer.close()
        self.stream_writer = None
        self.record_count = 0
        self.pack_sealed = False

    def get_byte_array(self):
        self.pack_sealed = True
        if self.stream_writer is not None:
            self.stream_writer.close()
            bytes = self.stream_writer.get_bytes()
        self.stream_writer = None
        return bytes

    def get_record_count(self):
        return self.record_count

