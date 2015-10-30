from odps.models import Record
from odps.datahub.conf import Conf
from odps.datahub.errors import DatahubError
from odps.datahub.xstreampack_pb2 import XStreamPack
from odps.tunnel.reader import ProtobufInputReader
from odps.tunnel.conf import CompressOption
try :
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

FIRST_PACK_ID = '00000000000000000000000000000000'
LAST_PACK_ID = 'ffffffffffffffffffffffffffffffff'

class DatahubReader(object):
    def __init__(self, client, schema, shardid, params, headers, packid = FIRST_PACK_ID):
        self.client = client
        self.schema = schema
        self.shardid = shardid
        self.params = params
        self.headers = headers

        self.stream_reader = None
        self.last_pack_id = None
        self.next_pack_id = None
        self.seek(packid, 'SEEK_CUR')

    def read(self):
        record = None
        while True:
            if self.stream_reader is not None:
                try :
                    record = self.stream_reader.read()
                    if record is not None:
                        return record
                except IOError:
                    self.next_pack_id = self.last_pack_id
                    self.read_mode = 'SEEK_CUR'
                    raise
            if not self.get_pack():
                break
        return record

    def seek(self, pack_id, MODE):
        if pack_id is None and not (pack_id == FIRST_PACK_ID or pack_id == LAST_PACK_ID):
            e = DatahubError('Invalid pack id')
            raise e
        if MODE == 'SEEK_BEGIN':
            self.next_pack_id = FIRST_PACK_ID
        elif MODE == 'SEEK_END':
            self.next_pack_id = LAST_PACK_ID
        elif MODE == 'SEEK_CUR':
            self.next_pack_id = str(pack_id)
        elif MODE == 'SEEK_NEXT':
            self.next_pack_id = str(pack_id)
        else :
            raise DatahubError('Invalid pack read mode')

        self.read_mode = MODE
        self.stream_reader = None

    def get_pack(self):
        self.stream_reader = None
        params = dict(self.params)
        headers = dict(self.headers)
        if self.read_mode == 'SEEK_NEXT':
            mode = 'AFTER_PACKID'
        else :
            mode = 'AT_PACKID'
        params['packid'] = self.next_pack_id
        params['iteratemode'] = mode
        params['packnum'] = '1'
        resp = self.client.get(params = params, headers = headers)
        if not self.client.is_ok(resp):
            raise DatahubError('no more packs')

        bytes = resp.content
        pack = XStreamPack()
        pack.ParseFromString(bytes)
        fp = StringIO(pack.pack_data)
        self.stream_reader = ProtobufInputReader(self.schema, fp, CompressOption())

        npid = resp.headers['x-odps-next-packid']
        self.last_pack_id = resp.headers['x-odps-current-packid']
        if npid != LAST_PACK_ID :
            self.next_pack_id = npid
            self.read_mode = 'SEEK_CUR'
        else :
            self.next_pack_id = self.last_pack_id
            self.read_mode = 'SEEK_NEXT'

        return True
        
