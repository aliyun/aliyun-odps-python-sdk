import json
import struct
from md5 import md5
from odps.datahub.conf import Conf
from odps.datahub.xstreampack_pb2 import XStreamPack
try :
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

class DatahubWriter(object):
    def __init__(self, client, params, headers):
        self.client = client
        self.params = params
        self.headers = headers

    def write(self, record_pack, partition_spec = None, meta = None):
        params = dict(self.params)
        headers = dict(self.headers)
        headers['Content-Encoding'] = 'deflate'

        pack_data = record_pack.get_byte_array()
        pack = XStreamPack()
        pack.pack_data = pack_data
        if meta is not None:
            pack.pack_meta = meta
        bytes = pack.SerializeToString()
        headers['Content-MD5'] = md5(bytes).hexdigest()

        if partition_spec is not None and len(partition_spec) > 0:
            params['partition'] = str(partition_spec).replace("'", "")
        params['recordcount'] = str(record_pack.get_record_count())

        resp = self.client.put(bytes, params = params, headers = headers)
        if not self.client.is_ok(resp):
            e = DatahubError.parse(resp)
            raise e
        else :
            body = resp.json()
            self._parse(body)

        return self.last_pack_result

    def _parse(self, body):
        node = body.get(u'PackId')
        if node is not None:
            self.last_pack_result = WritePackResult(node)
        else :
            e = DatahubError('get pack id fail')
            raise e

class WritePackResult(object):
    def __init__(self, packid):
        if packid is None:
            e = DatahubError('Invalid pack string')
            raise e
        self.pack_id = packid

    def get_pack_id(self):
        return self.pack_id

