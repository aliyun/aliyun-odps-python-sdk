import json
import time

from odps.models import Schema
from odps.datahub.datahubwriter import DatahubWriter
from odps.datahub.datahubreader import DatahubReader
from odps.datahub.schema import DatahubSchema
from odps.datahub.conf import Conf
from odps.datahub.errors import DatahubError

class DatahubClient(object):
    def __init__(self, odps, project_name, table_name, endpoint):
        if endpoint is None:
            raise DatahubError('Invalid datahub endpoint')
        self.conf = Conf(odps, endpoint)
        self.project_name = project_name
        self.table_name = table_name

        self.headers = {}
        self.headers['Content-Length'] = 0
        self.headers['x-odps-tunnel-stream-version'] = '1'
        self.init()

    def load_shard(self, shard_number):
        if shard_number < 0:
            raise DatahubError('invalid shard number')
        params = {'shardnumber': shard_number}
        headers = {'x-odps-tunnel-version': '4'}
        rest = self.conf.get_resource(self.project_name, self.table_name)
        resp = rest.post({}, params = params, headers = headers)
        
        if not rest.is_ok(resp):
            e = DatahubError.parse(resp)
            raise e

    def wait_for_shard_load(self, timeout):
        if timeout <= 0:
            raise DatahubEror('Invalid waiting time')
        wait_time = timeout
        if wait_time > 120:
            wait_time = 120
        now = time.time()
        end = now + wait_time
        while now < end:
            if self.shard_load_completed():
                break
            print 'wait for shards load: 10 secs'
            time.sleep(10)
            now = time.time()
        if not self.shard_load_completed():
            raise DatahubError('load shard timeout')

    def shard_load_completed(self):
        shard_status = self.get_shard_status()
        for shard_id in shard_status:
            if shard_status[shard_id] != 'LOADED':
                return False
        return True

    def get_shard_status(self):
        params = {'shardstatus': 'null'}
        headers = dict(self.headers)
        headers['x-odps-tunnel-version'] = '4'
        rest = self.conf.get_resource(self.project_name, self.table_name)
        resp = rest.get(params = params, headers = headers)
        if not rest.is_ok(resp):
            e = DatahubError.parse(resp)
            raise e
        return self._parse_shard_status(resp)     

    def open_datahub_reader(self, shardid, packid = None):
        params = {}
        headers = dict(self.headers)
        headers['x-odps-tunnel-version'] = '4'
        client = self.conf.get_resource(self.project_name, self.table_name, str(shardid))
        if packid is None:
            return DatahubReader(client, self.schema, shardid, params, headers)
        else :
            return DatahubReader(client, self.schema, shardid, params, headers, packid)

    def open_datahub_writer(self, shard_id):
        params = {}
        headers = dict(self.headers)
        headers['Content-Type'] = 'application/octet-stream'
        headers['x-odps-tunnel-version'] = '4'
        client = self.conf.get_resource(self.project_name, self.table_name, shard_id)
        return DatahubWriter(client, params, headers)

    def get_stream_schema(self):
        return self.schema

    def init(self):
        params = {}
        params['query'] = 'meta'
        params['type'] = 'stream'

        rest = self.conf.get_resource(self.project_name, self.table_name)
        resp = rest.get(params=params, headers=self.headers)
        if rest.is_ok(resp):
            self._parse(resp)
        else :
            e = DatahubError.parse(resp)
            raise e

    def _parse(self, xml):
        root = json.loads(xml.content)
        node = root.get('Schema')
        if node is None:
            raise DatahubError('get table schema fail')
        self.schema = Schema.parse(json.dumps(node))

        node = root.get('Shards')
        if node is None:
            raise DatahubError('get shard fail')
        self.shards = [int(shardid) for shardid in node]

    def _parse_shard_status(self, xml):
        shard_status = {}
        root = json.loads(xml.content)
        node = root.get('ShardStatus');
        if node is not None:
            for shard in node:
                shard_status[int(shard['ShardId'])] = shard['State'].upper()
        return shard_status

