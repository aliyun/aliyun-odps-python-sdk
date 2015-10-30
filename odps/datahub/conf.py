#!/usr/bin/env python

import platform
from urlparse import urlparse

from odps import rest
from odps.tunnel.conf import CompressOption

class Conf(object):
    # 上传数据的默认块大小(单位字节)。默认值按照MTU－4设置，MTU值1500
    DEFAULT_CHUNK_SIZE = 1496

    # 底层网络连接的默认超时时间，180秒
    DEFAULT_SOCKET_CONNECT_TIMEOUT = 180

    # 底层网络默认超时时间，300秒
    DEFAULT_SOCKET_TIMEOUT = 300
   
    chunk_size = DEFAULT_CHUNK_SIZE
    socket_connect_timeout = DEFAULT_SOCKET_CONNECT_TIMEOUT
    socket_timeout = DEFAULT_SOCKET_TIMEOUT
    
    def __init__(self, odps, endpoint):
        self.odps = odps
        self.client = odps.rest
        self.user_agent = 'TunnelSDK/0.12.0;%s' % \
            ' '.join((platform.system(), platform.release()))
        # self.client.set_user_agent(self.user_agent)
        self.option = CompressOption()
        # datahub endpoint
        if endpoint is not None or len(endpoint) != 0:
            self.endpoint = endpoint
        else :
            self.endpoint = None

    def get_endpoint(self, project_name):
        return self.endpoint
    
    def set_endpoint(self, endpoint):
        self.endpoint = endpoint

    def get_uri(self, project_name, table_name):
        endpoint = self.get_endpoint(project_name).geturl()
        endpoint = endpoint.rstrip('/')
        url = '%s/projects/%s/tables/%s' % (endpoint, project_name, table_name)
        return urlparse(url)
    
    def get_resource(self, project_name, table_name, shard_id = None):
        datahub_endpoint= self.get_endpoint(project_name)
        if datahub_endpoint is None or len(datahub_endpoint) == 0:
            raise Exception("datahub endpoint not set yet!")
        datahub_client = rest.RestClient(self.odps.account, datahub_endpoint)
        res = datahub_client.projects[project_name].tables[table_name]
        if shard_id is None:
            res['shards']
        else :
            res.shards[shard_id]
        return res

