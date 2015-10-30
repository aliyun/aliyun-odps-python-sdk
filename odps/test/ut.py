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

import os
import logging
import logging.config
import unittest
import ConfigParser

from odps import (ODPS, accounts)
from odps.tunnel.tabletunnel import TableTunnel

LOGGING_CONFIG = {
    'version': 1,
    "filters": {
        "odps": {
            "name": "odps"
        },
    },
    "formatters": {
            "msgonly": {
                "format": "%(message)s"
            },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": 'INFO',
            "formatter": "msgonly",
            "filters": ["odps",],
        },
    },
    "root": {
            "level": "NOTSET",
            "handlers": ["console"]
        },
    "disable_existing_loggers": False
}


class Config(object):
    config = None
    odps = None
    tunnel = None
    admin = None


def get_config():
    global LOGGING_LEVEL
    if not Config.config:
        config = ConfigParser.ConfigParser()
        Config.config = config
        config_path = os.path.join(os.path.dirname(__file__), 'test.conf')
        if not os.path.exists(config_path):
            raise Exception('Please configure test.conf (you can rename'
                            ' test.conf.template)')
        config.read(config_path)
        access_id = config.get("odps", "access_id")
        secret_access_key = config.get("odps", "secret_access_key")
        project = config.get("odps", "project")
        endpoint = config.get("odps", "endpoint")
        try:
            tunnel_endpoint = config.get("tunnel", "endpoint")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            tunnel_endpoint = None

        try:
            datahub_endpoint = config.get("datahub", "endpoint")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            datahub_endpoint = None

        config.odps = ODPS(access_id, secret_access_key, project, endpoint)
        config.tunnel = TableTunnel(config.odps, tunnel_endpoint)
        config.datahub_endpoint = datahub_endpoint
        logging_level = config.get('test', 'logging_level', 'INFO')
        LOGGING_CONFIG['handlers']['console']['level'] = logging_level
    else:
        config = Config.config
    logging.config.dictConfig(LOGGING_CONFIG)
    return config


class TestBase(unittest.TestCase):

    def setUp(self):
        self.config = get_config()
        self.odps = self.config.odps
        self.tunnel = self.config.tunnel
        self.datahub_endpoint = self.config.datahub_endpoint
        self.setup()

    def tearDown(self):
        self.teardown()

    def setup(self):
        pass

    def teardown(self):
        pass
