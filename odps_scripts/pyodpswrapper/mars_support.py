# -*- coding: utf-8 -*-
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

import json
import logging
import os
import pickle
import sys
import types
import warnings

logger = logging.getLogger(__name__)


_mars_version_reloads = {
    "0.9": "0.8",
    "0.10": "0.8",
}


def config_mars_version(mars_version_str):
    mars_version = mars_version_str or "0.6"
    mars_version = _mars_version_reloads.get(mars_version, mars_version)
    mars_path = "/opt/taobao/tbdpapp/pyodps/mars/%s" % mars_version
    if not os.path.isdir(mars_path):
        if mars_version_str is not None:
            warnings.warn(
                "Illegal Mars version %s specified, may use site version instead."
                % mars_version_str,
                ImportWarning,
            )
    else:
        sys.path = [mars_path] + sys.path


def create_sign_server(*args, **kw):
    from odps.accounts import SignServer

    class CupidSignServer(SignServer):
        class SignServerHandler(SignServer.SignServerHandler):
            def _do_POST(self):
                from odps.compat import six

                try:
                    from odps.compat import cgi
                except (AttributeError, ImportError, TypeError):
                    import cgi

                ctype, pdict = cgi.parse_header(self.headers.get("content-type"))
                if ctype == "multipart/form-data":
                    postvars = cgi.parse_multipart(self.rfile, pdict)
                elif ctype == "application/x-www-form-urlencoded":
                    length = int(self.headers.get("content-length"))
                    postvars = six.moves.urllib.parse.parse_qs(
                        self.rfile.read(length), keep_blank_values=1
                    )
                else:
                    self.send_response(400)
                    self.end_headers()
                    return

                if b"cupid_task" in postvars:
                    if postvars[b"cupid_task"][0] == b"create_mars_cluster":
                        self._create_mars_cluster(postvars)
                    elif postvars[b"cupid_task"][0] == b"get_proxied_url":
                        self._get_proxied_url(postvars)
                    else:
                        self.send_response(400)
                        self.end_headers()
                        return
                else:
                    self._sign(postvars)

            def _get_proxied_url(self, postvars):
                from cupid import CupidSession
                from odps import ODPS, utils

                instance_id = utils.to_str(postvars[b"instance_id"][0])
                app_name = utils.to_str(postvars[b"app_name"][0])

                access_id = utils.to_str(postvars[b"access_id"][0])
                project = utils.to_str(postvars[b"project"][0])
                endpoint = utils.to_str(postvars[b"endpoint"][0])
                odps_args = [
                    access_id,
                    self.server._accounts[access_id],
                    project,
                    endpoint,
                ]

                o = ODPS(*odps_args)
                cupid_session = CupidSession(o)
                try:
                    url = cupid_session.get_proxied_url(instance_id, app_name)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/json")
                    self.end_headers()

                    self.wfile.write(utils.to_binary(url))
                except:
                    self.send_response(500)
                    self.end_headers()

            def _create_mars_cluster(self, postvars):
                from odps import ODPS, utils

                args = json.loads(utils.to_str(postvars[b"args"][0]))
                kwargs = json.loads(utils.to_str(postvars[b"kwargs"][0]))
                kwargs["async_"] = True

                access_id = utils.to_str(postvars[b"access_id"][0])
                project = utils.to_str(postvars[b"project"][0])
                endpoint = utils.to_str(postvars[b"endpoint"][0])
                odps_args = [
                    access_id,
                    self.server._accounts[access_id],
                    project,
                    endpoint,
                ]
                o = ODPS(*odps_args)

                try:
                    client = o.create_mars_cluster(*args, **kwargs)
                except:
                    exc_info = sys.exc_info()
                    self.send_response(500)
                    self.end_headers()

                    try:
                        self.wfile.write(pickle.dumps(exc_info))
                    except:
                        logger.exception(
                            "Unexpected error encountered in sign server:",
                            exc_info=exc_info,
                        )
                    raise

                self.send_response(200)
                self.send_header("Content-Type", "text/json")
                self.end_headers()

                self.wfile.write(utils.to_binary(client._kube_instance.id))

    return CupidSignServer(*args, **kw)


def create_mars_cluster(odps, *args, **kwargs):
    from odps.mars_extension import CUPID_APP_NAME, NOTEBOOK_NAME, MarsCupidClient

    try:
        from odps.lib import requests
    except (AttributeError, ImportError, TypeError):
        import requests

    sign_server_account = odps.account
    project = odps.project
    endpoint = odps.endpoint
    post_data = dict(
        access_id=sign_server_account.access_id,
        project=project,
        endpoint=endpoint,
        cupid_task="create_mars_cluster",
        args=json.dumps(args),
        kwargs=json.dumps(kwargs),
    )
    resp = sign_server_account.session.request(
        "post", "http://%s:%s" % sign_server_account.sign_endpoint, data=post_data
    )
    if resp.status_code > 200:
        if len(resp.content) > 0:
            exc_info = pickle.loads(resp.content)
            raise exc_info[1].with_traceback(exc_info[2])
        else:
            raise SystemError(
                "Cannot create Mars cluster, see sign server logs for details"
            )
    instance_id = resp.text

    inst = odps.get_instance(instance_id)
    logger.info(inst.get_logview_address())
    client = MarsCupidClient(odps, inst)

    def _get_mars_endpoint(self):
        p_data = dict(
            access_id=sign_server_account.access_id,
            project=project,
            endpoint=endpoint,
            cupid_task="get_proxied_url",
            instance_id=self._kube_instance.id,
            app_name=CUPID_APP_NAME,
        )
        resp = sign_server_account.session.request(
            "post", "http://%s:%s" % sign_server_account.sign_endpoint, data=p_data
        )
        if resp.status_code > 200:
            raise requests.exceptions.ConnectionError
        else:
            return resp.text

    def _get_notebook_endpoint(self):
        p_data = dict(
            access_id=sign_server_account.access_id,
            project=project,
            endpoint=endpoint,
            cupid_task="get_proxied_url",
            instance_id=self._kube_instance.id,
            app_name=NOTEBOOK_NAME,
        )
        resp = sign_server_account.session.request(
            "post", "http://%s:%s" % sign_server_account.sign_endpoint, data=p_data
        )
        if resp.status_code > 200:
            raise requests.exceptions.ConnectionError
        else:
            return resp.text

    client.get_mars_endpoint = types.MethodType(_get_mars_endpoint, client)
    client.get_notebook_endpoint = types.MethodType(_get_notebook_endpoint, client)
    with_notebook = kwargs.get("with_notebook", None)
    default_num = args[0] if len(args) > 0 else 1
    min_worker_num = kwargs.get("min_worker_num", None) or default_num
    client.submit(min_worker_num=min_worker_num, with_notebook=with_notebook)
    return client
