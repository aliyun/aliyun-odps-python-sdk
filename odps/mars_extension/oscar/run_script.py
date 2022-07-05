#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import os
import logging

from mars.utils import to_binary


logger = logging.getLogger(__name__)

from mars.remote.run_script import RunScript


def _build_locals_in_odps(cls, ctx, op):
    from odps import ODPS

    sess = ctx.get_current_session().as_default()
    local = {"session": sess}
    if op.data is not None:
        local.update(op.data)
    o = ODPS.from_environments()
    return dict(o=o, session=sess, odps=o)


def execute_with_odps_context(f):
    def wrapper(ctx, op):
        from .cupid_service import CupidServiceClient
        from mars.utils import to_str

        if "CUPID_SERVICE_SOCKET" not in os.environ:
            f(ctx, op)
        else:
            old_envs = os.environ.copy()
            try:
                env = os.environ

                logger.debug("Get bearer token from cupid.")
                bearer_token = CupidServiceClient().get_bearer_token()
                env["ODPS_BEARER_TOKEN"] = to_str(bearer_token)
                if "endpoint" in op.extra_params:
                    env["ODPS_ENDPOINT"] = os.environ.get(
                        "ODPS_RUNTIME_ENDPOINT"
                    ) or str(op.extra_params["endpoint"])
                if ("project" in op.extra_params) and ("ODPS_PROJECT_NAME" not in env):
                    env["ODPS_PROJECT_NAME"] = str(op.extra_params["project"])
                f(ctx, op)
                for out in op.outputs:
                    if ctx[out.key] is None:
                        ctx[out.key] = {"status": "OK"}
            finally:
                os.environ = old_envs

    return wrapper


RunScript._build_locals = classmethod(_build_locals_in_odps)


def run_script(
    script,
    data=None,
    n_workers=1,
    mode="exec",
    command_argv=None,
    retry_when_fail=False,
    odps_params=None,
    session=None,
    run_kwargs=None,
):
    from mars.remote.run_script import _extract_inputs

    if hasattr(script, "read"):
        code = script.read()
    else:
        with open(os.path.abspath(script), "rb") as f:
            code = f.read()
    inputs = _extract_inputs(data)
    op = RunScript(
        data=data,
        code=to_binary(code),
        world_size=n_workers,
        retry_when_fail=retry_when_fail,
        command_args=command_argv,
    )

    op.extra_params["project"] = odps_params["project"]
    op.extra_params["endpoint"] = (
        os.environ.get("ODPS_RUNTIME_ENDPOINT") or odps_params["endpoint"]
    )
    return (
        op(inputs).execute(session=session, **(run_kwargs or {})).fetch(session=session)
    )
