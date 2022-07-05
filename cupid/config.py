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

from odps import options


options.register_option('cupid.major_task_version', 'cupid_v2')
options.register_option('cupid.wait_am_start_time', 600)
options.register_option('cupid.use_bearer_token', None)
options.register_option('cupid.settings', None)
options.register_option('cupid.mp_buffer_size', 1024 * 64)

options.register_option('cupid.proxy_endpoint', 'open.maxcompute.aliyun.com')
options.register_option('cupid.worker.virtual_resource', None)
options.register_option('cupid.master.virtual_resource', None)
options.register_option('cupid.master_type', 'kubernetes')
options.register_option('cupid.application_type', 'mars')
options.register_option('cupid.engine_running_type', 'default')
options.register_option('cupid.container_node_label', None)
options.register_option('cupid.job_duration_hours', 25920)
options.register_option('cupid.channel_init_timeout_seconds', 120)
options.register_option('cupid.kube.master_mode', 'cupid')
options.register_option('cupid.runtime.endpoint', None)

# mars app config
options.register_option('cupid.image_prefix', None)
options.register_option('cupid.image_version', 'v0.11.1')
