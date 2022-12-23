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

version_info = (0, 11, 2, 3)
_num_index = max(idx if isinstance(v, int) else 0
                 for idx, v in enumerate(version_info))
__version__ = '.'.join(map(str, version_info[:_num_index + 1])) + \
              ''.join(version_info[_num_index + 1:])
