# -*- coding: utf-8 -*-
# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from .instancetunnel import InstanceDownloadSession, InstanceTunnel
from .io import (
    ArrowWriter,
    BufferedArrowWriter,
    BufferedRecordWriter,
    CompressOption,
    RecordWriter,
    TunnelArrowReader,
    TunnelRecordReader,
    Upsert,
)
from .io.reader import TunnelArrowReader, TunnelRecordReader
from .tabletunnel import (
    TableDownloadSession,
    TableStreamUploadSession,
    TableTunnel,
    TableUploadSession,
    TableUpsertSession,
)
from .volumetunnel import (
    VolumeDownloadSession,
    VolumeFSTunnel,
    VolumeTunnel,
    VolumeUploadSession,
)

TableUploadStatus = TableUploadSession.Status
TableDownloadStatus = TableDownloadSession.Status
VolumeUploadStatus = VolumeUploadSession.Status
VolumeDownloadStatus = VolumeDownloadSession.Status
InstanceDownloadStatus = InstanceDownloadSession.Status
