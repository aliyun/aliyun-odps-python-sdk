#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import sys
import warnings

from .core import RestModel
from .function import Function
from .functions import Functions
from .instance import Instance
from .instances import Instances
from .project import Project
from .projects import Projects
from .quota import Quota
from .quotas import Quotas
from .record import Record
from .resource import *
from .resources import Resources
from .session import InSessionInstance, SessionInstance, SessionMethods
from .table import Table, TableSchema
from .tableio import TableIOMethods
from .tables import Tables
from .tasks import *
from .tenant import Tenant
from .volume_ext import ExternalVolume, ExternalVolumeDir, ExternalVolumeFile
from .volume_fs import FSVolume, FSVolumeDir, FSVolumeFile
from .volume_parted import PartedVolume, VolumePartition
from .volumes import *
from .worker import Worker
from .xflow import XFlow
from .xflows import XFlows

if sys.version_info[:2] < (3, 7):
    Schema = TableSchema  # Schema is to keep compatible
else:

    def __getattr__(name):
        if name != "Schema":
            raise AttributeError(name)

        from .. import utils

        warnings.warn(
            "Importing Schema from odps.models is deprecated, "
            "please use odps.models.TableSchema instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        utils.add_survey_call("odps.models.Schema")
        return TableSchema


Column = TableSchema.TableColumn
Partition = TableSchema.TablePartition

# keep renames compatible
VolumeFSDir = FSVolumeDir
VolumeFSFile = FSVolumeFile
