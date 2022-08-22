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

from .core import RestModel
from .projects import Projects
from .project import Project
from .tables import Tables
from .table import Table, TableSchema
from .instances import Instances
from .instance import Instance
from .functions import Functions
from .function import Function
from .resources import Resources
from .resource import *
from .volumes import *
from .volume_parted import PartedVolume, VolumePartition
from .volume_fs import FSVolume, VolumeFSDir, VolumeFSFile
from .xflows import XFlows
from .xflow import XFlow
from .tasks import *
from .record import Record
from .worker import Worker

Schema = TableSchema  # Schema is to keep compatible
Column = Schema.TableColumn
Partition = Schema.TablePartition
