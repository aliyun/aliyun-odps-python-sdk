# encoding: utf-8
# copyright 1999-2022 alibaba group holding ltd.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#      http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import sys
import types

from .classification import *
from .regression import *
from .clustering import *
from .scorer import SCORERS

from ..algolib.loader import load_defined_algorithms

# required by autodoc of sphinx
__all__ = [k for k, v in globals().items()
           if not k.startswith('_') and isinstance(v, (types.FunctionType, type))]

load_defined_algorithms(sys.modules[__name__], 'metrics')
