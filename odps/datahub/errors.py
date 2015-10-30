#!/usr/bin/env python

import json
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from odps import errors


class DatahubError(RuntimeError):

    @classmethod
    def parse(cls, resp):
        try:
            error = DatahubError(resp.content)
        except:
            raise
        return error
