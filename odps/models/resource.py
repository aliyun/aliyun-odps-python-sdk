#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from enum import Enum
import six

from .core import LazyLoad
from .cache import cache
from .. import serializers, utils, types, errors, compat


RESOURCE_SIZE_MAX = 512 * 1024 * 1024  # a single resource's size must be at most 512M


class Resource(LazyLoad):
    __slots__ = 'content_md5', 'is_temp_resource', 'volumn_path', '_type_indicator'

    class Type(Enum):
        FILE = 'FILE'
        JAR = 'JAR'
        PY = 'PY'
        ARCHIVE = 'ARCHIVE'
        TABLE = 'TABLE'
        VOLUMEFILE = 'VOLUMEFILE'
        VOLUMEARCHIVE = 'VOLUMEARCHIVE'
        UNKOWN = 'UNKOWN'

    _type_indicator = 'type'

    name = serializers.XMLNodeField('Name')
    owner = serializers.XMLNodeField('Owner')
    comment = serializers.XMLNodeField('Comment')
    type = serializers.XMLNodeField('ResourceType', parse_callback=lambda t: Resource.Type(t.upper()))
    creation_time = serializers.XMLNodeField('CreationTime', parse_callback=utils.parse_rfc822)
    last_modified_time = serializers.XMLNodeField('LastModifiedTime', parse_callback=utils.parse_rfc822)
    last_updator = serializers.XMLNodeField('LastUpdator')
    size = serializers.XMLNodeField('ResourceSize', parse_callback=int)
    source_table_name =  serializers.XMLNodeField('TableName')

    @classmethod
    def _get_cls(cls, typo):
        if typo is None:
            return cls

        if isinstance(typo, six.string_types):
            typo = Resource.Type(typo.upper())

        clazz = lambda name: globals()[name]
        if typo == Resource.Type.FILE:
            return clazz('FileResource')
        elif typo == Resource.Type.JAR:
            return clazz('JarResource')
        elif typo == Resource.Type.PY:
            return clazz('PyResource')
        elif typo == Resource.Type.ARCHIVE:
            return clazz('ArchiveResource')
        elif typo == Resource.Type.TABLE:
            return clazz('TableResource')
        else:
            return cls

    @cache
    def __new__(cls, *args, **kwargs):
        typo = kwargs.get('type')
        if typo is not None or (cls != Resource and issubclass(cls, Resource)):
            return object.__new__(cls._get_cls(typo))

        kwargs['type'] = Resource.Type.UNKOWN
        obj = Resource(**kwargs)
        obj.reload()
        for attr in obj.__slots__:
            kwargs[attr] = getattr(obj, attr, None)
        return Resource(**kwargs)

    def __init__(self, **kwargs):
        typo = kwargs.get('type')
        if isinstance(typo, six.string_types):
            kwargs['type'] = Resource.Type(typo.upper())
        super(Resource, self).__init__(**kwargs)

    @property
    def _project(self):
        return self._parent._parent.name

    def reload(self):
        url = self.resource()
        resp = self._client.head(url)
        self.owner = resp.headers.get('x-odps-owner')
        resource_type = resp.headers.get('x-odps-resource-type')
        self.type = Resource.Type(resource_type.upper())
        self.comment = resp.headers.get('x-odps-comment')
        self.last_updator = resp.headers.get('x-odps-updator')

        size = resp.headers.get('x-odps-resource-size')
        self.size = None if size is None else int(size)

        self.creation_time = utils.parse_rfc822(
            resp.headers.get('x-odps-creation-time'))
        self.last_modified_time = utils.parse_rfc822(
            resp.headers.get('Last-Modified'))

        self.source_table_name = resp.headers.get('x-odps-copy-table-source')
        self.volumn_path = resp.headers.get('x-odps-copy-file-source')
        self.content_md5 = resp.headers.get('Content-MD5')

        self._loaded = True

    def update(self, **kw):
        raise NotImplementedError

    def drop(self):
        return self.parent.delete(self)


class FileResource(Resource):
    __slots__ = '_fp', '_mode', '_opened', '_size', '_need_commit', \
                '_open_binary', '_encoding'

    class Mode(Enum):
        READ = 'r'
        WRITE = 'w'
        APPEND = 'a'
        READWRITE = 'r+'
        TRUNCEREADWRITE = 'w+'
        APPENDREADWRITE = 'a+'

    def __init__(self, **kw):
        super(FileResource, self).__init__(**kw)
        self.type = Resource.Type.FILE

        self._fp = None
        self._mode = FileResource.Mode.READ
        self._open_binary = False
        self._encoding = None
        self._size = 0
        self._opened = False
        self._need_commit = False

    def _is_create(self):
        if self._loaded:
            return False
        try:
            self.reload()
            return False
        except errors.NoSuchObject:
            return True

    def open(self, mode='r', encoding='utf-8'):
        # TODO: when reading, do not read all the data at once

        if 'b' in mode:
            self._open_binary = True
        mode = mode.replace('b', '')
        self._mode = FileResource.Mode(mode)
        self._encoding = encoding

        if self._mode in (FileResource.Mode.WRITE, FileResource.Mode.TRUNCEREADWRITE):
            io_clz = six.BytesIO if self._open_binary else six.StringIO
            self._fp = io_clz()
            self._size = 0
        else:
            self._fp = self.parent.read_resource(
                self, text_mode=not self._open_binary, encoding=self._encoding)
            self.reload()
            self._sync_size()

        self._opened = True

        return self

    def _check_read(self):
        if not self._opened:
            raise IOError('I/O operation on non-open resource')
        if self._mode in (FileResource.Mode.WRITE, FileResource.Mode.APPEND):
            raise IOError('Resource not open for reading')

    def _sync_size(self):
        curr_pos = self.tell()
        self.seek(0, compat.SEEK_END)
        self._size = self.tell()
        self.seek(curr_pos)

    def read(self, size=-1):
        self._check_read()
        return self._fp.read(size)

    def readline(self, size=-1):
        self._check_read()
        return self._fp.readline(size)

    def readlines(self, sizehint=-1):
        self._check_read()
        return self._fp.readlines(sizehint)

    def _check_write(self):
        if not self._opened:
            raise IOError('I/O operation on non-open resource')
        if self._mode == FileResource.Mode.READ:
            raise IOError('Resource not open for writing')

    def _check_size(self):
        if self._size > RESOURCE_SIZE_MAX:
            raise IOError('Single resource\'s max size is %sM' %
                          (RESOURCE_SIZE_MAX / (1024 ** 2)))

    def _convert(self, content):
        if self._open_binary and isinstance(content, six.text_type):
            return content.encode(self._encoding)
        elif not self._open_binary and isinstance(content, six.binary_type):
            return content.decode(self._encoding)
        return content

    def write(self, content):
        content = self._convert(content)

        length = len(content)
        self._check_write()
        if self._mode in (FileResource.Mode.APPEND, FileResource.Mode.APPENDREADWRITE):
            self.seek(0, compat.SEEK_END)

        if length > 0:
            self._need_commit = True

        res = self._fp.write(content)
        self._sync_size()
        self._check_size()
        return res

    def writelines(self, seq):
        seq = [self._convert(s) for s in seq]

        length = sum(len(s) for s in seq)
        self._check_write()
        if self._mode in (FileResource.Mode.APPEND, FileResource.Mode.APPENDREADWRITE):
            self.seek(0, compat.SEEK_END)

        if length > 0:
            self._need_commit = True

        res = self._fp.writelines(seq)
        self._sync_size()
        self._check_size()
        return res

    def seek(self, pos, whence=compat.SEEK_SET):  # io.SEEK_SET
        return self._fp.seek(pos, whence)

    def tell(self):
        return self._fp.tell()

    def truncate(self, size=None):
        self._check_write()

        curr_pos = self.tell()
        self._fp.truncate(size)

        self.seek(0, compat.SEEK_END)
        self._size = self.tell()

        self.seek(curr_pos)

        self._need_commit = True

    def flush(self):
        if self._need_commit:
            is_create = self._is_create()

            resources = self.parent

            if is_create:
                resources.create(obj=self, file_obj=self._fp)
            else:
                resources.update(obj=self, file_obj=self._fp)

            self._need_commit = False

    def close(self):
        self.flush()

        self._fp = None
        self._size = 0
        self._need_commit = False
        self._opened = False

    def __iter__(self):
        self._check_read()
        return self._fp.__iter__()

    def __next__(self):
        self._check_read()
        return next(self._fp)

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def update(self, file_obj):
        return self._parent.update(self, file_obj=file_obj)


class JarResource(FileResource):
    def __init__(self, **kw):
        super(JarResource, self).__init__(**kw)
        self.type = Resource.Type.JAR


class PyResource(FileResource):
    def __init__(self, **kw):
        super(PyResource, self).__init__(**kw)
        self.type = Resource.Type.PY


class ArchiveResource(FileResource):
    def __init__(self, **kw):
        super(ArchiveResource, self).__init__(**kw)
        self.type = Resource.Type.ARCHIVE


class TableResource(Resource):
    def __init__(self, **kw):
        project_name = kw.pop('project_name', None)
        table_name = kw.pop('table_name', None)
        partition_spec = kw.pop('partition', None)

        super(TableResource, self).__init__(**kw)

        self._init(project_name=project_name, table_name=table_name,
                   partition=partition_spec)

    def _init(self, project_name=None, table_name=None, partition=None):
        project_name = project_name or self._project
        if project_name is not None and project_name != self._project:
            from .projects import Projects
            self._parent = Projects(_client=self._client)[project_name].resources

        if table_name is not None:
            self.source_table_name = '%s.%s' % (project_name, table_name)

        if partition is not None:
            if not isinstance(partition, types.PartitionSpec):
                partition_spec = types.PartitionSpec(partition)
            self.source_table_name = '%s partition(%s)' \
                                     % (self.source_table_name, partition_spec)

    def get_source_table(self):
        if self.source_table_name is None:
            return

        splits = self.source_table_name.split(' partition(')
        src = splits[0]

        if '.' not in src:
            raise ValueError('Malformed source table name: %s' % src)

        project_name, table_name = tuple(src.split('.', 1))

        from .projects import Projects
        return Projects(client=self._client)[project_name].tables[table_name]

    def get_source_table_partition(self):
        if self.source_table_name is None:
            return

        splits = self.source_table_name.split(' partition(')
        if len(splits) < 2:
            return

        partition = splits[1].split(')', 1)[0].strip()
        return types.PartitionSpec(partition)

    def update(self, project_name=None, table_name=None, partition=None):
        self._init(project_name=project_name, table_name=table_name,
                   partition=partition)
        resources = self.parent
        return resources.update(self)
