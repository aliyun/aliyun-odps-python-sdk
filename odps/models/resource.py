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

from .core import LazyLoad
from .cache import cache
from .. import serializers, utils, types, errors, compat
from ..compat import Enum, six


RESOURCE_SIZE_MAX = 512 * 1024 * 1024  # a single resource's size must be at most 512M


class Resource(LazyLoad):
    """
    Resource is useful when writing UDF or MapReduce. This is an abstract class.

    Basically, resource can be either a file resource or a table resource.
    File resource can be ``file``, ``py``, ``jar``, ``archive`` in details.

    .. seealso:: :class:`odps.models.FileResource`, :class:`odps.models.PyResource`,
                 :class:`odps.models.JarResource`, :class:`odps.models.ArchiveResource`,
                 :class:`odps.models.TableResource`
    """

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
    """
    File resource represents for a file.

    Use ``open`` method to open this resource as an file-like object.
    """

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
        """
        The argument ``mode`` stands for the open mode for this file resource.
        It can be binary mode if the 'b' is inside. For instance,
        'rb' means opening the resource as read binary mode
        while 'r+b' means opening the resource as read+write binary mode.
        This is most import when the file is actually binary such as tar or jpeg file,
        so be aware of opening this file as a correct mode.

        Basically, the text mode can be 'r', 'w', 'a', 'r+', 'w+', 'a+'
        just like the builtin python ``open`` method.

        * ``r`` means read only
        * ``w`` means write only, the file will be truncated when opening
        * ``a`` means append only
        * ``r+`` means read+write without constraint
        * ``w+`` will truncate first then opening into read+write
        * ``a+`` can read+write, however the written content can only be appended to the end

        :param mode: the mode of opening file, described as above
        :param encoding: utf-8 as default
        :return: file-like object

        :Example:

        >>> with resource.open('r') as fp:
        >>>     fp.read(1)  # read one unicode character
        >>>     fp.write('test')  # wrong, cannot write under read mode
        >>>
        >>> with resource.open('wb') as fp:
        >>>     fp.readlines() # wrong, cannot read under write mode
        >>>     fp.write('hello world') # write bytes
        >>>
        >>> with resource.open('test_resource', 'r+') as fp: # open as read-write mode
        >>>     fp.seek(5)
        >>>     fp.truncate()
        >>>     fp.flush()
        """

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
        """
        Read the file resource, read all as default.

        :param size: unicode or byte length depends on text mode or binary mode.
        :return: unicode or bytes depends on text mode or binary mode
        :rtype: str or unicode(Py2), bytes or str(Py3)
        """

        self._check_read()
        return self._fp.read(size)

    def readline(self, size=-1):
        """
        Read a single line.

        :param size: If the size argument is present and non-negative,
                     it is a maximum byte count (including the trailing newline)
                     and an incomplete line may be returned.
                     When size is not 0,
                     an empty string is returned only when EOF is encountered immediately
        :return: unicode or bytes depends on text mode or binary mode
        :rtype: str or unicode(Py2), bytes or str(Py3)
        """

        self._check_read()
        return self._fp.readline(size)

    def readlines(self, sizehint=-1):
        """
        Read as lines.

        :param sizehint: If the optional sizehint argument is present, instead of reading up to EOF,
                     whole lines totalling approximately sizehint bytes
                     (possibly after rounding up to an internal buffer size) are read.
        :return: lines
        :rtype: list
        """

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
        """
        Write content into the file resource

        :param content: content to write
        :return: None
        """

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
        """
        Write lines into the file resource.

        :param seq: lines
        :return: None
        """

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
        """
        Seek to some place.

        :param pos: position to seek
        :param whence: if set to 2, will seek to the end
        :return: None
        """

        return self._fp.seek(pos, whence)

    def tell(self):
        """
        Tell the current position

        :return: current position
        """

        return self._fp.tell()

    def truncate(self, size=None):
        """
        Truncate the file resource's size.

        :param size: If the optional size argument is present,
                     the file is truncated to (at most) that size.
                     The size defaults to the current position.
        :return: None
        """

        self._check_write()

        curr_pos = self.tell()
        self._fp.truncate(size)

        self.seek(0, compat.SEEK_END)
        self._size = self.tell()

        self.seek(curr_pos)

        self._need_commit = True

    def flush(self):
        """
        Commit the change to ODPS if any change happens.
        Close will do this automatically.

        :return: None
        """

        if self._need_commit:
            is_create = self._is_create()

            resources = self.parent

            if is_create:
                resources.create(obj=self, file_obj=self._fp)
            else:
                resources.update(obj=self, file_obj=self._fp)

            self._need_commit = False

    def close(self):
        """
        Close this file resource.

        :return: None
        """

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
    """
    File resource representing for the .jar file.
    """

    def __init__(self, **kw):
        super(JarResource, self).__init__(**kw)
        self.type = Resource.Type.JAR


class PyResource(FileResource):
    """
    File resource representing for the .py file.
    """

    def __init__(self, **kw):
        super(PyResource, self).__init__(**kw)
        self.type = Resource.Type.PY


class ArchiveResource(FileResource):
    """
    File resource representing for the compressed file like .zip/.tgz/.tar.gz/.tar/jar
    """

    def __init__(self, **kw):
        super(ArchiveResource, self).__init__(**kw)
        self.type = Resource.Type.ARCHIVE


class TableResource(Resource):
    """
    Take a table as a resource.
    """

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
                                     % (self.source_table_name.split(' partition(')[0],
                                        partition_spec)

    def get_source_table(self):
        """
        Get the table object.

        :return: source table
        :rtype: :class:`odps.models.Table`

        .. seealso:: :class:`odps.models.Table`
        """

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
        """
        Get the source table partition.

        :return: the source table partition
        """

        if self.source_table_name is None:
            return

        splits = self.source_table_name.split(' partition(')
        if len(splits) < 2:
            return

        partition = splits[1].split(')', 1)[0].strip()
        return types.PartitionSpec(partition)

    def update(self, project_name=None, table_name=None, partition=None):
        """
        Update this resource.

        :param project_name: the source table's project
        :param table_name: the source table's name
        :param partition: the source table's partition
        :return: self
        """

        self._init(project_name=project_name, table_name=table_name,
                   partition=partition)
        resources = self.parent
        return resources.update(self)
