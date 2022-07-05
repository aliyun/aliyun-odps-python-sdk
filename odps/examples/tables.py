# encoding: utf-8
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
import glob
import codecs
import shutil
import tarfile
import gzip
import warnings

from itertools import groupby, product

from ..compat import pickle, urlretrieve, six, OrderedDict
from ..tunnel import TableTunnel
from ..utils import load_static_text_file, build_pyodps_dir

USER_DATA_REPO = build_pyodps_dir('data')


class TestDataMixIn(object):
    def after_create_test_data(self, table_name):
        pass


def table_creator(func):
    """
    Decorator for table creating method
    """
    def method(self, table_name, **kwargs):
        if self.odps.exist_table(table_name):
            return
        if kwargs.get('project', self.odps.project) != self.odps.project:
            tunnel = TableTunnel(self.odps, project=kwargs['project'])
        else:
            tunnel = self.tunnel
        func(self.odps, table_name, tunnel=tunnel, **kwargs)
        self.after_create_test_data(table_name)

    method.__name__ = func.__name__
    setattr(TestDataMixIn, func.__name__, method)

    return func


"""
Simple DataFrames
"""


@table_creator
def create_ionosphere(odps, table_name, tunnel=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    fields = ','.join('a%02d double' % i for i in range(1, 35)) + ', class bigint'
    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, fields, project=project)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for line in load_static_text_file('data/ionosphere.txt').splitlines():
        rec = upload_ss.new_record()
        cols = [float(c) if rec._columns[i].type == 'double' else int(c) for i, c in enumerate(line.split(','))]
        [rec.set(i, val) for i, val in enumerate(cols)]
        writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])


@table_creator
def create_ionosphere_one_part(odps, table_name, partition_count=3, tunnel=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    fields = ','.join('a%02d double' % i for i in range(1, 35)) + ', class bigint'
    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, (fields, 'part bigint'), project=project)
    for part_id in range(partition_count):
        odps.execute_sql('alter table %s add if not exists partition (part=%d)' % (table_name, part_id), project=project)

    upload_sses = [tunnel.create_upload_session(table_name, 'part=%d' % part_id) for part_id in range(partition_count)]
    writers = [session.open_record_writer(0) for session in upload_sses]

    for line_no, line in enumerate(load_static_text_file('data/ionosphere.txt').splitlines()):
        part_id = line_no % partition_count
        rec = upload_sses[part_id].new_record()
        cols = [float(c) if rec._columns[i].type == 'double' else int(c) for i, c in enumerate(line.split(','))]
        cols.append(part_id)
        [rec.set(i, val) for i, val in enumerate(cols)]
        writers[part_id].write(rec)
    [writer.close() for writer in writers]
    [upload_ss.commit([0, ]) for upload_ss in upload_sses]


@table_creator
def create_ionosphere_two_parts(odps, table_name, partition1_count=2, partition2_count=3, tunnel=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    fields = ','.join('a%02d double' % i for i in range(1, 35)) + ', class bigint'
    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, (fields, 'part1 bigint, part2 bigint'), project=project)
    for id1, id2 in product(range(partition1_count), range(partition2_count)):
        odps.execute_sql('alter table %s add if not exists partition (part1=%d, part2=%d)' % (table_name, id1, id2),
                         project=project)

    upload_sses = [[tunnel.create_upload_session(table_name, 'part1=%d,part2=%d' % (id1, id2))
                    for id2 in range(partition2_count)] for id1 in range(partition1_count)]
    writers = [[session.open_record_writer(0) for session in sessions] for sessions in upload_sses]

    for line_no, line in enumerate(load_static_text_file('data/ionosphere.txt').splitlines()):
        id1, id2 = line_no % partition1_count, line_no % partition2_count
        rec = upload_sses[id1][id2].new_record()
        cols = [float(c) if rec._columns[i].type == 'double' else int(c) for i, c in enumerate(line.split(','))]
        cols.extend([id1, id2])
        [rec.set(i, val) for i, val in enumerate(cols)]
        writers[id1][id2].write(rec)
    [writer.close() for ws in writers for writer in ws]
    [upload_ss.commit([0, ]) for upload_sss in upload_sses for upload_ss in upload_sss]


@table_creator
def create_iris(odps, table_name, tunnel=None, project=None, lifecycle=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    odps.delete_table(table_name, if_exists=True, project=project)
    table = odps.create_table(table_name, 'sepal_length double, sepal_width double, petal_length double, '
                              + 'petal_width double, category string', project=project, lifecycle=lifecycle)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for line in load_static_text_file('data/iris.txt').splitlines():
        rec = upload_ss.new_record()
        line_parts = line.split(',')
        cols = [float(c) for c in line_parts[:-1]]
        cols.append(line_parts[4])
        [rec.set(i, val) for i, val in enumerate(cols)]
        writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])
    return table


@table_creator
def create_iris_kv(odps, table_name, tunnel=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, 'content string, category bigint', project=project)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for line in load_static_text_file('data/iris.txt').splitlines():
        rec = upload_ss.new_record()
        line_parts = line.split(',')
        rec.set(0, ','.join('%s:%s' % (idx, c) for idx, c in enumerate(line_parts[:-1])))
        rec.set(1, 0 if 'setosa' in line_parts[-1] else 1)
        writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])


@table_creator
def create_corpus(odps, table_name, tunnel=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, 'id string, content string', project=project)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for line_no, line in enumerate(load_static_text_file('data/splited_words.txt').splitlines()):
        rec = upload_ss.new_record()
        cols = [line_no + 1, line.replace('####', '')]
        [rec.set(i, val) for i, val in enumerate(cols)]
        writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])


@table_creator
def create_word_triple(odps, table_name, tunnel=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, 'id string, word string, count bigint', project=project)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for line_no, line in enumerate(load_static_text_file('data/splited_words.txt').splitlines()):
        line = line.strip()
        if not line:
            break
        for word, group in groupby(sorted(line.split('####'))):
            rec = upload_ss.new_record()
            cols = [str(line_no + 1), word, len(list(group))]
            [rec.set(i, val) for i, val in enumerate(cols)]
            writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])


@table_creator
def create_splited_words(odps, table_name, joined=False, tunnel=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, 'doc_id string, content string', project=project)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for line_no, line in enumerate(load_static_text_file('data/splited_words.txt').splitlines()):
        if not line.strip():
            break
        if joined:
            rec = upload_ss.new_record()
            cols = [line_no + 1, line.replace('####', ' ')]
            [rec.set(i, val) for i, val in enumerate(cols)]
            writer.write(rec)
        else:
            for word in line.split('####'):
                if not word:
                    continue
                rec = upload_ss.new_record()
                cols = [line_no + 1, word]
                [rec.set(i, val) for i, val in enumerate(cols)]
                writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])


@table_creator
def create_weighted_graph_edges(odps, table_name, tunnel=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    data_rows = [
        ['1', '1', '2', '1', 0.7], ['1', '1', '3', '1', 0.7], ['1', '1', '4', '1', 0.6], ['2', '1', '3', '1', 0.7],
        ['2', '1', '4', '1', 0.6], ['3', '1', '4', '1', 0.6], ['4', '1', '6', '5', 0.3], ['5', '5', '6', '5', 0.6],
        ['5', '5', '7', '5', 0.7], ['5', '5', '8', '5', 0.7], ['6', '5', '7', '5', 0.6], ['6', '5', '8', '5', 0.6],
        ['7', '5', '8', '5', 0.7]
    ]
    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, 'flow_out_id string, group_out_id string, flow_in_id string, ' +
                      'group_in_id string, edge_weight double', project=project)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for rd in data_rows:
        rec = upload_ss.new_record()
        [rec.set(i, val) for i, val in enumerate(rd)]
        writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])


@table_creator
def create_tree_graph(odps, table_name, tunnel=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    data_rows = [
        ['0', '1'], ['0', '2'], ['1', '3'], ['1', '4'], ['2', '4'], ['2', '5'], ['4', '6'], ['a', 'b'], ['a', 'c'],
        ['c', 'd'], ['c', 'e']
    ]
    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, 'flow_out_id string, flow_in_id string', project=project)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for rd in data_rows:
        rec = upload_ss.new_record()
        [rec.set(i, val) for i, val in enumerate(rd)]
        writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])


@table_creator
def create_weighted_graph_vertices(odps, table_name, tunnel=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    data_rows = [
        ['1', '1', 0.7, 1.0], ['2', '1', 0.7, 1.0], ['3', '1', 0.7, 1.0], ['4', '1', 0.5, 1.0], ['5', '5', 0.7, 1.0],
        ['6', '5', 0.5, 1.0], ['7', '5', 0.7, 1.0], ['8', '5', 0.7, 1.0]
    ]
    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, 'node string, label string, node_weight double, label_weight double',
                      project=project)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for rd in data_rows:
        rec = upload_ss.new_record()
        [rec.set(i, val) for i, val in enumerate(rd)]
        writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])


@table_creator
def create_user_item_table(odps, table_name, tunnel=None, mode=None, project=None):
    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)
    data_rows = [
        ['CST0000', 'a', 0], ['CST0000', 'b', 1], ['CST0000', 'c', 2], ['CST0000', 'd', 3], ['CST0001', 'a', 0],
        ['CST0001', 'b', 0], ['CST0001', 'a', 1], ['CST0001', 'b', 1], ['CST0001', 'c', 1], ['CST0001', 'b', 2],
        ['CST0001', 'c', 2], ['CST0001', 'd', 2], ['CST0001', 'e', 3], ['CST0002', 'a', 0], ['CST0002', 'c', 0],
        ['CST0002', 'b', 1], ['CST0002', 'a', 2], ['CST0002', 'b', 2], ['CST0002', 'c', 2], ['CST0002', 'a', 3],
        ['CST0000', 'a', 0], ['CST0000', 'b', 1], ['CST0000', 'c', 2], ['CST0000', 'd', 3], ['CST0001', 'a', 0],
        ['CST0001', 'b', 0], ['CST0001', 'a', 1], ['CST0001', 'b', 1], ['CST0001', 'c', 1], ['CST0001', 'b', 2],
        ['CST0001', 'c', 2], ['CST0001', 'd', 2], ['CST0001', 'e', 3], ['CST0002', 'a', 0], ['CST0002', 'c', 0],
        ['CST0002', 'b', 1], ['CST0002', 'a', 2], ['CST0002', 'b', 2], ['CST0002', 'c', 2], ['CST0002', 'a', 3],
        ['CST0000', 'a', 0], ['CST0000', 'b', 1], ['CST0000', 'c', 2], ['CST0000', 'd', 3], ['CST0001', 'a', 0],
        ['CST0001', 'b', 0], ['CST0001', 'a', 1], ['CST0001', 'b', 1], ['CST0001', 'c', 1], ['CST0001', 'b', 2],
        ['CST0001', 'c', 2], ['CST0001', 'd', 2], ['CST0001', 'e', 3], ['CST0002', 'a', 0], ['CST0002', 'c', 0],
        ['CST0002', 'b', 1], ['CST0002', 'a', 2], ['CST0002', 'b', 2], ['CST0002', 'c', 2], ['CST0002', 'a', 3],
        ['CST0000', 'a', 0], ['CST0000', 'b', 1], ['CST0000', 'c', 2], ['CST0000', 'd', 3], ['CST0001', 'a', 0],
        ['CST0001', 'b', 0], ['CST0001', 'a', 1], ['CST0001', 'b', 1], ['CST0001', 'c', 1], ['CST0001', 'b', 2],
        ['CST0001', 'c', 2], ['CST0001', 'd', 2], ['CST0001', 'e', 3], ['CST0002', 'a', 0], ['CST0002', 'c', 0],
        ['CST0002', 'b', 1], ['CST0002', 'a', 2], ['CST0002', 'b', 2], ['CST0002', 'c', 2], ['CST0002', 'a', 3]
    ]
    odps.delete_table(table_name, if_exists=True, project=project)
    if mode == 'agg':
        data_rows = [k + (len(list(p)), ) for k, p in groupby(sorted(data_rows, key=lambda item: (item[0], item[1])), lambda item: (item[0], item[1]))]
        odps.create_table(table_name, 'user string, item string, payload bigint', project=project)
    elif mode == 'exist':
        items = dict()
        for k, g1 in groupby(sorted(data_rows, key=lambda it: it[0]), key=lambda it: it[0]):
            items[k] = dict((k2, 1) for k2 in set(it2[1] for it2 in g1))
        products = set(it[1] for it in data_rows)
        for k, g in six.iteritems(items):
            unexist_set = products - set(six.iterkeys(g))
            for it in unexist_set:
                g[it] = 0
        data_rows = [(u, p, l) for u, ud in six.iteritems(items) for p, l in six.iteritems(ud)]
        odps.create_table(table_name, 'user string, item string, label bigint', project=project)
    elif mode == 'kv':
        data_rows = [(k[0], '{0}:{1}'.format(k[1], len(list(p))))
                     for k, p in groupby(sorted(data_rows, key=lambda item: (item[0], item[1])), lambda item: (item[0], item[1]))]
        data_rows = [(k, ','.join((v[1] for v in p))) for k, p in groupby(data_rows, key=lambda item: item[0])]
        odps.create_table(table_name, 'user string, item_list string', project=project)
    else:
        odps.create_table(table_name, 'user string, item string, time bigint', project=project)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for rd in data_rows:
        rec = upload_ss.new_record()
        [rec.set(i, val) for i, val in enumerate(rd)]
        writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])


@table_creator
def create_dow_jones(odps, table_name, tunnel=None, project=None):
    fields = """quarter stock date open high low close volume percent_change_price percent_change_volume_over_last_wk
    previous_weeks_volume next_weeks_open next_weeks_close percent_change_next_weeks_price days_to_next_dividend
    percent_return_next_dividend""".strip().split()
    field_types = OrderedDict([(fname, 'double') for fname in fields])
    field_types['quarter'] = 'bigint'
    field_types['stock'] = 'string'
    field_types['date'] = 'string'
    fields_str = ','.join('`{0}` {1}'.format(k, v) for k, v in six.iteritems(field_types))

    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, fields_str, project=project)

    def iter_lines():
        for line in load_static_text_file('data/dow_jones.txt').splitlines():
            ldata = line.split(',')
            ldata[0] = int(ldata[0])
            ldata[3:] = [float(v) if v else None for v in ldata[3:]]
            yield ldata

    odps.write_table(table_name, iter_lines())


"""
20 Newsgroups
"""

NEWSGROUP_URL = 'http://repo.aliyun.com/shared_data/20news-bydate.tar.gz'
NEWSGROUP_DATA_NAME = '20news-bydate'
NEWSGROUP_ARCHIVE_NAME = '20news-bydate.tar.gz'
NEWSGROUP_CACHE_NAME = '20news-bydate.pkz'
NEWSGROUP_TRAIN_DIR = "20news-bydate-train"
NEWSGROUP_TEST_DIR = "20news-bydate-test"
NEWSGROUP_TRAIN_FOLDER = "20news-bydate-train"
NEWSGROUP_TEST_FOLDER = "20news-bydate-test"


def download_newsgroup(target_dir, cache_dir):
    target_tar = os.path.join(os.path.expanduser('~'), NEWSGROUP_ARCHIVE_NAME)
    urlretrieve(NEWSGROUP_URL, target_tar)
    cache_newsgroup_tar(target_tar, target_dir, cache_dir)


def cache_newsgroup_tar(target_tar, target_dir, cache_dir):
    tarfile.open(target_tar, 'r:gz').extractall(path=target_dir)
    os.unlink(target_tar)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    train_path = os.path.join(target_dir, NEWSGROUP_TRAIN_FOLDER)
    test_path = os.path.join(target_dir, NEWSGROUP_TEST_FOLDER)
    cache_path = os.path.join(cache_dir, NEWSGROUP_CACHE_NAME)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    def load_files(path, encoding):
        objs = []
        for fn in glob.glob(os.path.join(path, '*')):
            file_cat = os.path.basename(os.path.normpath(fn))
            for sfn in glob.glob(os.path.join(fn, '*')):
                file_id = os.path.basename(os.path.normpath(sfn))
                with open(sfn, 'rb') as f:
                    objs.append((file_id, file_cat, f.read().decode(encoding)))

        return objs

    # Store a zipped pickle
    cache = dict(train=load_files(train_path, encoding='latin1'),
                 test=load_files(test_path, encoding='latin1'))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

    shutil.rmtree(target_dir)


@table_creator
def create_newsgroup_table(odps, table_name, tunnel=None, data_part='train', project=None):
    cache_file = os.path.join(USER_DATA_REPO, NEWSGROUP_CACHE_NAME)
    if not os.path.exists(USER_DATA_REPO):
        os.makedirs(USER_DATA_REPO)
    if not os.path.exists(cache_file):
        warnings.warn('We need to download data from ' + NEWSGROUP_URL + '.')
        download_newsgroup(os.path.join(USER_DATA_REPO, NEWSGROUP_DATA_NAME), USER_DATA_REPO)

    with open(cache_file, 'rb') as f:
        cache = pickle.loads(codecs.decode(f.read(), 'zlib_codec'))

    if tunnel is None:
        tunnel = TableTunnel(odps, project=project)

    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, 'id string, category string, message string', project=project)

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for line in cache[data_part]:
        rec = upload_ss.new_record()
        [rec.set(i, six.text_type(val)) for i, val in enumerate(line)]
        writer.write(rec)
    writer.close()
    upload_ss.commit([0, ])


"""
MNIST
"""

MNIST_PICKLED_URL = 'http://repo.aliyun.com/shared_data/mnist.pkl.gz'
MNIST_FILE = 'mnist.pkl.gz'
mnist_unpickled = None


def load_mnist_data():
    global mnist_unpickled

    if mnist_unpickled is not None:
        return mnist_unpickled

    mnist_file = os.path.join(USER_DATA_REPO, MNIST_FILE)
    if not os.path.exists(USER_DATA_REPO):
        os.makedirs(USER_DATA_REPO)
    if not os.path.exists(mnist_file):
        warnings.warn('We need to download data from ' + MNIST_PICKLED_URL + '.')
        urlretrieve(MNIST_PICKLED_URL, mnist_file)

    with gzip.open(mnist_file, 'rb') as fobj:
        mnist_unpickled = pickle.load(fobj)
        fobj.close()

    return mnist_unpickled


@table_creator
def create_mnist_table(odps, table_name, part_id=0, tunnel=None, project=None):
    train_data = load_mnist_data()[part_id]

    if tunnel is None:
        tunnel = TableTunnel(odps)

    odps.delete_table(table_name, if_exists=True, project=project)
    odps.create_table(table_name, 'feature string, class string')

    upload_ss = tunnel.create_upload_session(table_name)
    writer = upload_ss.open_record_writer(0)

    for feature, label in zip(train_data[0], train_data[1]):
        rec = upload_ss.new_record()
        rec.set(0, six.text_type(','.join(str(n) for n in feature)))
        rec.set(1, six.text_type(label))
        writer.write(rec)

    writer.close()
    upload_ss.commit([0, ])
