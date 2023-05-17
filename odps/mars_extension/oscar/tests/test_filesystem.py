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

import numpy as np
import pytest

try:
    import mars
    from mars.lib.filesystem import get_fs
except ImportError:
    mars = None

from ..filesystem import VolumeFileSystem


@pytest.mark.skipif(mars is None, reason="mars not installed")
def test_volume_filesystem(odps):
    volume_name = "test_mars_volume"
    filename = "test_data"
    content = np.random.bytes(30)
    filename2 = "test_data2"
    content2 = np.random.bytes(40)

    if odps.exist_volume(volume_name):
        odps.delete_volume(volume_name)

    path = "odps:///{}/volumes/{}".format(odps.project, volume_name)
    fs = get_fs(path, {"odps": odps})
    assert isinstance(fs, VolumeFileSystem)

    fs.mkdir("/")
    files = fs.ls("/")
    assert len(files) == 0

    f = fs.open(filename, "wb")
    with f:
        f.write(content)

    files = fs.ls("/")
    assert len(files) == 1
    assert filename in files[0]

    f = fs.open(filename, "rb")
    with f:
        assert f.read() == content

    fs2 = get_fs(path, {"odps": odps, "chunk_size": 25})

    f = fs2.open(filename2, "wb")
    with f:
        f.write(content)
        f.write(content2)

    assert len(list(odps.list_volume_files(volume_name, filename2))) > 1

    f = fs2.open(filename2, "rb")
    with f:
        assert f.read() == content + content2

    fs2.delete(filename2)
    assert odps.exist_volume_partition(volume_name, filename2) is False

    with pytest.raises(FileNotFoundError):
        fs.ls("odps:///{}/volumes/{}".format(odps.project, "not_exist"))
