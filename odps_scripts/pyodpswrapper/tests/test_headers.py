# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

import mock
import pytest

from ..headers import (
    DUMP_TRACEBACK_RE,
    MARS_VERSION_RE,
    PROFILE_RE,
    RESOURCE_PACK_RE,
    USE_SPAWN_METHOD_RE,
    _config_run_flag,
    _run_flags,
    _scan_file_comments,
    config_with_headers,
    get_run_flags,
    use_spawn_method,
)


def test_mars_version_regex():
    """Test MARS_VERSION_RE regex pattern"""
    # Test valid patterns
    assert MARS_VERSION_RE.match("# mars_version: 1.0.0")
    assert MARS_VERSION_RE.match("# mars_version = 1.0.0")
    assert MARS_VERSION_RE.match("#  mars_version  :  1.0.0  ")
    assert MARS_VERSION_RE.match("  #\tmars_version:\t1.0.0\t")

    # Test invalid patterns
    assert not MARS_VERSION_RE.match("# mars_version 1.0.0")
    assert not MARS_VERSION_RE.match("# mars_version:")
    assert not MARS_VERSION_RE.match("mars_version: 1.0.0")


def test_dump_traceback_regex():
    """Test DUMP_TRACEBACK_RE regex pattern"""
    # Test valid patterns
    assert DUMP_TRACEBACK_RE.match("# dump_traceback: true")
    assert DUMP_TRACEBACK_RE.match("# dump_traceback = false")
    assert DUMP_TRACEBACK_RE.match("#  dump_traceback  :  1  ")
    assert DUMP_TRACEBACK_RE.match("  #\tdump_traceback:\t0\t")

    # Test invalid patterns
    assert not DUMP_TRACEBACK_RE.match("# dump_traceback true")
    assert not DUMP_TRACEBACK_RE.match("# dump_traceback:")
    assert not DUMP_TRACEBACK_RE.match("dump_traceback: true")


def test_profile_regex():
    """Test PROFILE_RE regex pattern"""
    # Test valid patterns
    assert PROFILE_RE.match("# profile: true")
    assert PROFILE_RE.match("# profile = false")
    assert PROFILE_RE.match("#  profile  :  1  ")
    assert PROFILE_RE.match("  #\tprofile:\t0\t")

    # Test invalid patterns
    assert not PROFILE_RE.match("# profile true")
    assert not PROFILE_RE.match("# profile:")
    assert not PROFILE_RE.match("profile: true")


def test_resource_pack_regex():
    """Test RESOURCE_PACK_RE regex pattern"""
    # Test valid patterns
    assert RESOURCE_PACK_RE.match("# resource_pack: pack1")
    assert RESOURCE_PACK_RE.match("# resource_pack = pack1,pack2")
    assert RESOURCE_PACK_RE.match("#  resource_pack  :  pack1 , pack2  ")
    assert RESOURCE_PACK_RE.match("  #\tresource_pack:\tpack1,pack2\t")

    # Test captures content correctly
    match = RESOURCE_PACK_RE.match("# resource_pack: pack1,pack2")
    assert match.group(1) == "pack1,pack2"


def test_use_spawn_method_regex():
    """Test USE_SPAWN_METHOD_RE regex pattern"""
    # Test valid patterns
    assert USE_SPAWN_METHOD_RE.match("# use_spawn_method: true")
    assert USE_SPAWN_METHOD_RE.match("# use_spawn_method = false")
    assert USE_SPAWN_METHOD_RE.match("#  use_spawn_method  :  1  ")
    assert USE_SPAWN_METHOD_RE.match("  #\tuse_spawn_method:\t0\t")

    # Test invalid patterns
    assert not USE_SPAWN_METHOD_RE.match("# use_spawn_method true")
    assert not USE_SPAWN_METHOD_RE.match("# use_spawn_method:")
    assert not USE_SPAWN_METHOD_RE.match("use_spawn_method: true")


@pytest.mark.skipif(
    sys.version_info[0] >= 3, reason="Python 3 handles strings differently"
)
def test_scan_file_comments_python2():
    """Test _scan_file_comments function for Python 2"""
    code = """# mars_version: 1.0.0
# dump_traceback: true
# profile: false
# resource_pack: pack1,pack2
# use_spawn_method: 1"""

    result = _scan_file_comments(
        code,
        MARS_VERSION_RE,
        DUMP_TRACEBACK_RE,
        PROFILE_RE,
        USE_SPAWN_METHOD_RE,
        RESOURCE_PACK_RE,
    )

    assert result[0] == "1.0.0"
    assert result[1] == "true"
    assert result[2] == "false"
    assert result[3] == "1"
    assert result[4] == "pack1,pack2"


def test_config_run_flag():
    """Test _config_run_flag function"""
    # Clear flags first
    _run_flags.clear()

    # Test None value
    _config_run_flag("test_flag1", None, default=True)
    assert _run_flags["test_flag1"] is True

    _config_run_flag("test_flag2", None, default=False)
    assert _run_flags["test_flag2"] is False

    # Test explicit false values
    _config_run_flag("test_flag3", "0")
    assert _run_flags["test_flag3"] is False

    _config_run_flag("test_flag4", "false")
    assert _run_flags["test_flag4"] is False

    _config_run_flag("test_flag5", "False")
    assert _run_flags["test_flag5"] is False

    # Test truthy values
    _config_run_flag("test_flag6", "1")
    assert _run_flags["test_flag6"] is True

    _config_run_flag("test_flag7", "true")
    assert _run_flags["test_flag7"] is True


@pytest.mark.skipif(sys.version_info[0] < 3, reason="Python 2 does not support Mars")
def test_config_with_headers():
    """Test config_with_headers function"""
    code = """# mars_version: 1.0.0
# dump_traceback: true
# profile: false
# resource_pack: pack1,pack2
# use_spawn_method: 1"""

    _run_flags.clear()

    # Patch the functions in their original modules
    with mock.patch(
        'odps_scripts.pyodpswrapper.mars_support.config_mars_version'
    ) as mock_config_mars, mock.patch(
        'odps_scripts.pyodpswrapper.resource.load_packages_in_subprocess'
    ) as mock_load_packages:
        config_with_headers(code)

        # Check that the mocks were called with correct arguments
        mock_config_mars.assert_called_once_with("1.0.0")
        mock_load_packages.assert_called_once_with("pack1,pack2")

        # Check that flags were set correctly
        assert _run_flags["dump_traceback"] is True
        assert _run_flags["profile"] is False
        assert _run_flags["use_spawn_method"] is True


@pytest.fixture(autouse=True)
def reset_run_flags():
    """Reset _run_flags before and after each test"""
    _run_flags.clear()
    try:
        yield
    finally:
        _run_flags.clear()


def test_get_run_flags():
    """Test get_run_flags function"""
    _run_flags.clear()
    _run_flags.update({"flag1": True, "flag2": False})

    flags = get_run_flags()
    assert flags == _run_flags
    assert flags["flag1"] is True
    assert flags["flag2"] is False


def test_use_spawn_method():
    """Test use_spawn_method function"""
    # Test when flag is explicitly set
    _run_flags.clear()
    _run_flags["use_spawn_method"] = True
    assert use_spawn_method() is True

    _run_flags["use_spawn_method"] = False
    assert use_spawn_method() is False

    # Test when flag is not set (should depend on Python version)
    _run_flags.clear()
    # Delete the key to simulate it not being set
    _run_flags.pop("use_spawn_method", None)
    assert use_spawn_method() == (sys.version_info[0] >= 3)
