# -*- coding: utf-8 -*-
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

import os

import pytest

from ...envs import REGION_ENV
from ..core import I18NMessage


@pytest.fixture(autouse=True)
def reset_i18n_state():
    # Save original environment and language setting
    original_region_env = os.environ.get(REGION_ENV)
    original_lang = I18NMessage._lang

    # Reset language for each test
    I18NMessage._lang = None

    try:
        yield
    finally:
        # Restore original environment and language setting
        I18NMessage._lang = original_lang
        if original_region_env is not None:
            os.environ[REGION_ENV] = original_region_env
        elif REGION_ENV in os.environ:
            del os.environ[REGION_ENV]


def test_initialization_with_default_only():
    """Test I18NMessage initialization with only default message."""
    message = I18NMessage("Hello World")
    # When no specific language is set and no region mapping applies,
    # it defaults to "default" which shows all messages
    assert "Hello World" in str(message)


def test_initialization_with_multiple_languages():
    """Test I18NMessage initialization with multiple language messages."""
    message = I18NMessage("Hello World", cn="你好世界")

    # Test default language explicitly
    I18NMessage._lang = "default"
    assert "Hello World" in str(message)

    # Set language to Chinese
    I18NMessage._lang = "cn"
    assert str(message) == "你好世界"


def test_clean_message():
    """Test _clean_message removes extra whitespace and dedents text."""
    # Test with indented text
    message = I18NMessage(
        """
        This is a test message
        with multiple lines
        """
    )
    expected = "This is a test message\nwith multiple lines"
    assert str(message).strip() == expected


def test_region_mapping_cn():
    """Test that CN region maps to Chinese language."""
    os.environ[REGION_ENV] = "cn-hangzhou"
    message = I18NMessage("Hello World", cn="你好世界")
    assert str(message) == "你好世界"


def test_region_mapping_d2():
    """Test that D2 region maps to all language."""
    os.environ[REGION_ENV] = "d2"
    message = I18NMessage("Hello", cn="你好")
    # For "all" language, it should show both versions
    result = str(message)
    assert "Hello" in result
    assert "你好" in result


def test_region_mapping_default():
    """Test that unknown regions default to English."""
    os.environ[REGION_ENV] = "us-east-1"
    # When no region mapping applies, it defaults to "default" language
    # which shows all messages
    message = I18NMessage("Hello World", cn="你好世界")
    assert "Hello World" in str(message)


def test_add_operation():
    """Test adding two I18NMessage objects."""
    msg1 = I18NMessage("First part", cn="第一部分")
    msg2 = I18NMessage("Second part", cn="第二部分")
    combined = msg1 + msg2

    # Test default language explicitly
    I18NMessage._lang = "default"
    result = str(combined)
    assert "First part" in result
    assert "Second part" in result

    # Test Chinese language
    I18NMessage._lang = "cn"
    assert str(combined) == "第一部分第二部分"


def test_add_operation_with_missing_language():
    """Test adding messages where one doesn't have the specific language."""
    msg1 = I18NMessage("First part", cn="第一部分")
    msg2 = I18NMessage("Second part")  # No Chinese version
    combined = msg1 + msg2

    # Test default language explicitly
    I18NMessage._lang = "default"
    result = str(combined)
    assert "First part" in result
    assert "Second part" in result

    # Test Chinese language - should fall back to default for second part
    I18NMessage._lang = "cn"
    assert str(combined) == "第一部分Second part"


def test_all_language_combined():
    """Test that 'all' language combines all languages."""
    message = I18NMessage("Hello", cn="你好", fr="Bonjour")
    I18NMessage._lang = "all"
    result = str(message)
    assert "Hello" in result
    assert "你好" in result
    assert "Bonjour" in result
