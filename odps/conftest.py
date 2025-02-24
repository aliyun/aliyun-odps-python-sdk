# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import pytest

from odps.tests.core import drop_test_tables, get_config


@pytest.fixture(scope="session")
def odps():
    cfg = get_config()
    return cfg.odps


@pytest.fixture(scope="session")
def odps_daily():
    try:
        return get_config().odps_daily
    except AttributeError:
        pytest.skip("ODPS project with schema not defined")


@pytest.fixture(scope="session")
def odps_with_storage_tier():
    try:
        return get_config().odps_with_storage_tier
    except AttributeError:
        pytest.skip("ODPS project with schema not defined")


@pytest.fixture(scope="session")
def odps_with_schema():
    try:
        return get_config().odps_with_schema
    except AttributeError:
        pytest.skip("ODPS project with schema not defined")


@pytest.fixture(scope="session")
def odps_with_mcqa2():
    try:
        return get_config().odps_with_mcqa2
    except AttributeError:
        pytest.skip("ODPS project with schema not defined")


@pytest.fixture(scope="session")
def odps_with_tunnel_quota():
    try:
        return get_config().odps_with_tunnel_quota
    except AttributeError:
        pytest.skip("ODPS project with quota not defined")


@pytest.fixture(scope="session")
def odps_with_long_string():
    try:
        return get_config().odps_with_long_string
    except AttributeError:
        pytest.skip("ODPS project with quota not defined")


@pytest.fixture(scope="session")
def config():
    return get_config()


@pytest.fixture(scope="session")
def delete_test_tables(odps):
    try:
        yield
    finally:
        drop_test_tables(odps)


@pytest.fixture(scope="session")
def tunnel():
    return get_config().tunnel


@pytest.mark.optionalhook
def pytest_html_results_table_html(report, data):
    """
    Hide captured logs for successful tests
    """
    try:
        from py.xml import html
    except ImportError:
        return

    if report.passed:
        del data[:]
        data.append(html.div("Logs disabled for passed tests.", **{"class": "log"}))
