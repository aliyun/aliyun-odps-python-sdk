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
