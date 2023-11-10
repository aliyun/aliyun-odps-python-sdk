import pytest

from odps.tests.core import get_config, drop_test_tables


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
def odps_with_schema_tenant():
    try:
        return get_config().odps_with_schema_tenant
    except AttributeError:
        pytest.skip("ODPS project with schema configured on tenants not defined")


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
        data.append(
            html.div("Logs disabled for passed tests.", **{"class": "log"})
        )
