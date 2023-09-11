import pytest

def pytest_addoption(parser):
    parser.addoption("--largemodel", action="store_true", default = False)
    parser.addoption("--slow", action="store_true", default = False)


def pytest_configure(config):
    config.addinivalue_line("markers", "largemodel: mark test as largemodels")
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    skip_largemodel = pytest.mark.skip(reason="need --largemodel option to run")
    skip_slow = pytest.mark.skip(reason="need --slow option to run")

    for item in items:
        if "largemodel" in item.keywords and not config.getoption("--largemodel"):
            item.add_marker(skip_largemodel)
        if "slow" in item.keywords and not config.getoption("--slow"):
            item.add_marker(skip_slow)