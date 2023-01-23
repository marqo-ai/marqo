import pytest

def pytest_addoption(parser):
    parser.addoption("--largemodel", action="store_true", default = False)


def pytest_configure(config):
    config.addinivalue_line("markers", "largemodel: mark test as largemodels")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --largemodel given in cli: do not skip largemodel tests
        return
    skip_largemodel = pytest.mark.skip(reason="need --largemodel option to run")
    for item in items:
        if "largemodel" in item.keywords:
            item.add_marker(skip_largemodel)