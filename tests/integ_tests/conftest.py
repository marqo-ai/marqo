import pytest


def pytest_addoption(parser):
    parser.addoption("--largemodel", action="store_true", default=False)
    parser.addoption("--multinode", action="store_true", default=False, help="Run tests that have multiple Vespa nodes")


def pytest_configure(config):
    config.addinivalue_line("markers", "largemodel: mark test as largemodels")
    config.addinivalue_line("markers", "cpu_only: mark test as cpu_only")
    config.addinivalue_line("markers", "unittest: mark test as unit test, it does not require vespa to run")
    config.addinivalue_line("markers", "skip_for_multinode: mark test as multinode, it requires multiple Vespa nodes to run")


def pytest_collection_modifyitems(config, items):
    skip_largemodel = pytest.mark.skip(reason="need --largemodel option to run")
    skip_cpu_only = pytest.mark.skip(reason="skip in --largemodel mode when cpu_only is present")
    skip_multinode = pytest.mark.skip(reason="Skipped because --multinode was used")

    if config.getoption("--largemodel"):
        # --largemodel given in cli: only run tests that have largemodel marker
        for item in items:
            if "largemodel" not in item.keywords:
                item.add_marker(skip_cpu_only)
    else:
        for item in items:
            if "largemodel" in item.keywords:
                item.add_marker(skip_largemodel)

    if config.getoption("--multinode"):
        for item in items:
            if "skip_for_multinode" in item.keywords:
                item.add_marker(skip_multinode)