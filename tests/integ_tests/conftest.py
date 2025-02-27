import pytest


def pytest_addoption(parser):
    parser.addoption("--largemodel", action="store_true", default=False)
    parser.addoption("--multinode", action="store_true", default=False, help="Run tests that have multiple Vespa nodes")
    parser.addoption("--split-n", action="store", default=1, help="Split tests into N parts")
    parser.addoption("--split-part", action="store", default=1, help="Run tests for part N")


def pytest_configure(config):
    config.addinivalue_line("markers", "largemodel: mark test as largemodels")
    config.addinivalue_line("markers", "cpu_only: mark test as cpu_only")
    config.addinivalue_line("markers", "unittest: mark test as unit test, it does not require vespa to run")
    config.addinivalue_line("markers", "skip_for_multinode: mark test as multinode, it requires multiple Vespa nodes to run")


def pytest_collection_modifyitems(config, items):
    parts = config.getoption("--split-n")
    part = config.getoption("--split-part")
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

    class_to_tests = {}
    for item in items:
        class_name = item.parent.name
        if class_name not in class_to_tests:
            class_to_tests[class_name] = []
        class_to_tests[class_name].append(item)

    # Sort classes and divide into n parts
    sorted_classes = sorted(class_to_tests.keys())
    chunk_size = max(1, len(sorted_classes) // int(parts))

    # Determine the range of classes to keep
    start_idx = (int(part)) * chunk_size
    end_idx = start_idx + chunk_size

    if int(part) + 1 == int(parts):
        end_idx = len(sorted_classes)

    selected_classes = set(sorted_classes[start_idx:end_idx])

    # Modify the list of collected tests
    items[:] = [item for item in items if item.parent.name in selected_classes]