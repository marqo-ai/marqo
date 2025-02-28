import pytest


def pytest_addoption(parser):
    parser.addoption("--largemodel", action="store_true", default=False)
    parser.addoption("--multinode", action="store_true", default=False, help="Run tests that have multiple Vespa nodes")
    parser.addoption("--split-n", action="store", type=int, default=1, help="Split tests into N parts")
    parser.addoption("--split-part", action="store", type=int, default=0, help="Run tests for part N (zero-based index)")


def pytest_configure(config):
    config.addinivalue_line("markers", "largemodel: mark test as largemodels")
    config.addinivalue_line("markers", "cpu_only: mark test as cpu_only")
    config.addinivalue_line("markers", "unittest: mark test as unit test, it does not require vespa to run")
    config.addinivalue_line("markers", "skip_for_multinode: mark test as multinode, it requires multiple Vespa nodes to run")


def pytest_collection_modifyitems(config, items):
    parts = config.getoption("--split-n")
    part = config.getoption("--split-part")

    filtered_items = []

    # Step 1: **Pre-filter tests that would be skipped**
    for item in items:
        # Skip tests that are cpu_only if --largemodel is set
        if config.getoption("--largemodel") and (
                "largemodel" not in item.keywords or "cpu_only" in item.keywords or "skip" in item.keywords
        ):
            continue # Skip adding this test to filtered_items
        # Skip tests that are largemodel if --largemodel is not set
        if not config.getoption("--largemodel") and "largemodel" in item.keywords:
            continue # Skip adding this test to filtered_items

        if config.getoption("--multinode") and (
                "skip_for_multinode" in item.keywords or "skip" in item.keywords
        ):
            continue  # Skip adding this test to filtered_items

        filtered_items.append(item)

    # Step 2: **Determine size of a partition**
    chunk_size = max(1, len(filtered_items) // parts)
    print(f"Total tests: {len(filtered_items)}, Splitting into {parts} parts, each with {chunk_size} tests")

    # Step 3: **Determine the range of classes to keep in this partition**
    start_idx = part * chunk_size
    end_idx = start_idx + chunk_size

    if part + 1 == parts:
        end_idx = len(filtered_items)  # Include all remaining classes in the last partition
    print(f"Running tests from index {start_idx} to {end_idx}")


    # Step 4: **Modify the list of collected tests**
    items[:] = filtered_items[start_idx:end_idx]