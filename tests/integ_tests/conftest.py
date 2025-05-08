CLASSES_SPLIT = "classes"
TESTS_SPLIT = "tests"


def pytest_addoption(parser):
    parser.addoption("--largemodel", action="store_true", default=False)
    parser.addoption("--multinode", action="store_true", default=False, help="Run tests that have multiple Vespa nodes")
    parser.addoption("--split-n", action="store", type=int, default=1, help="Split tests into N parts")
    parser.addoption("--split-part", action="store", type=int, default=0, help="Run tests for part N (zero-based index)")
    # add option to split either by classes or tests
    parser.addoption(
        "--split-by", action="store", type=str, default=TESTS_SPLIT, help=f"Split tests by {CLASSES_SPLIT} or {TESTS_SPLIT}"
    )
    parser.addoption(
        "--suppress-skips", action="store_true", default=False,
        help="Suppress run of skipped tests. Useful for split runs: only split tests that are not skipped"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "largemodel: mark test as largemodels")
    config.addinivalue_line("markers", "cpu_only: mark test as cpu_only")
    config.addinivalue_line("markers", "unittest: mark test as unit test, it does not require vespa to run")
    config.addinivalue_line("markers", "skip_for_multinode: mark test as multinode, it requires multiple Vespa nodes to run")


def pytest_collection_modifyitems(config, items):
    parts = config.getoption("--split-n")
    part = config.getoption("--split-part")
    split_by = config.getoption("--split-by")

    filtered_items = []

    # Step 1: **Pre-filter tests that would be skipped**
    for item in items:
        # Filter out skipped tests if --suppress-skips is set
        if config.getoption("--suppress-skips") and "skip" in item.keywords:
            continue
        # Skip tests that are cpu_only if --largemodel is set
        if config.getoption("--largemodel") and (
                "largemodel" not in item.keywords or "cpu_only" in item.keywords
        ):
            continue # Skip adding this test to filtered_items
        # Skip tests that are largemodel if --largemodel is not set
        if not config.getoption("--largemodel") and "largemodel" in item.keywords:
            continue # Skip adding this test to filtered_items

        if config.getoption("--multinode") and "skip_for_multinode" in item.keywords:
            continue  # Skip adding this test to filtered_items

        filtered_items.append(item)

    if split_by == CLASSES_SPLIT:
        classes_to_tests = {}
        for item in filtered_items:
            class_name = item.parent.name
            if class_name not in classes_to_tests:
                classes_to_tests[class_name] = []
            classes_to_tests[class_name].append(item)
        sorted_classes = sorted(classes_to_tests.keys())

        # Step 2: **Determine size of a partition**
        chunk_size = max(1, len(sorted_classes) // parts)
        last_idx = len(sorted_classes)

    else:
        # Step 2: **Determine size of a partition**
        chunk_size = max(1, len(filtered_items) // parts)
        last_idx = len(filtered_items)

    print(f"Total tests: {len(filtered_items)}, Splitting into {parts} parts, each with {chunk_size} {split_by}")

    # Step 3: **Determine the range of tests/classes to keep in this partition**
    start_idx = part * chunk_size
    end_idx = start_idx + chunk_size

    if part + 1 == parts:
        # Include all remaining classes/tests in the last partition
        end_idx = last_idx
    print(f"Running tests from index {start_idx} to {end_idx}")


    # Step 4: **Modify the list of collected tests**
    if split_by == CLASSES_SPLIT:
        items[:] = [item for item in filtered_items if item.parent.name in set(sorted_classes[start_idx:end_idx])]
    else:
        items[:] = filtered_items[start_idx:end_idx]