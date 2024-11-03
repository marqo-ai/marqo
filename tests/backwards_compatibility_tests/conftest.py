import pytest
import semver

def pytest_addoption(parser):
    parser.addoption("--from_version", action="store", default="2.7", help="version to start from")
    parser.addoption("--to_version", action="store", default="2.8", help="version to migrate to")
    print("Added pytest options for from_version and to_version")

@pytest.fixture
def from_version(request):
    return request.config.getoption("--from_version")

@pytest.fixture
def to_version(request):
    return request.config.getoption("--to_version")

def pytest_collection_modifyitems(config, items):
    from_version = config.getoption("--from_version")

    for item in items:
        version_marker = item.get_closest_marker("marqo_version")

        if version_marker:
            test_version = version_marker.args[0]
            print(f"Checking test: {item.name} with version: {test_version}")
            # Compare the test's required version with the from_version
            print(f"Test version: {test_version}, from_version: {from_version}")
            test_version = semver.VersionInfo.parse(test_version)
            from_version = semver.VersionInfo.parse(from_version)
            # TODO: review this logic to see if it is correct
            if test_version > from_version:
                item.add_marker(pytest.mark.skip(reason=f"Test requires marqo_version {test_version} which is not greater than from_version {from_version}. Skipping."))

    print(f"Total collected tests: {len(items)}")
