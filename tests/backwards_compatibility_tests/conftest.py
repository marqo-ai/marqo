import pytest
import semver
from tests.backwards_compatibility_tests.compatibility_test_logger import get_logger

logger = get_logger(__name__)

def pytest_addoption(parser):
    parser.addoption("--version_to_compare_against", action="store", default="2.7", help="version to start from")

@pytest.fixture
def version_to_compare_against(request):
    return request.config.getoption("--version_to_compare_against")


def pytest_collection_modifyitems(config, items):
    version = config.getoption("--version_to_compare_against") # This version will help us determine which test to skip v/s which test to collect.
    # The actual value inside the version can be from_version value (in case of test run where we run prepare on a from_version marqo instance,
    # and tests on a to_version marqo instance) or a to_version value (in case of a full test run where we run prepare and test on the same Marqo instance)

    for item in items:
        test_case_version_marker = item.get_closest_marker("marqo_version")

        if test_case_version_marker:
            test_case_version = test_case_version_marker.args[0] #test_case_version is the version defined as the argument in the "marqo_version" marker above each compatibility test
            logger.debug(f"Checking test: {item.name} with version: {test_case_version}")
            # Compare the test's required version with the version
            logger.debug(f"Test version: {test_case_version}, v/s version supplied in pytest arguments: {version}")
            test_case_version = semver.VersionInfo.parse(test_case_version)
            if test_case_version > version:
                item.add_marker(pytest.mark.skip(reason=f"Test requires marqo_version {test_case_version} which is not greater than version supplied {version}. Skipping."))