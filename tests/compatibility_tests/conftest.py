import pytest
import semver
from tests.compatibility_tests.compatibility_test_logger import get_logger

logger = get_logger(__name__)

def pytest_addoption(parser):
    parser.addoption("--version_to_compare_against", action="store", default="2.7", help="version to start from")

@pytest.fixture
def version_to_compare_against(request):
    return request.config.getoption("--version_to_compare_against")


def pytest_collection_modifyitems(config, items):
    version_to_test_against = semver.VersionInfo.parse(config.getoption("--version_to_compare_against")) # version_to_test_against will help us determine which test to skip v/s which test to collect.
    # The actual value inside the version_to_test_against can be from_version value (in case of test run where we run prepare on a from_version marqo instance,
    # and tests on a to_version marqo instance) or a to_version value (in case of a full test run where we run prepare and test on the same Marqo instance)
    # version_to_test_against = semver.VersionInfo.parse(version_to_test_against)
    for item in items:
        test_case_version_marker = item.get_closest_marker("marqo_version")
        skip_marqo_version_mark = item.get_closest_marker("skip_marqo_version")

        if skip_marqo_version_mark:
            skip_marqo_versions = skip_marqo_version_mark.args
            if str(version_to_test_against) in skip_marqo_versions:
                logger.debug(f"Testcase: {item.name} marked with skip_marqo_version: {skip_marqo_versions}. Skipping.")
                item.add_marker(pytest.mark.skip(reason=f"Testcase: {item.name} marked with skip_marqo_version: {skip_marqo_versions}. Skipping."))
                continue
        elif test_case_version_marker:
            test_case_version = test_case_version_marker.args[0] #test_case_version is the version_to_test_against defined as the argument in the "marqo_version" marker above each compatibility test
            # Compare the test's required version_to_test_against with the version_to_test_against
            logger.debug(f"Testcase: {item.name}, with marqo_version: {test_case_version}, v/s version_to_test_against supplied in pytest arguments: {version_to_test_against}")
            test_case_version = semver.VersionInfo.parse(test_case_version)
            if test_case_version.compare(version_to_test_against) > 0:
                logger.debug(f"marqo_version ({test_case_version}) should be <= supplied version_to_test_against: ({version_to_test_against}). Skipping.")
                item.add_marker(pytest.mark.skip(reason=f"marqo_version ({test_case_version}) should be <= supplied version_to_test_against: ({version_to_test_against}). Skipping."))
        else:
            logger.debug(f"Test class: {item.name} not marked with marqo_version. Skipping.")
            item.add_marker(pytest.mark.skip(reason=f"Testcase: {item.name} not marked with marqo_version. Skipping."))
