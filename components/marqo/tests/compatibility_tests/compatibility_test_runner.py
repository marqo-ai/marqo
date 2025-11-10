import argparse
import importlib
import pkgutil
import pytest
import requests
import semver
import sys
from enum import Enum
from typing import Set

from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase
from tests.compatibility_tests.compatibility_test_logger import get_logger
from tests.compatibility_tests.docker_manager import DockerManager

# Marqo changed how it transfers state post version 2.9.0, this variable stores that context
marqo_transfer_state_version = semver.VersionInfo.parse("2.9.0")

# Global set to track imported modules
_imported_modules = set()


class Mode(Enum):
    PREPARE = "prepare"
    TEST = "test"

# Keep track of containers that need cleanup
containers_to_cleanup: Set[str] = set()
volumes_to_cleanup: Set[str] = set()

logger = get_logger(__name__)

docker_manager = DockerManager()

def load_all_subclasses(package_name):
    """
    Dynamically load all subclasses within a specified package,
    including those in its subdirectories.

    Args:
        package_name (str): The top-level package name to search for subclasses.
    """
    global _imported_modules
    package = importlib.import_module(package_name)
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, f"{package_name}."):
        logger.debug(f"Processing subclass: {name}, is package? -> {is_pkg}")
        if is_pkg:
            continue
        if name in _imported_modules:
            logger.debug(f"Skipping already imported module: {name}")
            continue
        try:
            importlib.import_module(name)
            _imported_modules.add(name)
            logger.debug(f"Imported module with name {name}")
        except ImportError as e:
            logger.error(f"Could not import module with {name}. Original error: {e}", exc_info=True)

def run_prepare_mode(version_to_test_against: str):
    logger.info(f"===================================== RUN PREPARE MODE BEGINS =================================================")
    version_to_test_against = semver.VersionInfo.parse(version_to_test_against)
    logger.debug(f"Printing all test cases defined under tests/compatibility_tests/: {BaseCompatibilityTestCase.__subclasses__()}")
    errors = []

    collected_classes = []
    # Skip any tests that have already been prepared
    seen_classes = set()
    for test_class in BaseCompatibilityTestCase.__subclasses__():
        if test_class.__name__ in seen_classes:
            logger.info(f"Skipping duplicate test class {test_class.__name__} as it has already been processed")
            continue

        # Log to confirm no duplicates
        logger.info(f"{test_class.__name__} has NOT been processed yet. Processed classes: {seen_classes}. Processing now.")
        seen_classes.add(test_class.__name__)
        
        logger.info(f"========================================================================================")
        markers = getattr(test_class, "pytestmark", [])
        # Check for specific markers
        marqo_version_marker = next( # Checks what version a compatibility test is marked with (ex: @pytest.mark.marqo_version('2.11.0')). If no version is marked, it will skip the test
            (marker for marker in markers if marker.name == "marqo_version"),
            None
        )
        skip_marker = next( # Checks if a compatibility test is marked with @pytest.mark.skip
            (marker for marker in markers if marker.name == "skip"),
            None
        )
        skip_marqo_version_marker = next( # Checks if a compatibility test is marked with @pytest.mark.skip
            (marker for marker in markers if marker.name == "skip_marqo_version"),
            None
        )
        # To check for cases if a test case is not marked with marqo_version OR if it is marked with skip. In that case we skip running prepare mode on that test case.
        if not marqo_version_marker:
            logger.info(f"No marqo_version marker detected for class {test_class.__name__}, skipping prepare mode for this test class")
            continue

        if skip_marker:
            logger.info(f"Detected 'skip' marker for class {test_class.__name__}, skipping prepare mode for this test class")
            continue

        if skip_marqo_version_marker and (str(version_to_test_against) in skip_marqo_version_marker.args):
            logger.info(
                f"Detected 'skip_marqo_version' marker for class {test_class.__name__}. "
                f"These Marqo versions are skipped: {skip_marqo_version_marker.args}. "
                f"Skipping prepare mode for this test class as we are running on version {version_to_test_against}"
            )
            continue

        # TODO: Raname this to minimal version
        marqo_version = marqo_version_marker.args[0]
        logger.info(f"Detected marqo_version '{marqo_version}' for testcase: {test_class.__name__}")
        try:
            if semver.VersionInfo.parse(marqo_version).compare(version_to_test_against) <= 0:
                logger.info(f"Running prepare mode on testcase: {test_class.__name__}")
                test_class.setUpClass() #setUpClass will be used to create Marqo client
                test_instance = test_class()
                test_instance.prepare() #Prepare method will be used to create index and add documents
                collected_classes.append(test_instance)
            else: # Skip the test if the version_to_test_against is greater than the version the test is marked
                logger.info(f"Skipping testcase {test_class.__name__} as {marqo_version} > {version_to_test_against}")
        except Exception as e:
            logger.error(f"Failed to run prepare mode on testcase: {test_class.__name__}, when test mode runs on this test case, it is expected to fail. The exception was {e}", exc_info=True)
            errors.append(f"Failed to run prepare mode on testcase: {test_class.__name__}, when test mode runs on this test case, it is expected to fail. Search the class name in the logs to find the exact error.")
        logger.info(f"##################################################################################################")

    if errors:
        raise RuntimeError(f"Some errors occurred while running prepare mode on test cases: {errors}")

def construct_pytest_arguments(version_to_test_against) -> list[str]:
    pytest_args = [
        f"--version_to_compare_against={version_to_test_against}",
        "-m", f"marqo_version",
        "-s",
        "tests/compatibility_tests"
    ]
    return pytest_args

def run_test_mode(version_to_test_against):
    logger.info(f"Beginning test mode on all test cases for version: {version_to_test_against}")
    pytest_args = construct_pytest_arguments(version_to_test_against)
    pytest_result = pytest.main(pytest_args)

    if pytest_result == 0:
        logger.info(f"Successfully ran test mode on all test cases")
    elif pytest_result == 1:
        raise RuntimeError(f"Failed to run test mode on some test cases. Check pyTest output for exactly which test cases failed")

def trigger_rollback_endpoint():
    logger.info(f"Triggering rollback endpoint.")

    response = requests.post('http://localhost:8882/rollback-vespa')
    if response.status_code == 200:
        logger.info("Rollback endpoint triggered successfully")

def backwards_compatibility_test(
        from_version: str, to_version: str, to_api_image: str, to_inference_orchestrator_image: str,
        to_model_management_image: str
    ):
    """
    Perform a backwards compatibility test between two versions of Marqo.

    This function starts a container with the from_version, runs tests in prepare mode, stops the container,
    starts a container with the to_version by transferring state from from_version container, and runs tests in test mode.

    Since 2.25.0, Marqo uses separate images for API, Inference Orchestrator, and Model Management.
    Therefore, this function accepts separate image identifiers for each component to ensure compatibility during the upgrade process

    Args:
        from_version (str): The source version of the Marqo container.
        to_version (str): The target version of the Marqo container.
        to_api_image (str): The target API version of the Marqo container.
        to_inference_orchestrator_image (str): The target Inference Orchestrator version of the Marqo container.
        to_model_management_image (str): The target Model Management version of the Marqo container.

    Raises:
        ValueError: If the major versions of from_version and to_version are incompatible.
        Exception: If there is an error during the test process.
    """
    try:
        load_all_subclasses("tests.compatibility_tests")
        # Step 1: Start from_version container and run tests in prepare mode
        logger.info(f"Starting backwards compatibility tests with from_version: {from_version}, to_version: {to_version}")

        #Start from_version container
        docker_manager.start_marqo_container(from_version)
        logger.info(f"Started Marqo container {from_version}")

        try:
            run_prepare_mode(from_version)
        except Exception as e:
            raise RuntimeError(f"Error running tests in 'prepare' mode across versions on from_version: {from_version}") from e
        # Step 2: Stop from_version container (but don't remove it)
        docker_manager.stop_marqo_container(from_version)

        # Step 3: Start to_version container by transferring state
        logger.debug(f"Starting Marqo to_version: {to_version} container by transferring state from version {from_version} to {to_version}")

        docker_manager.start_marqo_container(
            to_version, to_api_image, to_inference_orchestrator_image, to_model_management_image
        )

        logger.info(f"Started Marqo to_version: {to_version} container by transferring state")
        # Step 4: Run tests
        try:
            run_test_mode(from_version)
        except Exception as e:
            raise RuntimeError(f"Error running tests across versions in 'test' mode on from_version: {from_version}") from e
        logger.info("Finished running tests in Test mode. THIS MARKS THE END OF BACKWARDS COMPATIBILITY TESTS ACROSS TWO CONTAINERS WITH DIFFERENT VERSIONS")
        # Step 5: Do a full test run which includes running tests in prepare and test mode on the same container
        try:
            run_prepare_mode(to_version)
            run_test_mode(to_version)
        except Exception as e:
            raise RuntimeError(f"Error running tests in full test run, on to_version: {to_version}.") from e
    except Exception as e:
        raise RuntimeError(f"An error occurred while executing backwards compatibility tests, on from_version: {from_version}, to_version: {to_version}") from e
    finally:
        # Stop the to_version container (but don't remove it yet)
        logger.info(f"Stopping Marqo to_version ({to_version}) container " + str(to_version))
        docker_manager.stop_marqo_container(to_version)
        # Clean up all containers at the end
        docker_manager.cleanup_containers()
        docker_manager.cleanup_volumes()

def rollback_test(
        from_version: str, to_version: str, to_api_image: str, to_inference_orchestrator_image: str,
        to_model_management_image: str
    ):
    """
    Perform a rollback test between two versions of Marqo.
    This function first runs test cases in prepare mode on from_version Marqo container, then upgrades it to to_version Marqo container,
    It then downgrades (rollback) to from_version container again where it runs test cases in test mode. Finally, it triggers rollback endpoint
    to rollback vespa application (this only happens if the Marqo version running is >=2.13.0) and runs the complete test suite again.

    Args:
        to_version (str): The target version of the Marqo container.
        from_version (str): The source version of the Marqo container.
        to_api_image (str): The target API version of the Marqo container.
        to_inference_orchestrator_image (str): The target Inference Orchestrator version of the Marqo container.
        to_model_management_image (str): The target Model Management version of
        the Marqo container
    """
    logger.info(f"Starting Marqo rollback tests with from_version: {from_version}, to_version: {to_version}")
    try:
        load_all_subclasses("tests.compatibility_tests")

        # Step 1: Start a Marqo container using from_version
        docker_manager.start_marqo_container(from_version)
        logger.info(f"Step 1: Started Marqo container {from_version}")

        # Step 2: Run prepare mode
        logger.info("Step 2: Running prepare mode on initial from_version container")
        run_prepare_mode(from_version)

        # Step 3: Stop Marqo from_version container started in Step #1.
        docker_manager.stop_marqo_container(from_version)
        logger.info("Step 3: Stopped Marqo container from Step #1")

        # Step 4: Upgrade to to_version container by transferring state
        logger.info(f"Step 4: Starting Marqo to_version: {to_version} container by transferring state from version: "
                    f"{from_version} to version: {to_version}")
        docker_manager.start_marqo_container(
            to_version, to_api_image, to_inference_orchestrator_image, to_model_management_image
        )

        #Step 5: Stop Marqo container from Step #4
        logger.info("Step 5: Stopping Marqo container from Step #4")
        docker_manager.stop_marqo_container(to_version)

        #Step 6: Again start a Marqo container using from_version (i.e Rollback marqo version),
        # transferring state from container in Step 4.
        logger.info(f"Step 6: Going back to marqo from_version."
                    f"Starting Marqo from_version: {from_version} container again, "
                    f"by transferring state from to_version, which was {to_version}")
        # TODO: Check from_version_volume for the case where the two versions are before and after 2.9 since we create a new volume in that case.
        docker_manager.start_marqo_container(from_version)

        # Step 7: Run test mode
        logger.info(f"Step 7: Running tests in test mode on from_version: {from_version}")
        run_test_mode(from_version) # This will validate results from the older indexes added as part of the PREPARE mode above.

        # Step 8: Run prepare and test mode again, on the from_version container.
        logger.info(f"Step 8: Running prepare and test mode on the same from_version: {from_version} container")
        run_prepare_mode(from_version)
        run_test_mode(from_version) # This will validate results by creating newer indexes and adding documents to them. This is required just so that we know that even after transferring state from an older version, we are able to create new indexes in the older state seamlessly.

        # Only execute the following if Marqo version >= 2.13.0. This is because the rollback endpoint is only
        # available in these versions.

        # Step 9: Trigger rollback Vespa endpoint
        if semver.VersionInfo.parse(from_version) >= semver.VersionInfo.parse("2.13.0"):
            trigger_rollback_endpoint()

            # Step 10: Run full test suite again after Vespa rollback
            try:
                logger.info(f"Running full test suite with from_version: {from_version}")
                run_prepare_mode(from_version)
                run_test_mode(from_version)
            except Exception as e:
                raise RuntimeError(
                    f"Error when running full test suite in rollback tests after rolling back vespa application, "
                    f"with from_version: {from_version}, to_version: {to_version}") from e

    finally:
        # Stop the final container (but don't remove it yet)
        logger.debug("Stopping marqo container")
        docker_manager.stop_marqo_container(from_version)
        # Clean up all containers and volumes at the end
        logger.debug("Cleaning up containers and volumes")
        docker_manager.cleanup_containers()
        docker_manager.cleanup_volumes()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marqo Testing Runner")
    parser.add_argument("--mode", choices=["backwards_compatibility", "rollback"], required=True)
    parser.add_argument("--from_version", required=True)
    parser.add_argument("--to_version", required=True)
    parser.add_argument("--to_api_image", required=True)
    parser.add_argument("--to_inference_orchestrator_image", required=True)
    parser.add_argument("--to_model_management_image", required=True)
    args = parser.parse_args()
    try:
        from_version = semver.VersionInfo.parse(args.from_version)
        to_version = semver.VersionInfo.parse(args.to_version)

        # Basic validation that verifies: from_version shouldn't be greater than or equal to to_version
        if from_version >= to_version:
            logger.error("from_version should be less than to_version")
            raise ValueError(f"from_version: {from_version} should be less than to_version: {to_version}")

        #If from major version & to major version aren't the same we cannot run backwards compatibility tests or rollback tests
        if from_version.major != to_version.major:
            logger.error(f"from_version {from_version} & to_version {to_version} cannot "
                         f"be used for running backwards compatibility tests or rollback tests"
                         f"since they are from different major versions")
            raise ValueError(f"from_version {from_version} & to_version {to_version} cannot "
                         f"be used for running backwards compatibility tests or rollback tests"
                         f"since they are from different major versions")

    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    try:
        if args.mode == "backwards_compatibility":
            backwards_compatibility_test(
                args.from_version, args.to_version, args.to_api_image,
                args.to_inference_orchestrator_image, args.to_model_management_image
            )
        elif args.mode == "rollback":
            rollback_test(
                args.from_version, args.to_version, args.to_api_image,
                args.to_inference_orchestrator_image, args.to_model_management_image
            )

    except Exception as e:
        logger.exception(f"Encountered an exception: {e} while running tests in mode {args.mode}, exiting", exc_info=True)
        sys.exit(1)