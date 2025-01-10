import argparse
import importlib
import pkgutil
import time

import pytest
from typing import Set
import subprocess
import sys
import requests
import semver

from compatibility_test_logger import get_logger
from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase
from enum import Enum
from tests.compatibility_tests.docker_manager import DockerManager

# Marqo changed how it transfers state post version 2.9.0, this variable stores that context
marqo_transfer_state_version = semver.VersionInfo.parse("2.9.0")


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
    package = importlib.import_module(package_name)
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, f"{package_name}."):
        logger.debug(f"Processing this {name}, {is_pkg}")
        if is_pkg:
            continue
        try:
            importlib.import_module(name)
            logger.debug(f"Imported module with name {name}")
        except ImportError as e:
            logger.error(f"Could not import module with {name}")

def run_tests_in_mode(mode: Mode, from_version: str):
    """
    This method will be used to either run tests in prepare mode (i.e run prepare method of the resp. test case)
    or run tests in test mode (i.e run test methods of the resp. test case).
    """
    logger.info(f"Running tests in '{mode}' mode with from_version: {from_version}")

    if mode == Mode.PREPARE:
        run_prepare_mode(from_version)
    elif mode == Mode.TEST:
        run_test_mode(from_version)

def run_full_test_suite(from_version: str, to_version: str):
    logger.info(f"Running full test suite with from_version: {from_version}, to_version: {to_version}")
    run_prepare_mode(from_version)
    run_test_mode(from_version)
    logger.info(
        "Finished tests across versions as part of Rollback test's full tests suite. THIS MARKS THE END OF ROLLBACK TESTS ACROSS TWO CONTAINERS WITH DIFFERENT VERSIONS")
    full_test_run(to_version)

def full_test_run(marqo_version: str):
    """
    This method will run tests on a single Marqo version container, which means it will run both prepare and tests on the
    to_version Marqo container. Note that to_version Marqo container has been created by transferring instance from a
    previous from_version Marqo container.
    """
    logger.info(f"Running full_test_run with version: {marqo_version}. THIS WILL RUN PREPARE METHOD AND TEST METHODS ON THE SAME {marqo_version} VERSION CONTAINER")
    #Step 1: Run tests in prepare mode
    run_prepare_mode(marqo_version)
    #Step 2: Run tests in test mode
    run_test_mode(marqo_version)

def run_prepare_mode(version_to_test_against: str):

    version_to_test_against = semver.VersionInfo.parse(version_to_test_against)
    load_all_subclasses("tests.compatibility_tests")
    logger.debug(f"Printing all test cases defined under tests/compatibility_tests/: {BaseCompatibilityTestCase.__subclasses__()}")
    for test_class in BaseCompatibilityTestCase.__subclasses__():
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
        # To check for cases if a test case is not marked with marqo_version OR if it is marked with skip. In that case we skip running prepare mode on that test case.
        if not marqo_version_marker or skip_marker:
            if not marqo_version_marker:
                logger.info(f"No marqo_version marker detected for class {test_class.__name__}, skipping prepare mode for this test class")
            elif skip_marker:
                logger.info(f"Detected 'skip' marker for class {test_class.__name__}, skipping prepare mode for this test class")
            continue

        marqo_version = marqo_version_marker.args[0]
        logger.info(f"Detected marqo_version '{marqo_version}' for testcase: {test_class.__name__}")
        try:
            if semver.VersionInfo.parse(marqo_version).compare(version_to_test_against) <= 0:
                logger.info(f"Running prepare mode on testcase: {test_class.__name__} with version: {marqo_version}")
                test_class.setUpClass() #setUpClass will be used to create Marqo client
                test_instance = test_class()
                test_instance.prepare() #Prepare method will be used to create index and add documents
            else: # Skip the test if the version_to_test_against is greater than the version the test is marked
                logger.info(f"Skipping testcase {test_class.__name__} with version {marqo_version} as it is greater than {version_to_test_against}")
        except Exception as e:
            logger.error(f"Failed to run prepare mode on testcase: {test_class.__name__} with version: {marqo_version}, when test mode runs on this test case, it is expected to fail. The exception was {e}", exc_info=True)
        logger.info(f"##################################################################################################")

def construct_pytest_arguments(version_to_test_against):
    pytest_args = [
        f"--version_to_compare_against={version_to_test_against}",
        "-m", f"marqo_version",
        "-s",
        "tests/compatibility_tests"
    ]
    return pytest_args

def run_test_mode(version_to_test_against):
    pytest_args = construct_pytest_arguments(version_to_test_against)
    pytest.main(pytest_args)

def trigger_rollback_endpoint(from_version: str):
    logger.info(f"Triggering rollback endpoint with from_version: {from_version}")
    import requests

    response = requests.post('http://localhost:8882/rollback-vespa')
    if response.status_code == 200:
        logger.info("Rollback endpoint triggered successfully")

def backwards_compatibility_test(from_version: str, to_version: str, to_version_image: str):
    """
    Perform a backwards compatibility test between two versions of Marqo.

    This function starts a container with the from_version, runs tests in prepare mode, stops the container,
    starts a container with the to_version by transferring state from from_version container, and runs tests in test mode.

    Args:
        from_version (str): The source version of the Marqo container.
        to_version (str): The target version of the Marqo container.
        to_version_image (str): The unique identifier for a to_version image. It can be either be the fully qualified image name with the tag
                                (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests:abcdefgh1234)
                                or the fully qualified image name with the digest (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests@sha256:1234567890abcdef).
                                This is constructed in build_push_image.yml workflow and will be the qualified image name with digest for an automatically triggered workflow.

    Raises:
        ValueError: If the major versions of from_version and to_version are incompatible.
        Exception: If there is an error during the test process.
    """
    try:
        # Step 1: Start from_version container and run tests in prepare mode
        logger.info(f"Starting backwards compatibility tests with from_version: {from_version}, to_version: {to_version}, to_version_image: {to_version_image}")

        # Generate a volume name to be used with the "from_version" Marqo container for state transfer.
        from_version_volume = docker_manager.get_volume_name_from_marqo_version(from_version)

        #Start from_version container
        docker_manager.start_marqo_container(from_version, from_version_volume)
        logger.info(f"Started Marqo container {from_version}")

        try:
            run_tests_in_mode(Mode.PREPARE, from_version)
        except Exception as e:
            raise RuntimeError(f"Error running tests in 'prepare' mode across versions on from_version: {from_version}") from e
        # Step 2: Stop from_version container (but don't remove it)
        docker_manager.stop_marqo_container(from_version)

        # Step 3: Start to_version container by transferring state
        logger.debug(f"Starting Marqo to_version: {to_version} container by transferring state from version {from_version} to {to_version}")
        docker_manager.start_marqo_container_by_transferring_state(to_version, from_version, from_version_volume,
                                                    to_version_image, "ECR")

        logger.info(f"Started Marqo to_version: {to_version} container by transferring state")
        # Step 4: Run tests
        try:
            run_tests_in_mode(Mode.TEST, from_version)
        except Exception as e:
            raise RuntimeError(f"Error running tests across versions in 'test' mode on from_version: {from_version}") from e
        logger.info("Finished running tests in Test mode. THIS MARKS THE END OF BACKWARDS COMPATIBILITY TESTS ACROSS TWO CONTAINERS WITH DIFFERENT VERSIONS")
        # Step 5: Do a full test run which includes running tests in prepare and test mode on the same container
        try:
            full_test_run(to_version)
        except Exception as e:
            raise RuntimeError(f"Error running tests in full test run, on to_version: {to_version}.") from e
    except Exception as e:
        raise RuntimeError(f"An error occurred while executing backwards compatibility tests, on from_version: {from_version}, to_version: {to_version}, to_version_image: {to_version_image}") from e
    finally:
        # Stop the to_version container (but don't remove it yet)
        logger.info(f"Stopping Marqo to_version ({to_version}) container " + str(to_version))
        docker_manager.stop_marqo_container(to_version)
        # Clean up all containers at the end
        docker_manager.cleanup_containers()
        docker_manager.cleanup_volumes()

def rollback_test(to_version: str, from_version: str, to_version_image: str):
    """
    Perform a rollback test between two versions of Marqo.
    This function first runs test cases in prepare mode on from_version Marqo container, then upgrades it to to_version Marqo container,
    It then downgrades (rollback) to from_version container again where it runs test cases in test mode. Finally, it triggers rollback endpoint
    to rollback vespa application (this only happens if the Marqo version running is >=2.13.0) and runs the complete test suite again.

    Args:
        to_version (str): The target version of the Marqo container.
        from_version (str): The source version of the Marqo container.
        to_version_image (str): The unique identifier for a to_version image. It can be either be the fully qualified image name with the tag
    """
    logger.info(f"Starting Marqo rollback tests with from_version: {from_version}, to_version: {to_version}, to_version_image: {to_version_image}")
    try:
        # Step 0: Generate a volume name to be used with the "from_version" Marqo container for state transfer.
        from_version_volume = docker_manager.get_volume_name_from_marqo_version(from_version)

        # Step 1: Start a Marqo container using from_version and run tests in prepare mode.
        docker_manager.start_marqo_container(from_version, from_version_volume)
        logger.info(f"Started Marqo container {from_version}")

        # Run tests in prepare mode
        try:
            run_tests_in_mode(Mode.PREPARE, from_version)
        except Exception as e:
            raise RuntimeError(f"Error while running tests across versions in 'prepare' mode.") from e

        # Step 2: Stop Marqo from_version container started in Step #1.
        docker_manager.stop_marqo_container(from_version)

        # Step 3: Start to_version container by transferring state
        logger.info(f"Starting Marqo to_version: {to_version} container by transferring state from version: {from_version} to version: {to_version}")
        docker_manager.start_marqo_container_by_transferring_state(to_version, from_version, from_version_volume,
                                                    to_version_image, "ECR")
        logger.info(f"Started Marqo to_version: {to_version} container by transferring state")

        #Step 4: Stop Marqo container from Step #3
        docker_manager.stop_marqo_container(to_version)

        #Step 5: Again start a Marqo container using from_version (i.e Rollback to from_version), transferring state from container in Step 4.
        logger.info(f"Starting Marqo from_version: {from_version} container again, by transferring state from to version: {to_version} to version: {from_version}")
        # TODO: Check from_version_volume for the case where the two versions are before and after 2.9 since we create a new volume in that case.
        prepare_volume_for_rollback(target_version=from_version, source_volume=from_version_volume, source="docker")
        docker_manager.start_marqo_container_by_transferring_state(target_version=from_version, source_version=to_version,
                                                    source_volume=from_version_volume, source="docker")

        #Step 6: Run tests in test mode, then run full test run
        try:
            run_tests_in_mode(Mode.TEST, from_version) # This will validate results from the older indexes added as part of the PREPARE mode above.
        except Exception as e:
            raise RuntimeError(f"Error in rollback tests while running tests across versions in 'test' mode on version: {from_version}") from e
        logger.info("Finished running tests in Test mode. THIS MARKS THE END OF ROLLBACK TESTS ACROSS TWO CONTAINERS WITH DIFFERENT VERSIONS")
        try:
            full_test_run(from_version) # This will validate results by creating newer indexes and adding documents to them. This is required just so that we know that even after transferring state from an older version, we are able to create new indexes in the older state seamlessly.
        except Exception as e:
            raise RuntimeError(f"Error in rollback tests while running tests in full test run on version: {to_version}") from e

        #Step 7: Trigger rollback endpoint, and run full test suite again. We only need to run full test suite in case we are able to trigger rollback endpoint (i.e in case the Marqo version is >=2.13.0)
        run_full_test_suite_after_triggering_rollback_endpoint(from_version, to_version)

    finally:
        # Stop the final container (but don't remove it yet)
        logger.debug("Stopping marqo container")
        docker_manager.stop_marqo_container(from_version)
        # Clean up all containers and volumes at the end
        logger.debug("Cleaning up containers and volumes")
        docker_manager.cleanup_containers()
        docker_manager.cleanup_volumes()

def run_full_test_suite_after_triggering_rollback_endpoint(from_version: str, to_version: str):
    if semver.VersionInfo.parse(from_version) < semver.VersionInfo.parse("2.13.0"):
        return
    trigger_rollback_endpoint(from_version)

    try:
        run_full_test_suite(from_version, to_version)
    except Exception as e:
        raise RuntimeError(
            f"Error when running full test suite in rollback tests after rolling back vespa application, with from_version: {from_version}, to_version: {to_version}") from e


def prepare_volume_for_rollback(target_version: str, source_volume: str, target_version_image_name: str = None,
                                source="docker"):
    """
    This method is used to run a command that adjusts the permissions of files or directories inside a Docker volume,
    making them accessible to a specific user (vespa) and group (vespa) that the container expects to interact with.
    """
    logger.info(f"Preparing volume for rollback with target_version: {target_version}, source_volume: {source_volume}, target_version_image_name: {target_version_image_name}, source: {source}")
    if source == "docker": # In case the source is docker, we will directly pull the image using version (ex: marqoai/marqo:2.13.0)
        image_name = f"marqoai/marqo:{target_version}"
    else:
        image_name = target_version_image_name

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{source_volume}:/opt/vespa/var",
        "--entrypoint", "/bin/sh",  # Override entrypoint with a shell
        image_name,
        "-c", "chown -R vespa:vespa /opt/vespa/var"
    ]

    logger.info(f"Running this command: {' '.join(cmd)} to prepare volume for rollback using from_version: {target_version}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to run command: {' '.join(cmd)} when preparing volume for rollback: {e}") from e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marqo Testing Runner")
    parser.add_argument("--mode", choices=["backwards_compatibility", "rollback"], required=True)
    parser.add_argument("--from_version", required=True)
    parser.add_argument("--to_version", required=True)
    parser.add_argument("--to_image", required=True)
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
            backwards_compatibility_test(args.from_version, args.to_version, args.to_image)
        elif args.mode == "rollback":
            rollback_test(args.to_version, args.from_version, args.to_image)

    except Exception as e:
        logger.exception(f"Encountered an exception: {e} while running tests in mode {args.mode}, exiting", exc_info=True)
        sys.exit(1)