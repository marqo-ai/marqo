import argparse
import importlib
import pkgutil
import subprocess
import sys
import os
import glob
import inspect
import unittest
from enum import Enum
from typing import Set

import pytest
import requests
import semver

from tests.compatibility_tests.compatibility_test_logger import get_logger
from tests.compatibility_tests.base_test_case.base_compatibility_test import BaseCompatibilityTestCase
from tests.compatibility_tests.docker_manager import DockerManager

# Marqo changed how it transfers state post version 2.9.0, this variable stores that context
marqo_transfer_state_version = semver.VersionInfo.parse("2.9.0")

# Global set to track imported modules
_imported_modules = set()

_CLASSES_TO_PREPARE = None

class Mode(Enum):
    PREPARE = "prepare"
    TEST = "test"

# Keep track of containers that need cleanup
containers_to_cleanup: Set[str] = set()
volumes_to_cleanup: Set[str] = set()

logger = get_logger(__name__)

docker_manager = DockerManager()

def split_and_prefix_path_to_test(path_to_test: str) -> list:
    """
    Split a string of class names separated by spaces and prefix each with 'tests/compatibility_tests/'.

    Args:
        path_to_test (str): A string of class names separated by spaces.

    Returns:
        list: A list of fully qualified class names.
    """
    if not path_to_test:
        # By default, test this whole dir
        return ["tests/compatibility_tests"]
    return [f"tests/compatibility_tests/{cls.strip()}" for cls in path_to_test.split()]

def run_prepare_mode(version_to_test_against: str, test_classes_to_prepare: list):
    logger.info(f"===================================== RUN PREPARE MODE BEGINS =================================================")
    version_to_test_against = semver.VersionInfo.parse(version_to_test_against)
    logger.info(f"Printing all test cases to prepare: {test_classes_to_prepare}")
    errors = []

    # Skip any tests that have already been prepared
    seen_classes = set()
    for test_class in test_classes_to_prepare:
        # Log to confirm no duplicates
        logger.info(f"{test_class.__name__} has NOT been processed yet. Processing now.")
        
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

        # TODO: Rename this to minimal version
        marqo_version = marqo_version_marker.args[0]
        logger.info(f"Detected marqo_version '{marqo_version}' for testcase: {test_class.__name__}")
        try:
            if semver.VersionInfo.parse(marqo_version).compare(version_to_test_against) <= 0:
                logger.info(f"Running prepare mode on testcase: {test_class.__name__}")
                test_class.setUpClass() #setUpClass will be used to create Marqo client
                test_instance = test_class()
                test_instance.prepare() #Prepare method will be used to create index and add documents
            else: # Skip the test if the version_to_test_against is greater than the version the test is marked
                logger.info(f"Skipping testcase {test_class.__name__} as {marqo_version} > {version_to_test_against}")
        except Exception as e:
            logger.error(f"Failed to run prepare mode on testcase: {test_class.__name__}, when test mode runs on this test case, it is expected to fail. The exception was {e}", exc_info=True)
            errors.append(f"Failed to run prepare mode on testcase: {test_class.__name__}, when test mode runs on this test case, it is expected to fail. Search the class name in the logs to find the exact error.")
        logger.info(f"##################################################################################################")

    if errors:
        raise RuntimeError(f"Some errors occurred while running prepare mode on test cases: {errors}")

def construct_pytest_arguments(version_to_test_against, path_to_test):
    pytest_args = [
        f"--version_to_compare_against={version_to_test_against}",
        "-m", f"marqo_version",
        "-s"
    ]

    pytest_args += split_and_prefix_path_to_test(path_to_test)

    return pytest_args

def run_test_mode(version_to_test_against, path_to_test):
    logger.info(f"TEST MODE START all test cases for version: {version_to_test_against}")
    pytest_args = construct_pytest_arguments(version_to_test_against, path_to_test)
    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    subprocess.run(cmd, check=True)

def trigger_rollback_endpoint():
    logger.info(f"Triggering rollback endpoint.")

    response = requests.post('http://localhost:8882/rollback-vespa')
    if response.status_code == 200:
        logger.info("Rollback endpoint triggered successfully")

def determine_test_classes_to_prepare(path_to_test: str = None) -> list:
    """
    Determine a list of test classes to run prepare mode on, given a string-separated path to test files
    or directories. This function will:
    1. Search all python files or classes in the specified path(s)
    2. Import each file as a module
    3. Find all subclasses of `unittest.TestCase` that start with 'Test'
    4. Return a list of these classes
    """
    global _CLASSES_TO_PREPARE
    if _CLASSES_TO_PREPARE is not None:
        return _CLASSES_TO_PREPARE

    dirs_to_check = split_and_prefix_path_to_test(path_to_test)
    test_classes = []
    
    for path_item in dirs_to_check:
        logger.debug(f"Processing path: {path_item}")
        
        # Check if path contains specific test class (format: file.py::TestClassName)
        specific_class = None
        if "::" in path_item:
            path_item, specific_class = path_item.split("::", 1)
            logger.debug(f"Specific class requested: {specific_class}")
        
        # Determine if path is a file or directory
        if os.path.isfile(path_item):
            if path_item.endswith('.py'):
                test_classes.extend(_import_and_find_test_classes(path_item, specific_class))
        elif os.path.isdir(path_item):
            # Search for all Python files in the directory recursively
            python_files = glob.glob(os.path.join(path_item, "**", "*.py"), recursive=True)
            for py_file in python_files:
                test_classes.extend(_import_and_find_test_classes(py_file, specific_class))
        else:
            logger.warning(f"Path does not exist or is not a file/directory: {path_item}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_classes = []
    for cls in test_classes:
        if cls not in seen:
            seen.add(cls)
            unique_classes.append(cls)
    
    _CLASSES_TO_PREPARE = unique_classes
    logger.debug(f"Found {len(unique_classes)} test classes: {[cls.__name__ for cls in unique_classes]}")
    return _CLASSES_TO_PREPARE


def _import_and_find_test_classes(file_path: str, specific_class: str = None) -> list:
    """
    Import a Python file as a module and find test classes in it.
    
    Args:
        file_path: Path to the Python file
        specific_class: If provided, only return this specific class
    
    Returns:
        List of test class objects
    """
    global _imported_modules
    test_classes = []
    
    try:
        # Convert file path to module name
        # Remove .py extension and convert path separators to dots
        module_name = file_path.replace('/', '.').replace('\\', '.')
        if module_name.endswith('.py'):
            module_name = module_name[:-3]
        
        # Skip if already imported
        if module_name in _imported_modules:
            logger.debug(f"Module {module_name} already imported, skipping")
            # Still need to get classes from the already imported module
            module = sys.modules.get(module_name)
        else:
            logger.debug(f"Importing module: {module_name}")
            module = importlib.import_module(module_name)
            _imported_modules.add(module_name)
        
        if module is None:
            logger.warning(f"Could not import or find module: {module_name}")
            return test_classes
        
        # Find all classes in the module that are test classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if it's a test class
            if (name.startswith('Test') and 
                issubclass(obj, BaseCompatibilityTestCase) and 
                obj != BaseCompatibilityTestCase):
                
                # If specific class is requested, only return that one
                if specific_class:
                    if name == specific_class:
                        test_classes.append(obj)
                        logger.debug(f"Found specific test class: {name}")
                        break
                else:
                    test_classes.append(obj)
                    logger.debug(f"Found test class: {name}")
        
        if specific_class and not test_classes:
            logger.warning(f"Specific test class '{specific_class}' not found in {file_path}")
            
    except ImportError as e:
        logger.error(f"Could not import module from {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
    
    return test_classes

def backwards_compatibility_test(from_version: str, to_version: str, to_version_image: str, path_to_test: str):
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
        path_to_test (str): The path to the test file/dir to be executed with pytest.

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
            test_classes_to_prepare = determine_test_classes_to_prepare(path_to_test)
            run_prepare_mode(from_version, test_classes_to_prepare)
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
            run_test_mode(from_version, path_to_test)
        except Exception as e:
            raise RuntimeError(f"Error running tests across versions in 'test' mode on from_version: {from_version}") from e
        logger.info("Finished running tests in Test mode. THIS MARKS THE END OF BACKWARDS COMPATIBILITY TESTS ACROSS "
                    "TWO CONTAINERS WITH DIFFERENT VERSIONS")
        # Step 5: Do a full test run which includes running tests in prepare and test mode on the same container
        try:
            logger.info(
                "Running final prepare then test mode on the same (to_version) container."
            )
            run_prepare_mode(to_version, test_classes_to_prepare)
            run_test_mode(to_version, path_to_test)
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

def rollback_test(to_version: str, from_version: str, to_version_image: str, path_to_test: str):
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
        # load_all_subclasses("tests.compatibility_tests")
        # Step 0: Generate a volume name to be used with the "from_version" Marqo container for state transfer.
        from_version_volume = docker_manager.get_volume_name_from_marqo_version(from_version)
        logger.info(f"Generated volume name: {from_version_volume} for from_version: {from_version}")

        # Step 1: Start a Marqo container using from_version
        docker_manager.start_marqo_container(from_version, from_version_volume)
        logger.info(f"Step 1: Started Marqo container {from_version}")

        # Step 2: Run prepare mode
        logger.info("Step 2: Running prepare mode on initial from_version container")
        test_classes_to_prepare = determine_test_classes_to_prepare(path_to_test)
        run_prepare_mode(from_version, test_classes_to_prepare)

        # Step 3: Stop Marqo from_version container started in Step #1.
        docker_manager.stop_marqo_container(from_version)
        logger.info("Step 3: Stopped Marqo container from Step #1")

        # Step 4: Upgrade to to_version container by transferring state
        logger.info(f"Step 4: Starting Marqo to_version: {to_version} container by transferring state from version: "
                    f"{from_version} to version: {to_version}")
        docker_manager.start_marqo_container_by_transferring_state(to_version, from_version, from_version_volume,
                                                    to_version_image, "ECR")

        #Step 5: Stop Marqo container from Step #4
        logger.info("Step 5: Stopping Marqo container from Step #4")
        docker_manager.stop_marqo_container(to_version)

        #Step 6: Again start a Marqo container using from_version (i.e Rollback marqo version),
        # transferring state from container in Step 4.
        logger.info(f"Step 6: Going back to marqo from_version."
                    f"Starting Marqo from_version: {from_version} container again, "
                    f"by transferring state from to_version, which was {to_version}")
        # TODO: Check from_version_volume for the case where the two versions are before and after 2.9 since we create a new volume in that case.
        prepare_volume_for_rollback(target_version=from_version, source_volume=from_version_volume, source="docker")
        docker_manager.start_marqo_container_by_transferring_state(target_version=from_version, source_version=to_version,
                                                    source_volume=from_version_volume, source="docker")

        # Step 7: Run test mode
        logger.info(f"Step 7: Running tests in test mode on from_version: {from_version}")
        run_test_mode(from_version, path_to_test) # This will validate results from the older indexes added as part of the PREPARE mode above.

        # Step 8: Run prepare and test mode again, on the from_version container.
        logger.info(f"Step 8: Running prepare and test mode on the same from_version: {from_version} container")
        run_prepare_mode(from_version, test_classes_to_prepare)
        run_test_mode(from_version, path_to_test) # This will validate results by creating newer indexes and adding documents to them. This is required just so that we know that even after transferring state from an older version, we are able to create new indexes in the older state seamlessly.

        # Only execute the following if Marqo version >= 2.13.0. This is because the rollback endpoint is only
        # available in these versions.

        # Step 9: Trigger rollback Vespa endpoint
        if semver.VersionInfo.parse(from_version) >= semver.VersionInfo.parse("2.13.0"):
            trigger_rollback_endpoint()

            # Step 10: Run full test suite again after Vespa rollback
            try:
                logger.info(f"Running full test suite with from_version: {from_version}")
                run_prepare_mode(from_version, test_classes_to_prepare)
                run_test_mode(from_version, path_to_test)
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
    parser.add_argument("--path_to_test", required=True, default="")
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
            backwards_compatibility_test(args.from_version, args.to_version, args.to_image, args.path_to_test)
        elif args.mode == "rollback":
            rollback_test(args.to_version, args.from_version, args.to_image, args.path_to_test)

    except Exception as e:
        logger.exception(f"Encountered an exception: {e} while running tests in mode {args.mode}, exiting", exc_info=True)
        sys.exit(1)