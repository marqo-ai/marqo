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

# Marqo changed how it transfers state post version 2.9.0, this variable stores that context
marqo_transfer_state_version = semver.VersionInfo.parse("2.9.0")

from base_compatibility_test_case import BaseCompatibilityTestCase
from enum import Enum

class Mode(Enum):
    PREPARE = "prepare"
    TEST = "test"

# Keep track of containers that need cleanup
containers_to_cleanup: Set[str] = set()
volumes_to_cleanup: Set[str] = set()

logger = get_logger(__name__)

def load_all_subclasses(package_name):
    package = importlib.import_module(package_name)
    package_path = package.__path__

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        importlib.import_module(f"{package_name}.{module_name}")


#TODO: Explore using docker python SDK docker-py to replace the subprocess call, https://github.com/marqo-ai/marqo/pull/1024#discussion_r1841689970
def pull_remote_image_from_ecr(image_name: str):
    """
    Pulls a Docker image from Amazon ECR using the image_name and optionally retags it locally.

    Args:
        image_name (str): The unique identifier for a to_version image. It can be either be the fully qualified image name with the tag
                                (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests:abcdefgh1234)
                                or the fully qualified image name with the digest (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests@sha256:1234567890abcdef).
                                This is constructed in build_push_image.yml workflow and will be the qualified image name with digest for an automatically triggered workflow.

    Returns:
        str: The local tag of the pulled and retagged Docker image.

    Raises:
        RuntimeError: If there is an error during the Docker image pull or retagging process.
    """
    ecr_registry = "424082663841.dkr.ecr.us-east-1.amazonaws.com"

    try:
        # Log in to ECR
        login_password = subprocess.run(
            ["aws", "ecr", "get-login-password", "--region", "us-east-1"],
            check=True,
            stdout=subprocess.PIPE
        ).stdout.decode('utf-8')
        subprocess.run(
            ["docker", "login", "--username", "AWS", "--password-stdin", ecr_registry],
            input=login_password.encode('utf-8'),
            check=True
        )
        # Pull the Docker image from ECR
        image_full_name = image_name
        logger.info(f"Pulling image: {image_full_name}")
        subprocess.run(["docker", "pull", image_full_name], check=True)

        # Optionally retag the image locally to marqo-ai/marqo
        hash_part = image_name.split(":")[1] if ":" in image_name else image_name
        local_tag = f"marqo-ai/marqo:{hash_part}" #it should now be called marqo-ai/marqo:sha-token for image with image digest or marqo-ai/marqo:github.sha for an image with image tag
        logger.info(f"Re-tagging image to: {local_tag}")
        subprocess.run(["docker", "tag", image_full_name, local_tag], check=True)
        return local_tag
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Command '{e.cmd}' failed with return code {e.returncode}. "
            f"Output: {e.output.decode('utf-8') if e.output else 'No output'}"
        )
        logger.exception("Docker command execution failed")
        raise RuntimeError(f"Failed to pull Docker image '{image_name}' from ECR due to a subprocess error.") from e

    except Exception as e:
        logger.exception(f"An unexpected error occurred while pulling the Docker image: {image_name} from ECR")
        raise RuntimeError(f"Failed to pull Docker image '{image_name}' from ECR due to an unexpected error.") from e


def pull_marqo_image(image_name: str, source: str):
    """
    Pull the specified Marqo Docker image.

    Args:
        image_name (str): The identifier with which to pull the docker image.
                                It can simply be the image name if pulling from DockerHub,
                                or it can be the image digest if pulling from ECR
        source (str): The source from which to pull the image.
                      It can be either 'docker' for DockerHub or 'ECR' for Amazon ECR.

    Returns:
        str: The name of the pulled Docker image.

    Raises:
        Exception: If there is an error during the Docker image pull process.
    """
    try:
        if source == "docker":
            logger.info(f"pulling this image: {image_name} from Dockerhub")
            subprocess.run(["docker", "pull", image_name], check=True)
            return image_name
        elif source == "ECR":
            return pull_remote_image_from_ecr(image_name)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to pull Docker image: {image_name}, from source: {source}.")


def start_marqo_container(version: str, volume_name: str):
    """
    Start a Marqo container after pulling the required image from docker and creating a volume.
    The volume is mounted to a specific point such that it can be later used to transfer state to a different version of Marqo.
    This method is usually used to start a Marqo container of an already released image.

    Args:
        version (str): The version of the Marqo container to start.
        volume_name: The volume to use for the container.
    """

    source = "docker" # The source would always be docker because this method is supposed to be downloading and running an already released docker image

    logger.info(f"Starting Marqo container with version: {version}, volume_name: {volume_name}, source: {source}")
    image_name = f"marqoai/marqo:{version}"
    container_name = f"marqo-{version}"

    logger.info(f"Using image: {image_name} with container name: {container_name}")

    # Pull the image before starting the container
    pull_marqo_image(image_name, source)

    # Stop and remove the container if it exists
    try:
        subprocess.run(["docker", "rm", "-f", container_name], check=True)
    except Exception as e:
        logger.warning(f"Container {container_name} not found, skipping removal.")

    # Prepare the docker run command
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", "8882:8882",
        "-e", "MARQO_ENABLE_BATCH_APIS=TRUE",
        "-e", "MARQO_MAX_CPU_MODEL_MEMORY=1.6"
    ]

    # Handle version-specific volume mounting

    # Mounting volumes for Marqo >= 2.9
    # Use the provided volume for state transfer
    volume_name = create_volume_for_marqo_version(version, volume_name)
    logger.info(f"from_version volume name = {volume_name}")
    if version >= marqo_transfer_state_version:
        # setting volume to be mounted at /opt/vespa/var because starting from 2.9, the state is stored in /opt/vespa/var
        cmd.extend(["-v", f"{volume_name}:/opt/vespa/var"])
    else:
        # setting volume to be mounted at /opt/vespa because before 2.9, the state was stored in /opt/vespa
        cmd.extend(["-v", f"{volume_name}:/opt/vespa"])  # volume name will be marqo_2_12_0_volume

    # Append the image
    cmd.append(image_name)
    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        # Run the docker command
        subprocess.run(cmd, check=True)
        containers_to_cleanup.add(container_name)

        # Follow docker logs
        log_cmd = ["docker", "logs", "-f", container_name]
        log_process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for the Marqo service to start
        logger.info("Waiting for Marqo to start...")
        while True:
            try:
                response = requests.get("http://localhost:8882", verify=False)
                if "Marqo" in response.text:
                    logger.info("Marqo started successfully.")
                    break
            except requests.ConnectionError:
                pass
            output = log_process.stdout.readline()
            if output:
                logger.debug(output.strip())
            time.sleep(0.1)

        # Stop following logs after Marqo starts
        log_process.terminate()
        log_process.wait()
        logger.debug("Stopped following docker logs.")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to start Docker container {container_name}, with version: {version}, and volume_name: {volume_name}") from e

    # Show the running containers
    try:
        subprocess.run(["docker", "ps"], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to list Docker containers: {e}")

def start_marqo_container_by_transferring_state(target_version: str, source_version: str, source_volume: str,
                                                target_version_image: str = None, source: str = "docker"):
    """
    Start a Marqo container for the specified target_version, transferring state from the source_version container.
    The state is transferred by copying the state from the source_version container (denoted by 'source_volume') to the target_version container, by re-using the
    source_volume created when starting source_version container.
    Note: This method is used both in backwards compatibility and rollback testing scenarios.
    Args:
        target_version (str): The target version of the Marqo container to start. This variable will contain 'to_version' in case of a backwards compatibility test
                                    whereas it will contain both 'to_version' and 'from_version' (albeit in different method calls) in case of rollback tests.
        source_version (str): The source version of the Marqo container. The source_marqo_version parameter is later used to determine how we transfer state.
                                    This variable will contain 'from_version' in case of a backwards compatibility test whereas it will contain both 'from_version'
                                    and 'to_version' (albeit in different method calls) in case of rollback tests.
        source_volume (str): The volume to use for the container.
        target_version_image (str): The unique identifier for a target_version image. It can be either be the fully qualified image name with the tag (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests:abcdefgh1234)
                                    or the fully qualified image name with the digest (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests@sha256:1234567890abcdef).
                                    This is constructed in build_push_image.yml workflow and will be the qualified image name with digest for an automatically triggered workflow.
                                    Imp. Note: This parameter only needs to be passed in case we are trying to start a container that doesn't exist on dockerHub, essentially only pass this parameter if the source is ECR.
        source (str): The source from which to pull the image. It can be either 'docker' for Docker Hub or 'ECR' for Amazon ECR. Not to be confused with source_version.
    """

    logger.info(
        f"Starting Marqo container with target version: {target_version}, "
        f"source version: {source_version} "
        f"source_volume: {source_volume}, target_version_image: {target_version_image}, source: {source}")
    container_name = f"marqo-{target_version}" # target_version is from_version
    # source_version is to_version

    target_version = semver.VersionInfo.parse(target_version)
    source_version = semver.VersionInfo.parse(source_version)

    logger.info(f"Using image: {target_version_image} with container name: {container_name}")

    if source == "docker": # In case the source is docker, we will directly pull the image using version (ex: marqoai/marqo:2.13.0)
        image_name = f"marqoai/marqo:{target_version}"
    else:
        image_name = target_version_image
    # Pull the image before starting the container
    target_version_image_name = pull_marqo_image(image_name, source)
    logger.info(f" Printing image name {target_version_image_name}")
    try:
        subprocess.run(["docker", "rm", "-f", container_name], check=True)
    except subprocess.CalledProcessError:
        logger.warning(f"Container {container_name} not found, skipping removal.")

    # Prepare the docker run command
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", "8882:8882",
        "-e", "MARQO_ENABLE_BATCH_APIS=TRUE",
        "-e", "MARQO_MAX_CPU_MODEL_MEMORY=1.6"
    ]

    if source_version >= marqo_transfer_state_version and target_version >= marqo_transfer_state_version:
        # Use the provided volume for state transfer
        # setting volume to be mounted at /opt/vespa/var because starting from 2.9, the state is stored in /opt/vespa/var
        cmd.extend(["-v", f"{source_volume}:/opt/vespa/var"])
    elif source_version < marqo_transfer_state_version and target_version < marqo_transfer_state_version:
        # setting volume to be mounted at /opt/vespa because before 2.9, the state was stored in /opt/vespa
        cmd.extend(["-v", f"{source_volume}:/opt/vespa"])
    elif source_version < marqo_transfer_state_version <= target_version:
        # Case when from_version is <2.9 and to_version is >=2.9
        # Here you need to explicitly copy
        target_version_volume = create_volume_for_marqo_version(str(target_version), None)
        copy_state_from_container(source_volume, target_version_volume, target_version_image_name)
        cmd.extend(["-v", f"{target_version_volume}:/opt/vespa/var"])

    cmd.append(target_version_image_name)

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        # Run the docker command
        subprocess.run(cmd, check=True)
        containers_to_cleanup.add(container_name)

        # Follow docker logs
        log_cmd = ["docker", "logs", "-f", container_name]
        log_process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for the Marqo service to start
        logger.info("Waiting for Marqo to start...")
        while True:
            try:
                response = requests.get("http://localhost:8882", verify=False)
                if "Marqo" in response.text:
                    logger.info("Marqo started successfully.")
                    break
            except requests.ConnectionError:
                pass
            output = log_process.stdout.readline()
            if output:
                logger.debug(output.strip())
            time.sleep(0.1)

        # Stop following logs after Marqo starts
        log_process.terminate()
        log_process.wait()
        logger.debug("Stopped following docker logs.")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to start Docker container {container_name} by transferring state with target_version: {target_version}, source_version: {source_version}, source_volume: {source_volume}") from e

    # Show the running containers
    try:
        subprocess.run(["docker", "ps"], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to list Docker containers: {e}")

def stop_marqo_container(version: str):
    """
    Stop a Marqo container but don't remove it yet.

    Args:
        version (str): The version of the Marqo container to stop.
    """
    container_name = f"marqo-{version}"
    logger.info(f"Stopping container with container name {container_name}")
    try:
        subprocess.run(["docker", "stop", container_name], check=True)
        logger.info(f"Successfully stopped container {container_name}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Warning: Failed to stop container {container_name}")


def cleanup_containers():
    """
    Remove all containers that were created during the test.

    This function iterates over the set of containers to clean up and attempts to remove each one using the Docker CLI.
    If a container cannot be removed, a warning message is printed.

    Raises:
        subprocess.CalledProcessError: If there is an error during the container removal process.
    """
    for container_name in containers_to_cleanup:
        try:
            subprocess.run(["docker", "rm", "-f", container_name], check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Warning: Failed to remove container {container_name}: {e}")
    containers_to_cleanup.clear()

def cleanup_volumes():
    """
    Remove all Docker volumes that were created during the test.

    This function iterates over the set of volumes to clean up and attempts to remove each one using the Docker CLI.
    If a volume cannot be removed, a warning message is printed.

    Raises:
        subprocess.CalledProcessError: If there is an error during the volume removal process.
    """
    for volume_name in volumes_to_cleanup:
        try:
            subprocess.run(["docker", "volume", "rm", volume_name], check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Warning: Failed to remove volume {volume_name}: {e}")
    volumes_to_cleanup.clear()

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
    full_test_run(to_version)

def full_test_run(marqo_version: str):
    """
    This method will run tests on a single Marqo version container, which means it will run both prepare and tests on the
    to_version Marqo container. Note that to_version Marqo container has been created by transferring instance from a
    previous from_version Marqo container.
    """
    logger.info(f"Running full_test_run with version: {marqo_version}")
    #Step 1: Run tests in prepare mode
    run_prepare_mode(marqo_version)
    #Step 2: Run tests in test mode
    run_test_mode(marqo_version)

def run_prepare_mode(version_to_test_against: str):
    version_to_test_against = semver.VersionInfo.parse(version_to_test_against)
    load_all_subclasses("tests.backwards_compatibility_tests")
    # Get all subclasses of `BaseCompatibilityTestCase` that match the `version_to_test_against` criterion
    # The below condition also checks if the test class is not marked to be skipped
    for test_class in BaseCompatibilityTestCase.__subclasses__():
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

        if semver.VersionInfo.parse(marqo_version).compare(version_to_test_against) <= 0:
            logger.info(f"Running prepare mode on testcase: {test_class.__name__} with version: {marqo_version}")
            test_class.setUpClass() #setUpClass will be used to create Marqo client
            test_instance = test_class()
            test_instance.prepare() #Prepare method will be used to create index and add documents
        else: # Skip the test if the version_to_test_against is greater than the version the test is marked
            logger.info(f"Skipping testcase {test_class.__name__} with version {marqo_version} as it is greater than {version_to_test_against}")

def construct_pytest_arguments(version_to_test_against):
    pytest_args = [
        f"--version_to_compare_against={version_to_test_against}",
        "-m", f"marqo_version",
        "-s",
        "tests/backwards_compatibility_tests"
    ]
    return pytest_args

def run_test_mode(version_to_test_against):
    pytest_args = construct_pytest_arguments(version_to_test_against)
    pytest.main(pytest_args)

def create_volume_for_marqo_version(version: str, volume_name: str = None):
    """
    Create a Docker volume for the specified Marqo version.

    This function replaces dots with underscores in the version string to format the volume name.
    If no volume name is provided, it generates one based on the version.

    Args:
        version (str): The version of the Marqo container.
        volume_name (str): The name of the Docker volume to create. If None, a name is generated based on the version.

    Returns:
        str: The name of the created Docker volume.

    Raises:
        subprocess.CalledProcessError: If there is an error during the Docker volume creation process.
    """
    # Replace dots with underscores to format the volume name
    if volume_name is None:
        volume_name = get_volume_name_from_marqo_version(version)

    # Create the Docker volume using the constructed volume name
    try:
        subprocess.run(["docker", "volume", "create", "--name", volume_name], check=True)
        logger.info(f"Successfully created volume: {volume_name}")
        volumes_to_cleanup.add(volume_name)
        return volume_name
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to create volume: {volume_name}") from e

def get_volume_name_from_marqo_version(version):
    """
    Generate a Docker volume name based on the Marqo version.

    This function replaces dots with underscores in the version string to format the volume name.

    Args:
        version (str): The version of the Marqo container.

    Returns:
        str: The formatted Docker volume name.
    """
    volume_name = f"marqo_{version.replace('.', '_')}_volume"
    return volume_name


def copy_state_from_container(
        from_version_volume: str, to_version_volume: str, image: str):
    """
    Copy the state from one Docker volume to another using a specified Docker image.

    This function runs a Docker container with the specified image, mounts the source and target volumes,
    and copies the contents from the source volume to the target volume. It is specifically used
    in case when from_version is <2.9 and to_version is >=2.9.

    Args:
        from_version_volume (str): The name of the source Docker volume.
        to_version_volume (str): The name of the target Docker volume.
        image (str): The Docker image to use for the container.

    Raises:
        subprocess.CalledProcessError: If there is an error during the Docker run or copy process.
    """

    cmd = ["docker", "run", "--rm", "-it", "--entrypoint=''",
           "-v", f"{from_version_volume}:/opt/vespa_old",
           "-v", f"{to_version_volume}:/opt/vespa/var",
           f"{image}",
           "sh", "-c", 'cd /opt/vespa_old && cp -a . /opt/vespa/var']
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully copied state from {from_version_volume} to {to_version_volume}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to copy state from {from_version_volume} to {to_version_volume}.") from e

def trigger_rollback_endpoint(from_version: str):
    if semver.VersionInfo.parse(from_version) < semver.VersionInfo.parse("2.13.0"):
        return
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
        from_version_volume = get_volume_name_from_marqo_version(from_version)

        #Start from_version container
        start_marqo_container(from_version, from_version_volume)
        logger.info(f"Started Marqo container {from_version}")

        try:
            run_tests_in_mode(Mode.PREPARE, from_version)
        except Exception as e:
            raise RuntimeError(f"Error running tests in 'prepare' mode across versions on from_version: {from_version}") from e
        # Step 2: Stop from_version container (but don't remove it)
        stop_marqo_container(from_version)

        # Step 3: Start to_version container by transferring state
        logger.info(f"Starting Marqo to_version: {to_version} container by transferring state from version {from_version} to {to_version}")
        start_marqo_container_by_transferring_state(to_version, from_version, from_version_volume,
                                                    to_version_image, "ECR")
        logger.info(f"Started Marqo to_version: {to_version} container by transferring state")
        # Step 4: Run tests
        try:
            run_tests_in_mode(Mode.TEST, from_version)
        except Exception as e:
            raise RuntimeError(f"Error running tests across versions in 'test' mode on from_version: {from_version}") from e
        logger.info("Finished running tests in Test mode")
        # Step 5: Do a full test run which includes running tests in prepare and test mode on the same container
        try:
            full_test_run(to_version)
        except Exception as e:
            raise RuntimeError(f"Error running tests in full test run, on to_version: {to_version}.") from e
    except Exception as e:
        raise RuntimeError(f"An error occurred while executing backwards compatibility tests, on from_version: {from_version}, to_version: {to_version}, to_version_image: {to_version_image}") from e
    finally:
        # Stop the to_version container (but don't remove it yet)
        logger.error("Calling stop_marqo_container with " + str(to_version))
        stop_marqo_container(to_version)
        # Clean up all containers at the end
        cleanup_containers()
        cleanup_volumes()



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
        from_version_volume = get_volume_name_from_marqo_version(from_version)

        # Step 1: Start a Marqo container using from_version and run tests in prepare mode.
        start_marqo_container(from_version, from_version_volume)
        logger.info(f"Started Marqo container {from_version}")

        # Run tests in prepare mode
        try:
            run_tests_in_mode(Mode.PREPARE, from_version)
        except Exception as e:
            raise RuntimeError(f"Error while running tests across versions in 'prepare' mode.") from e

        # Step 2: Stop Marqo from_version container started in Step #1.
        stop_marqo_container(from_version)

        # Step 3: Start to_version container by transferring state
        logger.info(f"Starting Marqo to_version: {to_version} container by transferring state from version: {from_version} to version: {to_version}")
        start_marqo_container_by_transferring_state(to_version, from_version, from_version_volume,
                                                    to_version_image, "ECR")
        logger.info(f"Started Marqo to_version: {to_version} container by transferring state")

        #Step 4: Stop Marqo container from Step #3
        stop_marqo_container(to_version)

        #Step 5: Again start a Marqo container using from_version (i.e Rollback to from_version), transferring state from container in Step 4.
        logger.info(f"Starting Marqo from_version: {from_version} container again, by transferring state from to version: {to_version} to version: {from_version}")
        # TODO: Check from_version_volume for the case where the two versions are before and after 2.9 since we create a new volume in that case.
        prepare_volume_for_rollback(target_version=from_version, source_volume=from_version_volume, source="docker")
        start_marqo_container_by_transferring_state(target_version=from_version, source_version=to_version,
                                                    source_volume=from_version_volume, source="docker")

        #Step 6: Run tests in test mode, then run full test run
        try:
            run_tests_in_mode(Mode.TEST, from_version)
        except Exception as e:
            raise RuntimeError(f"Error in rollback tests while running tests across versions in 'test' mode on version: {from_version}") from e
        try:
            full_test_run(to_version)
        except Exception as e:
            raise RuntimeError(f"Error in rollback tests while running tests in full test run on version: {to_version}") from e

        #Step 7: Trigger rollback endpoint
        trigger_rollback_endpoint(from_version)

        #Step 8:
        try:
            run_full_test_suite(from_version, to_version)
        except Exception as e:
            raise RuntimeError(f"Error when running full test suite in rollback tests after rolling back vespa application, with from_version: {from_version}, to_version: {to_version}") from e

    finally:
        # Stop the final container (but don't remove it yet)
        logger.debug("Stopping marqo container")
        stop_marqo_container(from_version)
        # Clean up all containers and volumes at the end
        logger.debug("Cleaning up containers and volumes")
        cleanup_containers()
        cleanup_volumes()

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
        logger.error(f"Encountered an exception: {e} while running tests in mode {args.mode}, exiting")
        sys.exit(1)