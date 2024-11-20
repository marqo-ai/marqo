import argparse
import importlib
import pkgutil
import time

import pytest
from typing import Optional, Set
import subprocess
import sys
import requests
import semver
import traceback

from compatibility_test_logger import get_logger

marqo_transfer_state_version = semver.VersionInfo.parse("2.9.0")

from base_compatibility_test_case import BaseCompatibilityTestCase

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
def pull_remote_image_from_ecr(image_identifier: str):
    """
    Pulls a Docker image from Amazon ECR using the image_identifier and optionally retags it locally.

    Args:
        image_identifier (str): The unique identifier for a to_version image. It can be either be the fully qualified image name with the tag
                                (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests:abcdefgh1234)
                                or the fully qualified image name with the digest (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests@sha256:1234567890abcdef).
                                This is constructed in build_push_image.yml workflow and will be the qualified image name with digest for an automatically triggered workflow.

    Returns:
        str: The local tag of the pulled and retagged Docker image.

    Raises:
        Exception: If there is an error during the Docker image pull or retagging process.
    """
    ecr_registry = "424082663841.dkr.ecr.us-east-1.amazonaws.com"
    image_repo = "marqo-compatibility-tests"

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
        image_full_name = image_identifier
        logger.debug(f"Pulling image: {image_full_name}")
        subprocess.run(["docker", "pull", image_full_name], check=True)

        # Optionally retag the image locally to marqo-ai/marqo
        hash_part = image_identifier.split(":")[1] if ":" in image_identifier else image_identifier
        local_tag = f"marqo-ai/marqo:{hash_part}" #it should now be called marqo-ai/marqo:sha-token for image with image digest or marqo-ai/marqo:github.sha for an image with image tag
        logger.debug(f"Retagging image to: {local_tag}")
        subprocess.run(["docker", "tag", image_full_name, local_tag], check=True)
        return local_tag
    except subprocess.CalledProcessError as e:
        logger.debug(f"Command '{e.cmd}' failed with return code {e.returncode}")
        logger.debug("Error output:", e.output.decode() if e.output else "No output")
        traceback.print_exc()  # Print the full stack trace for debugging
        raise Exception(f"Failed to pull Docker image {image_identifier}: {e}")
    except Exception as e:
        logger.debug("An unexpected error occurred while pulling the Docker image.")
        traceback.print_exc()  # Print full stack trace for debugging
        raise e

def pull_marqo_image(image_identifier: str, source: str):
    """
    Pull the specified Marqo Docker image.

    Args:
        image_identifier (str): The identifier with which to pull the docker image.
                                It can simply be the image name if pulling from DockerHub,
                                or it can be the image digest if pulling from ECR
        source (str): The source from which to pull the image.
                      It can be either 'docker' for Docker Hub or 'ECR' for Amazon ECR.

    Returns:
        str: The name of the pulled Docker image.

    Raises:
        Exception: If there is an error during the Docker image pull process.
    """
    try:
        if source == "docker":
            logger.debug(f"pulling this image: {image_identifier} from Dockerhub")
            subprocess.run(["docker", "pull", image_identifier], check=True)
            return image_identifier
        elif source == "ECR":
            return pull_remote_image_from_ecr(image_identifier)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to pull Docker image {image_identifier}: {e}")


def start_marqo_from_version_container(from_version: str, from_version_volume, env_vars: Optional[list] = None):
    """
    Start a Marqo container after pulling the required image from docker and creating a volume.
    The volume is mounted to a specific point such that it can be later used to transfer state to a different version of Marqo.

    Args:
        from_version (str): The version of the Marqo container to start.
        from_version_volume: The volume to use for the container.
        from_version_image (Optional[str]): The specific image to use for the container. Defaults to None.
        env_vars (Optional[list]): A list of environment variables to set in the container. Defaults to None.
    """

    source = "docker" #The source for from_version image would always be docker because it's supposed to be an already released docker image

    logger.debug(f"Starting Marqo container with from_version: {from_version}, from_version_volume: {from_version_volume}, source: {source}")
    from_version_image = f"marqoai/marqo:{from_version}"
    container_name = f"marqo-{from_version}"

    logger.debug(f"Using image: {from_version_image} with container name: {container_name}")

    # Pull the image before starting the container
    pull_marqo_image(from_version_image, source)

    # Stop and remove the container if it exists
    try:
        subprocess.run(["docker", "rm", "-f", container_name], check=True)
    except Exception as e:
        logger.debug(f"Container {container_name} not found, skipping removal.")

    # Prepare the docker run command
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", "8882:8882",
        "-e", "MARQO_ENABLE_BATCH_APIS=TRUE",
        "-e", "MARQO_MAX_CPU_MODEL_MEMORY=1.6"
    ]

    # Append environment variables passed via the method
    if env_vars:
        for var in env_vars:
            cmd.extend(["-e", var])

    # Handle version-specific volume mounting

    # Mounting volumes for Marqo >= 2.9
    # Use the provided volume for state transfer
    from_version_volume = create_volume_for_marqo_version(from_version, from_version_volume)
    logger.debug(f"from_version volume = {from_version_volume}")
    if from_version >= marqo_transfer_state_version:
        # setting volume to be mounted at /opt/vespa/var because starting from 2.9, the state is stored in /opt/vespa/var
        cmd.extend(["-v", f"{from_version_volume}:/opt/vespa/var"])
    else:
        # setting volume to be mounted at /opt/vespa because before 2.9, the state was stored in /opt/vespa
        cmd.extend(["-v", f"{from_version_volume}:/opt/vespa"])  # volume name will be marqo_2_12_0_volume

    # Append the image
    cmd.append(from_version_image)
    logger.debug(f"Running command: {' '.join(cmd)}")

    try:
        # Run the docker command
        subprocess.run(cmd, check=True)
        containers_to_cleanup.add(container_name)

        # Follow docker logs
        log_cmd = ["docker", "logs", "-f", container_name]
        log_process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for the Marqo service to start
        logger.debug("Waiting for Marqo to start...")
        while True:
            try:
                response = requests.get("http://localhost:8882", verify=False)
                if "Marqo" in response.text:
                    logger.debug("Marqo started successfully.")
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
        logger.debug(f"Failed to start Docker container {container_name}: {e}")
        raise

    # Show the running containers
    try:
        subprocess.run(["docker", "ps"], check=True)
    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to list Docker containers: {e}")
        raise

def start_marqo_to_version_container(to_version: str, from_version: str, from_version_volume: str,
                                     to_version_identifier: str, env_vars: Optional[list] = None):
    """
    Start a Marqo container for the specified to_version, transferring state from the from_version container.
    The state is transferred by copying the state from the from_version container to the to_version container, by re-using the
    from_version_volume created when starting from_version container.
    Args:
        to_version (str): The target version of the Marqo container to start.
        from_version (str): The source version of the Marqo container. The from_version parameter is later used to determine how we transfer state.
        from_version_volume (str): The volume to use for the container.
        to_version_identifier (str): The unique identifier for a to_version image. It can be either be the fully qualified image name with the tag (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests:abcdefgh1234)
                                        or the fully qualified image name with the digest (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests@sha256:1234567890abcdef).
                                        This is constructed in build_push_image.yml workflow and will be the qualified image name with digest for an automatically triggered workflow.
        env_vars (Optional[list]): A list of environment variables to set in the container. Defaults to None.
    """
    source = "ECR" #Source of a to_version image will always be ECR because we build and push unpublished & to be tested images to ECR
    logger.debug(
        f"Starting Marqo container with to_version: {to_version}, "
        f"from_version: {from_version} "
        f"from_version_volume: {from_version_volume}, to_version_identifier: {to_version_identifier}, source: {source}")
    container_name = f"marqo-{to_version}"
    to_version = semver.VersionInfo.parse(to_version)
    from_version = semver.VersionInfo.parse(from_version)

    logger.debug(f"Using image: {to_version_identifier} with container name: {container_name}")

    # Pull the image before starting the container
    to_version_image_name = pull_marqo_image(to_version_identifier, source)
    logger.debug(f" Printing image name {to_version_image_name}")
    try:
        subprocess.run(["docker", "rm", "-f", container_name], check=True)
    except subprocess.CalledProcessError:
        logger.debug(f"Container {container_name} not found, skipping removal.")

    # Prepare the docker run command
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", "8882:8882",
        "-e", "MARQO_ENABLE_BATCH_APIS=TRUE",
        "-e", "MARQO_MAX_CPU_MODEL_MEMORY=1.6"
    ]
    # Append environment variables passed via the method
    if env_vars:
        for var in env_vars:
            cmd.extend(["-e", var])


    if from_version >= marqo_transfer_state_version and to_version >= marqo_transfer_state_version:
        # Use the provided volume for state transfer
        cmd.extend(["-v", f"{from_version_volume}:/opt/vespa/var"]) #setting volume to be mounted at /opt/vespa/var because starting from 2.9, the state is stored in /opt/vespa/var
    elif from_version < marqo_transfer_state_version and to_version < marqo_transfer_state_version:
        cmd.extend(["-v", f"{from_version_volume}:/opt/vespa"]) #setting volume to be mounted at /opt/vespa because before 2.9, the state was stored in /opt/vespa
    elif from_version < marqo_transfer_state_version <= to_version:     # Case when from_version is <2.9 and to_version is >=2.9
    # Here you need to explicitly copy
        to_version_volume = create_volume_for_marqo_version(str(to_version), None)
        copy_state_from_container(from_version_volume, to_version_volume, to_version_image_name)
        cmd.extend(["-v", f"{to_version_volume}:/opt/vespa/var"])

    cmd.append(to_version_image_name)

    logger.debug(f"Running command: {' '.join(cmd)}")

    try:
        # Run the docker command
        subprocess.run(cmd, check=True)
        containers_to_cleanup.add(container_name)

        # Follow docker logs
        log_cmd = ["docker", "logs", "-f", container_name]
        log_process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for the Marqo service to start
        logger.debug("Waiting for Marqo to start...")
        while True:
            try:
                response = requests.get("http://localhost:8882", verify=False)
                if "Marqo" in response.text:
                    logger.debug("Marqo started successfully.")
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
        logger.debug(f"Failed to start Docker container {container_name}: {e}")
        raise

    # Show the running containers
    try:
        subprocess.run(["docker", "ps"], check=True)
    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to list Docker containers: {e}")
        raise


def stop_marqo_container(version: str):
    """
    Stop a Marqo container but don't remove it yet.

    Args:
        version (str): The version of the Marqo container to stop.
    """
    container_name = f"marqo-{version}"
    logger.debug(f"Stopping container with container name {container_name}")
    try:
        subprocess.run(["docker", "stop", container_name], check=True)
        logger.debug(f"Successfully stopped container {container_name}")
    except subprocess.CalledProcessError as e:
        logger.debug(f"Warning: Failed to stop container {container_name}: {e}")


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
            logger.debug(f"Warning: Failed to remove container {container_name}: {e}")
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
            logger.debug(f"Warning: Failed to remove volume {volume_name}: {e}")
    volumes_to_cleanup.clear()

def backwards_compatibility_test(from_version: str, to_version: str, to_version_image_idenfitifer: str):
    """
    Perform a backwards compatibility test between two versions of Marqo.

    This function starts a container with the from_version, runs tests in prepare mode, stops the container,
    starts a container with the to_version by transferring state from from_version container, and runs tests in test mode.

    Args:
        from_version (str): The source version of the Marqo container.
        to_version (str): The target version of the Marqo container.
        to_version_image_idenfitifer (str): The unique identifier for a to_version image. It can be either be the fully qualified image name with the tag
                                (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests:abcdefgh1234)
                                or the fully qualified image name with the digest (ex: 424082663841.dkr.ecr.us-east-1.amazonaws.com/marqo-compatibility-tests@sha256:1234567890abcdef).
                                This is constructed in build_push_image.yml workflow and will be the qualified image name with digest for an automatically triggered workflow.

    Raises:
        ValueError: If the major versions of from_version and to_version are incompatible.
        Exception: If there is an error during the test process.
    """
    try:
        # Step 1: Start from_version container and run tests in prepare mode
        logger.debug(f"Starting backwards compatibility tests with from_version: {from_version}, to_version: {to_version}, to_version_image_idenfitifer: {to_version_image_idenfitifer}")
        # Check for version compatibility
        from_major_version = int(from_version.split('.')[0])
        logger.debug(f"from major version = {from_major_version}")
        to_major_version = int(to_version.split('.')[0])
        if from_major_version != to_major_version:
            logger.debug(f"from version & to_version can be tested for backwards_compatibility")
            raise ValueError("Cannot transfer state between incompatible major versions of Marqo.")
        logger.debug(f"Transferring state from version {from_version} to {to_version}")

        from_version_volume = get_volume_name_from_marqo_version(from_version)
        start_marqo_from_version_container(from_version, from_version_volume)
        logger.debug(f"Started marqo container {from_version}")

        try:
            run_tests_across_versions("prepare", from_version, to_version)
        except Exception as e:
            logger.error(f"Error running tests across versions in 'prepare' mode: {e}")
            raise e
        # Step 2: Stop from_version container (but don't remove it)
        stop_marqo_container(from_version)

        # Step 3: Start to_version container, transferring state
        start_marqo_to_version_container(to_version, from_version, from_version_volume, to_version_image_idenfitifer)
        logger.debug(f"Started marqo to_version: {to_version} container by transferring state")
        # Step 4: Run tests
        try:
            run_tests_across_versions("test", from_version, to_version)
        except Exception as e:
            logger.error(f"Error running tests across versions in 'test' mode: {e}")
            raise e
        logger.debug("Finished running tests in Test mode")
        # Step 5: Do a full test run which includes running tests in prepare and test mode on the same container
        try:
            full_test_run(to_version)
        except Exception as e:
            logger.error(f"Error running tests in full test run mode: {e}")
            raise e
    except Exception as e:
        logger.debug(f"An error occurred while executing backwards compatibility tests: {e}")
        raise e
    finally:
        # Stop the to_version container (but don't remove it yet)
        logger.debug("Calling stop_marqo_container with " + str(to_version))
        stop_marqo_container(to_version)
        # Clean up all containers at the end
        cleanup_containers()
        cleanup_volumes()



def rollback_test(to_version: str, from_version: str, to_version_digest, from_image: Optional[str] = None,
                  to_image: Optional[str] = None):
    """
    Perform a rollback test between two versions of Marqo.

    This function first performs a backwards compatibility test from the from_version to the to_version.
    Then, it stops the to_version container, starts the from_version container again, and runs tests in test mode.

    Args:
        to_version (str): The target version of the Marqo container.
        from_version (str): The source version of the Marqo container.
        to_version_digest: The specific image digest to use for the to_version container.
        from_image (Optional[str]): The specific image to use for the from_version container. Defaults to None.
        to_image (Optional[str]): The specific image to use for the to_version container. Defaults to None.
    """
    try:
        backwards_compatibility_test(from_version, to_version, None)

        stop_marqo_container(to_version)

        start_marqo_from_version_container(from_version, None)

        run_tests_across_versions("test", from_version, to_version)
    finally:
        # Stop the final container (but don't remove it yet)
        stop_marqo_container(from_version)
        # Clean up all containers at the end
        cleanup_containers()

def run_tests_across_versions(mode: str, from_version: str, to_version: str):
    """
    This method will run tests across two Marqo versions, meaning it will run prepare on a Marqo from_version instance,
    and run tests on a Marqo to_version instance.
    """
    logger.debug(f"Running tests across versions with mode: {mode}, from_version: {from_version}, to_version: {to_version}")

    if mode == "prepare":
        run_prepare_mode(from_version)
    elif mode == "test":
        run_test_mode(from_version)

def full_test_run(to_version: str):
    """
    This method will run tests on a single marqo version container, which means it will run both prepare and tests on the
    to_version Marqo container. Note that to_version Marqo container has been created by transferring instance from a
    previous from_version Marqo container.
    """
    logger.debug(f"Running full_test_run with to_version: {to_version}")
    #Step 1: Run tests in prepare mode
    run_prepare_mode(to_version)
    #Step 2: Run tests in test mode
    run_test_mode(to_version)

def run_prepare_mode(version_to_test_against: str):
    load_all_subclasses("tests.backwards_compatibility_tests")
    # Get all subclasses of `BaseCompatibilityTestCase` that match the `version_to_test_against` criterion
    # The below condition also checks if the test class is not marked to be skipped
    tests = [test_class for test_class in BaseCompatibilityTestCase.__subclasses__()
             if (getattr(test_class, 'marqo_version', '0') <= version_to_test_against and getattr(test_class, 'skip', False) == False)]
    for test_class in tests:
        test_class.setUpClass() #setUpClass will be used to create Marqo client
        test_instance = test_class()
        test_instance.prepare() #Prepare method will be used to create index and add documents

def construct_pytest_arguments(version_to_test_against):
    pytest_args = [
        f"--version_to_compare_against={version_to_test_against}",
        "-m", f"marqo_version",
        "tests/backwards_compatibility_tests"
    ]
    return pytest_args

def run_test_mode(version_to_test_against):
    pytest_args = construct_pytest_arguments(version_to_test_against)
    pytest.main(pytest_args)

def create_volume_for_marqo_version(version: str, volume_name: str):
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
        logger.debug(f"Successfully created volume: {volume_name}")
        volumes_to_cleanup.add(volume_name)
        return volume_name
    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to create volume: {volume_name}. Error: {e}")


    #TODO: Make it compatible for when you directly pass and image and no version is passed.
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
        logger.debug(f"Successfully copied state from {from_version_volume} to {to_version_volume}")
    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to copy state from {from_version_volume} to {to_version_volume}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marqo Testing Runner")
    parser.add_argument("--mode", choices=["backwards_compatibility", "rollback"], required=True)
    parser.add_argument("--from_version", required=True)
    parser.add_argument("--to_version", required=True)
    parser.add_argument("--to_version_image_identifier", required=True)
    args = parser.parse_args()

    from_version = semver.VersionInfo.parse(args.from_version)
    to_version = semver.VersionInfo.parse(args.to_version)
    if from_version >= to_version:
        logger.debug("from_version should be less than to_version")
        sys.exit(0) # TODO: figure out if we should just quit.

    if args.mode == "backwards_compatibility":
        backwards_compatibility_test(args.from_version, args.to_version, args.to_version_image_identifier)
    elif args.mode == "rollback":
        rollback_test(args.to_version, args.from_version, args.to_version_image_identifier)
