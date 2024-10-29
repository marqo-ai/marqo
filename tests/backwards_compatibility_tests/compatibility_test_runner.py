import argparse
import time

import pytest
from typing import Optional, Set
import subprocess
import sys
import os
import requests
import semver

marqo_transfer_state_version = semver.VersionInfo.parse("2.9.0")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base_test_case import BaseTestCase
from test_vector_normalisation import TestVectorNormalisation


# Keep track of containers that need cleanup
containers_to_cleanup: Set[str] = set()
volumes_to_cleanup: Set[str] = set()

def pull_remote_image_from_ecr(image_tag: str):
    ecr_registry = "424082663841.dkr.ecr.us-east-1.amazonaws.com"
    image_repo = "marqo-compatibility-tests"

    # Log in to ECR
    subprocess.run(
        ["aws", "ecr", "get-login-password", "--region", "us-east-1"],
        check=True,
        stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", ecr_registry],
        check=True
    )
    # Pull the Docker image from ECR
    image_full_name = f"{ecr_registry}/{image_repo}:{image_tag}"
    print(f"Pulling image: {image_full_name}")
    subprocess.run(["docker", "pull", image_full_name], check=True)

    # Optionally retag the image locally to marqo-ai/marqo
    local_tag = f"marqo-ai/marqo:{image_tag}" #it should now be called marqo-ai/marqo:sha-token
    print(f"Retagging image to: {local_tag}")
    subprocess.run(["docker", "tag", image_full_name, local_tag], check=True)
    return local_tag

    # Now you can use the image as "marqo-ai/marqo:{image_tag}"

def pull_marqo_image(image: str, source: str):
    """Pull the specified Marqo Docker image."""
    try:
        if source == "docker":
            print(f"pulling this image from dockerhub {image}")
            subprocess.run(["docker", "pull", image], check=True)
            return image
        elif source == "ECR":
            return pull_remote_image_from_ecr(image_tag=image)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to pull Docker image {image}: {e}")


def start_marqo_from_version_container(version: str, from_version_volume, from_version_image: Optional[str] = None,
                                       transfer_state: Optional[str] = None, env_vars: Optional[list] = None):
    source = "docker"
    """Start a Marqo container after pulling the required image and apply all provided environment variables."""
    print(f"Starting Marqo container with version {version}, from_version_image {from_version_image}, from_version_volume {from_version_image}, transfer_state {transfer_state}, source {source}")
    from_version_image = from_version_image or f"marqoai/marqo:{version}"
    container_name = f"marqo-{version}"

    print(f"Using image: {from_version_image} with container name: {container_name}")

    # Pull the image before starting the container
    pull_marqo_image(from_version_image, source)

    # Stop and remove the container if it exists
    try:
        subprocess.run(["docker", "rm", "-f", container_name], check=True)
    except Exception as e:
        print(f"Container {container_name} not found, skipping removal.")

    # Prepare the docker run command
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-it", "-p", "8882:8882",
        "-e", "MARQO_ENABLE_BATCH_APIS=TRUE",
        "-e", "MARQO_MAX_CPU_MODEL_MEMORY=1.6"
    ]

    # Append environment variables passed via the method
    if env_vars:
        for var in env_vars:
            cmd.extend(["-e", var])

    # Handle version-specific volume mounting
    # Mounting volumes for Marqo >= 2.9")
    # Use the provided volume for state transfer
    from_version_volume = create_volume_for_marqo_version(version, from_version_volume)
    print(f"from version volume = {from_version_volume}")
    if version >= marqo_transfer_state_version:
        # setting volume to be mounted at /opt/vespa/var because starting from 2.9, the state is stored in /opt/vespa/var
        cmd.extend(["-v", f"{from_version_volume}:/opt/vespa/var"])
        # subprocess.run(["mkdir", "/logs"]) #TODO: Check if this is required or not
        # cmd.extend(["-v", "$(pwd)/logs:/opt/vespa/logs"])  # Optional logs for debugging
    else:
        # setting volume to be mounted at /opt/vespa because before 2.9, the state was stored in /opt/vespa
        cmd.extend(["-v", f"{from_version_volume}:/opt/vespa"])  # volume name will be marqo_2_12_0_volume

    # Append the image
    cmd.append(from_version_image)
    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run the docker command
        subprocess.run(cmd, check=True)
        containers_to_cleanup.add(container_name)
        print(f"Going to start {container_name}.")

        # Follow docker logs
        log_cmd = ["docker", "logs", "-f", container_name]
        log_process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for the Marqo service to start
        print("Waiting for Marqo to start...")
        while True:
            try:
                response = requests.get("http://localhost:8882", verify=False)
                if "Marqo" in response.text:
                    print("Marqo started successfully.")
                    break
            except requests.ConnectionError:
                pass
            output = log_process.stdout.readline()
            if output:
                print(output.strip())
            time.sleep(0.1)

        # Stop following logs after Marqo starts
        log_process.terminate()
        log_process.wait()
        print("Stopped following docker logs.")

    except subprocess.CalledProcessError as e:
        print(f"Failed to start Docker container {container_name}: {e}")
        raise

    # Show the running containers
    try:
        subprocess.run(["docker", "ps"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to list Docker containers: {e}")
        raise

def start_marqo_to_version_container(to_version: str, from_version: str, from_version_volume: str,
                                     to_version_tag: str, env_vars: Optional[list] = None):
    source = "ECR"
    print(
        f"Starting Marqo container with to_version {to_version}, "
        f"from_version: {from_version} "
        f"from_version_volume {from_version_volume}, to_version_tag, {to_version_tag}, source {source}")
    container_name = f"marqo-{to_version}"
    to_version = semver.VersionInfo.parse(to_version)
    from_version = semver.VersionInfo.parse(from_version)

    print(f"Using image: {to_version_tag} with container name: {container_name}")

    # Pull the image before starting the container
    to_version_image_name = pull_marqo_image(to_version_tag, source)
    print(f" Printing image name {to_version_image_name}")
    try:
        subprocess.run(["docker", "rm", "-f", container_name], check=True)
    except subprocess.CalledProcessError:
        print(f"Container {container_name} not found, skipping removal.")

    # Prepare the docker run command
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-it", "-p", "8882:8882",
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

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run the docker command
        subprocess.run(cmd, check=True)
        containers_to_cleanup.add(container_name)
        print(f"Going to start {container_name}.")

        # Follow docker logs
        log_cmd = ["docker", "logs", "-f", container_name]
        log_process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for the Marqo service to start
        print("Waiting for Marqo to start...")
        while True:
            try:
                response = requests.get("http://localhost:8882", verify=False)
                if "Marqo" in response.text:
                    print("Marqo started successfully.")
                    break
            except requests.ConnectionError:
                pass
            output = log_process.stdout.readline()
            if output:
                print(output.strip())
            time.sleep(0.1)

        # Stop following logs after Marqo starts
        log_process.terminate()
        log_process.wait()
        print("Stopped following docker logs.")

    except subprocess.CalledProcessError as e:
        print(f"Failed to start Docker container {container_name}: {e}")
        raise

    # Show the running containers
    try:
        subprocess.run(["docker", "ps"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to list Docker containers: {e}")
        raise


def stop_marqo_container(version: str):
    """Stop a Marqo container but don't remove it yet."""
    print("in here with version " + version)
    container_name = f"marqo-{version}"
    print(f"Stopping container with container name {container_name}")
    try:
        subprocess.run(["docker", "stop", container_name], check=True)
        print(f"Successfully stopped container {container_name}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to stop container {container_name}: {e}")


def cleanup_containers():
    """Remove all containers that were created during the test."""
    for container_name in containers_to_cleanup:
        try:
            subprocess.run(["docker", "rm", "-f", container_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to remove container {container_name}: {e}")
    containers_to_cleanup.clear()

def cleanup_volumes():
    for volume_name in volumes_to_cleanup:
        try:
            subprocess.run(["docker", "volume", "rm", volume_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to remove volume {volume_name}: {e}")
    volumes_to_cleanup.clear()

def backwards_compatibility_test(from_version: str, to_version: str, to_version_tag: str, from_image: Optional[str] = None,
                                 to_image: Optional[str] = None):
    try:
        # Step 1: Start from_version container and run tests in prepare mode
        print("In here with from_version:" + from_version + " to_version: " + to_version + " ");
            # Check for version compatibility
        from_major_version = int(from_version.split('.')[0])
        print(f"from major version = {from_major_version}")
        to_major_version = int(to_version.split('.')[0])
        if from_major_version != to_major_version:
            print(f"from version & to_version can be tested for backwards_compatibility")
            raise ValueError("Cannot transfer state between incompatible major versions of Marqo.")
        print(f"Transferring state from version {from_version} to {to_version}")

        from_version_volume = get_volume_name_from_marqo_version(from_version)
        start_marqo_from_version_container(from_version, from_version_volume, from_image)
        print("Started marqo container" + from_version)

        run_tests("prepare", from_version, to_version, "http://localhost:8882")
        # Step 2: Stop from_version container (but don't remove it)
        stop_marqo_container(from_version)

        # Step 3: Start to_version container, transferring state
        start_marqo_to_version_container(to_version, from_version, from_version_volume, to_image)
        print(f"Started marqo container in to_version {to_version} by transferring state")
        # Step 4: Run tests
        run_tests("test", from_version, to_version, "http://localhost:8882")
        print("Ran tests in test mode")
    except Exception as e:
        print(f"Error: {e}, {e.__class__.__name__}, {e.__traceback__}, {e.__traceback__.__class__}, {e.__traceback__.tb_lineno}")
        raise e
    finally:
        # Stop the to_version container (but don't remove it yet)
        print("Calling stop_marqo_container with" + str(to_version))
        stop_marqo_container(to_version)
        # Clean up all containers at the end
        cleanup_containers()
        cleanup_volumes()



def rollback_test(to_version: str, from_version: str, to_version_tag, from_image: Optional[str] = None,
                  to_image: Optional[str] = None):
    try:
        # Steps 1-3: Same as backwards_compatibility_test
        backwards_compatibility_test(from_version, to_version, None, from_image, to_image)

        # Step 4: Stop to_version container (but don't remove it)
        stop_marqo_container(to_version)

        # Step 5: Start from_version container, transferring state back
        start_marqo_from_version_container(from_version, None, from_image, transfer_state=f"marqo-{to_version}")

        # Step 6: Run tests
        run_tests("test", from_version, to_version, "http://localhost:8882")
    finally:
        # Stop the final container (but don't remove it yet)
        stop_marqo_container(from_version)
        # Clean up all containers at the end
        cleanup_containers()

def run_tests(mode: str, from_version: str, to_version: str, marqo_api: str):
    if mode == "prepare":
        tests = []
        for test in BaseTestCase.__subclasses__():
            if getattr(test, 'marqo_from_version', '0') <= from_version:
                tests.append(test)
        for test in tests:
            test.setUpClass()
            test_instance = test()
            test_instance.prepare()
            test.tearDownClass()
    elif mode == "test":
        pytest_args = [
            f"--from_version={from_version}",
            f"--to_version={to_version}",
            "-m", f"marqo_version"
        ]
        pytest.main(pytest_args)

def create_volume_for_marqo_version(version: str, volume_name: str):
    # Replace dots with underscores to format the volume name
    if volume_name is None:
        volume_name = get_volume_name_from_marqo_version(version)

    # Create the Docker volume using the constructed volume name
    try:
        subprocess.run(["docker", "volume", "create", "--name", volume_name], check=True)
        print(f"Successfully created volume: {volume_name}")
        volumes_to_cleanup.add(volume_name)
        return volume_name
    except subprocess.CalledProcessError as e:
        print(f"Failed to create volume: {volume_name}. Error: {e}")


    #TODO: Make it compatible for when you directly pass and image and no version is passed.
def get_volume_name_from_marqo_version(version):
    volume_name = f"marqo_{version.replace('.', '_')}_volume"
    return volume_name


def copy_state_from_container(
        from_version_volume: str, to_version_volume: str, image: str):
    cmd = ["docker", "run", "--rm", "-it", "--entrypoint=''",
           "-v", f"{from_version_volume}:/opt/vespa_old",
           "-v", f"{to_version_volume}:/opt/vespa/var",
           f"{image}",
           "sh", "-c", 'cd /opt/vespa_old && cp -a . /opt/vespa/var']
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully copied state from {from_version_volume} to {to_version_volume}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to copy state from {from_version_volume} to {to_version_volume}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marqo Testing Runner")
    parser.add_argument("--mode", choices=["backwards_compatibility", "rollback"], required=True)
    parser.add_argument("--from_version", required=True)
    parser.add_argument("--to_version", required=True)
    parser.add_argument("--to_version_tag", required=True)
    parser.add_argument("--from_image", required=False, default=None, help='Specify the source image')
    parser.add_argument("--to_image", required=False, default=None, help='Specify the target image')
    args = parser.parse_args()

    from_version = semver.VersionInfo.parse(args.from_version)
    to_version = semver.VersionInfo.parse(args.to_version)
    if from_version >= to_version:
        print("from_version should be less than to_version")
        sys.exit(0) # TODO: figure out if we should just quit.

    if args.mode == "backwards_compatibility":
        backwards_compatibility_test(args.from_version, args.to_version, args.to_version_tag, args.from_image, args.to_image)
    elif args.mode == "rollback":
        rollback_test(args.to_version, args.from_version, args.to_version_tag, args.from_image, args.to_image)
