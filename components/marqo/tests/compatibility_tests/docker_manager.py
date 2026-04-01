import base64
import time

import boto3
import docker
import requests
import semver
from botocore.exceptions import BotoCoreError, ClientError
from docker.errors import NotFound, APIError, ContainerError, ImageNotFound
from pathlib import Path
import tempfile
import sys
import subprocess

from tests.compatibility_tests.compatibility_test_logger import get_logger
import os
import yaml


MARQO_TRITON_VERSION = semver.VersionInfo.parse("2.25.0")
FILE_PATH = Path(__file__)


class DockerManager:
    def __init__(self):
        self.containers_to_cleanup = set()
        self.volumes_to_cleanup = set()
        self.docker_client = docker.from_env()
        self.logger = get_logger(__name__)
        self.marqo_transfer_state_version = semver.VersionInfo.parse("2.9.0")
        self.compose_file_dict = dict()

    def get_volume_name_from_marqo_version(self, version: str) -> str:
        """
        Generate a Docker volume name based on the Marqo version.

        Args:
            version (str): The Marqo version.

        Returns:
            str: A Docker-compatible volume name.
        """
        return f"marqo_{version.replace('.', '_')}_volume"

    def create_volume_for_marqo_version(self, version: str, volume_name: str = None) -> str:
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
            RuntimeError: If there is an error during the Docker volume creation process.
        """
        # Generate a volume name if not provided
        if volume_name is None:
            volume_name = self.get_volume_name_from_marqo_version(version)

        # Create the Docker volume
        try:
            self.logger.debug(f"Creating Docker volume: {volume_name}")
            self.docker_client.volumes.create(name=volume_name)
            self.volumes_to_cleanup.add(volume_name)
            self.logger.info(f"Successfully created volume: {volume_name}")
            return volume_name
        except APIError as e:
            self.logger.exception(f"Failed to create Docker volume: {volume_name}")
            raise RuntimeError(f"Failed to create volume: {volume_name}") from e

    def pull_remote_image_from_ecr(self, image_name: str):
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
        region = "us-east-1"

        try:
            # Get the ECR login password
            self.logger.debug("Retrieving ECR login credentials")
            ecr_client = boto3.client("ecr", region_name=region)
            auth_data = ecr_client.get_authorization_token()["authorizationData"][0]
            token = auth_data["authorizationToken"]
            decoded_token = base64.b64decode(token).decode('utf-8')

            username, password = decoded_token.split(":")
            # Get the ECR login password
            self.logger.debug(f"Logging into ECR registry: {ecr_registry}")
            resp = self.docker_client.login(username=username, password=password, registry=ecr_registry)

            # Pull the Docker image from ECR
            self.logger.debug(f"Pulling image: {image_name} from ECR registry")
            image = self.docker_client.images.pull(image_name, auth_config={'username': username, 'password': password})

            # Optionally retag the image locally to marqo-ai/marqo
            hash_part = image_name.split(":")[1] if ":" in image_name else image_name
            local_tag = f"marqo-ai/marqo:{hash_part}"  # it should now be called marqo-ai/marqo:sha-token or marqo-ai/marqo:GitHub.sha
            self.logger.info(f"Re-tagging image to: {local_tag}")
            image.tag(local_tag)

            return local_tag

        except (BotoCoreError, ClientError) as e:
            self.logger.exception(f"Failed to retrieve ECR authorization token: {str(e)}")
            raise RuntimeError("Failed to authenticate with ECR.") from e
        except docker.errors.APIError as e:
            self.logger.exception(f"Failed to pull or tag the image: {str(e)}")
            raise RuntimeError(
                f"Failed to pull or tag the Docker image '{image_name}' due to a Docker API error.") from e
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred while pulling the Docker image: {image_name} from ECR")
            raise RuntimeError(
                f"Failed to pull Docker image '{image_name}' from ECR due to an unexpected error.") from e

    def pull_marqo_image(self, image_name: str, source: str):
        """
        Pull the specified Marqo Docker image.

        Args:
            image_name (str): The identifier with which to pull the Docker image.
                              It can simply be the image name if pulling from DockerHub,
                              or it can be the image digest if pulling from ECR.
            source (str): The source from which to pull the image.
                          It can be either 'docker' for DockerHub or 'ECR' for Amazon ECR.

        Returns:
            str: The name of the pulled Docker image.

        Raises:
            Exception: If there is an error during the Docker image pull process.
        """
        try:
            if source == "docker":
                self.logger.info(f"Pulling image: {image_name} from DockerHub")
                self.pull_image_from_dockerhub(image_name)
                return image_name
            elif source == "ECR":
                return self.pull_remote_image_from_ecr(image_name)
            else:
                raise ValueError(f"Invalid source specified: {source}. Must be 'docker' or 'ECR'.")
        except docker.errors.APIError as e:
            self.logger.exception(f"Failed to pull image: {image_name} from source: {source}")
            raise Exception(f"Failed to pull Docker image: {image_name} from source: {source}. Error: {str(e)}") from e

    def start_marqo_container(
            self, version: str, to_api_image: str = None, to_inference_orchestrator_image: str = None,
            to_model_management_image: str = None
    ):
        if semver.VersionInfo.parse(version) < MARQO_TRITON_VERSION:
            self._start_marqo_container_before_2250(version)
        else:
            self._start_marqo_container_post_2250(version, to_api_image, to_inference_orchestrator_image, to_model_management_image)

    def _start_marqo_container_post_2250(
            self, version: str,
            api_image: str = None, inference_orchestrator_image: str = None, model_management_image: str = None
    ):
        if semver.VersionInfo.parse(version) < MARQO_TRITON_VERSION:
            raise ValueError(f"Version {version} is less than {MARQO_TRITON_VERSION}, cannot use this method.")

        os_ecr_name_space = "424082663841.dkr.ecr.us-east-1.amazonaws.com/marqoai"
        provided_images = [api_image, inference_orchestrator_image, model_management_image]
        num_provided = sum(img is not None for img in provided_images)

        if num_provided == 0:
            # No images provided → use defaults
            self.logger.info(f"Starting Marqo container with ECR images for version: {version}")
            api_image = f"{os_ecr_name_space}/api:{version}-cloud"
            inference_orchestrator_image = f"{os_ecr_name_space}/inference-orchestrator:{version}-cloud"
            model_management_image = f"{os_ecr_name_space}/model-management:{version}-cloud"
        elif num_provided == 3:
            # All provided → use as-is
            self.logger.info("Starting Marqo container with all custom images.")
        else:
            # Partial → configuration error
            raise ValueError(
                "Either all or none of api_image, inference_orchestrator_image, and "
                "model_management_image must be provided."
            )

        compose_file = os.path.join(FILE_PATH.resolve().parents[4], "compose.yaml")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as fp:
            with open(compose_file, 'r') as compose_fp:
                compose_content = yaml.safe_load(compose_fp)

            compose_content['services']['api']['image'] = api_image
            compose_content['services']['mioc']['image'] = inference_orchestrator_image
            compose_content['services']['mmc']['image'] = model_management_image

            yaml.dump(compose_content, fp)
            fp.flush()
            self.logger.info(f"Marqo backwards compatibility test using compose file: {compose_content}")
            self.compose_file_dict[version] = Path(fp.name).absolute()

            subprocess.run(
                [
                    "docker",
                    "compose",
                    "--profile",
                    "cpu",
                    "-f",
                    str(self.compose_file_dict[version]),
                    "up",
                    "-d",
                    "--force-recreate",
                    "--no-build", # To ignore any build instructions in the compose file
                    "--quiet-pull"
                ],
                check=True,
                timeout=60
            )

        for _ in range(10):
            try:
                response_1 = requests.get("http://localhost:8882/health")
                response_2 = requests.get("http://localhost:8884/healthz")
                if response_1.status_code == 200 and response_2.status_code == 200:
                    self.logger.info("Marqo server started successfully.")
                    return
                else:
                    time.sleep(5)
                    continue
            except requests.ConnectionError:
                pass
            time.sleep(5)

        raise RuntimeError(f"Marqo server failed to start within the expected time. Check images:"
                           f"{api_image}, {inference_orchestrator_image}, {model_management_image} for issues ")

    def _start_marqo_container_before_2250(self, version: str):
        """
        Start a Marqo container after pulling the required image and creating a volume.

        Args:
            version (str): The version of the Marqo container to start.
        """
        source = "docker"  # Always DockerHub for released images
        image_name = f"marqoai/marqo:{version}"
        container_name = f"marqo-{version}"
        self.logger.info(f"Starting Marqo container with version: {version}")

        # Pull the image
        self.pull_marqo_image(image_name, source)

        # Stop and remove the container if it exists
        try:
            container = self.docker_client.containers.get(container_name)
            self.logger.debug(f"Stopping and removing existing container: {container_name}")
            container.stop()
            container.remove()
        except NotFound:
            self.logger.warning(f"Container {container_name} does not exist. Skipping removal.")

        # Start the container
        try:
            self.logger.info(f"Starting container: {container_name} with image: {image_name}")
            container = self.docker_client.containers.run(
                image=image_name,
                name=container_name,
                detach=True,
                ports={"8882/tcp": 8882},
                environment={
                    "MARQO_ENABLE_BATCH_APIS": "TRUE",
                    "MARQO_MAX_CPU_MODEL_MEMORY": "4",
                    "VESPA_CONFIG_URL": "http://host.docker.internal:19071",
                    "VESPA_DOCUMENT_URL": "http://host.docker.internal:8080",
                    "VESPA_QUERY_URL": "http://host.docker.internal:8080",
                    "ZOOKEEPER_HOSTS": "host.docker.internal:2181"
                },
                extra_hosts={
                    "host.docker.internal": "host-gateway"
                }
            )
            log_stream = container.logs(stream=True, follow=True)
            self.containers_to_cleanup.add(container_name)

            # Wait for the Marqo service to start
            self.logger.debug("Waiting for Marqo to start...")
            while True:
                try:
                    response = requests.get("http://localhost:8882", verify=False)
                    if "Marqo" in response.text:
                        self.logger.info("Marqo server started successfully.")
                        break
                except requests.ConnectionError:
                    pass
                # Read and log container output
                try:
                    log_line = next(log_stream)
                    if log_line:
                        log_text = log_line.decode("utf-8").strip()
                        self.logger.debug(log_text)
                except StopIteration:
                    self.logger.warning("Log stream unexpectedly ended.")
                    break
                time.sleep(0.5)

            #Stop following logs after Marqo starts
            self.logger.debug("Stopped following docker logs")

        except APIError as e:
            raise RuntimeError(
                f"Failed to start Docker container {container_name}, with version: {version}."
            ) from e

    def stop_marqo_container(self, version: str):
        if semver.VersionInfo.parse(version) < MARQO_TRITON_VERSION:
            self._stop_marqo_container_before_2250(version)
        else:
            self._stop_marqo_container_post_2250(version)

    def _stop_marqo_container_post_2250(self, version: str):
        """
        Stop a Marqo container but don't remove it yet.

        Args:
            version (str): The version of the Marqo container to stop.

        Raises:
            RuntimeError: If there is an unexpected error during the container stop process.
        """
        container_name = f"marqo-{version}"
        self.logger.info(f"Stopping container with container name {container_name}")

        try:
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "--profile",
                    "cpu",
                    "-f",
                    str(self.compose_file_dict[version]),
                    "down",
                ],
                check=True,
                timeout=60
            )
            self.logger.debug(f"Successfully stopped container {container_name}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to stop container {container_name}") from e

    def _stop_marqo_container_before_2250(self, version: str):
        """
        Stop a Marqo container but don't remove it yet.

        Args:
            version (str): The version of the Marqo container to stop.

        Raises:
            RuntimeError: If there is an unexpected error during the container stop process.
        """
        container_name = f"marqo-{version}"
        self.logger.info(f"Stopping container with container name {container_name}")

        try:
            # Get the container by name
            container = self.docker_client.containers.get(container_name)

            # Stop the container
            container.stop(timeout=60)  # Increase the timeout from default 10 seconds to 60 seconds
            self.logger.debug(f"Successfully stopped container {container_name}")

        except NotFound:
            self.logger.warning(f"Warning: Container {container_name} not found. It may not be running.")
        except APIError as e:
            raise RuntimeError(f"Failed to stop container {container_name}") from e

    def cleanup_containers(self):
        """
        Remove all containers that were created during the test.

        This function iterates over the set of containers to clean up and attempts to remove each one using the Docker SDK.
        If a container cannot be removed, a warning message is logged.
        """

        for container_name in list(self.containers_to_cleanup):
            try:
                # Get the container by name
                container = self.docker_client.containers.get(container_name)

                # Remove the container
                container.remove(force=True)
                self.logger.debug(f"Successfully removed container {container_name}")
                self.containers_to_cleanup.remove(container_name)
            except NotFound:
                self.logger.warning(f"Warning: Container {container_name} not found. It may already have been removed.")
            except APIError as e:
                self.logger.warning(f"Warning: Failed to remove container {container_name}: {e}")

        # Clear any remaining entries in the cleanup set
        self.containers_to_cleanup.clear()

    def cleanup_volumes(self):
        """
        Remove all Docker volumes that were created during the test.

        This function iterates over the set of volumes to clean up and attempts to remove each one using the Docker SDK.
        If a volume cannot be removed, a warning message is logged.
        """

        for volume_name in list(self.volumes_to_cleanup):
            try:
                # Get the volume by name
                volume = self.docker_client.volumes.get(volume_name)

                # Remove the volume
                volume.remove(force=True)
                self.logger.info(f"Successfully removed volume {volume_name}")
                self.volumes_to_cleanup.remove(volume_name)
            except NotFound:
                self.logger.warning(f"Warning: Volume {volume_name} not found. It may already have been removed.")
            except APIError as e:
                self.logger.warning(f"Warning: Failed to remove volume {volume_name}: {e}")

        # Clear any remaining entries in the cleanup set
        self.volumes_to_cleanup.clear()

    def prepare_volume_for_rollback(self, target_version: str, source_volume: str, target_version_image_name: str = None,
                                    source="docker"):
        """
        Adjust the permissions of files or directories inside a Docker volume to be accessible
        by the specific user (vespa) and group (vespa) that the container expects to interact with.

        Args:
            target_version (str): The target version of the container.
            source_volume (str): The name of the source Docker volume.
            target_version_image_name (str): The name of the Docker image for the target version.
            source (str): The source to pull the image from ('docker' for Docker Hub or 'ECR').
        """
        self.logger.info(
            f"Preparing volume for rollback with target_version: {target_version}, source_volume: {source_volume}, target_version_image_name: {target_version_image_name}, source: {source}")

        # Determine the image to use
        if source == "docker":
            image_name = f"marqoai/marqo:{target_version}"
        else:
            image_name = target_version_image_name


        try:
            # Pull the image if not already available locally
            self.logger.info(f"Pulling image {image_name}...")
            self.docker_client.images.pull(image_name)
            self.logger.debug(f"Image {image_name} pulled successfully.")

            # Run a container with the provided image and the required command
            self.logger.info(f"Starting container to adjust permissions on volume {source_volume}...")
            container = self.docker_client.containers.run(
                image=image_name,
                name=f"prepare-rollback-{target_version}",
                command=["/bin/sh", "-c", "chown -R vespa:vespa /opt/vespa/var"],  # Using verified shell path
                volumes={source_volume: {'bind': '/opt/vespa/var', 'mode': 'rw'}},
                remove=True,
                detach=False
            )
            self.logger.info(f"Volume {source_volume} prepared successfully for rollback.")
        except APIError as e:
            raise RuntimeError(
                f"Failed to prepare volume {source_volume} for rollback using image {image_name}: {e}") from e

    def _check_image_exists_on_dockerhub(self, image_name: str) -> bool:
        """Check if a Docker image exists on DockerHub."""
        try:
            namespace, repo_tag = image_name.split("/")
            repo, tag = repo_tag.split(":")
        except ValueError:
            raise ValueError(f"Invalid image name format: {image_name}. Expected format: namespace/repo:tag")

        url = f"https://hub.docker.com/v2/repositories/{namespace}/{repo}/tags/{tag}"
        response = requests.get(url)
        return response.status_code == 200


    def pull_image_from_dockerhub(self, image_name: str):
        """
        Pull a Docker image using the Docker SDK.

        Starting from 2.17.0, the image name will have a "-cloud" suffix. E.g., "marqoai/marqo:2.17.0-cloud".

        Args:
            image_name (str): The name of the Docker image to pull.

        Raises:
            RuntimeError: If the image cannot be pulled.
        """

        variants = [
            image_name,
            image_name + "-cloud",
        ]

        available_variants = [v for v in variants if self._check_image_exists_on_dockerhub(v)]
            
        if not available_variants:
            raise RuntimeError(f"Image {image_name} and its variants = {variants} do not exist on DockerHub.")

        if len(available_variants) > 1:
            self.logger.warning(f"Multiple variants exist on DockerHub: {available_variants}. We will use the first one "
                                f"{available_variants[0]}.")

        target_image = available_variants[0]
        try:
            self.logger.debug(f"Trying to pull image: {target_image}")
            self.docker_client.images.pull(target_image)
            self.logger.info(f"Successfully pulled image: {target_image}")
            image = self.docker_client.images.get(target_image)
            # Retag to base image name
            image.tag(image_name.split(":")[0], image_name.split(":")[1])
        except APIError as e:
            raise RuntimeError(f"Failed to pull image {target_image} from DockerHub.")