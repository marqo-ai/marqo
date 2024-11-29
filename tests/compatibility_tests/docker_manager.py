import time

import boto3
import docker
import requests
from botocore.exceptions import BotoCoreError, ClientError
from docker.errors import NotFound, APIError

from compatibility_test_logger import get_logger
from tests.compatibility_tests.compatibility_test_runner import marqo_transfer_state_version


class DockerManager:
    def __init__(self):
        self.containers_to_cleanup = set()
        self.volumes_to_cleanup = set()
        self.docker_client = docker.from_env()
        self.logger = get_logger(__name__)

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
            self.logger.info(f"Creating Docker volume: {volume_name}")
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
            self.logger.info("Retrieving ECR login credentials")
            ecr_client = boto3.client("ecr", region_name=region)
            login_password = ecr_client.get_authorization_token()["authorizationData"][0]["authorizationToken"]

            # Get the ECR login password
            self.logger.info(f"Logging into ECR registry: {ecr_registry}")
            self.docker_client.login(username="AWS", password=login_password, registry=ecr_registry)

            # Pull the Docker image from ECR
            self.logger.info(f"Pulling image: {image_name}")
            image = self.docker_client.images.pull(image_name)

            # Optionally retag the image locally to marqo-ai/marqo
            hash_part = image_name.split(":")[1] if ":" in image_name else image_name
            local_tag = f"marqo-ai/marqo:{hash_part}"  # it should now be called marqo-ai/marqo:sha-token or marqo-ai/marqo:github.sha
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
                self.docker_client.images.pull(image_name)
                return image_name
            elif source == "ECR":
                return self.pull_remote_image_from_ecr(image_name)
            else:
                raise ValueError(f"Invalid source specified: {source}. Must be 'docker' or 'ECR'.")
        except docker.errors.APIError as e:
            self.logger.exception(f"Failed to pull image: {image_name} from source: {source}")
            raise Exception(f"Failed to pull Docker image: {image_name} from source: {source}. Error: {str(e)}") from e


    def start_marqo_container(self, version: str, volume_name: str):
        """
        Start a Marqo container after pulling the required image and creating a volume.

        Args:
            version (str): The version of the Marqo container to start.
            volume_name: The volume to use for the container.
        """
        source = "docker"  # Always DockerHub for released images
        image_name = f"marqoai/marqo:{version}"
        container_name = f"marqo-{version}"
        self.logger.info(f"Starting Marqo container with version: {version}, volume_name: {volume_name}, source: {source}")

        # Pull the image
        self.pull_marqo_image(image_name, source)

        # Stop and remove the container if it exists
        try:
            container = self.docker_client.containers.get(container_name)
            self.logger.info(f"Stopping and removing existing container: {container_name}")
            container.stop()
            container.remove()
        except NotFound:
            self.logger.info(f"Container {container_name} does not exist. Skipping removal.")

        # Create volume and configure mounting
        volume_name = self.create_volume_for_marqo_version(version, volume_name)
        if version >= marqo_transfer_state_version:
            volume_mount_path = "/opt/vespa/var"
        else:
            volume_mount_path = "/opt/vespa"
        self.logger.info(f"Mounting volume: {volume_name} to {volume_mount_path}")

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
                    "MARQO_MAX_CPU_MODEL_MEMORY": "1.6"
                },
                volumes={volume_name: {"bind": volume_mount_path, "mode": "rw"}}
            )
            self.containers_to_cleanup.add(container_name)

            # Wait for the Marqo service to start
            self.logger.info("Waiting for Marqo to start...")
            while True:
                try:
                    response = requests.get("http://localhost:8882", verify=False)
                    if "Marqo" in response.text:
                        self.logger.info("Marqo started successfully.")
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(0.5)

        except APIError as e:
            self.logger.exception(f"Failed to start container: {container_name}")
            raise RuntimeError(
                f"Failed to start Docker container {container_name}, with version: {version}, and volume_name: {volume_name}"
            ) from e

        # Show running containers
        self.logger.info("Listing running containers...")
        for container in self.docker_client.containers.list():
            self.logger.info(f"Container: {container.name} | Status: {container.status}")
