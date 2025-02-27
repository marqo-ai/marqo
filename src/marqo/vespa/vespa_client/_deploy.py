"""
This file contains the `VespaDeployMixin` class, which provides methods to **deploy, update, and manage**
Vespa applications.

Functions Included:
--------------------
1. **Application Deployment**
   - `deploy_application()` → Deploys a new Vespa application.
   - `download_application()` → Downloads the currently deployed application.
   - `prepare()` → Prepares a deployment session.
   - `activate()` → Activates a deployment session.
   - `get_content_url()` → Returns the URL for a content path.
   - `list_contents()` → Lists all content paths.
   - `get_text_content()` → Fetches text content.
   - `get_binary_content()` → Fetches binary content.
   - `put_content()` → Uploads content.
   - `delete_content()` → Deletes content.


2. **Application Lifecycle Management**
   - `create_deployment_session()` → Creates a new Vespa deployment session.
   - `check_for_application_convergence()` → Checks if the application has fully deployed.
   - `wait_for_application_convergence()` → Waits for the application to be fully deployed.
   - `get_application_generation()` → Retrieves the current application generation.
   - `get_application_has_converged()` → Returns whether the application has finished deploying.
   - `get_vespa_version()` → Fetches the current Vespa version.


3. **Application Metadata**
   - `get_application_generation()` → Retrieves the current application generation.
   - `get_application_has_converged()` → Returns whether the application has finished deploying.
   - `get_vespa_version()` → Fetches the current Vespa version.
   - `get_metrics()` → Fetches application metrics.


This mixin **relies** on `http_client` and `config_url` from `VespaClientBase` to send HTTP requests.

Usage Example:
--------------
vespa = VespaClient(...)
vespa.deploy_application(application="path/to/app")
vespa.wait_for_application_convergence()
"""
import io
import os
import tarfile
import tempfile
import time
from json import JSONDecodeError
from typing import Tuple, TYPE_CHECKING, List, Union, Dict
from urllib.parse import urlparse

import httpcore
import httpx

import marqo.logging
from marqo.vespa.exceptions import (VespaError, VespaNotConvergedError)
from ..models.application_metrics import ApplicationMetrics

if TYPE_CHECKING:
    from ._client_base import VespaClientBase

logger = marqo.logging.get_logger(__name__)


class VespaDeployMixin:
    class _ConvergenceStatus:
        def __init__(self, current_generation: int, wanted_generation: int, converged: bool):
            self.current_generation = current_generation
            self.wanted_generation = wanted_generation
            self.converged = converged

    def deploy_application(self: "VespaClientBase", application: str, timeout: int = 60) -> None:
        """
        Deploy a Vespa application.
        Args:
            application: Path to the Vespa application root directory
            timeout: Timeout in seconds
        """
        endpoint = f'{self.config_url}/application/v2/tenant/default/prepareandactivate'

        gzip_stream = self._gzip_compress(application)

        response = self.http_client.post(
            endpoint, headers={
                'Content-Type': 'application/x-gzip'
            }, data=gzip_stream.read(), timeout=timeout
        )

        self._raise_for_status(response)

    def create_deployment_session(self: "VespaClientBase", check_for_application_convergence: bool = True) -> Tuple[str, str]:
        """
        Create a Vespa deployment session.
        Args:
            check_for_application_convergence: check for the application to converge before create a deployment session.

        Returns:
            Tuple[str, str]:
             - content_base_url is the base url for contents in this session
             - prepare_url is the url for prepare this session

        Please note that the session created is local in one config server and will be replicated to multiple servers
        via Zookeeper. Following requests should use content_base_url and prepare_url to make sure it can hit the right
        config server that this session is created on.
        """
        if check_for_application_convergence:
            self.check_for_application_convergence()

        res = self._create_deploy_session(self.http_client)
        content_base_url = res['content']
        prepare_url = res['prepared']
        return content_base_url, prepare_url

    def download_application(self: "VespaClientBase", check_for_application_convergence: bool = False) -> str:
        """
        Args:
            check_for_application_convergence: check for the application to converge before downloading.

        Download the Vespa application. If wait_for_application_convergence is True, this method will wait for the
        application to converge before downloading.

        Application download happens in two steps:
        1. Create a session
        2. Download the application using the session ID

        The session created in step 1 is local to the config node that created it and subsequent requests will return a
        404 error if the request is routed to a different config node. This method attempts to ensure the same config
        node is used for all requests by using the same httpx client for all requests. However, this is not guaranteed.

        The likelihood of getting a 404 error is further reduced if config cluster uses a load balancer with sticky
        sessions. Since we are using a single httpx client, cookie-based sticky sessions will work with this
        implementation.

        Returns:
            Path to the downloaded application
        """
        if check_for_application_convergence:
            self.check_for_application_convergence()

        with httpx.Client() as httpx_client:
            session_id = self._create_deploy_session(httpx_client)['session-id']
            return self._download_application(session_id, httpx_client)

    def check_for_application_convergence(self: "VespaClientBase") -> None:
        """
        Check if the Vespa application has converged and raise an exception if it has not.

        Raises:
            VespaNotConvergedError: If the application has not converged
        """
        if not self.get_application_has_converged():
            raise VespaNotConvergedError('Vespa application has not converged')

    def get_application_generation(self: "VespaClientBase") -> int:
        """
        Get the current application generation.

        Returns:
            Current application generation
        """
        return self._get_convergence_status().current_generation

    def get_application_has_converged(self: "VespaClientBase") -> bool:
        """
        Get the current application convergence status.

        Application convergence is asynchronous following a deployment. This method can be used to check if the
        application has converged after a deployment.

        Returns:
            True if the application is converged, False otherwise
        """
        return self._get_convergence_status().converged

    def wait_for_application_convergence(self: "VespaClientBase", timeout: float = 120) -> None:
        """
        Wait for Vespa application to converge, checking every second.

        Args:
            timeout: Timeout in seconds
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.get_application_has_converged():
                    return
                else:
                    logger.debug('Waiting for Vespa application to converge')
                    time.sleep(1)
            # TODO Find out what exceptions is raised here
            except (httpx.TimeoutException, httpcore.TimeoutException):
                logger.error("Marqo timed out waiting for Vespa application to converge. Will retry.")

        raise VespaError(f"Vespa application did not converge within {timeout} seconds. "
                         f"The convergence status is {self._get_convergence_status()}")

    def get_vespa_version(self: "VespaClientBase") -> str:
        endpoint = f'{self.config_url}/state/v1/version'

        response = self.http_client.get(endpoint)

        self._raise_for_status(response)

        return response.json()['version']

    def prepare(self, prepare_url: str, timeout: int):
        response = self.http_client.put(prepare_url, timeout=timeout)

        self._raise_for_status(response)

        return response.json()

    def activate(self, activate_url: str, timeout: int):
        response = self.http_client.put(activate_url, timeout=timeout)

        self._raise_for_status(response)

        return response.json()

    def get_content_url(self: "VespaClientBase", content_base_url: str, *paths: str) -> str:
        return f'{content_base_url}{"/".join(paths)}'

    def list_contents(self: "VespaClientBase", content_base_url: str) -> List[str]:
        endpoint = f'{content_base_url}?recursive=true'

        response = self.http_client.get(endpoint)

        self._raise_for_status(response)

        return response.json()

    def get_text_content(self: "VespaClientBase", content_base_url: str, *path: str) -> str:
        endpoint = f'{content_base_url}{"/".join(path)}'

        response = self.http_client.get(endpoint)

        self._raise_for_status(response)

        return response.text

    def get_binary_content(self: "VespaClientBase", content_base_url: str, *path: str) -> bytes:
        endpoint = f'{content_base_url}{"/".join(path)}'

        response = self.http_client.get(endpoint)

        self._raise_for_status(response)

        return response.content

    def put_content(self: "VespaClientBase", content_base_url: str, content: Union[str, bytes], *path: str) -> None:
        endpoint = f'{content_base_url}{"/".join(path)}'

        response = self.http_client.put(endpoint, content=content)

        self._raise_for_status(response)

    def delete_content(self: "VespaClientBase", content_base_url: str, *path: str) -> None:
        endpoint = f'{content_base_url}{"/".join(path)}'

        response = self.http_client.delete(endpoint)

        self._raise_for_status(response)

    def _gzip_compress(self: "VespaClientBase", directory: str) -> io.BytesIO:
        """
        Gzip all files in the given directory and return an in-memory byte buffer.
        """
        byte_stream = io.BytesIO()
        with tarfile.open(fileobj=byte_stream, mode='w:gz') as tar:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, directory)  # archive name should be relative
                    tar.add(file_path, arcname=arcname)

        byte_stream.seek(0)
        return byte_stream

    def _create_deploy_session(self: "VespaClientBase", httpx_client: httpx.Client) -> Dict:
        endpoint = f'{self.config_url}/application/v2/tenant/default/session?from=' \
                   f'{self.config_url}/application/v2/tenant/default/application/default/environment' \
                   f'/default/region/default/instance/default'

        response = httpx_client.post(endpoint)

        self._raise_for_status(response)

        return response.json()

    def _download_application(self: "VespaClientBase", session_id: int, httpx_client: httpx.Client) -> str:
        endpoint = f'{self.config_url}/application/v2/tenant/default/session/{session_id}/content/?recursive=true'

        response = httpx_client.get(endpoint)

        self._raise_for_status(response)

        urls = response.json()

        logger.debug(f'URLs: {urls}')

        def is_file(url: str) -> bool:
            last_component = urlparse(url).path.split('/')[-1]
            return '.' in last_component

        temp_dir = tempfile.mkdtemp()

        logger.debug(f'Downloading application to {temp_dir}')

        for url in urls:
            if not is_file(url):
                continue  # Skip directories

            # Parse the URL
            parsed = urlparse(url)
            path_parts = parsed.path.split('/')

            # Find the index for 'content' and use it as root
            content_index = path_parts.index('content')
            rel_path = os.path.join(*path_parts[content_index + 1:])
            abs_path = os.path.join(temp_dir, rel_path)

            # Ensure directory exists before downloading
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            response = httpx_client.get(url)
            self._raise_for_status(response)

            # Save the downloaded content
            with open(abs_path, 'wb') as f:
                f.write(response.content)

        return temp_dir

    def _get_convergence_status(self: "VespaClientBase") -> "_ConvergenceStatus":
        endpoint = f'{self.config_url}/application/v2/tenant/default/application/default/environment/default/region/' \
                   f'default/instance/default/serviceconverge'

        response = self.http_client.get(endpoint)

        self._raise_for_status(response)

        try:
            json = response.json()
            return self._ConvergenceStatus(
                current_generation=json['currentGeneration'],
                wanted_generation=json['wantedGeneration'],
                converged=json['converged']
            )

        except (JSONDecodeError, KeyError) as e:
            raise VespaError(f'Unexpected response: {response.text}') from e

    def get_metrics(self) -> ApplicationMetrics:
        """
        Get metrics for every service on all nodes for the application.

        See https://docs.vespa.ai/en/operations-selfhosted/monitoring.html#metrics-v2-values for more information.

        Returns:
             A selected set of metrics for every service on all nodes for the application
        """
        try:
            resp = self.http_client.get(f'{self.document_url}/metrics/v2/values')
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)

        return ApplicationMetrics(**resp.json())