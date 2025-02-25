"""
This file contains the `VespaDeployMixin` class, which provides methods to **deploy, update, and manage**
Vespa applications.

Functions Included:
--------------------
1. **Application Deployment**
   - `deploy_application()` → Deploys a new Vespa application.
   - `download_application()` → Downloads the currently deployed application.

2. **Application Lifecycle Management**
   - `create_deployment_session()` → Creates a new Vespa deployment session.
   - `check_for_application_convergence()` → Checks if the application has fully deployed.
   - `wait_for_application_convergence()` → Waits for the application to be fully deployed.

3. **Application Metadata**
   - `get_application_generation()` → Retrieves the current application generation.
   - `get_application_has_converged()` → Returns whether the application has finished deploying.
   - `get_vespa_version()` → Fetches the current Vespa version.

This mixin **relies** on `http_client` and `config_url` from `VespaClientBase` to send HTTP requests.

Usage Example:
--------------
vespa = VespaClient(...)
vespa.deploy_application(application="path/to/app")
vespa.wait_for_application_convergence()
"""

import time
from typing import Tuple, TYPE_CHECKING

import httpcore
import httpx

import marqo.logging
from marqo.vespa.exceptions import (VespaError, VespaNotConvergedError)
if TYPE_CHECKING:
    from ._client_base import VespaClientBase

logger = marqo.logging.get_logger(__name__)


class VespaDeployMixin:
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