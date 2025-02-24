"""
This file contains the `VespaUtilsMixin` class, which provides **general utility methods**
that are used across different functionalities.

Functions Included:
--------------------
1. **Content Management**
   - `get_content_url()` → Constructs a URL for accessing Vespa content.
   - `list_contents()` → Lists available files and directories in Vespa.
   - `get_text_content()` → Retrieves text-based content.
   - `get_binary_content()` → Retrieves binary data.
   - `put_content()` → Uploads content to Vespa.
   - `delete_content()` → Deletes content from Vespa.

2. **Resource Management**
   - `close()` → Closes the HTTP client and releases resources.

3. **Compression & Deployment Helpers**
   - `_gzip_compress()` → Compresses directories into a Gzip file for deployment.
   - `_create_deploy_session()` → Creates a Vespa deployment session.
   - `_download_application()` → Downloads the current deployment files.

4. **URL & Query Helpers**
   - `_add_query_params()` → Appends query parameters to a URL.

This mixin is **not tied to search, indexing, or deployment** but supports general operations across all modules.

Usage Example:
--------------
vespa = VespaClient(...)
vespa.close()  # Clean up resources
"""

import io
import os
import tarfile
import tempfile
from json import JSONDecodeError
from typing import Dict, List, Union, TYPE_CHECKING
from urllib.parse import urlparse

import httpx

import marqo.logging
from marqo.vespa.exceptions import VespaError

if TYPE_CHECKING:
    from ._client_base import VespaClientBase

logger = marqo.logging.get_logger(__name__)


class VespaUtilsMixin:
    class _ConvergenceStatus:
        def __init__(self, current_generation: int, wanted_generation: int, converged: bool):
            self.current_generation = current_generation
            self.wanted_generation = wanted_generation
            self.converged = converged

    def close(self: "VespaClientBase"):
        """
        Close the VespaClient object.
        """
        self.http_client.close()

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

    def _add_query_params(self: "VespaClientBase", url: str, query_params: Dict[str, str]) -> str:
        if not query_params:
            return url

        query_string = '&'.join([f'{key}={value}' for key, value in query_params.items() if value])
        return f'{url.strip("?")}?{query_string}'

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