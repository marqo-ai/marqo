import httpx

from httpx import TimeoutException, ConnectError, NetworkError, HTTPError, HTTPStatusError
from .errors import ModelLoadingError
from marqo_model_management_container.errors.common import DependencyTimeoutError, DependencyUnavailableError, \
    DependencyBadGatewayError


class TritonClient:

    def __init__(self, url: str):
        self.url = url
        self.client = httpx.Client()

    def load_model(self, model_name: str):
        try:
            res = self.client.post(f"{self.url}/v2/repository/models/{model_name}/load", timeout=httpx.Timeout(5, read=30))
        except TimeoutException:
            raise DependencyTimeoutError('Triton timed out when trying to load model ')
        except (ConnectError, NetworkError) as e:
            raise DependencyUnavailableError('Triton is unavailable') from e
        except HTTPError as e:
            raise DependencyBadGatewayError('Triton is unavailable') from e

        try:
            res.raise_for_status()
        except HTTPStatusError as e:
            raise ModelLoadingError(f'Failed to load model. Original error: {res.json()["error"]}') from e

    def unload_model(self, model_name: str):
        try:
            res = self.client.post(f"{self.url}/v2/repository/models/{model_name}/unload",
                                   timeout=httpx.Timeout(5, read=30))
        except TimeoutException:
            raise DependencyTimeoutError('Triton timed out when trying to load model ')
        except (ConnectError, NetworkError) as e:
            raise DependencyUnavailableError('Triton is unavailable') from e
        except HTTPError as e:
            raise DependencyBadGatewayError('Triton is unavailable') from e

        try:
            res.raise_for_status()
        except HTTPStatusError as e:
            raise ModelLoadingError(f'Failed to unload model. Original error: {res.json()["error"]}') from e

    def get_loaded_models(self) -> list[str]:
        """
        There is an issue with check loaded model API in Triton server.

        Here is the issue link:
        https://github.com/triton-inference-server/server/issues/7066
        """
        raise NotImplementedError