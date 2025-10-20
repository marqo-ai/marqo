import httpx
from httpx import ConnectError, HTTPStatusError, NetworkError, TimeoutException
from model_management.services.errors import (
    TritonCommunicationError,
    TritonModelLoadError,
)


class TritonClient:
    def __init__(self, url: str):
        self.url = url
        self.client = httpx.Client()

    def load_model(self, model_name: str):
        try:
            res = self.client.post(
                f"{self.url}/v2/repository/models/{model_name}/load",
                timeout=httpx.Timeout(5, read=30),
            )
        except TimeoutException:
            raise TritonCommunicationError(
                "Triton timed out when trying to load model "
            )
        except (ConnectError, NetworkError) as e:
            raise TritonCommunicationError("Triton is unavailable") from e

        try:
            res.raise_for_status()
        except HTTPStatusError as e:
            raise TritonModelLoadError(
                f"Failed to load model. Original error: {res.json().get('error')}"
            ) from e

    def unload_model(self, model_name: str):
        try:
            res = self.client.post(
                f"{self.url}/v2/repository/models/{model_name}/unload",
                timeout=httpx.Timeout(5, read=30),
            )
        except TimeoutException:
            raise TritonCommunicationError(
                "Triton timed out when trying to unload model "
            )
        except (ConnectError, NetworkError) as e:
            raise TritonCommunicationError("Triton is unavailable") from e

        try:
            res.raise_for_status()
        except HTTPStatusError as e:
            raise TritonModelLoadError(
                f'Failed to unload model "{model_name}". '
                f"Original error: {res.json().get('error')}"
            ) from e

    def get_loaded_models(self) -> list[str]:
        """
        There is an issue with check loaded model API in Triton server.

        Here is the issue link:
        https://github.com/triton-inference-server/server/issues/7066
        """
        raise NotImplementedError
