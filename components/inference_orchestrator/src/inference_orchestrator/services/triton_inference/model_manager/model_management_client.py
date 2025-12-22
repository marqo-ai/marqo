import httpx
from httpx import ConnectError, HTTPStatusError, NetworkError, TimeoutException

from inference_orchestrator.services.errors import (
    ModelManagementServiceUnavailableError,
    TritonModelLoadError,
)


class ModelManagementClient:
    """
    A class to communicate with marqo model management container to load/eject models in Triton Inference Server.
    """

    def __init__(self, url: str):
        self.url = url
        self.client = httpx.Client()

    def load_model(self, model_properties: dict, timeout: float = 600):
        """
        Load a model into Triton Inference Server via the model management container.

        Args:
            model_properties (dict): The properties of the model to be loaded. This is the TritonModelProperties schema,
                not the EmbeddingModelConfig schema.
            timeout (float): The timeout for the request in seconds. Default is 600 seconds to allow
                for long model downloads.
        """
        pay_load = {
            "tritonModelProperties": model_properties,
        }

        try:
            res = self.client.post(
                url=f"{self.url}/v1/models/load", json=pay_load, timeout=timeout
            )
        except (ConnectError, NetworkError) as e:
            raise ModelManagementServiceUnavailableError(
                f"Model management container is unavailable. Original error: {e}"
            ) from e
        except TimeoutException as e:
            raise ModelManagementServiceUnavailableError(
                f"Request to model management container timed out. Original error: {e}. The service is either "
                f"taking too long to respond or is unavailable. Possible causes include very long model downloading time "
            ) from e

        try:
            res.raise_for_status()
        except HTTPStatusError as e:
            raise TritonModelLoadError(
                f"Failed to load the Triton model with properties {model_properties}. Original error: {e}. "
            ) from e

    def unload_model(
        self, model_name: str, remove_files: bool = False, timeout: float = 60
    ):
        try:
            res = self.client.post(
                url=f"{self.url}/v1/models/{model_name}/unload?remove-files={remove_files}",
                timeout=timeout,
            )
        except (ConnectError, NetworkError) as e:
            raise ModelManagementServiceUnavailableError(
                f"Model management container is unavailable. Original error: {e} "
            ) from e
        except TimeoutException as e:
            raise ModelManagementServiceUnavailableError(
                f"Request to model management container timed out. Original error: {e} "
            ) from e

        try:
            res.raise_for_status()
        except HTTPStatusError as e:
            raise TritonModelLoadError(
                f"Failed to unload the Triton model: {model_name}. Original error: {e} "
            ) from e
