import httpx

from marqo import logging
from marqo.core.inference.api import ModelManager, ModelError

logger = logging.get_logger(__name__)


class ModelManagerClient(ModelManager):

    def __init__(self, base_url: str, timeout: float = 10.0):
        """
        Args:
            base_url (str): The base URL of the remote inference service.
            timeout (float): Timeout for HTTP requests in seconds. 10 seconds should be enough to eject a model
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def get_loaded_models(self) -> dict:
        """
        Retrieves the loaded models from the remote inference service.

        Returns:
            dict: A dictionary of loaded models.

        Raises:
            httpx.HTTPError: If an HTTP error occurs.
            Exception: For any other exceptions.
        """
        try:
            response = self.client.get("/models")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as http_err:
            if http_err.response.status_code == 400:
                error_detail = http_err.response.json().get('detail', 'Bad Request')
                raise ModelError(f"Failed to retrieve loaded models: {error_detail}") from http_err
            else:
                # Re-raise the original HTTPStatusError for other status codes
                raise

    def eject_model(self, model_name: str, device: str) -> dict:
        """
        Ejects a specified model from the given device on the remote inference service.

        Args:
            model_name (str): The name of the model to eject.
            device (str): The device from which to eject the model.

        Returns:
            dict: A dictionary containing the result of the ejection.

        Raises:
            ModelError: If an error occurs while ejecting the model.
            Exception: For any other exceptions.
        """
        params = {
            "model_name": model_name,
            "model_device": device
        }
        try:
            response = self.client.delete("/models", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as http_err:
            if http_err.response.status_code == 400:
                error_detail = http_err.response.json().get('detail', 'Bad Request')
                raise ModelError(
                    f"Failed to eject model '{model_name}' from device '{device}': {error_detail}") from http_err
            else:
                # Re-raise the original HTTPStatusError for other status codes
                raise
