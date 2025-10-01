from marqo_model_management_container.core.settings import get_settings, Settings
from marqo_model_management_container.service.triton.triton_client import TritonClient
from .service.model_manager.model_manager import ModelManager


class Config:
    """
    Basic configuration class for the Marqo Model Management Container.
    """

    def __init__(self, settings: Settings):
        self.triton_client = TritonClient(url=settings.triton_url)
        self.model_manager = ModelManager(model_base_dir=settings.model_base_dir, triton_client=self.triton_client)

    def _instantiate_triton_client(self):
        self.triton_client = TritonClient()


_config = Config


def get_config() -> Config:
    return _config(get_settings())
