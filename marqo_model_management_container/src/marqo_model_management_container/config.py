from marqo_model_management_container.core.settings import get_settings, Settings
from marqo_model_management_container.services.triton.triton_client import TritonClient
from .services.model_manager.model_manager import ModelManager


class Config:
    """
    Basic configuration class for the Marqo Model Management Container.
    """

    def __init__(self, settings: Settings):
        self.triton_client = TritonClient(url=settings.triton_url)
        self.model_manager = ModelManager(model_base_dir=settings.model_base_dir, triton_client=self.triton_client)


_config = Config(get_settings())


def get_config() -> Config:
    return _config
