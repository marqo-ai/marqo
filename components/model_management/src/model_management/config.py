from model_management.core.settings import Settings, get_settings
from model_management.services.triton.triton_client import TritonClient

from .services.model_manager.model_manager import ModelManager


class Config:
    """
    Basic configuration class for the Marqo Model Management Container.
    """

    def __init__(self, settings: Settings):
        self.triton_client = TritonClient(url=settings.marqo_triton_rest_url)
        self.model_manager = ModelManager(
            marqo_model_cache_path=settings.marqo_model_cache_path,
            triton_client=self.triton_client,
        )


_config = Config(get_settings())


def get_config() -> Config:
    return _config
