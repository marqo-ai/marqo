from .service.model_manager.model_manager import ModelManager
from marqo_model_management_container.service.triton.triton_client import TritonClient
from marqo_model_management_container.core.settings import get_settings, Settings
from functools import lru_cache


class Config:
    """
    Basic configuration class for the Marqo Model Management Container.
    """
    def __init__(self, settings: Settings):
        self.triton_client = TritonClient(url=settings.triton_url)
        self.model_manager = ModelManager(model_base_dir=settings.model_base_dir, triton_client=self.triton_client)

    def _instantiate_triton_client(self):
        self.triton_client = TritonClient()


@lru_cache()
def get_config()-> Config:
    return Config(settings=get_settings())
