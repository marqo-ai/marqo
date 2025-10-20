from inference_orchestrator.core.settings import Settings, get_settings

from .core.logging import get_logger
from .services.inference_cache.caching_inference import CachingInference
from .services.triton_inference.model_manager.model_management_client import (
    ModelManagementClient,
)
from .services.triton_inference.triton.triton_grpc_client import TritonGRPCClient
from .services.triton_inference.triton_inference import TritonInference

logger = get_logger(__name__)


class Config:
    def __init__(self, settings: Settings):
        self._settings = settings
        self.triton_client: TritonGRPCClient = self._instantiate_triton_grpc_client()
        self.model_management_client: ModelManagementClient = (
            self._instantiate_model_management_client()
        )
        self.inference = self._instantiate_inference()

    def _instantiate_triton_grpc_client(self) -> TritonGRPCClient:
        triton_url = self._settings.marqo_triton_url
        return TritonGRPCClient(
            url=triton_url, triton_channel_args=self._settings.channel_args
        )

    def _instantiate_model_management_client(self) -> ModelManagementClient:
        model_management_url = self._settings.marqo_model_management_container_url
        return ModelManagementClient(url=model_management_url)

    def _instantiate_inference(self):
        inference = TritonInference(
            model_management_client=self.model_management_client,
            triton_client=self.triton_client,
        )

        # initialise inference cache
        inference_cache_size = self._settings.marqo_inference_cache_size
        if inference_cache_size > 0:  # enable inference cache
            inference_cache_type = self._settings.marqo_inference_cache_type
            return CachingInference(
                delegate=inference,
                cache_size=inference_cache_size,
                cache_type=inference_cache_type,
            )
        else:
            return inference


_config = Config(settings=get_settings())


def get_config() -> Config:
    return _config
