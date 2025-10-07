from marqo_inference_container.core.settings import get_settings, Settings
from .core.logging import get_logger
from .services.inference_cache.caching_inference import CachingInference
from .services.triton_inference.model_manager.model_management_client import ModelManagementClient
from .services.triton_inference.triton.triton_grpc_client import TritonGRPCClient
from .services.triton_inference.triton_inference import TritonInference

settings = get_settings()

logger = get_logger(__name__)


class Config:
    def __init__(self, settings: Settings):
        # TODO load env vars to this class and expose them as properties
        self.triton_client: TritonGRPCClient = self._instantiate_triton_grpc_client()
        self.model_management_client: ModelManagementClient = self._instantiate_model_management_client()
        inference = TritonInference(
            model_management_client=self.model_management_client, triton_client=self.triton_client,
        )

        # initialise inference cache
        inference_cache_size = settings.marqo_inference_cache_size
        if inference_cache_size > 0:  # enable inference cache
            inference_cache_type = settings.marqo_inference_cache_type
            self.local_inference = CachingInference(
                delegate=inference,
                cache_size=inference_cache_size,
                cache_type=inference_cache_type
            )
        else:
            self.local_inference = inference

    def _instantiate_triton_grpc_client(self) -> TritonGRPCClient:
        triton_url = settings.marqo_triton_url
        return TritonGRPCClient(url=triton_url, channel_args=settings.channel_args)

    def _instantiate_model_management_client(self) -> ModelManagementClient:
        model_management_url = settings.marqo_model_management_container_url
        return ModelManagementClient(url=model_management_url)


_config = Config(settings=get_settings())


def get_config() -> Config:
    return _config