from marqo import logging
from marqo.core.inference.api import ModelManager
from marqo.inference.inference_cache.caching_inference import CachingInference
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.triton_inference.model_manager.model_manager import TritonModelManager
from marqo.inference.native_inference.local_inference import NativeInferenceLocal
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.inference.triton_inference.triton.triton_grpc_client import TritonGRPCClient
from marqo.inference.triton_inference.triton.channel_args import ChannelArgs
import json


logger = logging.get_logger(__name__)


class Config:
    def __init__(self):
        # TODO load env vars to this class and expose them as properties
        triton_grpc_client: TritonGRPCClient = self._instantiate_triton_grpc_client()
        self.model_manager: TritonModelManager = TritonModelManager(triton_grpc_client)
        inference = NativeInferenceLocal(model_manager=self.model_manager, triton_grpc_client=triton_grpc_client)

        # initialise inference cache
        inference_cache_size = utils.read_env_vars_and_defaults_ints(EnvVars.MARQO_INFERENCE_CACHE_SIZE)
        if inference_cache_size > 0:  # enable inference cache
            inference_cache_type = utils.read_env_vars_and_defaults(EnvVars.MARQO_INFERENCE_CACHE_TYPE)
            self.local_inference = CachingInference(
                delegate=inference,
                cache_size=inference_cache_size,
                cache_type=inference_cache_type
            )
        else:
            self.local_inference = inference

    def _instantiate_triton_grpc_client(self) -> TritonGRPCClient:
        triton_url = utils.read_env_vars_and_defaults(EnvVars.MARQO_TRITON_URL)
        if not triton_url:
            raise ValueError(f"{EnvVars.MARQO_TRITON_URL} is not set, cannot instantiate TritonGRPCClient")
        channel_args = utils.read_env_vars_and_defaults(EnvVars.MARQO_TRITON_GRPC_CLIENT_CONFIGS)
        if channel_args is None:
            channel_args = {}
        else:
            try:
                channel_args = json.loads(channel_args)
                if not isinstance(channel_args, dict):
                    raise ValueError(f"{EnvVars.MARQO_TRITON_GRPC_CLIENT_CONFIGS} must be a JSON object")
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse {EnvVars.MARQO_TRITON_GRPC_CLIENT_CONFIGS}: {str(e)}") from e
        channel_args = ChannelArgs(**channel_args)
        return TritonGRPCClient(url=triton_url, channel_args=channel_args)