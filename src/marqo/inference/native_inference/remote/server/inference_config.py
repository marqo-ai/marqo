from marqo import logging
from marqo.core.inference.api import ModelManager
from marqo.inference.inference_cache.caching_inference import CachingInference
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.load_model import NativeModelManager
from marqo.inference.native_inference.local_inference import NativeInferenceLocal
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars

logger = logging.get_logger(__name__)


class Config:
    def __init__(self):
        # TODO load env vars to this class and expose them as properties
        self.model_manager: ModelManager = NativeModelManager()
        self.device_manager: DeviceManager = DeviceManager()
        inference = NativeInferenceLocal(device_manager=self.device_manager)

        # initialise inference cache
        inference_cache_size = utils.read_env_vars_and_defaults_ints(EnvVars.MARQO_INFERENCE_CACHE_SIZE)
        if inference_cache_size > 0:
            inference_cache_type = utils.read_env_vars_and_defaults(EnvVars.MARQO_INFERENCE_CACHE_TYPE)
            inference = CachingInference(
                delegate=inference,
                cache_size=inference_cache_size,
                cache_type=inference_cache_type
            )

        self.local_inference = inference
