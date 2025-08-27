from marqo import logging
from marqo.core.inference.api import ModelManager
from marqo.inference.inference_cache.caching_inference import CachingInference
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.load_model import NativeModelManager
from marqo.inference.native_inference.local_inference import NativeInferenceLocal
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.inference.triton_inference.triton_inference import TritonInference
from marqo.inference.triton_inference.triton_inference_client import TritonInferenceClient

logger = logging.get_logger(__name__)


class Config:
    def __init__(self):
        # TODO load env vars to this class and expose them as properties
        self.model_manager: ModelManager = NativeModelManager()
        self.device_manager: DeviceManager = DeviceManager()

        marqo_mode = utils.read_env_vars_and_defaults(EnvVars.MARQO_MODE)
        marqo_mode = marqo_mode.upper() if marqo_mode else None
        triton_inference_url = utils.read_env_vars_and_defaults(EnvVars.TRITON_INFERENCE_URL)
        if not triton_inference_url and marqo_mode == "INFERENCE":
            raise ValueError(
                f"Environment variable {EnvVars.TRITON_INFERENCE_URL} is not set. "
                "Please set it to the URL of the Triton inference server."
            )

        triton_inference_client = TritonInferenceClient(triton_inference_url)
        self.triton_inference = TritonInference(triton_inference_client)