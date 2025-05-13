import time

from marqo import logging
from marqo.core.inference.api import ModelManager, Inference, InferenceRequest, InferenceResult
from marqo.inference.inference_cache.caching_inference import CachingInference
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.load_model import NativeModelManager, load_model

# TODO move device manager to native_inference
from marqo.inference.native_inference.local_inference import NativeInferenceLocal
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars

import numpy as np
import hashlib


logger = logging.get_logger(__name__)


class Config:
    def __init__(self):
        # TODO load env vars to this class and expose them as properties
        self.model_manager: ModelManager = NativeModelManager()
        self.device_manager: DeviceManager = DeviceManager()

        if utils.read_env_vars_and_defaults("MARQO_INFERENCE_STUB") == "TRUE":
            stub_latency_ms = utils.read_env_vars_and_defaults_ints("MARQO_INFERENCE_STUB_LATENCY_MS") or 10
            inference = RandomInferenceStub(device_manager=self.device_manager, stub_latency_ms=stub_latency_ms)
            logger.info(f"RandomInferenceStub with latency {stub_latency_ms} ms is initialised")
        else:
            inference = NativeInferenceLocal(device_manager=self.device_manager)
            logger.info(f"NativeInferenceLocal is initialised")

        # initialise inference cache
        inference_cache_size = utils.read_env_vars_and_defaults_ints(EnvVars.MARQO_INFERENCE_SERVER_CACHE_SIZE)
        if inference_cache_size > 0:
            inference_cache_type = utils.read_env_vars_and_defaults(EnvVars.MARQO_INFERENCE_SERVER_CACHE_TYPE)
            inference = CachingInference(
                delegate=inference,
                cache_size=inference_cache_size,
                cache_type=inference_cache_type
            )

        self.local_inference = inference


class RandomInferenceStub(Inference):

    def __init__(self, device_manager: DeviceManager, stub_latency_ms: int):
        self.device_manager = device_manager
        self.stub_latency_ms = stub_latency_ms

    def random_ndarray(self, content: str, dimension: int, normalise: bool):
        seed = int(hashlib.sha256(content.encode("utf-8")).hexdigest(), 16) % 2**32
        rng = np.random.default_rng(seed)
        arr = rng.random((dimension, ), dtype=np.float32)
        if not normalise:
            return arr
        else:
            norm = np.linalg.norm(arr)
            return arr / norm

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        model = load_model(
            model_name=request.model_config.model_name,
            model_properties=request.model_config.model_properties,
            model_auth=request.model_config.model_auth,
            device=self.device_manager.pick_and_validate_device(device=request.device)
        )
        dimension = model.model_properties.dimensions

        now = time.perf_counter()
        result = InferenceResult(result=[[(content, self.random_ndarray(
            content, dimension, request.model_config.normalize_embeddings))] for content in request.contents])
        elapsed_ms = (time.perf_counter() - now) * 1000
        time.sleep((self.stub_latency_ms - elapsed_ms) / 1000)

        return result
