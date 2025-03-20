from typing import Optional

import numpy as np

from marqo.core.inference.api import Inference, InferenceRequest, InferenceResult, InferenceError

# TODO move device manager to native_inference
from marqo.core.inference.device_manager import DeviceManager


class Config:
    def __init__(
            self,
    ) -> None:

        # TODO [Refactoring device logic] deprecate default_device since it's not used
        # self.default_device = default_device if default_device is not None else (
        #     utils.read_env_vars_and_defaults(EnvVars.MARQO_BEST_AVAILABLE_DEVICE))

        # TODO load env vars to this class and expose them as properties

        self.device_manager: DeviceManager = DeviceManager()
        self.local_inference: Inference = DummyInference()


class DummyInference(Inference):
    def __init__(self, model_dimension: int = 512, chunks: int = 2, error_on_prefix: Optional[str] = None):
        self.model_dimension = model_dimension
        self.chunks = chunks
        self.error_on_prefix = error_on_prefix

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        results = []
        for content in request.contents:
            try:
                if self.error_on_prefix and content.startswith(self.error_on_prefix):
                    raise Exception(content)
                if request.preprocessing_config.should_chunk:
                    results.append([(f'chunk_{i}', np.random.rand(self.model_dimension)) for i in range(self.chunks)])
                else:
                    results.append([(content, np.random.rand(self.model_dimension))])
            except Exception as e:
                # If an error occurs for this specific content, add an InferenceError.
                results.append(InferenceError(f"Error processing content: {content}. Error: {str(e)}"))

        return InferenceResult(result=results)
