from abc import ABC, abstractmethod
from marqo.core.inference.api.inference import InferenceResult


class AbstractInferencePipeline(ABC):

    def __init__(self, model, inference_request):
        self.model = model
        self.inference_request = inference_request

    @abstractmethod
    def run_pipeline(self) -> InferenceResult:
        pass