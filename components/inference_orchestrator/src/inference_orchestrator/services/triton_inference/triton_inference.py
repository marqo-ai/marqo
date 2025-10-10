from inference_orchestrator.services.errors import InternalServerError
from inference_orchestrator.schemas.api import *
from inference_orchestrator.services.triton_inference.embedding_models import OpenCLIPModel, HuggingFaceModel, RandomModel
from inference_orchestrator.services.triton_inference.inference_pipelines.hugging_face_model_inference_pipeline import \
    HuggingFaceModelInferencePipeline
from inference_orchestrator.services.triton_inference.inference_pipelines.open_clip_model_inference_pipeline import (
    OpenCLIPModelInferencePipeline)
from inference_orchestrator.services.triton_inference.inference_pipelines.random_model_inference_pipeline import \
    RandomModelInferencePipeline
from inference_orchestrator.services.triton_inference.model_manager.model_manager import load_model


class TritonInference(Inference):

    def __init__(self, model_management_client, triton_client):
        self.model_management_client = model_management_client
        self.triton_client = triton_client

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        model = load_model(
            model_name=request.model_config_.model_name,
            model_properties=request.model_config_.model_properties,
            triton_client=self.triton_client,
            model_management_client=self.model_management_client,
        )

        if isinstance(model, OpenCLIPModel):
            return OpenCLIPModelInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, HuggingFaceModel):
            return HuggingFaceModelInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, RandomModel):
            return RandomModelInferencePipeline(model, request).run_pipeline()
        else:
            raise InternalServerError(f"Model not supported.")
