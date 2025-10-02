from marqo_inference_container.errors.common_errors import InternalError
from marqo_inference_container.schemas.api import *
from marqo_inference_container.services.triton_inference.embedding_models import OpenCLIPModel, HuggingFaceModel, \
    RandomModel
from marqo_inference_container.services.triton_inference.inference_pipeline.hugging_face_model_inference_pipeline import \
    HuggingFaceModelInferencePipeline
from marqo_inference_container.services.triton_inference.inference_pipeline.open_clip_model_inference_pipeline import (
    OpenCLIPModelInferencePipeline)
from marqo_inference_container.services.triton_inference.inference_pipeline.random_model_inference_pipeline import \
    RandomModelInferencePipeline
from marqo_inference_container.services.triton_inference.model_manager.load_model import load_model


class TritonInference(Inference):

    def __init__(self, model_manager, triton_client):
        self.model_manager = model_manager
        self.triton_client = triton_client

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        # TODO - Catch error here
        model = load_model(
            model_name=request.model_config_.model_name,
            model_properties=request.model_config_.model_properties,
            triton_client=self.triton_client,
            model_manager=self.model_manager,
        )

        if isinstance(model, OpenCLIPModel):
            return OpenCLIPModelInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, HuggingFaceModel):
            return HuggingFaceModelInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, RandomModel):
            return RandomModelInferencePipeline(model, request).run_pipeline()
        else:
            raise InternalError(f"Model type '{model.__name__}' not supported.")
