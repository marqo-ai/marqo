from marqo_inference_container.errors.inference_errors import InferenceError

from marqo_inference_container.schemas.api import *
from marqo_inference_container.services.triton_inference.embedding_models.hugging_face.hugging_face_model import \
    HuggingFaceModel
from marqo_inference_container.services.triton_inference.embedding_models.open_clip.open_clip_model import OpenCLIPModel
from marqo_inference_container.services.triton_inference.inference_pipeline.hugging_face_model_inference_pipeline import \
    HuggingFaceModelInferencePipeline
from marqo_inference_container.services.triton_inference.inference_pipeline.open_clip_model_inference_pipeline import (
    OpenCLIPModelInferencePipeline)
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
        else:
            raise ValueError(f"Model type {type(model)} not supported")
