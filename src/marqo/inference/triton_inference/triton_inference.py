import marqo.core.inference.api.exceptions as inference_api_exceptions
from marqo.core.exceptions import DeviceError
from marqo.inference.triton_inference.embedding_models.open_clip.open_clip_model import OpenCLIPModel
from marqo.inference.triton_inference.inference_pipeline.open_clip_model_inference_pipeline import (
    OpenCLIPModelInferencePipeline)
from marqo.inference.triton_inference.inference_pipeline.hugging_face_model_inference_pipeline import HuggingFaceModelInferencePipeline
from marqo.inference.triton_inference.model_manager.load_model import load_model
from marqo.inference.triton_inference.embedding_models.hugging_face.hugging_face_model import HuggingFaceModel
from marqo.inference.type import *
from marqo.s2_inference.errors import S2InferenceError



class TritonInference(Inference):

    def __init__(self, model_manager, triton_client):
        self.model_manager = model_manager
        self.triton_client = triton_client

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        try:
            model = load_model(
                model_name=request.model_config.model_name,
                model_properties=request.model_config.model_properties,
                triton_client=self.triton_client,
                model_manager=self.model_manager,
            )
        except (S2InferenceError, DeviceError) as e:
            raise inference_api_exceptions.ModelError(str(e)) from e

        if isinstance(model, OpenCLIPModel):
            return OpenCLIPModelInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, HuggingFaceModel):
            return HuggingFaceModelInferencePipeline(model, request).run_pipeline()
        else:
            raise ValueError(f"Model type {type(model)} not supported")