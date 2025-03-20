import marqo.core.inference.api.exceptions as inference_api_exceptions
from marqo.inference.native_inference.embedding_models.hugging_face_model import HuggingFaceModel
from marqo.inference.native_inference.embedding_models.open_clip_model import OpenCLIPModel
from marqo.inference.native_inference.embedding_models.random_model import RandomModel
from marqo.inference.native_inference.inference_pipeline.hugging_face_model_inference_pipeline import \
    HuggingFaceModelInferencePipeline
from marqo.inference.native_inference.inference_pipeline.open_clip_inference_pipeline import OpenCLIPInferencePipeline
from marqo.inference.native_inference.inference_pipeline.random_model_inference_pipeline import \
    RandomModelInferencePipeline
from marqo.inference.native_inference.load_model import load_model
from marqo.inference.type import *
from marqo.s2_inference.errors import S2InferenceError


class NativeInferenceLocal(Inference):

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        try:
            model = load_model(
                model_name=request.model_config.model_name,
                model_properties=request.model_config.model_properties,
                model_auth=request.model_config.model_auth,
                device=request.device
            )
        except S2InferenceError as e:
            raise inference_api_exceptions.ModelError(str(e)) from e

        if isinstance(model, OpenCLIPModel):
            return OpenCLIPInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, RandomModel):
            return RandomModelInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, HuggingFaceModel):
            return HuggingFaceModelInferencePipeline(model, request).run_pipeline()
        else:
            raise ValueError(f"Model type {type(model)} not supported")
