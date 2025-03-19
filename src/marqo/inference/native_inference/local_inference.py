from marqo.inference.native_inference.embedding_models.random_model import RandomModel
from marqo.inference.native_inference.inference_pipeline.open_clip_inference_pipeline import OpenCLIPInferencePipeline
from marqo.inference.native_inference.embedding_models.open_clip_model import OPEN_CLIP
from marqo.inference.native_inference.load_model import load_model
from marqo.inference.native_inference.embedding_models.random_model import RandomModel
from marqo.inference.native_inference.inference_pipeline.random_model_inference_pipeline import RandomModelInferencePipeline

from marqo.inference.type import *



class NativeInferenceLocal(Inference):

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        model = load_model(
            model_name=request.model_config.model_name,
            model_properties=request.model_config.model_properties,
            model_auth=request.model_config.model_auth,
            device=request.device
        )

        if isinstance(model, OPEN_CLIP):
            return OpenCLIPInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, RandomModel):
            return RandomModelInferencePipeline(model, request).run_pipeline()
        else:
            raise ValueError(f"Model type {type(model)} not supported")
