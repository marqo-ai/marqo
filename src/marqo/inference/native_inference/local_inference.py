import marqo.core.inference.api.exceptions as inference_api_exceptions
from marqo.core.exceptions import DeviceError
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.embedding_models.hugging_face_model import HuggingFaceModel
from marqo.inference.native_inference.embedding_models.open_clip_model import OpenCLIPModel
from marqo.inference.native_inference.embedding_models.random_model import RandomModel
from marqo.inference.native_inference.inference_pipeline.hugging_face_model_inference_pipeline import \
    HuggingFaceModelInferencePipeline
from marqo.inference.native_inference.inference_pipeline.open_clip_model_inference_pipeline import (
    OpenCLIPModelInferencePipeline)
from marqo.inference.native_inference.inference_pipeline.random_model_inference_pipeline import \
    RandomModelInferencePipeline
from marqo.inference.native_inference.load_model import load_model
from marqo.inference.type import *
from marqo.s2_inference.errors import S2InferenceError
from marqo.s2_inference.models.model_type import ModelType
from marqo.s2_inference.no_model_utils import NO_MODEL
from marqo.inference.native_inference.inference_pipeline.multilingual_inference_pipeline import MultilingualCLIPModelInferencePipeline
from marqo.inference.native_inference.embedding_models.multilingual_clip_model import MultilingualCLIPModel


class NativeInferenceLocal(Inference):

    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        try:
            model = load_model(
                model_name=request.model_config.model_name,
                model_properties=request.model_config.model_properties,
                model_auth=request.model_config.model_auth,
                device=self.device_manager.pick_and_validate_device(device=request.device)
            )
        except (S2InferenceError, DeviceError) as e:
            raise inference_api_exceptions.ModelError(str(e)) from e

        if isinstance(model, OpenCLIPModel):
            return OpenCLIPModelInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, RandomModel):
            return RandomModelInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, HuggingFaceModel):
            return HuggingFaceModelInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, MultilingualCLIPModel):
            return MultilingualCLIPModelInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, NO_MODEL):
            # TODO do we need to create a pipeline class for this?
            error = f"Cannot vectorise anything with '{ModelType.NO_MODEL}'. " \
                    f"This model is intended for adding documents and searching with custom vectors only. " \
                    f"If vectorisation is needed, please use a different model "
            if request.return_individual_error:
                return InferenceResult(result=[InferenceErrorModel(error_message=error)for _ in request.contents])
            else:
                raise InferenceError(error)
        else:
            raise ValueError(f"Model type {type(model)} not supported")