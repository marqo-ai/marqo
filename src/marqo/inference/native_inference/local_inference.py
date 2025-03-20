from marqo.core.inference.device_manager import DeviceManager
from marqo.inference.native_inference.inference_pipeline.open_clip_inference_pipeline import OpenCLIPInferencePipeline
from marqo.inference.native_inference.embedding_models.open_clip_model import OPEN_CLIP
from marqo.inference.native_inference.load_model import load_model
from marqo.inference.native_inference.embedding_models.random_model import RandomModel
from marqo.inference.native_inference.inference_pipeline.random_model_inference_pipeline import RandomModelInferencePipeline
from marqo.s2_inference.errors import S2InferenceError
import marqo.core.inference.api.exceptions as inference_api_exceptions

from marqo.inference.type import *
from marqo.s2_inference.models.model_type import ModelType
from marqo.s2_inference.no_model_utils import NO_MODEL


class NativeInferenceLocal(Inference):

    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        try:
            model = load_model(
                model_name=request.model_config.model_name,
                model_properties=request.model_config.model_properties,
                model_auth=request.model_config.model_auth,
                # TODO improve this to validate the device passed in from client
                device=request.device or self.device_manager.best_available_device_type.value
            )
        except S2InferenceError as e:
            raise inference_api_exceptions.ModelError(str(e)) from e

        if isinstance(model, OPEN_CLIP):
            return OpenCLIPInferencePipeline(model, request).run_pipeline()
        elif isinstance(model, RandomModel):
            return RandomModelInferencePipeline(model, request).run_pipeline()
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
