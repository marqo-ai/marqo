from marqo.core.inference.api.inference import *
from marqo.inference.triton_inference.triton_inference_client import TritonInferenceClient
from marqo.inference.native_inference.inference_pipeline.open_clip_model_triton_inference_pipeline import OpenCLIPModelTritonInferencePipeline
import marqo.core.inference.api.exceptions as inference_api_exceptions
from marqo.core.exceptions import DeviceError
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.embedding_models.hugging_face_model import HuggingFaceModel
from marqo.inference.native_inference.embedding_models.languagebind_model import LanguagebindModel
from marqo.inference.native_inference.embedding_models.multilingual_clip_model import MultilingualCLIPModel
from marqo.inference.native_inference.embedding_models.open_clip_model import OpenCLIPModel
from marqo.inference.native_inference.embedding_models.random_model import RandomModel
from marqo.inference.native_inference.inference_pipeline.hugging_face_model_inference_pipeline import \
    HuggingFaceModelInferencePipeline
from marqo.inference.native_inference.inference_pipeline.languagebind_model_inference_pipeline import \
    LanguagebindModelInferencePipeline
from marqo.inference.native_inference.inference_pipeline.multilingual_inference_pipeline import \
    MultilingualCLIPModelInferencePipeline
from marqo.inference.native_inference.inference_pipeline.open_clip_model_inference_pipeline import (
    OpenCLIPModelInferencePipeline)
from marqo.inference.native_inference.inference_pipeline.random_model_inference_pipeline import \
    RandomModelInferencePipeline
from marqo.inference.native_inference.load_model import load_model
from marqo.inference.type import *
from marqo.s2_inference.errors import S2InferenceError
from marqo.s2_inference.models.model_type import ModelType
from marqo.s2_inference.no_model_utils import NO_MODEL
from marqo.inference.triton_inference.triton_inference_client import TritonInferenceClient


class TritonInference(Inference):
    """
    This class is a placeholder for Triton Inference implementation.
    It should be implemented to handle inference requests using Triton Inference Server.
    """

    def __init__(self, triton_client: TritonInferenceClient):
        self.triton_client = triton_client

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        try:
            model = load_model(
                model_name="test-fn-model",
                model_properties={
                    "name": "hf-hub:timm/ViT-B-16-SigLIP",
                    "dimensions": 768,
                    "type": "open_clip"
                },
                model_auth=None,
                device="cpu"  # Triton Inference Server does not require a specific device
            )
        except (S2InferenceError, DeviceError) as e:
            raise inference_api_exceptions.ModelError(str(e)) from e

        model.model = None  # Set model to None to avoid using the local model in the pipeline

        if isinstance(model, OpenCLIPModel):
            return OpenCLIPModelTritonInferencePipeline(
                model=model,
                inference_request=request,
                triton_client=self.triton_client
            ).run_pipeline()
        else:
            raise ValueError(f"Unsupported model type for Triton Inference: {type(model)}. "
                             f"Currently, only OpenCLIPModel is supported for Triton Inference.")

