from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union

from numpy import ndarray

from marqo.base_model import ImmutableBaseModel
from marqo.core.inference.api import PreprocessingConfig, InferenceError
# TODO Ideally this should be in a shared module
from marqo.tensor_search.models.private_models import ModelAuth


class Modality(str, Enum):
    TEXT = "language"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ModelConfig(ImmutableBaseModel):
    model_name: str
    model_properties: Optional[Dict[str, Any]]
    model_auth: Optional[ModelAuth]
    normalize_embeddings: bool


class InferenceRequest(ImmutableBaseModel):
    modality: Modality
    contents: List[str]
    device: Optional[str]
    model_config: ModelConfig
    preprocessing_config: PreprocessingConfig
    use_inference_cache: bool


class InferenceResult(ImmutableBaseModel):
    result: List[Union[InferenceError, List[Tuple[str, ndarray]]]]


class Inference(ABC):

    @abstractmethod
    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        """
        The Inference interface is an abstraction for the embedding generation logic. It takes in a list of contents
        for a given modality (either a piece of text or a URL of a media files), downloads, chunks, preprocesses,
        and generates embeddings using the embedding model specified in the request.

        Args:
            request (InferenceRequest): the inference request

        Returns: (InferenceResult)
            The inference result, for each content, it's either an InferenceError or A list of tuples. Each tuple
            represents a chunk with a string-typed key and the embedding in ndarray format.

        Raises:
            InferenceError: if an error impacting the whole batch of contents occurs during inference.
        """
        pass

