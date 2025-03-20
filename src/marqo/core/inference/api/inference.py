from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union

import pydantic
from numpy import ndarray
from pydantic import StrictStr, root_validator

from marqo.base_model import ImmutableBaseModel
from marqo.core.inference.api import InferenceError, Modality, PreprocessingConfigType
# TODO Ideally this should be in a shared module
from marqo.tensor_search.models.private_models import ModelAuth


class ModelConfig(ImmutableBaseModel):
    model_name: StrictStr = pydantic.Field(alias='modelName')
    model_properties: Optional[Dict[str, Any]] = pydantic.Field(default=None, alias='modelProperties')
    model_auth: Optional[ModelAuth] = pydantic.Field(default=None, alias='modelAuth')
    normalize_embeddings: bool = pydantic.Field(default=True, alias='normalizeEmbeddings')


class InferenceRequest(ImmutableBaseModel):
    modality: Modality
    contents: List[str] = pydantic.Field(min_items=1)
    device: Optional[str] = pydantic.Field(default=None)
    model_config: ModelConfig = pydantic.Field(alias='modelConfig')
    preprocessing_config: PreprocessingConfigType = pydantic.Field(alias='preprocessingConfig')
    use_inference_cache: bool = pydantic.Field(default=False, alias='useInferenceCache')
    # whether we should return error for individual content, when set to false, any error should fail the whole batch
    return_individual_error: bool = pydantic.Field(default=True, alias='returnIndividualError')

    @root_validator(pre=False)
    def check_preprocessing_config_matches_modality(cls, values):
        modality: Modality = values.get('modality')
        preprocessing_config: PreprocessingConfigType = values.get('preprocessing_config')

        if not modality or modality.value != preprocessing_config.modality:
            raise ValueError(f"{type(preprocessing_config)} only supports modality: {preprocessing_config.modality}, "
                             f"but modality: {modality} is specified in the request")

        return values


class InferenceErrorModel(ImmutableBaseModel):
    """
    A model class to store error information for each individual content
    """
    status_code: int = pydantic.Field(default=400)
    error_code: str = pydantic.Field(default='inference_error')
    error_message: str


class InferenceResult(ImmutableBaseModel):
    result: List[Union[InferenceErrorModel, List[Tuple[str, ndarray]]]]

    class Config(ImmutableBaseModel.Config):
        arbitrary_types_allowed = True


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

