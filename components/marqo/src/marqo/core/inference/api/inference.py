from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union

from numpy import ndarray
from pydantic.v1 import StrictStr, root_validator, Field, validator

from marqo.base_model import ImmutableBaseModel
from marqo.core.inference.api import Modality, PreprocessingConfigType
# TODO Ideally this should be in a shared module
from marqo.tensor_search.models.private_models import ModelAuth


class EmbeddingModelConfig(ImmutableBaseModel):
    model_name: StrictStr = Field(alias='modelName')
    model_properties: Optional[Dict[str, Any]] = Field(default=None, alias='modelProperties')
    model_auth: Optional[ModelAuth] = Field(default=None, alias='modelAuth')
    normalize_embeddings: bool = Field(default=True, alias='normalizeEmbeddings')


class InferenceRequest(ImmutableBaseModel):
    modality: Modality
    contents: List[str] = Field(min_items=1)
    device: Optional[str] = Field(default=None)
    embedding_model_config: EmbeddingModelConfig = Field(alias='embeddingModelConfig')
    preprocessing_config: PreprocessingConfigType = Field(alias='preprocessingConfig')
    use_inference_cache: bool = Field(default=False, alias='useInferenceCache')
    # whether we should return error for individual content, when set to false, any error should fail the whole batch
    return_individual_error: bool = Field(default=True, alias='returnIndividualError')

    @root_validator(pre=False)
    def check_preprocessing_config_matches_modality(cls, values):
        modality: Modality = values.get('modality')
        preprocessing_config: PreprocessingConfigType = values.get('preprocessing_config')

        if not modality or not preprocessing_config:
            raise ValueError("Modality or preprocessing_config is missing")

        if modality.value != preprocessing_config.modality:
            raise ValueError(f"preprocessing config of type {type(preprocessing_config)} "
                             f"does not support modality: {modality}")

        return values


class InferenceErrorModel(ImmutableBaseModel):
    """
    A model class to store error information for each individual content
    """
    status_code: int = Field(default=400)
    error_code: str = Field(default='inference_error')
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


class ModelManager(ABC):
    @abstractmethod
    def get_loaded_models(self, detailed: bool=False) -> dict:
        """
        Retrieve information about models loaded in all devices

        Args:
            detailed (bool): whether to return detailed information about each model

        Returns: A dictionary containing the list of loaded models, in following format,
        e.g,
            {
                "models": [{"modelName": "model1||1234", "modelProperties": {...}, ...]
            } if detailed is True,

            {
                "models": [{"modelName": "model1||1234"}, {"modelName": "model2||5678"}, ...]
            } if detailed is False

        """
        pass

    @abstractmethod
    def eject_model(self, model_name: str) -> dict:
        """
        Eject a model from the model cache

        Args:
            model_name (str): the name of the model

        Returns: The result of the rejection, in following format:
          {"result": "success",
           "message": f"successfully eject model_name `{model_name}` "}

        Raises:
            ModelError: If model is not found or not in the model cache
        """
        pass
