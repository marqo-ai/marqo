from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from numpy import ndarray
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from .base_model import AppImmutableBaseModel


class Modality(str, Enum):
    TEXT = "language"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class PreprocessingConfig(AppImmutableBaseModel, ABC):
    """Parent class of preprocessing config for all modality types"""

    modality: str
    should_chunk: bool = Field(default=False, alias="shouldChunk")


class ChunkConfig(AppImmutableBaseModel):
    split_length: int = Field(gt=0, alias="splitLength")
    split_overlap: int = Field(ge=0, alias="splitOverlap")

    @model_validator(mode="after")
    def check_split_length_greater_than_overlap(self):
        split_length = self.split_length
        split_overlap = self.split_overlap
        if split_length is not None and split_overlap is not None:
            if split_length <= split_overlap:
                raise ValueError("split_length must be greater than split_overlap")
        return self


class TextChunkConfig(ChunkConfig):
    split_method: Literal["character", "word", "sentence", "passage"] = Field(
        alias="splitMethod"
    )


class TextPreprocessingConfig(PreprocessingConfig):
    """Preprocessing config for text modality"""

    modality: Literal[Modality.TEXT] = Modality.TEXT
    text_prefix: Optional[str] = Field(default=None, alias="textPrefix")
    chunk_config: Optional[TextChunkConfig] = Field(default=None, alias="chunkConfig")

    @model_validator(mode="after")
    def validate_chunk_config(self):
        if self.should_chunk and self.chunk_config is None:
            raise ValueError(
                "`chunk_config` must be provided when `should_chunk` is True."
            )
        if not self.should_chunk and self.chunk_config is not None:
            raise ValueError(
                "`chunk_config` must not be provided when `should_chunk` is False."
            )
        return self


class ImagePreprocessingConfig(PreprocessingConfig):
    """Preprocessing config for image modality"""

    modality: Literal[Modality.IMAGE] = Modality.IMAGE
    download_timeout_ms: int = Field(
        default=3000, alias="downloadTimeoutMs"
    )  # default to 3000ms
    download_thread_count: Optional[int] = Field(
        default=None, alias="downloadThreadCount"
    )
    download_header: Optional[Dict[str, str]] = Field(
        default=None, alias="downloadHeader"
    )

    # image chunking TODO this is going away in future versions
    patch_method: Optional[
        # TODO check if we need to support all methods in image_processor.chunk_image method
        Literal["simple", "frcnn", "dino-v1", "dino-v2", "marqo-yolo"]
    ] = Field(default=None, alias="patchMethod")

    @model_validator(mode="after")
    def validate_chunk_config(self):
        should_chunk = self.should_chunk
        patch_method = self.patch_method
        if should_chunk and patch_method is None:
            raise ValueError(
                "`patch_method` must be provided when `should_chunk` is True."
            )
        if not should_chunk and patch_method is not None:
            raise ValueError(
                "`patch_method` must not be provided when `should_chunk` is False."
            )
        return self


class AudioPreprocessingConfig(PreprocessingConfig):
    """Preprocessing config for audio modality"""

    modality: Literal[Modality.AUDIO] = Modality.AUDIO
    download_thread_count: Optional[int] = Field(
        default=None, alias="downloadThreadCount"
    )
    download_header: Optional[Dict[str, str]] = Field(
        default=None, alias="downloadHeader"
    )
    chunk_config: Optional[ChunkConfig] = Field(default=None, alias="chunkConfig")
    max_media_size_bytes: int = Field(
        ge=1, default=387973120, alias="maxMediaSizeBytes"
    )

    @model_validator(mode="after")
    def validate_chunk_config(self):
        should_chunk = self.should_chunk
        chunk_config = self.chunk_config
        if should_chunk and chunk_config is None:
            raise ValueError(
                "`chunk_config` must be provided when `should_chunk` is True."
            )
        if not should_chunk and chunk_config is not None:
            raise ValueError(
                "`chunk_config` must not be provided when `should_chunk` is False."
            )
        return self


class VideoPreprocessingConfig(PreprocessingConfig):
    """Preprocessing config for video modality"""

    modality: Literal[Modality.VIDEO] = Modality.VIDEO
    download_thread_count: Optional[int] = Field(
        default=None, alias="downloadThreadCount"
    )
    download_header: Optional[Dict[str, str]] = Field(
        default=None, alias="downloadHeader"
    )
    chunk_config: Optional[ChunkConfig] = Field(default=None, alias="chunkConfig")
    max_media_size_bytes: int = Field(
        ge=1, default=387973120, alias="maxMediaSizeBytes"
    )

    @model_validator(mode="after")
    def validate_chunk_config(self):
        should_chunk = self.should_chunk
        chunk_config = self.chunk_config
        if should_chunk and chunk_config is None:
            raise ValueError(
                "`chunk_config` must be provided when `should_chunk` is True."
            )
        if not should_chunk and chunk_config is not None:
            raise ValueError(
                "`chunk_config` must not be provided when `should_chunk` is False."
            )
        return self


PreprocessingConfigType = Union[
    TextPreprocessingConfig,
    ImagePreprocessingConfig,
    AudioPreprocessingConfig,
    VideoPreprocessingConfig,
]


class EmbeddingModelConfig(AppImmutableBaseModel):
    model_name: StrictStr = Field(alias="modelName")
    model_properties: Optional[Dict[str, Any]] = Field(
        default=None, alias="modelProperties"
    )
    normalize_embeddings: bool = Field(default=True, alias="normalizeEmbeddings")


class InferenceRequest(AppImmutableBaseModel):
    modality: Modality
    contents: List[str] = Field(min_length=1)
    embedding_model_config: EmbeddingModelConfig = Field(alias="embeddingModelConfig")
    preprocessing_config: Union[TextPreprocessingConfig, ImagePreprocessingConfig] = (
        Field(alias="preprocessingConfig", discriminator="modality")
    )
    use_inference_cache: bool = Field(default=False, alias="useInferenceCache")
    # whether we should return error for individual content, when set to false, any error should fail the whole batch
    return_individual_error: bool = Field(default=True, alias="returnIndividualError")

    @model_validator(mode="after")
    def check_preprocessing_config_matches_modality(self):
        modality: Modality = self.modality
        preprocessing_config: PreprocessingConfigType = self.preprocessing_config

        if not modality or not preprocessing_config:
            raise ValueError("Modality or preprocessing_config is missing")

        if modality.value != preprocessing_config.modality:
            raise ValueError(
                f"preprocessing config of type {type(preprocessing_config)} "
                f"does not support modality: {modality}"
            )

        return self


class InferenceErrorModel(AppImmutableBaseModel):
    """
    A model class to store error information for each individual content
    """

    status_code: int = Field(default=400)
    error_code: str = Field(default="inference_error")
    error_message: str


class InferenceResult(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        frozen=True,
        arbitrary_types_allowed=True,  # To allow msgpack ndarray type in the result
    )

    result: List[Union[InferenceErrorModel, List[Tuple[str, ndarray]]]]


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
    def get_loaded_models(self) -> dict:
        """
        Retrieve information about models loaded in all devices
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
           "message": f"successfully eject modelName `{model_name}`"}

        Raises:
            ModelError: If model is not found or not in the model cache
        """
        pass
