from abc import ABC, abstractmethod
from typing import Optional, Dict, Literal, List, Set, Union

import pydantic
from pydantic import Field, model_validator

from marqo.base_model import StrictBaseModel, ImmutableBaseModel
from marqo.core.inference.api.modality import Modality


class PreprocessingConfig(ImmutableBaseModel, ABC):
    """Parent class of preprocessing config for all modality types"""
    modality: str
    should_chunk: bool = Field(default=False, alias='shouldChunk')


class ChunkConfig(ImmutableBaseModel):
    split_length: int = Field(gt=0, alias='splitLength')
    split_overlap: int = Field(ge=0, alias='splitOverlap')

    @model_validator(mode='before')
    def check_split_length_greater_than_overlap(cls, values):
        split_length = values.get('split_length')
        split_overlap = values.get('split_overlap')
        if split_length is not None and split_overlap is not None:
            if split_length <= split_overlap:
                raise ValueError('split_length must be greater than split_overlap')
        return values


class TextChunkConfig(ImmutableBaseModel):
    """Text chunk configuration"""
    should_chunk: bool = Field(default=False, alias='shouldChunk')
    # required if should_chunk is True
    # the split parameters
    split_length: int = Field(gt=0, alias='splitLength')
    split_overlap: int = Field(ge=0, alias='splitOverlap')


class ChunkMethodConfig(ImmutableBaseModel):
    split_method: Literal['character', 'word', 'sentence', 'passage'] = Field(alias='splitMethod')


class TextPreprocessingConfig(ImmutableBaseModel):
    """Text preprocessing configuration"""
    # NOTE: modality must be set to text
    modality: str = "text"
    text_prefix: Optional[str] = Field(default=None, alias='textPrefix')
    chunk_config: Optional[TextChunkConfig] = Field(default=None, alias='chunkConfig')

    @model_validator(mode='before')
    def validate_chunk_config(cls, values: dict) -> dict:
        if not isinstance(values, dict):
            return values
            
        chunk_config = values.get('chunk_config') or values.get('chunkConfig')
        if not chunk_config:
            return values
        if not isinstance(chunk_config, dict):
            return values

        if chunk_config.get('should_chunk') or chunk_config.get('shouldChunk'):
            if not (chunk_config.get('split_length') or chunk_config.get('splitLength')):
                raise ValueError("split_length is required when should_chunk is True")
            split_length = chunk_config.get('split_length') or chunk_config.get('splitLength')
            split_overlap = chunk_config.get('split_overlap') or chunk_config.get('splitOverlap') or 0
            if split_length < split_overlap:
                raise ValueError(f"split_length ({split_length}) must be greater than or equal to split_overlap ({split_overlap})")
        return values


class ImageDownloadConfig(ImmutableBaseModel):
    """Image download configuration"""
    download_timeout_ms: int = Field(default=3000, alias='downloadTimeoutMs')  # default to 3000ms
    download_thread_count: Optional[int] = Field(default=None, alias='downloadThreadCount')
    download_header: Optional[Dict[str, str]] = Field(default=None, alias='downloadHeader')


class ChunkConfig(ImageDownloadConfig, ChunkMethodConfig):
    """
    Configuration for chunking images
    """
    pass


class ImagePreprocessingConfig(ImmutableBaseModel):
    """
    Configuration for preprocessing images
    """
    # NOTE: modality must be set to image
    modality: str = "image"
    download_thread_count: Optional[int] = Field(default=None, alias='downloadThreadCount')
    download_header: Optional[Dict[str, str]] = Field(default=None, alias='downloadHeader')
    chunk_config: Optional[ChunkConfig] = Field(default=None, alias='chunkConfig')


class MultiModalPreprocessingConfig(ImmutableBaseModel):
    """
    Configuration for preprocessing multi-modal (text+image) data
    """
    # NOTE: modality must be set to multimodal
    modality: str = "multimodal"
    download_thread_count: Optional[int] = Field(default=None, alias='downloadThreadCount')
    download_header: Optional[Dict[str, str]] = Field(default=None, alias='downloadHeader')
    chunk_config: Optional[ChunkConfig] = Field(default=None, alias='chunkConfig')


# PreprocessingConfigType is a sum type of all preprocessing configurations
# While pydantic doesn't support sum types, the inference code makes sure it only parses
# config of the right type based on the modality field
PreprocessingConfigType = Union[TextPreprocessingConfig, ImagePreprocessingConfig, MultiModalPreprocessingConfig]