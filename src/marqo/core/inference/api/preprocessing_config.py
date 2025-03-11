from abc import ABC, abstractmethod
from typing import Optional, Dict, Literal, List, Set

import pydantic
from pydantic import root_validator

from marqo.base_model import ImmutableBaseModel
from marqo.core.inference.api.modality import Modality


class PreprocessingConfig(ImmutableBaseModel, ABC):
    """Parent class of preprocessing config for all modality types"""
    should_chunk: bool = pydantic.Field(default=False, alias='shouldChunk')

    @abstractmethod
    def supported_modalities(self) -> Set[Modality]:
        pass


class ChunkConfig(ImmutableBaseModel):
    split_length: int = pydantic.Field(gt=0, alias='splitLength')
    split_overlap: int = pydantic.Field(ge=0, alias='splitOverlap')


class TextChunkConfig(ChunkConfig):
    split_method: Literal['character', 'word', 'sentence', 'passage'] = pydantic.Field(alias='splitMethod')


class TextPreprocessingConfig(PreprocessingConfig):
    """Preprocessing config for text modality"""
    text_prefix: Optional[str] = pydantic.Field(default=None, alias='textPrefix')
    chunk_config: Optional[TextChunkConfig] = pydantic.Field(default=None, alias='chunkConfig')

    @root_validator
    def validate_chunk_config(cls, values):
        should_chunk = values.get('should_chunk')
        chunk_config = values.get('chunk_config')
        if should_chunk and chunk_config is None:
            raise ValueError("`chunk_config` must be provided when `should_chunk` is True.")
        if not should_chunk and chunk_config is not None:
            raise ValueError("`chunk_config` must not be provided when `should_chunk` is False.")
        return values

    def supported_modalities(self) -> Set[Modality]:
        return {Modality.TEXT}


class ImagePreprocessingConfig(PreprocessingConfig):
    """Preprocessing config for image modality"""
    download_timeout_ms: Optional[int] = pydantic.Field(default=None, alias='downloadTimeoutMs')
    download_thread_count: Optional[int] = pydantic.Field(default=None, alias='downloadThreadCount')
    download_header: Optional[Dict[str, str]] = pydantic.Field(default=None, alias='downloadHeader')

    # image chunking TODO this is going away in future versions
    patch_method: Optional[
        # TODO check if we need to support all methods in image_processor.chunk_image method
        Literal['simple', 'frcnn', 'dino-v1', 'dino-v2', 'marqo-yolo']
    ] = pydantic.Field(
        default=None,
        alias='patchMethod'
    )

    @root_validator
    def validate_chunk_config(cls, values):
        should_chunk = values.get('should_chunk')
        patch_method = values.get('patch_method')
        if should_chunk and patch_method is None:
            raise ValueError("`patch_method` must be provided when `should_chunk` is True.")
        if not should_chunk and patch_method is not None:
            raise ValueError("`patch_method` must not be provided when `should_chunk` is False.")
        return values

    def supported_modalities(self) -> Set[Modality]:
        return {Modality.IMAGE}


class AudioVideoPreprocessingConfig(PreprocessingConfig):
    """Preprocessing config for audio and video modality"""
    download_thread_count: Optional[int] = pydantic.Field(default=None, alias='downloadThreadCount')
    download_header: Optional[Dict[str, str]] = pydantic.Field(default=None, alias='downloadHeader')
    chunk_config: Optional[ChunkConfig] = pydantic.Field(default=None, alias='chunkConfig')

    @root_validator
    def validate_chunk_config(cls, values):
        should_chunk = values.get('should_chunk')
        chunk_config = values.get('chunk_config')
        if should_chunk and chunk_config is None:
            raise ValueError("`chunk_config` must be provided when `should_chunk` is True.")
        if not should_chunk and chunk_config is not None:
            raise ValueError("`chunk_config` must not be provided when `should_chunk` is False.")
        return values

    def supported_modalities(self) -> Set[Modality]:
        return {Modality.AUDIO, Modality.VIDEO}

