from abc import ABC
from typing import Optional, Dict, Literal

import pydantic

from marqo.base_model import ImmutableBaseModel


class PreprocessingConfig(ImmutableBaseModel, ABC):
    """Parent class of preprocessing config for all modality types"""
    should_chunk: bool = pydantic.Field(default=True, alias='shouldChunk')


class ChunkConfig(ImmutableBaseModel):
    split_length: int = pydantic.Field(gt=0, alias='splitLength')
    split_overlap: int = pydantic.Field(ge=0, alias='splitOverlap')


class TextChunkConfig(ChunkConfig):
    split_method: Literal['character', 'word', 'sentence', 'passage'] = pydantic.Field(alias='splitMethod')


class TextPreprocessingConfig(PreprocessingConfig):
    """Preprocessing config for text modality"""
    text_prefix: Optional[str] = pydantic.Field(default=None, alias='textPrefix')
    chunk_config: Optional[TextChunkConfig] = pydantic.Field(default=None, alias='chunkConfig')


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


class AudioVideoPreprocessingConfig(PreprocessingConfig):
    """Preprocessing config for audio and video modality"""
    download_thread_count: Optional[int] = pydantic.Field(default=None, alias='downloadThreadCount')
    download_header: Optional[Dict[str, str]] = pydantic.Field(default=None, alias='downloadHeader')
    chunk_config: Optional[ChunkConfig] = pydantic.Field(default=None, alias='chunkConfig')

