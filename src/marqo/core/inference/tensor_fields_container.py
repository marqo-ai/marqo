import hashlib
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional, Any, Generator, Tuple, cast, TypeVar

import numpy as np
from PIL.Image import Image
from pydantic.main import BaseModel
from torch import Tensor

from marqo.core import constants
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.exceptions import AddDocumentsError, ModelError
from marqo.core.models.marqo_index import FieldType, TextPreProcessing, ImagePreProcessing
from marqo.s2_inference import errors as s2_inference_errors
from marqo.s2_inference import s2_inference
from marqo.s2_inference.multimodal_model_load import Modality
from marqo.s2_inference.processing import image as image_processor
from marqo.s2_inference.processing import text as text_processor

# TODO remove these deps
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.models.private_models import ModelAuth

# Content chunk of different modality can have different types
# - Text: str
# - Image: PIL.Image.Image if not chunked, torch.Tensor if chunked
# - Audio, Video: Dict[str, Tensor]
ContentChunkType = TypeVar('ContentChunkType', str, Image, Tensor, Dict[str, Tensor])

MODALITY_FIELD_TYPE_MAP = {
    Modality.TEXT: FieldType.Text,
    Modality.IMAGE: FieldType.ImagePointer,
    Modality.VIDEO: FieldType.VideoPointer,
    Modality.AUDIO: FieldType.AudioPointer,
}


class Chunker(ABC):
    @abstractmethod
    def chunk(self, field_content: str, single_chunk: bool = False) -> Tuple[List[str], List[ContentChunkType]]:
        """
        Chunks the given field content into chunks.

        Arguments:
            field_content: the content of a tensor field. For TextField, this is a string. For image, audio, video type,
                this is usually a URL or a path to file
            single_chunk: whether to chunk the field content into a single chunk. Single chunk is used for multimodal
                subfields if the subfield is text of image

        Returns:
              A tuple of chunks and content chunks. Chunks are stored in the doc and used for highlighting. Content
              chunks are used to generate embeddings. For text fields, chunks are split text, content chunks is the same
              piece of text with prefixes. For multimedia fields, chunks are usually metadata of the chunk(position of
              the section in the image, time sections in the audio/video, etc.), and content chunks are the tensor or
              image used by multimodal models.
        """
        pass


class TextChunker(Chunker):
    def __init__(self, text_preprocessing: TextPreProcessing, text_chunk_prefix: str):
        self.text_preprocessing = text_preprocessing
        self.text_chunk_prefix = text_chunk_prefix

    def chunk(self, field_content: str, single_chunk: bool = False) -> Tuple[List[str], List[str]]:
        chunks = [field_content] if single_chunk else (
            text_processor.split_text(text=field_content,
                                      split_by=self.text_preprocessing.split_method.value,
                                      split_length=self.text_preprocessing.split_length,
                                      split_overlap=self.text_preprocessing.split_overlap))
        return chunks, text_processor.prefix_text_chunks(chunks, self.text_chunk_prefix)


class ImageChunker(Chunker):
    def __init__(self, media_repo, image_preprocessing: ImagePreProcessing, device: Optional[str]):
        self.image_preprocessing = image_preprocessing
        self.device = device
        self.media_repo = media_repo

    def chunk(self, field_content: str, single_chunk: bool = False):
        image_method = self.image_preprocessing.patch_method
        url = field_content
        image_data = self.media_repo[url]
        if single_chunk or image_method is None:
            return [url], [image_data]

        try:
            content_chunks, text_chunks = image_processor.chunk_image(
                image_data, device=self.device, method=image_method.value)
            return text_chunks, content_chunks
        except s2_inference_errors.S2InferenceError as e:
            raise AddDocumentsError(e.message) from e


class AudioVideoChunker(Chunker):
    def __init__(self, media_repo):
        self.media_repo = media_repo

    def _chunk_id(self, media_chunk: dict) -> str:
        chunk_start = media_chunk['start_time']
        chunk_end = media_chunk['end_time']
        return f"{[chunk_start, chunk_end]}"

    def chunk(self, field_content: str, single_chunk: bool = False):
        url = field_content
        media_chunks = self.media_repo[url]

        # single_chunk does not apply to video and audio fields. Instead, it needs to calculate the
        # embedding for each chunk and average them
        if single_chunk:
            raise RuntimeError("Video and Audio chunker does not support single_chunk")

        text_chunks = [self._chunk_id(media_chunk) for media_chunk in media_chunks]
        content_chunks = [media_chunk['tensor'] for media_chunk in media_chunks]

        return text_chunks, content_chunks


class ModelConfig(BaseModel):
    model_name: str
    model_properties: Optional[Dict[str, Any]]
    model_auth: Optional[ModelAuth]
    device: Optional[str]
    normalize_embeddings: bool


class Vectoriser(ABC):
    @abstractmethod
    def vectorise(self, content_chunks: List[ContentChunkType]) -> List[List[float]]:
        """
        Generate embeddings from a list of content chunks.

        Arguments:
            content_chunks: a list of content chunks. This is generated by Chunkers. Different modality has different
                types of content chunks.

        Returns:
            A list of embeddings. The size of the list should match the size of the content chunks list.
        """
        pass

    def _s2inference_vectorise(self, content_chunks: List[ContentChunkType],
                               modality: Modality, model_config: ModelConfig):
        try:
            return s2_inference.vectorise(
                model_name=model_config.model_name,
                model_properties=model_config.model_properties,
                content=content_chunks,
                device=model_config.device,
                normalize_embeddings=model_config.normalize_embeddings,
                model_auth=model_config.model_auth,
                modality=modality
            )
        except (s2_inference_errors.UnknownModelError,
                s2_inference_errors.InvalidModelPropertiesError,
                s2_inference_errors.ModelLoadError,
                s2_inference.ModelDownloadError) as model_error:
            # Fail the whole batch due to a malfunctioning embedding model
            raise ModelError(f'Problem vectorising query. Reason: {str(model_error)}')
        except s2_inference_errors.S2InferenceError as e:
            raise AddDocumentsError(e.message) from e

    @classmethod
    def single_vectorisers_by_modality(cls, model_config: ModelConfig) -> Dict[FieldType, 'Vectoriser']:
        return {field_type: SingleVectoriser(modality, model_config)
                for modality, field_type in MODALITY_FIELD_TYPE_MAP.items()}

    @classmethod
    def batch_vectorisers_by_modality(cls, model_config: ModelConfig,
                                      chunks_to_vectorise: Dict[FieldType, List[ContentChunkType]]
                                      ) -> Dict[FieldType, 'Vectoriser']:
        return {field_type: BatchCachingVectoriser(modality, chunks_to_vectorise[field_type], model_config)
                for modality, field_type in MODALITY_FIELD_TYPE_MAP.items()
                if field_type in chunks_to_vectorise}


class SingleVectoriser(Vectoriser):
    """
    Generate the embeddings when vectorise method is called.
    """
    def __init__(self, modality: Modality, model_config: ModelConfig):
        self.modality = modality
        self.model_config = model_config

    def vectorise(self, content_chunks: List[ContentChunkType]) -> List[List[float]]:
        with RequestMetricsStore.for_request().time(f"add_documents.create_vectors"):
            if self.modality in [Modality.AUDIO, Modality.VIDEO]:
                # audio and video fields has to be vectorised chunk by chunk due to a limitation of languagebind
                return [vector for content_chunk in content_chunks for vector in
                        self._s2inference_vectorise([content_chunk], self.modality, self.model_config)]
            return self._s2inference_vectorise(content_chunks, self.modality, self.model_config)


class BatchCachingVectoriser(Vectoriser):
    """
    Generate embeddings when the class is initialised and cache them. When vectorise method is called, just return
    the cached embeddings.
    """
    def __init__(self, modality: Modality, chunks_to_vectorise: List[ContentChunkType], model_config: ModelConfig):
        self.modality = modality
        self.model_config = model_config
        self.embedding_cache = self._vectorise_and_cache(chunks_to_vectorise)

    def _dict_key(self, chunk: ContentChunkType):
        if isinstance(chunk, Image):
            chunk = chunk.convert('RGB')
            pixel_bytes = chunk.tobytes()
            # Use md5 hash for faster hashing.
            return hashlib.md5(pixel_bytes).hexdigest()
        elif isinstance(chunk, dict):
            # Generate a sorted key-value pairs to ensure consistency.
            return frozenset((k, self._dict_key(v)) for k, v in chunk.items())
        elif isinstance(chunk, Tensor):
            # Convert to a tuple to be hashable  # TODO find a more memory efficient way, maybe hashlib.md5?
            return tuple(chunk.flatten().tolist())
        else:
            return chunk

    def _vectorise_and_cache(self, chunks_to_vectorise: List[ContentChunkType]) -> dict:
        if not chunks_to_vectorise:
            return dict()

        # TODO this might be a breaking change since the create_vectors metrics is not consistent
        #   for individual tensor fields and subfields of multimodal fields
        with RequestMetricsStore.for_request().time(f"add_documents.create_vectors"):
            if self.modality in [Modality.AUDIO, Modality.VIDEO]:
                # audio and video fields has to be vectorised chunk by chunk due to a limitation of languagebind
                embeddings = [vector for content_chunk in chunks_to_vectorise for vector in
                              self._s2inference_vectorise([content_chunk], self.modality, self.model_config)]
            else:
                embeddings = self._s2inference_vectorise(chunks_to_vectorise, self.modality, self.model_config)
            return {self._dict_key(chunk): embeddings[i] for i, chunk in enumerate(chunks_to_vectorise)}

    def vectorise(self, content_chunks: List[ContentChunkType]) -> List[List[float]]:
        return [self.embedding_cache[self._dict_key(chunk)] for chunk in content_chunks]


class TensorFieldContent(BaseModel):
    field_content: str
    field_type: FieldType

    chunks: Optional[List[str]] = []
    content_chunks: Optional[List[ContentChunkType]] = []
    embeddings: Optional[List[List[float]]] = []

    # metadata fields
    is_tensor_field: bool  # whether this is a toplevel tensor field
    is_multimodal_subfield: bool = False
    is_resolved: bool = False
    tensor_field_chunk_count: int = 0

    class Config:
        arbitrary_types_allowed = True

    def _is_audio_or_video(self) -> bool:
        return self.field_type in [FieldType.AudioPointer, FieldType.VideoPointer]

    def _should_add_single_chunk(self) -> bool:
        # for text and image field, we need to add a single chunk if it's a multimodal subfield
        return self.is_multimodal_subfield and not self._is_audio_or_video()

    def populate_chunks_and_embeddings(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        """
        This method is called in two scenarios
        - When we collect custom vector fields, we can directly populate chunks and embeddings
        - When we populate embeddings from existing docs
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.tensor_field_chunk_count = len(chunks)

        if not self._should_add_single_chunk():
            # If we do not need to add another chunk for multimodal subfield, mark it as resolved.
            self.is_resolved = True

    def chunk(self, chunkers: Dict[FieldType, Chunker]):
        if self.field_type not in chunkers:
            raise AddDocumentsError(f'Chunking is not supported for field type: {self.field_type.name}')

        chunker = chunkers[self.field_type]

        # chunk top-level fields
        if not self.chunks and (self.is_tensor_field or self._is_audio_or_video()):
            chunks, content_chunks = chunker.chunk(self.field_content, single_chunk=False)
            self.chunks.extend(chunks)
            self.content_chunks.extend(content_chunks)
            self.tensor_field_chunk_count = len(chunks)

        # chunk subfields of multimodal combo fields
        if self._should_add_single_chunk():
            # We do not chunk subfields of a multimodal combo field, except for audio and video fields.
            # If this field is also a top level tensor field, it might already have the chunk we need.
            # So we check if the single chunk generated by chunker matches the last chunk of the tensor field.
            # If not, we attach the single chunk to the exiting chunk for embedding.
            chunks, content_chunks = chunker.chunk(self.field_content, single_chunk=True)

            if not self.chunks or chunks[0] != self.chunks[-1]:
                self.chunks.extend(chunks)
                self.content_chunks.extend(content_chunks)

    def vectorise(self, vectorisers: Dict[FieldType, Vectoriser]) -> None:
        if self.field_type not in vectorisers:
            raise AddDocumentsError(f'Vectorisation is not supported for field type: {self.field_type.name}')

        if not self.content_chunks:
            return

        embeddings = vectorisers[self.field_type].vectorise(self.content_chunks)
        self.embeddings.extend(embeddings)
        self.content_chunks = []  # drop it after vectorisation so memory can be freed
        self.is_resolved = True

    @property
    def tensor_field_chunks(self):
        return self.chunks[:self.tensor_field_chunk_count]

    @property
    def tensor_field_embeddings(self):
        return self.embeddings[:self.tensor_field_chunk_count]

    @property
    def sub_field_chunk(self):
        if not self.chunks or not self.is_multimodal_subfield:
            return None
        elif self._is_audio_or_video():
            return self.field_content
        else:
            return self.chunks[-1]

    @property
    def sub_field_embedding(self):
        if not self.embeddings or not self.is_multimodal_subfield:
            return None
        elif self._is_audio_or_video():
            return np.mean(np.array(self.embeddings), axis=0).tolist()
        else:
            return self.embeddings[-1]


class MultiModalTensorFieldContent(TensorFieldContent):
    weights: Dict[str, float]
    subfields: Dict[str, TensorFieldContent] = dict()
    normalize_embeddings: bool

    @property
    def tensor_field_chunks(self):
        if self.chunks:
            # populated from existing tensor
            return super().tensor_field_chunks

        if not self.subfields:
            return []

        subfield_chunks = {subfield: self.subfields[subfield].sub_field_chunk for subfield in self.weights.keys()
                           if subfield in self.subfields}
        return [json.dumps(subfield_chunks)]

    @property
    def tensor_field_embeddings(self):
        if self.embeddings:
            # populated from existing tensor
            return super().tensor_field_embeddings

        if not self.subfields:
            return []

        combo_embeddings = [
            np.array(self.subfields[subfield].sub_field_embedding) * weight for subfield, weight in self.weights.items()
            if subfield in self.subfields
        ]

        vector_chunk = np.squeeze(np.mean(combo_embeddings, axis=0))
        if self.normalize_embeddings:
            # TODO check if the norm can be 0
            vector_chunk = vector_chunk / np.linalg.norm(vector_chunk)

        return [vector_chunk.tolist()]


class TensorFieldsContainer:

    def __init__(self, tensor_fields: List[str], custom_vector_fields: List[str],
                 multimodal_combo_fields: dict, should_normalise_custom_vector: bool):
        self._tensor_field_map: Dict[str, Dict[str, TensorFieldContent]] = dict()
        self._tensor_fields = set(tensor_fields)
        self._custom_tensor_fields: Set[str] = set(custom_vector_fields)
        self._should_normalise_custom_vector = should_normalise_custom_vector
        self._multimodal_combo_fields = multimodal_combo_fields
        self._multimodal_sub_field_reverse_map: Dict[str, Set[str]] = dict()

        for field_name, weights in self._multimodal_combo_fields.items():
            for sub_field in weights.keys():
                if sub_field not in self._multimodal_sub_field_reverse_map:
                    self._multimodal_sub_field_reverse_map[sub_field] = set()
                self._multimodal_sub_field_reverse_map[sub_field].add(field_name)

    def is_custom_tensor_field(self, field_name: str) -> bool:
        return field_name in self._custom_tensor_fields

    def is_multimodal_field(self, field_name: str) -> bool:
        return field_name in self._multimodal_combo_fields

    def get_multimodal_field_mapping(self, field_name: str) -> Optional[dict]:
        return self._multimodal_combo_fields.get(field_name, None)

    def get_multimodal_sub_fields(self) -> Set[str]:
        return set(self._multimodal_sub_field_reverse_map.keys())

    def remove_doc(self, doc_id: str):
        if doc_id in self._tensor_field_map:
            del self._tensor_field_map[doc_id]

    def _add_tensor_field_content(self, doc_id: str, field_name: str, content: TensorFieldContent) -> None:
        if doc_id not in self._tensor_field_map:
            self._tensor_field_map[doc_id] = dict()
        self._tensor_field_map[doc_id][field_name] = content

    def tensor_fields_to_vectorise(self, *types: FieldType) -> Generator[str, str, TensorFieldContent]:
        for doc_id in list(self._tensor_field_map.keys()):
            fields = self._tensor_field_map[doc_id]
            for field_name, tensor_field_content in fields.items():
                if doc_id not in self._tensor_field_map:
                    # removed during interation due to error handling
                    break

                if tensor_field_content.field_type not in types:
                    # type does not match
                    continue

                if tensor_field_content.is_resolved:
                    # already vectorised (from existing tensor), skip
                    continue

                def need_vectorisation_as_toplevel_field():
                    return tensor_field_content.is_tensor_field and not tensor_field_content.embeddings

                def need_vectorisation_as_subfield():
                    return (tensor_field_content.is_multimodal_subfield and
                            any([not fields[combo_field].is_resolved
                                 for combo_field in self._multimodal_sub_field_reverse_map[field_name]
                                 if combo_field in fields]))

                if not need_vectorisation_as_toplevel_field() and not need_vectorisation_as_subfield():
                    continue

                yield doc_id, field_name, tensor_field_content

    def get_tensor_field_content(self, doc_id: str) -> Dict[str, TensorFieldContent]:
        return {field_name: content for field_name, content in self._tensor_field_map.get(doc_id, dict()).items()
                if content.is_tensor_field and content.tensor_field_chunks}

    def populate_tensor_from_existing_doc(self, existing_marqo_doc: Dict[str, Any],
                                          existing_multimodal_weights: Dict[str, Dict[str, float]]) -> None:
        doc_id = existing_marqo_doc[MARQO_DOC_ID]

        if doc_id not in self._tensor_field_map:
            return

        doc = self._tensor_field_map[doc_id]

        for field_name, tensor_content in doc.items():
            if not tensor_content.is_tensor_field:
                # If this is not top level tensor field, we do not populate from existing tensor
                # TODO confirm if this is expected for unstructured as well
                continue

            if tensor_content.embeddings:
                # Already populated, might be a custom vector
                continue

            if field_name in existing_multimodal_weights:
                # for multimodal_combo fields

                if tensor_content.field_type != FieldType.MultimodalCombination:
                    # Field with the same name is not a multimodal field in this batch
                    continue

                weights = cast(MultiModalTensorFieldContent, tensor_content).weights
                if existing_multimodal_weights[field_name] != weights:
                    # mapping config is different, need to re-vectorise
                    continue

                if any([sub_field not in existing_marqo_doc or sub_field not in doc or
                        existing_marqo_doc[sub_field] != doc[sub_field].field_content for sub_field in weights.keys()]):
                    # If content of any subfields does not match
                    continue

            else:
                # for other tensor fields

                if field_name not in existing_marqo_doc:
                    # This is a new field added to the doc, we need to vectorise it
                    continue

                if existing_marqo_doc[field_name] != tensor_content.field_content:
                    # Field content has changed, we need to re-vectorise
                    continue

            if (constants.MARQO_DOC_TENSORS not in existing_marqo_doc or
                    field_name not in existing_marqo_doc[constants.MARQO_DOC_TENSORS]):
                # This field is not a tensor field in existing doc, we need to vectorise
                continue

            existing_tensor = existing_marqo_doc[constants.MARQO_DOC_TENSORS][field_name]
            tensor_content.populate_chunks_and_embeddings(existing_tensor[constants.MARQO_DOC_CHUNKS],
                                                          existing_tensor[constants.MARQO_DOC_EMBEDDINGS])

    def collect(self, doc_id: str, field_name: str, field_content: Any, field_type: Optional[FieldType]) -> Any:
        if field_name not in self._tensor_fields and field_name not in self._multimodal_sub_field_reverse_map:
            # not tensor fields, no need to collect
            return field_content

        if self.is_custom_tensor_field(field_name):
            return self._collect_custom_vector_field(doc_id, field_name, field_content)

        if self.is_multimodal_field(field_name):
            raise AddDocumentsError(
                f"Field {field_name} is a multimodal combination field and cannot be assigned a value."
            )

        if not isinstance(field_content, str):
            raise AddDocumentsError(
                f'Invalid type {type(field_content)} for tensor field {field_name}'
            )

        self._add_tensor_field_content(
            doc_id, field_name, TensorFieldContent(
                field_content=field_content,
                field_type=field_type,
                is_tensor_field=field_name in self._tensor_fields,
                is_multimodal_subfield=field_name in self._multimodal_sub_field_reverse_map
            )
        )
        return field_content

    def _collect_custom_vector_field(self, doc_id, field_name, field_content):
        content = field_content['content']
        embedding = field_content['vector']

        if self._should_normalise_custom_vector:
            # normalise custom vector
            magnitude = np.linalg.norm(np.array(embedding), axis=-1, keepdims=True)
            if magnitude == 0:
                raise AddDocumentsError(f"Field {field_name} has zero magnitude vector, cannot normalize.")
            embedding = (np.array(embedding) / magnitude).tolist()

        tensor_field_content = TensorFieldContent(
            field_content=content,
            field_type=FieldType.CustomVector,
            is_tensor_field=True,
            is_multimodal_subfield=False,  # for now custom vectors can only be top level
        )

        tensor_field_content.populate_chunks_and_embeddings([content], [embedding])
        self._add_tensor_field_content(doc_id, field_name, tensor_field_content)

        return content

    def collect_multi_modal_fields(self, doc_id: str, normalize_embeddings: bool):
        for field_name, weights in self._multimodal_combo_fields.items():
            self._add_tensor_field_content(doc_id, field_name, MultiModalTensorFieldContent(
                weights=weights,
                field_content='',
                field_type=FieldType.MultimodalCombination,
                subfields={subfield: self._tensor_field_map[doc_id][subfield] for subfield in weights.keys()
                           if doc_id in self._tensor_field_map and subfield in self._tensor_field_map[doc_id]},
                is_tensor_field=True,
                is_multimodal_subfield=False,
                normalize_embeddings=normalize_embeddings
            ))
            yield field_name, weights
