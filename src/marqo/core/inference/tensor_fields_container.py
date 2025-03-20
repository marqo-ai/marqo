import json
from collections import defaultdict
from typing import List, Dict, Set, Optional, Any, cast, Callable

import numpy as np
from pydantic import BaseModel

from marqo.core import constants
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.exceptions import AddDocumentsError
from marqo.core.inference.api import Modality
from marqo.core.models.marqo_index import FieldType


class TensorField(BaseModel):
    doc_id: str
    field_name: str
    field_content: str
    field_type: Optional[FieldType] = None
    modality: Optional[Modality] = None  # Actual modality is inferred from the content

    is_top_level_tensor_field: bool
    is_multimodal_subfield: bool = False

    # set to True when all embeddings are populated (from existing tensor or vectorisation)
    is_resolved: bool = False

    # chunks and embeddings for top level tensor fields or audio and video fields
    chunks: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None

    # for text and image type, we need to calculate an extra embedding without chunking if it's a subfield
    multimodal_subfield_embedding: Optional[List[float]] = None

    class Config:
        extra = "forbid"
        # Please note we intentionally don't use StrictBaseModel here to avoid deep copy of TensorField when assigned
        # to a MultiModalTensorField. This is because the embeddings and chunks are populated post the construction
        # of a MultiModalTensorField. If deeply copied, these fields won't be updated in the MultiModalTensorField.
        # When validate_assignment set to True, the deep copy is enforced.
        validate_assignment = False

    def is_audio_or_video(self) -> bool:
        return self.modality in {Modality.AUDIO, Modality.VIDEO}

    def is_unresolved_top_level_field(self) -> bool:
        return self.is_top_level_tensor_field and not self.embeddings

    def is_unresolved_multimodal_subfield(self) -> bool:
        if not self.is_multimodal_subfield:
            return False

        if self.is_audio_or_video():
            # audio and video are always chunked
            return not self.embeddings
        else:
            # text and image fields stores extra embedding for subfield without chunking
            # in multimodal_subfield_embedding field
            return not self.multimodal_subfield_embedding

    def populate_chunks_and_embeddings(self, chunks: List[str], embeddings: List[List[float]],
                                       for_top_level_field: bool = True) -> None:
        """
        Populate chunks and embeddings to this tensor field. Chunks and embeddings can come from the following sources:
        - For custom vector field, field content is the single chunk, and single embedding is provided by user
        - For existing tensor, chunks and embeddings are from the original document
        - Vectorisation result from the inference process

        When all needed embeddings are populated, the tensor field is marked as resolved. We only select unresolved
        fields when doing vectorisation.

        Args:
            chunks (List[str]): List of chunk keys
            embeddings (List[List[float]]): List of embeddings
            for_top_level_field (bool): if the embedding is for top level field or subfields
        """
        if (not chunks or not embeddings
                or not isinstance(chunks, list) or not isinstance(chunks[0], str)
                or not isinstance(embeddings, list) or not isinstance(embeddings[0], list)
                or not isinstance(embeddings[0][0], (float, int))):  # custom vector can be integers
            raise ValueError(f'Invalid chunks and embeddings for doc: {self.doc_id}, field: {self.field_name}')

        if len(chunks) != len(embeddings):
            raise ValueError(f'Chunk and embedding size does not match for doc: {self.doc_id}, field: {self.field_name}'
                             f': chunk size: {len(chunks)}, embedding size: {len(embeddings)}')

        if for_top_level_field or self.is_audio_or_video():
            self.chunks = chunks
            self.embeddings = embeddings

            if not self.is_multimodal_subfield or self.is_audio_or_video():
                self.is_resolved = True
            else:
                # for image and text fields which are both top level field and subfield
                if len(chunks) == 1 and chunks[0] == self.field_content:
                    # This is an optimisation that if only one chunk is returned and the chunk equals the content
                    # the single embedding can be used as a subfield embedding as well, saving one vectorise call
                    self.multimodal_subfield_embedding = embeddings[0]
                    self.is_resolved = True
        else:
            # for image and text subfields only, it should not be chunked
            if len(chunks) != 1:
                raise ValueError(f'{self.field_name} of doc: {self.doc_id} is a subfield and should not be chunked')

            self.multimodal_subfield_embedding = embeddings[0]
            self.is_resolved = True

    @property
    def tensor_field_chunks(self) -> List[str]:
        if not self.is_top_level_tensor_field:
            raise ValueError(f'{self.field_name} of doc: {self.doc_id} is not a top level tensor field')

        return self.chunks

    @property
    def tensor_field_embeddings(self) -> List[List[float]]:
        if not self.is_top_level_tensor_field:
            raise ValueError(f'{self.field_name} of doc: {self.doc_id} is not a top level tensor field')

        return self.embeddings

    @property
    def subfield_chunk(self) -> Optional[str]:
        if not self.is_multimodal_subfield:
            raise ValueError(f'{self.field_name} of doc: {self.doc_id} is not a subfield')
        else:
            return self.field_content

    @property
    def subfield_embedding(self) -> Optional[List[float]]:
        if not self.is_multimodal_subfield:
            raise ValueError(f'{self.field_name} of doc: {self.doc_id} is not a subfield')
        elif self.is_audio_or_video():
            return np.mean(np.array(self.embeddings), axis=0).tolist()
        else:
            return self.multimodal_subfield_embedding


class MultiModalTensorField(TensorField):
    weights: Dict[str, float]
    subfields: Dict[str, TensorField] = dict()
    normalize_embeddings: bool

    @property
    def tensor_field_chunks(self):
        if self.chunks:
            # populated from existing tensor
            return self.chunks

        if not self.subfields:
            return []

        subfield_chunks = {subfield: self.subfields[subfield].subfield_chunk for subfield in self.weights.keys()
                           if subfield in self.subfields}
        return [json.dumps(subfield_chunks)]

    @property
    def tensor_field_embeddings(self):
        if self.embeddings:
            # populated from existing tensor
            return self.embeddings

        if not self.subfields:
            return []

        combo_embeddings = [
            np.array(self.subfields[subfield].subfield_embedding) * weight for subfield, weight in self.weights.items()
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
        self._tensor_field_map: Dict[str, Dict[str, TensorField]] = defaultdict(dict)
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

    def has_unresolved_parent_field(self, field: TensorField) -> bool:
        """
        Check if a field has unresolved parent field

        Returns:
            False if the field is not a subfield or all of its parent fields are resolved
            True if the field has any unresolved parent field
        """
        if not field.is_multimodal_subfield:
            return False

        return any([not self._tensor_field_map[field.doc_id][combo_field].is_resolved
                    for combo_field in self._multimodal_sub_field_reverse_map[field.field_name]
                    if field.doc_id in self._tensor_field_map
                    and combo_field in self._tensor_field_map[field.doc_id]])

    def select_unresolved_tensor_fields(self, predicate: Optional[Callable[[TensorField], bool]] = None) \
            -> List[TensorField]:
        return [tensor_field
                for doc_id, fields in self._tensor_field_map.items()
                for field_name, tensor_field in fields.items()
                if not tensor_field.is_resolved
                and tensor_field.field_type not in {FieldType.CustomVector, FieldType.MultimodalCombination}
                and (predicate is None or predicate(tensor_field))]

    def get_tensor_field_content(self, doc_id: str) -> Dict[str, TensorField]:
        return {field_name: content for field_name, content in self._tensor_field_map.get(doc_id, dict()).items()
                if content.is_top_level_tensor_field and content.tensor_field_chunks}

    def populate_tensor_from_existing_doc(self, existing_marqo_doc: Dict[str, Any],
                                          existing_multimodal_weights: Dict[str, Dict[str, float]]) -> None:
        doc_id = existing_marqo_doc[MARQO_DOC_ID]

        if doc_id not in self._tensor_field_map:
            return

        doc = self._tensor_field_map[doc_id]

        for field_name, tensor_content in doc.items():
            if not tensor_content.is_top_level_tensor_field:
                # If this is not top level tensor field, we do not populate from existing tensor
                continue

            if tensor_content.embeddings:
                # Already populated, might be a custom vector
                continue

            if field_name in existing_multimodal_weights:
                # for multimodal_combo fields

                if tensor_content.field_type != FieldType.MultimodalCombination:
                    # Field with the same name is not a multimodal field in this batch
                    continue

                weights = cast(MultiModalTensorField, tensor_content).weights
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

    def collect(self, doc_id: str, field_name: str, field_content: Any, field_type: Optional[FieldType] = None) -> Any:
        """
        Collect tensor field content from the document if it is a tensor field.

        Args:
            doc_id: document id
            field_name: name of the field
            field_content: content of the field
            field_type: type of field, only present for structured index
        Returns:
            The field content
        """
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

        field = TensorField(
            doc_id=doc_id,
            field_name=field_name,
            field_content=field_content,
            field_type=field_type,
            is_top_level_tensor_field=field_name in self._tensor_fields,
            is_multimodal_subfield=field_name in self._multimodal_sub_field_reverse_map
        )
        self._tensor_field_map[doc_id][field_name] = field
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

        field = TensorField(
            doc_id=doc_id,
            field_name=field_name,
            field_content=content,
            field_type=FieldType.CustomVector,
            is_top_level_tensor_field=True,
            is_multimodal_subfield=False,  # for now custom vectors can only be top level
        )

        field.populate_chunks_and_embeddings([content], [embedding])
        self._tensor_field_map[doc_id][field_name] = field

        return content

    def collect_multi_modal_fields(self, doc_id: str, normalize_embeddings: bool):
        for field_name, weights in self._multimodal_combo_fields.items():
            field = MultiModalTensorField(
                doc_id=doc_id,
                field_name=field_name,
                weights=weights,
                field_content='',
                field_type=FieldType.MultimodalCombination,
                subfields={subfield: self._tensor_field_map[doc_id][subfield] for subfield in weights.keys()
                           if doc_id in self._tensor_field_map and subfield in self._tensor_field_map[doc_id]},
                is_top_level_tensor_field=True,
                is_multimodal_subfield=False,
                normalize_embeddings=normalize_embeddings
            )
            self._tensor_field_map[doc_id][field_name] = field
            yield field_name, weights
