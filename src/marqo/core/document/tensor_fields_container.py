import json
from functools import cached_property
from typing import List, Dict, Set, Optional, Any, Generator, Union, Tuple, Protocol, cast

import numpy as np
from pydantic.main import BaseModel

from marqo.core import constants
from marqo.core.document.add_documents_handler import AddDocumentsError
from marqo.core.models.marqo_index import FieldType
from marqo.s2_inference.clip_utils import _is_image
from marqo.tensor_search import enums


class Chunker(Protocol):
    def __call__(self, field_content: str, single_chunk: bool = False) -> Tuple[List[str], List[str]]:
        ...


class Vectoriser(Protocol):
    def __call__(self, content_chunks: List[str], field_type: FieldType) -> List[List[float]]:
        ...


class TensorFieldContent(BaseModel):
    field_content: str
    field_type: FieldType
    chunks: Optional[List[str]] = None
    content_chunks: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None

    # metadata fields
    is_tensor_field: bool
    is_multimodal_subfield: bool
    is_resolved: bool = False
    tensor_field_chunk_count: int = 0

    def populate_from_existing_tensor(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        if not self.is_multimodal_subfield or len(chunks) == 1:
            self.is_resolved = True

    def chunk(self, chunkers: Dict[FieldType, Chunker]) -> None:
        if self.field_type not in chunkers or self.is_resolved:
            return

        if self.is_tensor_field and not self.chunks:
            self.chunks, self.content_chunks = chunkers[self.field_type](self.field_content, single_chunk=False)
            self.tensor_field_chunk_count = len(self.chunks)

        if self.is_multimodal_subfield:

            chunks, content_chunks = chunkers[self.field_type](self.field_content, single_chunk=True)
            if not self.chunks:
                self.chunks, self.content_chunks = chunks, content_chunks
            else:
                # attach the single chunk to chunks only if chunks do not match
                if chunks[0] != self.chunks[-1]:
                    self.chunks.extend(chunks)
                    self.content_chunks.extend(content_chunks)

    def vectorise(self, vectoriser: Vectoriser) -> None:
        pass

    @property
    def tensor_field_chunks(self):
        return self.chunks[:self.tensor_field_chunk_count] if self.chunks else []

    @property
    def tensor_field_embeddings(self):
        return self.embeddings[:self.tensor_field_chunk_count] if self.chunks else []

    @property
    def sub_field_chunk(self):
        return self.chunks[-1] if self.chunks else None

    @property
    def sub_field_embedding(self):
        return self.embeddings[-1] if self.embeddings else None


class MultiModalTensorFieldContent(TensorFieldContent):
    weights: Dict[str, float]
    subfields: Dict[str, TensorFieldContent] = dict()
    normalize_embeddings: bool

    @property
    def tensor_field_chunks(self):
        subfield_chunks = {subfield: self.subfields[subfield].sub_field_chunk for subfield in self.weights.keys()}
        return [json.dumps(subfield_chunks)]

    @property
    def tensor_field_embeddings(self):
        combo_embeddings = [
            np.array(self.subfields[subfield].sub_field_embedding) * weight for subfield, weight in self.weights.items()
        ]
        vector_chunk = np.squeeze(np.mean(combo_embeddings, axis=0))
        if self.normalize_embeddings:
            vector_chunk = vector_chunk / np.linalg.norm(vector_chunk)

        return [vector_chunk]


class TensorFieldsContainer:

    def __init__(self, tensor_fields: List[str], mappings: dict, treat_url_as_media: bool = False):
        self._tensor_field_map: Dict[str, Dict[str, TensorFieldContent]] = dict()
        self._tensor_fields = set(tensor_fields)
        self._custom_tensor_fields: Set[str] = set()
        self._multimodal_combo_fields = dict()
        self._multimodal_sub_fields = set()
        self._multimodal_sub_field_reverse_map: Dict[str, Set[str]] = dict()
        self._treat_url_as_media = treat_url_as_media

        for field_name, mapping in mappings.items():
            field_type = mapping.get("type", None)
            if field_type == enums.MappingsObjectType.custom_vector:
                self._custom_tensor_fields.add(field_name)
            elif field_type == enums.MappingsObjectType.multimodal_combination:
                self._multimodal_combo_fields[field_name] = mapping
                for sub_field in mapping["weights"].keys():
                    self._multimodal_sub_fields.add(sub_field)
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
        return self._multimodal_sub_fields

    def remove_doc(self, doc_id: str):
        if doc_id in self._tensor_field_map:
            del self._tensor_field_map[doc_id]

    def add_tensor_field_content(self, doc_id: str, field_name: str, content: TensorFieldContent) -> None:
        if doc_id not in self._tensor_field_map:
            self._tensor_field_map[doc_id] = dict()
        self._tensor_field_map[doc_id][field_name] = content

    def tensor_fields_to_vectorise(self, *types: FieldType) -> Generator[str, str, TensorFieldContent]:
        for doc_id, fields in self._tensor_field_map.items():
            for field_name, tensor_field_content in fields.items():
                if doc_id not in self._tensor_field_map:
                    # removed during interation due to error handling
                    break

                if tensor_field_content.field_type not in types:
                    # type does not match
                    continue

                if tensor_field_content.embeddings is not None:
                    # already vectorised (from existing tensor), skip
                    continue

                if (field_name not in self._tensor_fields and
                        field_name in self._multimodal_sub_field_reverse_map and
                        all([fields[field].embeddings is not None
                             for field in self._multimodal_sub_field_reverse_map[field_name]])):
                    # if field is only used by multimodal fields and all multimodal fields using it are vectorised
                    continue

                yield doc_id, field_name, tensor_field_content

    def get_tensor_field_content(self, doc_id: str) -> Dict[str, TensorFieldContent]:
        return {field_name: content for field_name, content in self._tensor_field_map.get(doc_id, dict()).items()
                if content.is_tensor_field}

    def populate_tensor_from_existing_doc(self, doc_id: str, existing_marqo_doc: Dict[str, Any],
                                          existing_multimodal_mappings: dict) -> None:
        if doc_id not in self._tensor_field_map:
            return
        doc = self._tensor_field_map[doc_id]

        for field_name, tensor_content in doc.items():
            if tensor_content.embeddings:
                # Already populated, might be a custom vector
                continue

            if field_name not in existing_marqo_doc:
                # This is a new field added to the doc, we need to vectorise it
                continue

            if field_name in existing_multimodal_mappings:
                if tensor_content.field_type != FieldType.MultimodalCombination:
                    # Field with the same name is not a multimodal field in this batch
                    continue

                weights = cast(MultiModalTensorFieldContent, tensor_content).weights
                if existing_multimodal_mappings[field_name]['weight'] != weights:
                    # mapping config is different, need to re-vectorise
                    continue

                if any([existing_marqo_doc[sub_field] != doc[sub_field].field_content for sub_field in weights.keys()]):
                    # If content of any subfields does not match
                    continue

            elif existing_marqo_doc[field_name] != tensor_content.field_content:
                # Field content has changed, we need to re-vectorise
                continue

            if (constants.MARQO_DOC_TENSORS not in existing_marqo_doc or
                    field_name in existing_marqo_doc[constants.MARQO_DOC_TENSORS]):
                # This field is not a tensor field in existing doc, we need to vectorise
                continue

            existing_tensor = existing_marqo_doc[constants.MARQO_DOC_TENSORS][field_name]
            tensor_content.chunks = existing_tensor[constants.MARQO_DOC_CHUNKS]
            tensor_content.embeddings = existing_tensor[constants.MARQO_DOC_EMBEDDINGS]

    def collect(self, doc_id: str, field_name: str, field_content: Any) -> Any:
        if field_name not in self._tensor_fields and field_name not in self._multimodal_sub_fields:
            # not tensor fields, no need to collect
            return field_content

        if self.is_custom_tensor_field(field_name):
            content = field_content['content']
            vector = field_content['vector']
            self.add_tensor_field_content(
                doc_id, field_name, TensorFieldContent(
                    field_content=content,
                    field_type=FieldType.CustomVector,
                    chunks=[content],
                    embeddings=[vector],
                    is_tensor_field=True,
                    is_multimodal_subfield=False
                )
            )
            return content

        if self.is_multimodal_field(field_name):
            raise AddDocumentsError(
                f"Multimodal_field {field_name} cannot have value assigned"
            )

        if not isinstance(field_content, str):
            raise AddDocumentsError(
                f"Field of type {type(field_content).__name__} cannot be tensor field"
            )

        self.add_tensor_field_content(
            doc_id, field_name, TensorFieldContent(
                field_content=field_content,
                field_type=self._infer_field_type(field_content),
                is_tensor_field=field_name in self._tensor_fields,
                is_multimodal_subfield=field_name in self._multimodal_sub_fields
            )
        )
        return field_content

    # TODO this logic is unstructured only. We need to extract out
    def _infer_field_type(self, field_content: str):
        if self._treat_url_as_media and _is_image(field_content):
            return FieldType.ImagePointer

        return FieldType.Text

    def collect_multi_modal_fields(self, doc_id: str, normalize_embeddings: bool):
        for field_name, mapping in self._multimodal_combo_fields.items():
            self.add_tensor_field_content(doc_id, field_name, MultiModalTensorFieldContent(
                weights=mapping['weights'], field_content='', field_type=FieldType.MultimodalCombination,
                subfields={subfield: self._tensor_field_map[doc_id][subfield] for subfield in mapping['weights'].keys()},
                is_tensor_field=True, is_multimodal_subfield=False, normalize_embeddings=normalize_embeddings
            ))
            yield field_name, mapping

