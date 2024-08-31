from typing import List, Dict, Tuple, Set, Optional, Any, Generator

from pydantic.main import BaseModel

from marqo.api.exceptions import InvalidArgError
from marqo.core import constants
from marqo.core.models.marqo_index import FieldType
from marqo.s2_inference.clip_utils import _is_image
from marqo.tensor_search import enums


class TensorFieldContent(BaseModel):
    field_content: str
    field_type: FieldType
    chunks: Optional[List[str]] = None
    content_chunks: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None


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

    def add_tensor_field_content(self, doc_id: str, field_name: str, field_content: str, field_type: FieldType,
                                 chunks: Optional[List[str]] = None,
                                 embeddings: Optional[List[List[float]]] = None) -> None:
        if doc_id not in self._tensor_field_map:
            self._tensor_field_map[doc_id] = dict()

        self._tensor_field_map[doc_id][field_name] = TensorFieldContent(
            field_content=field_content,
            field_type=field_type,
            chunks=chunks,
            embeddings=embeddings
        )

    def tensor_fields_to_vectorise(self, *types: FieldType) -> Generator[str, str, TensorFieldContent]:
        for doc_id, fields in self._tensor_field_map.items():
            for field_name, tensor_field_content in fields.items():
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
                if field_name in self._tensor_fields}

    def populate_tensor_from_existing_doc(self, doc_id, existing_marqo_doc):
        if doc_id not in self._tensor_field_map:
            return

        for field_name, tensor_content in self._tensor_field_map[doc_id].items():
            if tensor_content.embeddings:
                # Already populated, might be a custom vector
                continue

            if field_name not in existing_marqo_doc:
                # This is a new field added to the doc, we need to vectorise it
                continue

            if existing_marqo_doc[field_name] != tensor_content.field_content:
                # Field content has changed, we need to re-vectorise
                continue

            if (constants.MARQO_DOC_TENSORS not in existing_marqo_doc or
                    field_name in existing_marqo_doc[constants.MARQO_DOC_TENSORS]):
                # This field is not a tensor field in existing doc, we need to vectorise
                continue

            # TODO see if this handle multimodal fields

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
                doc_id, field_name, content, field_type=FieldType.CustomVector, chunks=[content], embeddings=[vector]
            )
            return content

        if self.is_multimodal_field(field_name):
            # TODO handle multimodal
            pass

        if not isinstance(field_content, str):
            # TODO better error message
            raise InvalidArgError(
                f"Field of type {type(field_content).__name__} cannot be tensor field"
            )

        self.add_tensor_field_content(
            doc_id, field_name, field_content, field_type=self._infer_field_type(field_content)
        )
        return field_content

    def _infer_field_type(self, field_content: str):
        if self._treat_url_as_media and _is_image(field_content):
            return FieldType.ImagePointer

        return FieldType.Text
