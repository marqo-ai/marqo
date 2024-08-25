import json
from typing import Dict, List

from marqo.core.exceptions import FieldTypeMismatchError
from marqo.core.models.marqo_index import FieldType, FieldFeature, SemiStructuredMarqoIndex, TensorField, Field
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.s2_inference.clip_utils import _is_image


class FieldRequestCollector:
    def __init__(self, marqo_index: SemiStructuredMarqoIndex, expected_tensor_fields: List[str]):
        self.marqo_index = marqo_index
        self.expected_tensor_fields = expected_tensor_fields
        self.fields_to_add: Dict[str, Field] = dict()
        self.tensor_fields_to_add = self._get_missing_tensor_fields(expected_tensor_fields)

    def collect_from_mappings(self, mappings: dict):
        for field_name, mapping in mappings.items():
            if mapping.get("type", None) == FieldType.MultimodalCombination:
                if self._should_add(field_name, FieldType.MultimodalCombination):
                    self.fields_to_add[field_name] = Field(name=field_name, type=FieldType.MultimodalCombination,
                                                           dependentFields=mapping["weights"])
            elif mapping.get("type", None) == FieldType.CustomVector:
                if self._should_add(field_name, FieldType.CustomVector):
                    self.fields_to_add[field_name] = Field(name=field_name, type=FieldType.CustomVector)
            else:
                raise ValueError(f'Unsupported field type {mapping["type"]} in {json.dumps(mapping)}')

    def collect_from_field(self, field_name: str, field_content: str) -> None:
        if self.marqo_index.treat_urls_and_pointers_as_images and _is_image(field_content):
            expected_type = FieldType.ImagePointer
        else:
            expected_type = FieldType.Text

        if self._should_add(field_name, expected_type):
            if expected_type == FieldType.ImagePointer:
                self.fields_to_add[field_name] = Field(name=field_name, type=FieldType.ImagePointer)
            else:
                self.fields_to_add[field_name] = Field(name=field_name, type=FieldType.Text,
                                                       features=[FieldFeature.LexicalSearch],
                                                       lexical_field_name=f'{SemiStructuredVespaSchema._FIELD_INDEX_PREFIX}{field_name}')

    def _should_add(self, field_name: str, expected_type: FieldType) -> bool:
        if field_name in self.fields_to_add:
            if self.fields_to_add[field_name].type != expected_type:
                raise FieldTypeMismatchError(
                    f'Field {field_name} already define with {self.fields_to_add[field_name].type} '
                    f'type in other documents')
            else:
                return False

        if field_name in self.marqo_index.all_field_map:
            existing_type = self.marqo_index.all_field_map[field_name].type
            if existing_type == expected_type:
                return False
            else:
                raise FieldTypeMismatchError(f'Field {field_name} exists in index {self.marqo_index.name}, '
                                             f'but with a different type type {existing_type}')

        return True

    def should_update_marqo_index(self):
        return self.fields_to_add or self.tensor_fields_to_add

    def updated_marqo_index(self) -> SemiStructuredMarqoIndex:
        self.marqo_index.fields.extend(self.fields_to_add.values())
        self.marqo_index.tensor_fields.extend(self.tensor_fields_to_add)
        return self.marqo_index

    def _get_missing_tensor_fields(self, expected_tensor_fields):
        missing_tensor_fields = []
        for field_name in expected_tensor_fields:
            if field_name not in self.marqo_index.tensor_field_map:
                missing_tensor_fields.append(TensorField(
                    name=field_name,
                    chunk_field_name=f'{SemiStructuredVespaSchema._FIELD_CHUNKS_PREFIX}{field_name}',
                    embeddings_field_name=f'{SemiStructuredVespaSchema._FIELD_EMBEDDING_PREFIX}{field_name}',
                ))
        return missing_tensor_fields
