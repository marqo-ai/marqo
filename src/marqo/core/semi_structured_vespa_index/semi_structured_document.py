import json
from typing import List, Dict, Any

from pydantic import Field

from marqo.base_model import MarqoBaseModel
from marqo.core import constants as index_constants, constants
from marqo.core.exceptions import VespaDocumentParsingError, MarqoDocumentParsingError, InvalidFieldNameError, \
    InvalidTensorFieldError
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.semi_structured_vespa_index import common


class SemiStructuredVespaDocumentFields(MarqoBaseModel):
    """A class with fields that are common to all Vespa documents."""
    marqo__id: str = Field(alias=common.VESPA_FIELD_ID)

    short_string_fields: Dict[str, str] = Field(default_factory=dict, alias=common.SHORT_STRINGS_FIELDS)
    string_arrays: List[str] = Field(default_factory=list, alias=common.STRING_ARRAY)
    int_fields: Dict[str, int] = Field(default_factory=dict, alias=common.INT_FIELDS)
    bool_fields: Dict[str, int] = Field(default_factory=dict, alias=common.BOOL_FIELDS)
    float_fields: Dict[str, float] = Field(default_factory=dict, alias=common.FLOAT_FIELDS)
    score_modifiers_fields: Dict[str, Any] = Field(default_factory=dict, alias=common.SCORE_MODIFIERS)
    vespa_multimodal_params: Dict[str, str] = Field(default_factory=str, alias=common.VESPA_DOC_MULTIMODAL_PARAMS)


class SemiStructuredVespaDocument(MarqoBaseModel):
    """A helper class to handle the conversion between Vespa and Marqo documents for an semi-structured index.
    The object can be instantiated from a Marqo document using the from_marqo_document method,
    or can be instantiated from a Vespa document using the from_vespa_document method.
    """
    id: str
    fixed_fields: SemiStructuredVespaDocumentFields = Field(default_factory=SemiStructuredVespaDocumentFields)
    text_fields: dict = Field(default_factory=dict)
    tensor_fields: dict = Field(default_factory=dict)
    vector_counts: int = Field(default=0, alias=common.FIELD_VECTOR_COUNT)
    match_features: Dict[str, Any] = Field(default_factory=dict, alias=common.VESPA_DOC_MATCH_FEATURES)

    # For hybrid search
    raw_tensor_score: float = None
    raw_lexical_score: float = None

    _VESPA_DOC_FIELDS = "fields"
    _VESPA_DOC_ID = "id"

    @classmethod
    def from_vespa_document(cls, document: Dict, marqo_index: SemiStructuredMarqoIndex) -> "SemiStructuredVespaDocument":
        """
        Instantiate an UnstructuredVespaDocument from a Vespa document.
        Used in get_document_by_id or get_documents_by_ids
        """
        fields = document.get(cls._VESPA_DOC_FIELDS, {})
        tensor_fields = {}
        text_fields = {}
        for field_name in fields:
            if field_name in marqo_index.tensor_subfield_map:
                tensor_fields[field_name] = fields[field_name]
            elif field_name in marqo_index.lexical_field_map:  # lexical fields are returned with prefixed name from get_by_ids
                text_field_name = marqo_index.lexical_field_map[field_name].name
                text_fields[text_field_name] = fields[field_name]
            elif field_name in marqo_index.field_map:   # lexical fields are returned with the original name from search
                text_fields[field_name] = fields[field_name]

        return cls(id=document[cls._VESPA_DOC_ID],
                   fixed_fields=SemiStructuredVespaDocumentFields(**fields),
                   tensor_fields=tensor_fields,
                   text_fields=text_fields,
                   raw_tensor_score=cls.extract_field(fields, common.VESPA_DOC_HYBRID_RAW_TENSOR_SCORE, None),
                   raw_lexical_score=cls.extract_field(fields, common.VESPA_DOC_HYBRID_RAW_LEXICAL_SCORE, None),
                   match_features=cls.extract_field(fields, common.VESPA_DOC_MATCH_FEATURES, dict()),
                   vector_counts=cls.extract_field(fields, common.FIELD_VECTOR_COUNT, 0))

    @classmethod
    def extract_field(cls, fields, name: str, default: Any):
        return fields[name] if name in fields else default

    @classmethod
    def from_marqo_document(cls, document: Dict, marqo_index: SemiStructuredMarqoIndex) -> "SemiStructuredVespaDocument":
        """Instantiate an UnstructuredVespaDocument from a valid Marqo document for feeding to Vespa"""

        if index_constants.MARQO_DOC_ID not in document:
            raise VespaDocumentParsingError(
                f"Unstructured Marqo document does not have a {index_constants.MARQO_DOC_ID} field. "
                f"This should be assigned for a valid document")

        doc_id = document[index_constants.MARQO_DOC_ID]
        instance = cls(id=doc_id, fixed_fields=SemiStructuredVespaDocumentFields(marqo__id=doc_id))

        for field_name, field_content in document.items():
            if field_name in [index_constants.MARQO_DOC_ID, constants.MARQO_DOC_TENSORS]:
                continue
            if isinstance(field_content, str):
                if field_name not in marqo_index.field_map:
                    # All string fields will be added to the index as lexical fields before this convertion happens
                    raise MarqoDocumentParsingError(f'Field {field_name} is not in index {marqo_index.name}')
                field = marqo_index.field_map[field_name]
                instance.text_fields[field.lexical_field_name] = field_content
                if len(field_content) <= marqo_index.filter_string_max_length:
                    instance.fixed_fields.short_string_fields[field_name] = field_content
            elif isinstance(field_content, bool):
                instance.fixed_fields.bool_fields[field_name] = int(field_content)
            elif isinstance(field_content, list) and all(isinstance(elem, str) for elem in field_content):
                instance.fixed_fields.string_arrays.extend([f"{field_name}::{element}" for element in field_content])
            elif isinstance(field_content, int):
                instance.fixed_fields.int_fields[field_name] = field_content
                instance.fixed_fields.score_modifiers_fields[field_name] = field_content
            elif isinstance(field_content, float):
                instance.fixed_fields.float_fields[field_name] = field_content
                instance.fixed_fields.score_modifiers_fields[field_name] = field_content
            elif isinstance(field_content, dict):
                for k, v in field_content.items():
                    if isinstance(v, int):
                        instance.fixed_fields.int_fields[f"{field_name}.{k}"] = v
                        instance.fixed_fields.score_modifiers_fields[f"{field_name}.{k}"] = v
                    elif isinstance(v, float):
                        instance.fixed_fields.float_fields[f"{field_name}.{k}"] = float(v)
                        instance.fixed_fields.score_modifiers_fields[f"{field_name}.{k}"] = v
            else:
                raise VespaDocumentParsingError(
                    f"In document {doc_id}, field {field_name} has an "
                    f"unsupported type {type(field_content)} which has not been validated in advance.")

            # Tensors
            vector_count = 0
            if constants.MARQO_DOC_TENSORS in document:
                for marqo_tensor_field in document[constants.MARQO_DOC_TENSORS]:
                    marqo_tensor_value = document[constants.MARQO_DOC_TENSORS][marqo_tensor_field]

                    cls._verify_marqo_tensor_field_name(marqo_tensor_field, marqo_index)
                    cls._verify_marqo_tensor_field(marqo_tensor_field, marqo_tensor_value)

                    # If chunking an image, chunks will be a list of tuples, hence the str(c)
                    chunks = [str(c) for c in marqo_tensor_value[constants.MARQO_DOC_CHUNKS]]
                    embeddings = marqo_tensor_value[constants.MARQO_DOC_EMBEDDINGS]
                    vector_count += len(embeddings)

                    index_tensor_field = marqo_index.tensor_field_map[marqo_tensor_field]

                    instance.tensor_fields[index_tensor_field.chunk_field_name] = chunks
                    instance.tensor_fields[index_tensor_field.embeddings_field_name] = \
                        {f'{i}': embeddings[i] for i in range(len(embeddings))}

            instance.vector_counts = vector_count

            instance.fixed_fields.vespa_multimodal_params = document.get(common.MARQO_DOC_MULTIMODAL_PARAMS, {})

        return instance

    def to_vespa_document(self) -> Dict[str, Any]:
        """Convert VespaDocumentObject to a Vespa document.
        Empty fields are removed from the document."""
        vespa_fields = {
            **{k: v for k, v in self.fixed_fields.dict(exclude_none=True, by_alias=True).items() if v or v == 0},
            **self.text_fields,
            **self.tensor_fields,
            common.FIELD_VECTOR_COUNT: self.vector_counts,
        }

        return {self._VESPA_DOC_ID: self.id, self._VESPA_DOC_FIELDS: vespa_fields}

    def to_marqo_document(self, marqo_index: SemiStructuredMarqoIndex) -> Dict[str, Any]:
        """Convert VespaDocumentObject to marqo document document structure."""
        marqo_document = {}
        for string_array in self.fixed_fields.string_arrays:
            key, value = string_array.split("::", 1)
            if key not in marqo_document:
                marqo_document[key] = []
            marqo_document[key].append(value)

        # marqo_document.update(self.fixed_fields.short_string_fields)
        marqo_document.update(self.fixed_fields.int_fields)
        marqo_document.update(self.fixed_fields.float_fields)
        marqo_document.update({k: bool(v) for k, v in self.fixed_fields.bool_fields.items()})
        marqo_document[index_constants.MARQO_DOC_ID] = self.fixed_fields.marqo__id

        # text fields
        for field_name, field_content in self.text_fields.items():
            marqo_document[field_name] = field_content

        # tensor fields
        for field_name, field_content in self.tensor_fields.items():
            tensor_field = marqo_index.tensor_subfield_map[field_name]

            if constants.MARQO_DOC_TENSORS not in marqo_document:
                marqo_document[constants.MARQO_DOC_TENSORS] = dict()
            if tensor_field.name not in marqo_document[constants.MARQO_DOC_TENSORS]:
                marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name] = dict()

            if field_name == tensor_field.chunk_field_name:
                marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name][constants.MARQO_DOC_CHUNKS] = field_content
            elif field_name == tensor_field.embeddings_field_name:
                try:
                    marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name][
                        constants.MARQO_DOC_EMBEDDINGS] = list(field_content['blocks'].values())
                except (KeyError, AttributeError, TypeError) as e:
                    raise VespaDocumentParsingError(
                        f'Cannot parse embeddings field {field_name} with value {field_content}'
                    ) from e

        if self.fixed_fields.vespa_multimodal_params:
            marqo_document[common.MARQO_DOC_MULTIMODAL_PARAMS] = dict()
            for multimodal_field_name, serialized_multimodal_params in self.fixed_fields.vespa_multimodal_params.items():
                marqo_document[common.MARQO_DOC_MULTIMODAL_PARAMS][multimodal_field_name] = \
                    json.loads(serialized_multimodal_params)

        # Hybrid search raw scores
        if self.raw_tensor_score is not None:
            marqo_document[index_constants.MARQO_DOC_HYBRID_TENSOR_SCORE] = self.raw_tensor_score
        if self.raw_lexical_score is not None:
            marqo_document[index_constants.MARQO_DOC_HYBRID_LEXICAL_SCORE] = self.raw_lexical_score

        return marqo_document

    @classmethod
    def _verify_marqo_tensor_field_name(cls, field_name: str, marqo_index: SemiStructuredMarqoIndex):
        tensor_field_map = marqo_index.tensor_field_map
        if field_name not in tensor_field_map:
            raise InvalidFieldNameError(f'Invalid tensor field name {field_name} for index {marqo_index.name}. '
                                        f'Valid tensor field names are {", ".join(tensor_field_map.keys())}')

    @classmethod
    def _verify_marqo_tensor_field(cls, field_name: str, field_value: Dict[str, Any]):
        if not set(field_value.keys()) == {constants.MARQO_DOC_CHUNKS, constants.MARQO_DOC_EMBEDDINGS}:
            raise InvalidTensorFieldError(f'Invalid tensor field {field_name}. '
                                          f'Expected keys {constants.MARQO_DOC_CHUNKS}, {constants.MARQO_DOC_EMBEDDINGS} '
                                          f'but found {", ".join(field_value.keys())}')