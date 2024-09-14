import json
from typing import List, Dict, Any

from pydantic import Field

from marqo.base_model import MarqoBaseModel
from marqo.core import constants as index_constants, constants
from marqo.core.document.add_documents_handler import ORIGINAL_ID
from marqo.core.exceptions import VespaDocumentParsingError
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, FieldType, logger
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
    vespa_multimodal_params: Dict[str, str] = Field(default_factory=str,
                                                    alias=common.VESPA_DOC_MULTIMODAL_PARAMS)


class SemiStructuredVespaDocument(MarqoBaseModel):
    """A helper class to handle the conversion between Vespa and Marqo documents for an unstructured index.
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
            elif field_name in marqo_index.lexical_field_map:  # TODO used by get_by_id, why?
                text_field_name = marqo_index.lexical_field_map[field_name].name
                text_fields[text_field_name] = fields[field_name]
            elif field_name in marqo_index.field_map:   # TODO used during search, why?
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
        return fields.pop(name) if name in fields else default

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
            if field_name in [index_constants.MARQO_DOC_ID, constants.MARQO_DOC_TENSORS, ORIGINAL_ID]:
                continue
            if isinstance(field_content, str):
                if field_name not in marqo_index.field_map:
                    raise ValueError()  # TODO find a better error

                field = marqo_index.field_map[field_name]
                if field.type == FieldType.Text:
                    instance.text_fields[field.lexical_field_name] = field_content
                    if len(field_content) <= marqo_index.filter_string_max_length:
                        instance.fixed_fields.short_string_fields[field_name] = field_content
                else:
                    instance.text_fields[field.name] = field_content
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

                    # self._verify_marqo_tensor_field_name(marqo_tensor_field)
                    # self._verify_marqo_tensor_field(marqo_tensor_field, marqo_tensor_value)

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

    def to_marqo_document(self, marqo_index: SemiStructuredMarqoIndex, return_highlights: bool = False) -> Dict[str, Any]:
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

        if return_highlights and self.match_features:
            marqo_document[index_constants.MARQO_DOC_HIGHLIGHTS] = self.extract_highlights(marqo_index)

        # Hybrid search raw scores
        if self.raw_tensor_score is not None:
            marqo_document[index_constants.MARQO_DOC_HYBRID_TENSOR_SCORE] = self.raw_tensor_score
        if self.raw_lexical_score is not None:
            marqo_document[index_constants.MARQO_DOC_HYBRID_LEXICAL_SCORE] = self.raw_lexical_score

        return marqo_document

    def extract_highlights(self, marqo_index: SemiStructuredMarqoIndex) -> List[Dict[str, str]]:
        # FIXME logic copied from structured
        # For each tensor field we will have closest(tensor_field) and distance(tensor_field) in match features
        # If a tensor field hasn't been searched, closest(tensor_field)[cells] will be empty and distance(tensor_field)
        # will be max double
        match_features = self.match_features

        min_distance = None
        closest_tensor_field = None
        for tensor_field in marqo_index.tensor_fields:
            closest_feature = f'closest({tensor_field.embeddings_field_name})'
            if closest_feature in match_features and len(match_features[closest_feature]['cells']) > 0:
                distance_feature = f'distance(field,{tensor_field.embeddings_field_name})'
                if distance_feature not in match_features:
                    raise VespaDocumentParsingError(
                        f'Expected {distance_feature} in match features but it was not found'
                    )
                distance = match_features[distance_feature]
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    closest_tensor_field = tensor_field

        if closest_tensor_field is None:
            return []
            # TODO verify if this is expected in unstructured search. Lexical search + tensor ranking can ended up in this situation. See test_hybrid_search.py test_hybrid_search_unstructured_highlights_for_lexical_tensor
            # raise VespaDocumentParsingError('Failed to extract highlights from Vespa document. Could not find '
            #                                 'closest tensor field in response')

        # Get chunk index
        chunk_index_str = next(iter(
            match_features[f'closest({closest_tensor_field.embeddings_field_name})']['cells']
        ))
        try:
            chunk_index = int(chunk_index_str)
        except ValueError as e:
            raise VespaDocumentParsingError(
                f'Expected integer as chunk index, but found {chunk_index_str}', cause=e
            ) from e

        # Get chunk value
        try:
            chunk_field_name = closest_tensor_field.chunk_field_name

            if chunk_field_name in self.tensor_fields:
                chunk = self.tensor_fields[chunk_field_name][chunk_index]
            else:
                # Note: WARN level will create verbose logs in production as this is per result
                logger.debug(f'Failed to extract highlights as Vespa document is missing chunk field '
                             f'{chunk_field_name}. This can happen if attributes_to_retrieve does not include '
                             f'all searchable tensor fields (searchable_attributes)')

                chunk = None

        except (KeyError, TypeError, IndexError) as e:
            raise VespaDocumentParsingError(
                f'Cannot extract chunk value from {closest_tensor_field.chunk_field_name}: {str(e)}',
                cause=e
            ) from e

        if chunk is not None:
            return [{closest_tensor_field.name: chunk}]
        else:
            return []