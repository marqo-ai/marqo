import json
from typing import Dict, Any

from marqo.core import constants as index_constants, constants
from marqo.core.constants import MARQO_DOC_HIGHLIGHTS
from marqo.core.exceptions import VespaDocumentParsingError
from marqo.core.semi_structured_vespa_index import common
from marqo.core.semi_structured_vespa_index.common import STRING_ARRAY
from marqo.core.semi_structured_vespa_index.semi_structured_document import _VESPA_DOC_FIELDS
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex


class SemiStructuredVespaIndexNoPydantic(SemiStructuredVespaIndex):

    def to_marqo_document(self, vespa_document: Dict[str, Any], return_highlights: bool = False) -> Dict[str, Any]:
        marqo_index = self.get_marqo_index()

        fields = vespa_document.get(_VESPA_DOC_FIELDS, {})
        tensor_fields = {}
        text_fields = {}

        tensor_subfield_map = marqo_index.tensor_subfield_map
        lexical_field_map = marqo_index.lexical_field_map
        field_map = marqo_index.field_map

        marqo_document = {}

        def process_field(field_name: str, fields: Dict) -> None:
            """Helper function to process individual tensor fields, lexical fields and populate the appropriate dictionaries"""
            if field_name in tensor_subfield_map:
                tensor_fields[field_name] = fields[field_name]
            elif field_name in lexical_field_map:
                # Lexical fields are returned with prefixed name from get_by_ids
                text_field_name = lexical_field_map[field_name].name
                text_fields[text_field_name] = fields[field_name]
            elif field_name in field_map:
                # Lexical fields are returned with original name from search
                text_fields[field_name] = fields[field_name]

        if marqo_index.index_supports_partial_updates:
            string_array_field_map = marqo_index.string_array_field_name_to_string_array_field_map
            string_array_prefix_length = len(STRING_ARRAY + '_')
            for field_name in fields:
                # Process tensor and text fields
                process_field(field_name, fields)

                # Handle string arrays separately
                # In case of indexes which  support partial updates (i.e indexes created with Marqo version post 2.16.0), string arrays are stored in Vespa like
                # 'marqo__string_array_field_name_1': ['element1', 'element2', ...]
                # 'marqo__string_array_field_name_2': ['element3', 'element4', ...]
                # Here we will collect all such string array fields and put them in string_arrays_dict, which will later be used  to construct the SemiStructuredVespaDocument object.
                if field_name in string_array_field_map:
                    string_array_field_key = field_name[string_array_prefix_length:]
                    string_array_field_value = fields[field_name]
                    marqo_document[string_array_field_key] = string_array_field_value

        else:
            # For older versions, just process tensor and text fields
            for field_name in fields:
                process_field(field_name, fields)

                if field_name == common.STRING_ARRAY:
                    for string_array in fields[field_name]:
                        string_array_key, string_array_value = string_array.split("::", 1)  # String_array_key will be string in this case, and string_array_value will be a single string in this case.
                        if string_array_key not in marqo_document:
                            marqo_document[string_array_key] = []
                        marqo_document[string_array_key].append(string_array_value)

        marqo__id = fields.get(common.VESPA_FIELD_ID, None)
        int_fields = fields.get(common.INT_FIELDS, dict())
        bool_fields = fields.get(common.BOOL_FIELDS, dict())
        float_fields = fields.get(common.FLOAT_FIELDS, dict())
        vespa_multimodal_params = fields.get(common.VESPA_DOC_MULTIMODAL_PARAMS, dict())
        raw_tensor_score = fields.get(common.VESPA_DOC_HYBRID_RAW_TENSOR_SCORE, None)
        raw_lexical_score = fields.get(common.VESPA_DOC_HYBRID_RAW_LEXICAL_SCORE, None)
        match_features = fields.get(common.VESPA_DOC_MATCH_FEATURES, dict())

        # Add int and float fields back
        # Please note that int-map and float-map fields are flattened in the result. The correct behaviour is to convert
        # them back to the format when they are indexed. We will keep the behaviour as is to avoid breaking changes.
        marqo_document.update(int_fields)
        marqo_document.update(float_fields)

        marqo_document.update({k: bool(v) for k, v in bool_fields.items()})
        marqo_document[index_constants.MARQO_DOC_ID] = marqo__id
        # Note: We are not adding field_types & version_uuid to the document because
        # it's a field for internal Marqo use only.

        # text fields
        for field_name, field_content in text_fields.items():
            marqo_document[field_name] = field_content

        # tensor fields
        for field_name, field_content in tensor_fields.items():
            tensor_field = tensor_subfield_map[field_name]

            if constants.MARQO_DOC_TENSORS not in marqo_document:
                marqo_document[constants.MARQO_DOC_TENSORS] = dict()
            if tensor_field.name not in marqo_document[constants.MARQO_DOC_TENSORS]:
                marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name] = dict()

            if field_name == tensor_field.chunk_field_name:
                marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name][
                    constants.MARQO_DOC_CHUNKS] = field_content
            elif field_name == tensor_field.embeddings_field_name:
                try:
                    marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name][
                        constants.MARQO_DOC_EMBEDDINGS] = list(field_content['blocks'].values())
                except (KeyError, AttributeError, TypeError) as e:
                    raise VespaDocumentParsingError(
                        f'Cannot parse embeddings field {field_name} with value {field_content}'
                    ) from e

        if vespa_multimodal_params:
            marqo_document[common.MARQO_DOC_MULTIMODAL_PARAMS] = dict()
            for multimodal_field_name, serialized_multimodal_params in vespa_multimodal_params.items():
                marqo_document[common.MARQO_DOC_MULTIMODAL_PARAMS][multimodal_field_name] = \
                    json.loads(serialized_multimodal_params)

        # Hybrid search raw scores
        if raw_tensor_score is not None:
            marqo_document[index_constants.MARQO_DOC_HYBRID_TENSOR_SCORE] = raw_tensor_score
        if raw_lexical_score is not None:
            marqo_document[index_constants.MARQO_DOC_HYBRID_LEXICAL_SCORE] = raw_lexical_score

        if return_highlights and match_features:
            # Since tensor fields are stored in each individual field, we need to use same logic in structured
            # index to extract highlights
            marqo_document[MARQO_DOC_HIGHLIGHTS] = StructuredVespaIndex._extract_highlights(
                self, vespa_document.get('fields', {}))

        return marqo_document
