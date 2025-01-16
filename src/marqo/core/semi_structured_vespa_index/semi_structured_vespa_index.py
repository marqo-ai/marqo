import json
from typing import Dict, Any, Optional, cast

from marqo.core.constants import MARQO_DOC_HIGHLIGHTS, MARQO_DOC_ID
from marqo.core.exceptions import MarqoDocumentParsingError
from marqo.core.models import MarqoQuery
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery
from marqo.core.semi_structured_vespa_index import common
from marqo.core.semi_structured_vespa_index.semi_structured_document import SemiStructuredVespaDocument
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.unstructured_vespa_index.unstructured_validation import validate_field_name
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.exceptions import InternalError
from marqo.tensor_search.validation import validate_map_numeric_field


class SemiStructuredVespaIndex(StructuredVespaIndex, UnstructuredVespaIndex):
    """
    An implementation of VespaIndex for SemiStructured indexes.
    """

    def __init__(self, marqo_index: SemiStructuredMarqoIndex):
        super().__init__(marqo_index)

    def get_marqo_index(self) -> SemiStructuredMarqoIndex:
        if isinstance(self._marqo_index, SemiStructuredMarqoIndex):
            return cast(SemiStructuredMarqoIndex, self._marqo_index)
        else:
            raise TypeError('Wrong type of marqo index')

    def to_vespa_document(self, marqo_document: Dict[str, Any]) -> Dict[str, Any]:
        return (SemiStructuredVespaDocument.from_marqo_document(
            marqo_document, marqo_index=self.get_marqo_index())).to_vespa_document()

    def to_marqo_document(self, vespa_document: Dict[str, Any], return_highlights: bool = False) -> Dict[str, Any]:
        vespa_doc = SemiStructuredVespaDocument.from_vespa_document(vespa_document, marqo_index=self.get_marqo_index())
        marqo_doc = vespa_doc.to_marqo_document(marqo_index=self.get_marqo_index())

        if return_highlights and vespa_doc.match_features:
            # Since tensor fields are stored in each individual field, we need to use same logic in structured
            # index to extract highlights
            marqo_doc[MARQO_DOC_HIGHLIGHTS] = StructuredVespaIndex._extract_highlights(
                self, vespa_document.get('fields', {}))

        return marqo_doc

    def to_vespa_query(self, marqo_query: MarqoQuery) -> Dict[str, Any]:
        # Verify attributes to retrieve, if defined
        if marqo_query.attributes_to_retrieve is not None:
            marqo_query.attributes_to_retrieve.append(common.VESPA_FIELD_ID)
            # add chunk field names for tensor fields
            marqo_query.attributes_to_retrieve.extend(
                [self.get_marqo_index().tensor_field_map[att].chunk_field_name
                 for att in marqo_query.attributes_to_retrieve
                 if att in self.get_marqo_index().tensor_field_map]
            )

        # Hybrid must be checked first since it is a subclass of Tensor and Lexical
        if isinstance(marqo_query, MarqoHybridQuery):
            return StructuredVespaIndex._to_vespa_hybrid_query(self, marqo_query)
        elif isinstance(marqo_query, MarqoTensorQuery):
            return StructuredVespaIndex._to_vespa_tensor_query(self, marqo_query)
        elif isinstance(marqo_query, MarqoLexicalQuery):
            return StructuredVespaIndex._to_vespa_lexical_query(self, marqo_query)

        else:
            raise InternalError(f'Unknown query type {type(marqo_query)}')

    @classmethod
    def _get_filter_term(cls, marqo_query: MarqoQuery) -> Optional[str]:
        # Reuse logic in UnstructuredVespaIndex to create filter term
        return UnstructuredVespaIndex._get_filter_term(marqo_query)

    # def to_vespa_partial_document(self, marqo_document: Dict[str, Any], # marqo_document is the one you have given in request
    #                               original_vespa_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: #original_vespa_document is the one you derive from vespa
    #     vespa_id: Optional[str] = None
    #     vespa_fields: Dict[str, Any] = dict()
    #
    #     if MARQO_DOC_ID not in marqo_document:
    #         raise MarqoDocumentParsingError(f"'{MARQO_DOC_ID}' is a required field but it does not exist")
    #     else:
    #         vespa_id = marqo_document[MARQO_DOC_ID]
    #         self._verify_id_field(vespa_id)
    #
    #     original_doc = SemiStructuredVespaDocument.from_vespa_document(original_vespa_document, self.get_marqo_index()) #this is made out of the document you get from vespa
    #     new_string_array = original_doc.fixed_fields.string_arrays.copy()
    #
    #     all_numeric_field_map = dict()
    #     all_numeric_field_map.update(original_doc.fixed_fields.int_fields)
    #     all_numeric_field_map.update(original_doc.fixed_fields.float_fields)
    #
    #     for marqo_field, value in marqo_document.items():
    #         if marqo_field == MARQO_DOC_ID:
    #             continue
    #
    #         # TODO move the validation logic out of validation.py
    #         validate_field_name(marqo_field)
    #
    #         if self._is_tensor_field(marqo_field, original_doc): # original document is used to determine if we're updating a tensor field. But we error out here anyway
    #             raise MarqoDocumentParsingError(f'field {marqo_field} is a tensor field or a dependant field of a '
    #                                             f'multimodal combo fields, we cannot update its value')
    #
    #         if value is None:
    #             # TODO handle removal of a field
    #             # https://docs.vespa.ai/en/reference/document-json-format.html#clearing-a-field
    #             pass
    #
    #         if isinstance(value, bool):
    #             # Assign values to a map: https://docs.vespa.ai/en/reference/document-json-format.html#assign-map-field
    #             if value != original_doc.fixed_fields.bool_fields.get(marqo_field, None):
    #                 field_name = f'{common.BOOL_FIELDS}{{{marqo_field}}}'
    #                 vespa_fields[field_name] = {"assign": int(value)}
    #
    #         # Handle numeric fields including numeric maps
    #         elif isinstance(value, dict):
    #             # numeric dict need to be handled separately since the original_marqo_doc has them flattened
    #             # TODO move the validation logic
    #             validate_map_numeric_field(value)
    #
    #             # remove all entries from the flattened map, and repopulate
    #             all_numeric_field_map = {key: value for key, value in all_numeric_field_map.items()
    #                                      if not key.startswith(f'{marqo_field}.')}
    #             # TODO what if v == None?
    #             all_numeric_field_map.update({f'{marqo_field}.{k}': v for k, v in value.items()})
    #
    #         elif isinstance(value, (int, float)):
    #             all_numeric_field_map[marqo_field] = value
    #
    #         # Handle string array fields
    #         elif isinstance(marqo_document[marqo_field], list): #original document isn't even used here
    #             # TODO move the validation logic
    #             if any(not isinstance(v, str) for v in marqo_document[marqo_field]):
    #                 raise MarqoDocumentParsingError('Only string array is supported')
    #
    #             new_string_array = [value for value in new_string_array if not value.startswith(f'{marqo_field}::')]
    #             new_string_array.extend([f'{marqo_field}::{value}' for value in marqo_document[marqo_field]])
    #
    #         # Handle string fields (lexical only)
    #         elif isinstance(marqo_document[marqo_field], str):
    #             # Handle lexical field change
    #             lexical_field_name = f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{marqo_field}'
    #
    #             if lexical_field_name not in self.get_marqo_index().lexical_field_map:
    #                 raise MarqoDocumentParsingError(f'{marqo_field} of type str does not exist in the original '
    #                                                 f'document. We do not support adding new lexical fields in '
    #                                                 f'partial updates')
    #
    #             if value == original_vespa_document.get(marqo_field, None):
    #                 # Skip changing this field
    #                 continue
    #
    #             # update the lexical field, please note that the lexical field name has a prefix
    #             # https://docs.vespa.ai/en/reference/document-json-format.html#single-field-value
    #             vespa_fields[lexical_field_name] = {"assign": value}
    #
    #             # Update the short string map
    #             #   short string -> long string, we'll need to remove it from short string map
    #             #   long string -> short string, we'll need to add it to the short string map
    #             #   short string -> short string, we'll need to update it in the short string map
    #             field_name = f'{common.SHORT_STRINGS_FIELDS}{{{marqo_field}}}'
    #             if len(marqo_document[marqo_field]) <= self.get_marqo_index().filter_string_max_length:
    #                 # Add or update the value in the map
    #                 # https://docs.vespa.ai/en/reference/document-json-format.html#assign-map-field
    #                 vespa_fields[field_name] = {"assign": marqo_document[marqo_field]}
    #             else:
    #                 # remove from map: https://docs.vespa.ai/en/reference/document-json-format.html#map-field-remove
    #                 vespa_fields[field_name] = {"remove": 0}
    #         else:
    #             raise MarqoDocumentParsingError(f'Unsupported field type {type(value)} '
    #                                             f'for field {marqo_field} in doc {vespa_id}')
    #
    #     # Handle string array change
    #     items_to_remove = set(original_doc.fixed_fields.string_arrays) - set(new_string_array)
    #     items_to_add = set(new_string_array) - set(original_doc.fixed_fields.string_arrays)
    #     if items_to_remove:
    #         # https://docs.vespa.ai/en/reference/document-json-format.html#array-field
    #         vespa_fields[common.STRING_ARRAY] = {"assign": new_string_array}
    #     elif items_to_add:
    #         # if we only need to add items, it can be appended to the array, which requires smaller change
    #         # https://docs.vespa.ai/en/reference/document-json-format.html#add-array-elements
    #         vespa_fields[common.STRING_ARRAY] = {"add": list(items_to_add)}
    #
    #     # Handle all numeric values (including score modifiers)
    #     int_fields_changed = self._update_numeric_field(int, all_numeric_field_map, original_doc, vespa_fields) #Here we compare the int field changes
    #     float_fields_changed = self._update_numeric_field(float, all_numeric_field_map, original_doc, vespa_fields) #here we compare the float field changes
    #
    #     if int_fields_changed or float_fields_changed: #Here we use int field changed or float field changed to determine if the score modifier should be updated.
    #         # TODO: @Aditya to find out why do we update score modifiers anyway?
    #         # TODO SCORE_MODIFIERS is a tensor, find out if it can be updated in the same way as a map, is it faster?
    #         #   If not, we copied the replace logic from structured_vespa_index, should we rather use assign here?
    #         # https://docs.vespa.ai/en/reference/document-json-format.html#tensor-field
    #         # https://docs.vespa.ai/en/reference/document-json-format.html#tensor-add
    #         # https://docs.vespa.ai/en/reference/document-json-format.html#tensor-remove
    #         # https://docs.vespa.ai/en/reference/document-json-format.html#tensor-modify
    #         vespa_fields[common.SCORE_MODIFIERS] = {
    #             "modify": {
    #                 "operation": "replace",
    #                 "cells": all_numeric_field_map
    #             }
    #         }
    #
    #     return {"id": vespa_id, "create_timestamp": original_doc.fixed_fields.create_timestamp,  "fields": vespa_fields}

    def to_vespa_partial_document(self, marqo_document: Dict[str, Any], original_vespa_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        vespa_id: Optional[str] = None
        vespa_fields: Dict[str, Any] = dict()
        vespa_field_types: Dict[str, str] = dict()

        if MARQO_DOC_ID not in marqo_document:
            raise MarqoDocumentParsingError(f"'{MARQO_DOC_ID}' is a required field but it does not exist")
        else:
            vespa_id = marqo_document[MARQO_DOC_ID]
            self._verify_id_field(vespa_id)

        all_numeric_field_map = {}
        new_string_array = []

        for marqo_field_key, marqo_field_value in marqo_document.items():
            if marqo_field_key == MARQO_DOC_ID:
                continue
            # TODO move the validation logic out of validation.py
            validate_field_name(marqo_field_key)

            if isinstance(marqo_field_value, bool):
                # Assign values to a map: https://docs.vespa.ai/en/reference/document-json-format.html#assign-map-field
                field_name = f'{common.BOOL_FIELDS}{{{marqo_field_key}}}'
                vespa_fields[field_name] = {"assign": int(marqo_field_value)}
                vespa_field_types[marqo_field_key] = 'bool'
            elif isinstance(marqo_field_value, dict):
                for key, value in marqo_field_value.items():
                    all_numeric_field_map[f'{marqo_field_key}.{key}'] = value
                    if isinstance(value, int):
                        vespa_field_types[f'{marqo_field_key}.{key}'] = 'int_map'
                    elif isinstance(value, float):
                        vespa_field_types[f'{marqo_field_key}.{key}'] = 'float_map'
            elif isinstance(marqo_field_value, int):
                all_numeric_field_map[marqo_field_key] = marqo_field_value
                vespa_field_types[marqo_field_key] = 'int'
            elif isinstance(marqo_field_value, float):
                all_numeric_field_map[marqo_field_key] = marqo_field_value
                vespa_field_types[marqo_field_key] = 'float'
            elif isinstance(marqo_document[marqo_field_key], list):
                if any(not isinstance(v, str) for v in marqo_document[marqo_field_key]):
                    raise MarqoDocumentParsingError('Only string array is supported')

                # new_string_array = [value for value in new_string_array if not value.startswith(f'{marqo_field_key}::')]
                new_string_array.extend([f'{marqo_field_key}::{value}' for value in marqo_document[marqo_field_key]])
                vespa_field_types[marqo_field_key] = 'string_array'

            elif isinstance(marqo_document[marqo_field_key], str):
                lexical_field_name = f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{marqo_field_key}'
                lexical_field_map = self.get_marqo_index().lexical_field_map
                if lexical_field_name not in lexical_field_map:
                    raise MarqoDocumentParsingError(f'{marqo_field_key} of type str does not exist in the original '
                                                    f'document. We do not support adding new lexical fields in '
                                                    f'partial updates')
                vespa_fields[lexical_field_name] = {"assign": marqo_field_value}

                field_name = f'{common.SHORT_STRINGS_FIELDS}{{{marqo_field_key}}}'
                if len(marqo_document[marqo_field_key]) <= self.get_marqo_index().filter_string_max_length:
                    vespa_fields[field_name] = {"assign": marqo_document[marqo_field_key]}
                else:
                    vespa_fields[field_name] = {"remove": 0}
                vespa_field_types[marqo_field_key] = 'string'
            else:
                raise MarqoDocumentParsingError(f'Unsupported field type {type(marqo_field_value)} for field {marqo_field_key} in doc {vespa_id}')

        if new_string_array: #Don't know what to do with this.
            vespa_fields[common.STRING_ARRAY] = {"assign": new_string_array}

        int_fields_changed = self._update_numeric_field(int, all_numeric_field_map, vespa_fields) #Here we compare the int field changes
        float_fields_changed = self._update_numeric_field(float, all_numeric_field_map, vespa_fields) #here we compare the float field changes
        #
        if int_fields_changed or float_fields_changed: #Here we use int field changed or float field changed to determine if the score modifier should be updated.
            # TODO: @Aditya to find out why do we update score modifiers anyway?
            # TODO SCORE_MODIFIERS is a tensor, find out if it can be updated in the same way as a map, is it faster?
            #   If not, we copied the replace logic from structured_vespa_index, should we rather use assign here?
            # https://docs.vespa.ai/en/reference/document-json-format.html#tensor-field
            # https://docs.vespa.ai/en/reference/document-json-format.html#tensor-add
            # https://docs.vespa.ai/en/reference/document-json-format.html#tensor-remove
            # https://docs.vespa.ai/en/reference/document-json-format.html#tensor-modify
            vespa_fields[common.SCORE_MODIFIERS] = {
                "modify": {
                    "operation": "replace",
                    "cells": all_numeric_field_map
                }
            }

        return {"id": vespa_id, "field_types": vespa_field_types,  "fields": vespa_fields}



    def _update_numeric_field(self, numeric_type, all_numeric_field_map, original_doc, vespa_fields) -> bool:
        field_name_prefix = common.INT_FIELDS if numeric_type == int else common.FLOAT_FIELDS
        original_fields = original_doc.fixed_fields.int_fields if numeric_type == int \
            else original_doc.fixed_fields.float_fields #original fields nikal li
        new_fields = {key: value for key, value in all_numeric_field_map.items() if isinstance(value, numeric_type)} #new_fields nikal li

        changed = False

        for k, v in new_fields.items(): # iterating over new fields
            if k not in original_fields or original_fields[k] != v: #if the field is not in original fields or the value is different -> only then create assign statement
                vespa_fields[f'{field_name_prefix}{{{k}}}'] = {"assign": v}
                changed = True #changed set to true

        for k, v in original_fields.items(): #iterating over original fields
            if k not in new_fields: # if the field is not in new fields -> remove the field
                vespa_fields[f'{field_name_prefix}{{{k}}}'] = {"remove": 0} #If the field not in new fields -> create remove statement.

                changed = True

        return changed

    def _update_numeric_field(self, numeric_type, all_numeric_field_map, vespa_fields) -> bool:
        field_name_prefix = common.INT_FIELDS if numeric_type == int else common.FLOAT_FIELDS
        # original_fields = original_doc.fixed_fields.int_fields if numeric_type == int \
        #     else original_doc.fixed_fields.float_fields
        new_fields = {key: value for key, value in all_numeric_field_map.items() if isinstance(value, numeric_type)}

        changed = False

        for k, v in new_fields.items():
            # if k not in original_fields or original_fields[k] != v:
            vespa_fields[f'{field_name_prefix}{{{k}}}'] = {"assign": v}
            changed = True

        # for k, v in original_fields.items():
        #     if k not in new_fields:
        #         vespa_fields[f'{field_name_prefix}{{{k}}}'] = {"remove": 0}
        #         changed = True

        return changed

    def _is_tensor_field(self, partial_update_field_name, original_doc):
        # Please note that we cannot rely on the tensor_fields property in the index to derive the tensor fields
        # info. tensor_fields is a superset for all tensor fields in the index. For each individual document, we
        # will need to derive the info based on whether chunks fields exists or if the field is a dependant field
        # of a multimodal combo field.
        potential_tensor_field_name = f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{partial_update_field_name}'
        if potential_tensor_field_name in original_doc.tensor_fields:
            return True

        if (original_doc.fixed_fields.vespa_multimodal_params and
            any([partial_update_field_name in json.loads(config)["weights"] for config
                 in original_doc.fixed_fields.vespa_multimodal_params.values()])):
            return True

        return False
