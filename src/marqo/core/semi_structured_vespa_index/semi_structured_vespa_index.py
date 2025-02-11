import json
from typing import Dict, Any, List, Optional, Union, cast

from marqo.core.constants import MARQO_DOC_HIGHLIGHTS, MARQO_DOC_ID
from marqo.core.exceptions import MarqoDocumentParsingError
from marqo.core.models import MarqoQuery
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery
from marqo.core.search import search_filter
from marqo.core.semi_structured_vespa_index import common
from marqo.core.semi_structured_vespa_index.common import VESPA_FIELD_ID, BOOL_FIELDS, SHORT_STRINGS_FIELDS, \
    STRING_ARRAY, INT_FIELDS, FLOAT_FIELDS
from marqo.core.semi_structured_vespa_index.semi_structured_document import SemiStructuredVespaDocument
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.unstructured_vespa_index.unstructured_validation import validate_field_name
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.core.semi_structured_vespa_index.marqo_field_types import MarqoFieldTypes
from marqo.exceptions import InternalError, InvalidArgumentError


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
        index_supports_partial_updates = self._marqo_index_version >= SemiStructuredVespaSchema.SEMISTRUCTURED_INDEX_PARTIAL_UPDATE_SUPPORT_VERSION
        return (SemiStructuredVespaDocument.from_marqo_document(
            marqo_document, marqo_index=self.get_marqo_index())).to_vespa_document(index_supports_partial_updates)

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
        index_supports_partial_updates = self._marqo_index_version >= SemiStructuredVespaSchema.SEMISTRUCTURED_INDEX_PARTIAL_UPDATE_SUPPORT_VERSION
        if marqo_query.attributes_to_retrieve is not None:
            if len(marqo_query.attributes_to_retrieve) > 0:
                if index_supports_partial_updates:
                    # Retrieve static fields content to extract non-string values from combined fields
                    marqo_query.attributes_to_retrieve.extend([
                        common.INT_FIELDS,
                        common.FLOAT_FIELDS,
                        common.BOOL_FIELDS,
                    ])
                else:
                    marqo_query.attributes_to_retrieve.extend([
                        common.STRING_ARRAY,
                        common.INT_FIELDS,
                        common.FLOAT_FIELDS,
                        common.BOOL_FIELDS,
                    ])

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


    def _get_filter_term(self, marqo_query: MarqoQuery) -> Optional[str]:
        # Reuse logic in UnstructuredVespaIndex to create filter term
        def escape(s: str) -> str:
            return s.replace('\\', '\\\\').replace('"', '\\"')

        def generate_equality_filter_string(node: search_filter.EqualityTerm) -> str:
            filter_parts = []
            index_supports_partial_updates = self._marqo_index_version >= SemiStructuredVespaSchema.SEMISTRUCTURED_INDEX_PARTIAL_UPDATE_SUPPORT_VERSION

            # Filter on `_id`
            if node.field == MARQO_DOC_ID:
                return f'({VESPA_FIELD_ID} contains "{escape(node.value)}")'

            # Bool Filter
            if node.value.lower() in self._FILTER_STRING_BOOL_VALUES:
                filter_value = int(True if node.value.lower() == "true" else False)
                bool_filter_string = (f'({BOOL_FIELDS} contains '
                                      f'sameElement(key contains "{node.field}", value = {filter_value}))')
                filter_parts.append(bool_filter_string)

            # Short String Filter
            short_string_filter_string = (f'({SHORT_STRINGS_FIELDS} '
                                          f'contains sameElement(key contains "{node.field}", '
                                          f'value contains "{escape(node.value)}"))')
            filter_parts.append(short_string_filter_string)

            # String Array Filter
            if index_supports_partial_updates:
                if node.field in self.get_marqo_index().name_to_string_array_field_map:
                    string_array_field_name = f'{STRING_ARRAY}_{node.field}'
                    string_array_filter_string = (f'({string_array_field_name} contains '
                                                  f'"{escape(node.value)}")')
                    filter_parts.append(string_array_filter_string)
            else:
                string_array_filter_string = (f'({STRING_ARRAY} contains '
                                              f'"{node.field}::{escape(node.value)}")')
                filter_parts.append(string_array_filter_string)

            # Numeric Filter
            numeric_filter_string = ""
            try:
                numeric_value = int(node.value)
                numeric_filter_string = (
                    f'({INT_FIELDS} contains sameElement(key contains "{node.field}", value = {numeric_value})) '
                    f'OR ({FLOAT_FIELDS} contains sameElement(key contains "{node.field}", value = {numeric_value}))')
            except ValueError:
                try:
                    numeric_value = float(node.value)
                    numeric_filter_string = f'({FLOAT_FIELDS} contains sameElement(key contains "{node.field}", value = {numeric_value}))'
                except ValueError:
                    pass

            if numeric_filter_string:
                filter_parts.append(numeric_filter_string)

            # Final Filter String
            final_filter_string = f"({' OR '.join(filter_parts)})"
            return final_filter_string

        def generate_range_filter_string(node: search_filter.RangeTerm) -> str:
            lower = f'value >= {node.lower}' if node.lower is not None else ""
            higher = f'value <= {node.upper}' if node.upper is not None else ""
            bound = f'{lower}, {higher}' if lower and higher else f'{lower}{higher}'
            if not bound:
                raise InternalError('RangeTerm has no lower or upper bound')

            float_field_string = (f'({FLOAT_FIELDS} contains '
                                  f'sameElement(key contains "{node.field}", {bound}))')

            int_field_string = (f'({INT_FIELDS} contains '
                                f'sameElement(key contains "{node.field}", {bound}))')

            return f'({float_field_string} OR {int_field_string})'

        def tree_to_filter_string(node: search_filter.Node) -> str:
            if isinstance(node, search_filter.Operator):
                if isinstance(node, search_filter.And):
                    operator = 'AND'
                elif isinstance(node, search_filter.Or):
                    operator = 'OR'
                else:
                    raise InternalError(f'Unknown operator type {type(node)}')
                return f'({tree_to_filter_string(node.left)} {operator} {tree_to_filter_string(node.right)})'
            elif isinstance(node, search_filter.Modifier):
                if isinstance(node, search_filter.Not):
                    return f'!({tree_to_filter_string(node.modified)})'
                else:
                    raise InternalError(f'Unknown modifier type {type(node)}')
            elif isinstance(node, search_filter.Term):
                if isinstance(node, search_filter.EqualityTerm):
                    return generate_equality_filter_string(node)
                elif isinstance(node, search_filter.RangeTerm):
                    return generate_range_filter_string(node)
                elif isinstance(node, search_filter.InTerm):
                    raise InvalidArgumentError("The 'IN' filter keyword is not yet supported for unstructured indexes")
            raise InternalError(f'Unknown node type {type(node)}')

        if marqo_query.filter is not None:
            return tree_to_filter_string(marqo_query.filter.root)

    def _extract_document_id(self, document: Dict[str, Any]) -> str:
        """Extract and validate document ID."""
        if "_id" not in document:
            raise MarqoDocumentParsingError("'_id' is a required field")
        doc_id = document["_id"]
        self._verify_id_field(doc_id)
        return doc_id

    def to_vespa_partial_document(self, marqo_document: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Marqo document to Vespa partial document format for updates.

        This method transforms a Marqo document into the format required by Vespa for partial document updates.
        It processes each field in the document according to its type and creates the appropriate Vespa field
        representations.

        Args:
            marqo_document: A dictionary containing the Marqo document to be converted. Must contain an '_id' field.

        Returns:
            Dict[str, Any]: A dictionary containing the Vespa partial document format with:
                - 'id': The document ID
                - 'field_types': Mapping of field names to their types
                - 'fields': The actual field values

        Raises:
            MarqoDocumentParsingError: If the '_id' field is missing from the document
            InvalidFieldNameError: If any field name is invalid
        """
        # Validate and extract document ID
        doc_id = self._extract_document_id(marqo_document)
        
        # Initialize result
        fields = {}
        field_types = {}

        # Initialize dictionary to be later used for updating score modifiers. 
        numeric_fields = {}

        # Process each field in the document
        for field_name, value in marqo_document.items():
            if field_name == MARQO_DOC_ID:
                continue

            validate_field_name(field_name)
            
            self._process_field(
                field_name=field_name,
                value=value,
                fields=fields,
                field_types=field_types,
                numeric_fields=numeric_fields,
                doc_id=doc_id
            )

        return {
            "id": doc_id,
            "field_types": field_types,
            "fields": fields
        }

    def _process_field(
        self,
        field_name: str,
        value: Any,
        fields: Dict[str, Any],
        field_types: Dict[str, Any],
        numeric_fields: Dict[str, Any],
        doc_id: str
    ) -> None:
        """Process a single field from a document based on its type.

        This method determines the type of the field value and delegates processing to the appropriate handler method.
        The field value is processed and added to the fields, field_types, and numeric_fields dictionaries as needed.

        Args:
            field_name: The name of the field being processed
            value: The value of the field, can be of the type bool, dict, int, float, list, or str
            fields: Dictionary to store the update statements corresponding to the processed fields 
            field_types: Dictionary mapping field names to their Marqo field types
            numeric_fields: Dictionary storing numeric field values for being later used to update score modifier 
            doc_id: The ID of the document containing this field

        Raises:
            MarqoDocumentParsingError: If the field value is of an unsupported type
        """
        if isinstance(value, bool):
            self._handle_boolean_field(field_name, value, fields, field_types)
        elif isinstance(value, dict):
            self._handle_dict_field(field_name, value, fields, field_types, numeric_fields, doc_id)
        elif isinstance(value, (int, float)):
            self._handle_numeric_field(field_name, value, field_types, fields, numeric_fields)
        elif isinstance(value, list):
            self._handle_string_array_field(field_name, value, fields, field_types)
        elif isinstance(value, str):
            self._handle_string_field(field_name, value, fields, field_types)
        else:
            raise MarqoDocumentParsingError(
                f'Unsupported field type {type(value)} for field {field_name} in doc {doc_id}'
            )

    def _handle_boolean_field(
        self,
        field_name: str,
        value: bool,
        fields: Dict[str, Any],
        field_types: Dict[str, Any]
    ) -> None:
        """Handle boolean field processing for document updates.

        This method processes a boolean field by:
        1. Creating an update statement for the field value
        2. Setting the field type metadata to BOOL
        3. Creating an update statement for the field type metadata

        Args:
            field_name: The name of the boolean field
            value: The boolean value to be stored
            fields: Dictionary to store the update statements for fields
            field_types: Dictionary mapping field names to their Marqo field types
        """
        self._create_update_statement_for_updating_field(fields, field_name, value)
        field_types[field_name] = MarqoFieldTypes.BOOL.value
        self._create_update_statement_for_updating_field_type_metadata(fields, field_types, field_name)

    def _handle_dict_field(
        self,
        field_name: str,
        value: Dict[str, Any],
        fields: Dict[str, Any],
        field_types: Dict[str, Any],
        numeric_fields: Dict[str, Any],
        doc_id: str
    ) -> None:
        """Handle dictionary field processing for document updates.

        This method processes a dictionary field by:
        1. Iterating through the dictionary key-value pairs
        2. Creating a full key by combining the field name and dictionary key
        3. For numeric values (int/float):
            - Updates numeric_fields dictionary for score modifier calculation
            - Sets appropriate field type (INT_MAP or FLOAT_MAP)
            - Creates update statements for the field value, field type metadata and score modifiers
        4. Raises error for non-numeric values

        Args:
            field_name: The name of the dictionary field
            value: The dictionary value to be processed
            fields: Dictionary to store the update statements for fields
            field_types: Dictionary mapping field names to their Marqo field types
            numeric_fields: Dictionary storing numeric field values for score modifier updates
            doc_id: The ID of the document containing this field

        Raises:
            MarqoDocumentParsingError: If any dictionary value is not numeric (int/float)
        """
        for key, val in value.items():
            full_key = f'{field_name}.{key}'
            if isinstance(val, (int, float)):
                # setting numeric_fields for score modifier updation
                numeric_fields[full_key] = val
                # setting field types for later creating pre-conditions
                field_types[full_key] = (
                    MarqoFieldTypes.INT_MAP.value if isinstance(val, int)
                    else MarqoFieldTypes.FLOAT_MAP.value
                )
                # Step for updating the actual field
                self._create_update_statement_for_updating_field(fields, full_key, val)
                # Step for updating field type metadata.
                self._create_update_statement_for_updating_field_type_metadata(fields, field_types, full_key)
                # Step for creating update statement for updating score modifiers
                self._create_update_statement_for_updating_score_modifiers(numeric_fields, fields)
            else:
                raise MarqoDocumentParsingError(
                    f'Unsupported field type {type(val)} for field {field_name} in doc {doc_id}'
                )


    def _handle_numeric_field(
        self,
        field_name: str,
        value: Union[int, float],
        field_types: Dict[str, Any],
        fields: Dict[str, Any],
        numeric_fields: Dict[str, Any]
    ) -> None:
        """Handle numeric field processing for document updates.

        This method processes a numeric field by:
        1. Storing the numeric value for score modifier calculation
        2. Setting the appropriate field type (INT or FLOAT)
        3. Creating update statements for:
           - The field value
           - Field type metadata
           - Score modifiers

        Args:
            field_name: The name of the numeric field
            value: The numeric value (integer or float) to be processed
            field_types: Dictionary mapping field names to their Marqo field types
            fields: Dictionary to store the update statements for fields
            numeric_fields: Dictionary storing numeric field values for score modifier updates
        """
        numeric_fields[field_name] = value
        field_types[field_name] = ( # setting field types for later creating pre-conditions
            MarqoFieldTypes.INT.value if isinstance(value, int)
            else MarqoFieldTypes.FLOAT.value
        )
        self._create_update_statement_for_updating_field(fields, field_name, value) # To create update statement for updating the actual field
        self._create_update_statement_for_updating_field_type_metadata(fields, field_types, field_name) # To create update statement for updating 'field type' metadata 
        self._create_update_statement_for_updating_score_modifiers(numeric_fields, fields) # To create update statement for updating score modifiers

    def _handle_string_array_field(
        self,
        field_name: str,
        value: List[Any],
        fields: Dict[str, Any],
        field_types: Dict[str, Any]
    ) -> None:
        """Handle string array field processing for document updates.

        This method processes a string array field by:
        1. Validating that all array elements are strings
        2. Setting the field type to STRING_ARRAY
        3. Creating update statements for:
           - The field value
           - Field type metadata

        Args:
            field_name: The name of the string array field
            value: The list of strings to be processed
            fields: Dictionary to store the update statements for fields
            field_types: Dictionary mapping field names to their Marqo field types

        Raises:
            MarqoDocumentParsingError: If any element in the array is not a string
        """
        if not all(isinstance(v, str) for v in value) or self.get_marqo_index().name_to_string_array_field_map.get(field_name) is None:
            raise MarqoDocumentParsingError('Only string arrays are supported')
        field_types[field_name] = MarqoFieldTypes.STRING_ARRAY.value # setting field types for later creating pre-conditions
        self._create_update_statement_for_updating_field(fields, field_name, value) # To create update statement for updating the actual field 
        self._create_update_statement_for_updating_field_type_metadata(fields, field_types, field_name) # To create update statement for updating 'field type' metadata

    def _handle_string_field(
        self,
        field_name: str,
        value: str,
        fields: Dict[str, Any],
        field_types: Dict[str, Any]
    ) -> None:
        """Handle string field processing for document updates.

        This method processes a string field by:
        1. Validating that the field exists in the lexical field map
        2. Creating update statements for:
           - The lexical field value
           - Short string field value (if string length is within limit)
           - Field type metadata

        Args:
            field_name: The name of the string field
            value: The string value to be processed
            fields: Dictionary to store the update statements for fields
            field_types: Dictionary mapping field names to their Marqo field types

        Raises:
            MarqoDocumentParsingError: If the field does not exist in the lexical field map
        """
        lexical_field_name = f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}'
        if lexical_field_name not in self.get_marqo_index().lexical_field_map:
            raise MarqoDocumentParsingError(
                f'{field_name} of type str does not exist in the original document. '
                'We do not support adding new lexical fields in partial updates'
            )

        fields[lexical_field_name] = {"assign": value} # To create update statement for updating the lexical fields
        
        short_string_field = f'{common.SHORT_STRINGS_FIELDS}{{{field_name}}}'
        if len(value) <= self.get_marqo_index().filter_string_max_length:
            fields[short_string_field] = {"assign": value} # To create update statement for updating the actual field
        else:
            fields[short_string_field] = {"remove": 0}
            
        field_types[field_name] = MarqoFieldTypes.STRING.value
        self._create_update_statement_for_updating_field_type_metadata(fields, field_types, field_name) # To create update statement for updating 'field type' metadata

    def _create_update_statement_for_updating_field_type_metadata(self, update_statement_fields, field_types,
                                                                  field_key):
        """Create update statement for updating field type metadata.

        This method creates an update statement to modify the field type metadata in Vespa.
        It assigns the field type value from field_types to a metadata field in the update statement.

        Args:
            update_statement_fields: Dictionary containing the update statements for fields
            field_types: Dictionary mapping field names to their Marqo field types
            field_key: The field name whose type metadata needs to be updated

        Example:
            If field_key is "title" and field_types["title"] is "string", this will add:
            {"marqo__field_type{title}": {"assign": "string"}} to update_statement_fields
        """
        update_field_type_metadata_key = f'{common.VESPA_DOC_FIELD_TYPE}{{{field_key}}}'
        update_statement_fields[update_field_type_metadata_key] = {"assign": field_types[field_key]}

    def _create_update_statement_for_updating_field(self, fields, key, val):
        """Create update statement for updating a field in Vespa.

        This method creates an update statement for a field based on its value type.
        For boolean values, it converts them to integers (0/1) before assigning.
        For other types (float, int, list), it assigns the value directly.

        Args:
            fields: Dictionary containing the update statements for fields
            key: The field name to be updated
            val: The value to assign to the field. Can be bool, float, int or list.

        Example:
            For a boolean field "active" with value True:
            fields["marqo__bool_fields{active}"] = {"assign": 1}

            For a float field "score" with value 0.95:
            fields["marqo__float_fields{score}"] = {"assign": 0.95}
        """
        vespa_doc_field_name = ""
        # Create the vespa doc field name
        if isinstance(val, bool):
            vespa_doc_field_name = f'{common.BOOL_FIELDS}{{{key}}}'
        elif isinstance(val, float):
            vespa_doc_field_name = f'{common.FLOAT_FIELDS}{{{key}}}'
        elif isinstance(val, int):
            vespa_doc_field_name = f'{common.INT_FIELDS}{{{key}}}'
        elif isinstance(val, list):
            vespa_doc_field_name = f'{common.STRING_ARRAY}_{key}'

        # Create the update statement
        if isinstance(val, bool):
            fields[vespa_doc_field_name] = {"assign": int(val)}
        else:
            fields[vespa_doc_field_name] = {"assign": val}

    def _create_update_statement_for_updating_score_modifiers(self, numeric_fields: Dict[str, Any], fields: Dict[str, Any]) -> None:
        """Create update statement for updating score modifiers in Vespa.

        This method creates an update statement to modify score modifiers in Vespa by replacing the existing
        score modifier values with new values from numeric_fields.

        Args:
            numeric_fields: Dictionary containing the new score modifier values to set
            fields: Dictionary that will be updated with the score modifier update statement

        Example:
            If numeric_fields is {"price": 100, "rating": 4.5}, this will add:
            {
                "marqo__score_modifiers": {
                    "modify": {
                        "operation": "replace",
                        "cells": {"price": 100, "rating": 4.5}
                    }
                }
            }
            to the fields dictionary
        """
        fields[common.SCORE_MODIFIERS] = {
            "modify": {
                "operation": "replace",
                "create": True,
                "cells": numeric_fields
            }
        }