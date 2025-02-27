import json
from typing import Dict, Any, List, Optional, Type, Union, cast

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
    index_supports_partial_updates: bool = False

    def __init__(self, marqo_index: SemiStructuredMarqoIndex):
        super().__init__(marqo_index)
        self.index_supports_partial_updates = self._marqo_index_version >= SemiStructuredVespaSchema.SEMISTRUCTURED_INDEX_PARTIAL_UPDATE_SUPPORT_VERSION

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
            if len(marqo_query.attributes_to_retrieve) > 0:
                if self.index_supports_partial_updates:
                    # Retrieve static fields content to extract non-string values from combined fields
                    marqo_query.attributes_to_retrieve.extend([
                        common.INT_FIELDS,
                        common.FLOAT_FIELDS,
                        common.BOOL_FIELDS,
                    ])
                    string_array_attributes_to_retrieve = self._get_string_array_attributes_to_retrieve(marqo_query.attributes_to_retrieve)
                    marqo_query.attributes_to_retrieve.extend(string_array_attributes_to_retrieve)
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

    def _get_string_array_attributes_to_retrieve(self, attributes_to_retrieve: List) -> List[str]:
        name_to_string_array_field_map = self.get_marqo_index().name_to_string_array_field_map
        return [name_to_string_array_field_map[att].string_array_field_name for att in attributes_to_retrieve if
                name_to_string_array_field_map.get(att)]

    def _get_filter_term(self, marqo_query: MarqoQuery) -> Optional[str]:
        # Reuse logic in UnstructuredVespaIndex to create filter term
        def escape(s: str) -> str:
            return s.replace('\\', '\\\\').replace('"', '\\"')

        def generate_equality_filter_string(node: search_filter.EqualityTerm) -> str:
            filter_parts = []

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
            if self.index_supports_partial_updates:
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

    def to_vespa_partial_document(self, marqo_document: Dict[str, Any], existing_vespa_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert a Marqo document to Vespa partial document format for updates.

        This method transforms a Marqo document into the format required by Vespa for partial document updates.
        It processes each field in the document according to its type and creates the appropriate Vespa field
        representations.

        Args:
            marqo_document: A dictionary containing the Marqo document to be converted. Must contain an '_id' field.
            existing_vespa_document: Optional existing Vespa document to be compared against while creating the update statement

        Returns:
            Dict containing the Vespa partial document format with:
            - 'id': Document ID
            - 'field_types': Field name to type mapping
            - 'fields': Field values
            - 'create_timestamp': Original document timestamp if it exists

        Raises:
            MarqoDocumentParsingError: If '_id' field is missing
            InvalidFieldNameError: If any field name is invalid
        """
        doc_id = self._extract_document_id(marqo_document)
        
        # Convert existing document if provided
        original_doc = None
        if existing_vespa_document:
            original_doc = SemiStructuredVespaDocument.from_vespa_document(
                existing_vespa_document, 
                marqo_index=self.get_marqo_index()
            )

        # Initialize tracking dictionaries
        vespa_fields = {}
        vespa_field_types = {}

        # Initialize dictionary to be later used for updating score modifiers. 
        numeric_fields = {}

        numeric_field_map: Dict[str, Any] = dict() # This map is used to store the numeric fields in the document. It is used to update the numeric fields & score modifiers later
        if original_doc:
            numeric_field_map.update(original_doc.fixed_fields.int_fields)
            numeric_field_map.update(original_doc.fixed_fields.float_fields)

        # Process each field in the document
        for field_name, value in marqo_document.items():
            if field_name == MARQO_DOC_ID:
                continue
                
            validate_field_name(field_name)

            # This method broadly processes the field based on its type and updates the vespa_fields,
            # vespa_field_types, numeric_fields, numeric_field_map dictionaries. Numeric fields and numeric field maps
            # are special cases and are processed later.
            self._process_field(
                field_name=field_name,
                value=value,
                fields=vespa_fields,
                field_types=vespa_field_types,
                numeric_fields=numeric_fields,
                numeric_field_map=numeric_field_map,
                doc_id=doc_id
            )

        # This method creates the update statement for updating int fields / int map fields.
        int_fields_changed = self._update_numeric_field(
            int, numeric_field_map, original_doc, vespa_fields, vespa_field_types
        )
        # This method creates the update statement for float numeric fields / float map fields.
        float_fields_changed = self._update_numeric_field(
            float, numeric_field_map, original_doc, vespa_fields, vespa_field_types
        )

        # Handle score modifier updates
        if int_fields_changed or float_fields_changed:
            self._update_score_modifiers(
                original_doc=original_doc,
                numeric_field_map=numeric_field_map,
                vespa_fields=vespa_fields
            )

        return {
            "id": doc_id,
            "fields": vespa_fields,
            "field_types": vespa_field_types,
            "create_timestamp": original_doc.fixed_fields.create_timestamp if original_doc else None
        }

    def _update_score_modifiers(self, original_doc: Optional[SemiStructuredVespaDocument], 
                              numeric_field_map: Dict[str, Any],
                              vespa_fields: Dict[str, Any]) -> None:
        """Updates score modifiers for numeric fields in Vespa documents.
        
        This method handles the updating of score modifiers for numeric fields during partial document updates.
        It identifies which score modifiers need to be removed (fields that existed in the original document
        but are no longer present) and which ones need to be modified (fields with new values).
        
        Args:
            original_doc: The original document before the update, if it exists
            numeric_field_map: Dictionary mapping field names to their numeric values (both int and float)
            vespa_fields: Dictionary to store the update statements for Vespa fields
            
        Returns:
            None
        
        Note:
            Score modifiers are only updated if there are changes to numeric fields.
            The method creates a Vespa update operation that either replaces or removes score modifiers.
        """
            
        original_fields = {}
        # Find score modifiers to remove
        score_modifier_to_be_removed = []
        if original_doc:
            original_fields.update(original_doc.fixed_fields.int_fields)
            original_fields.update(original_doc.fixed_fields.float_fields)
            score_modifier_to_be_removed = [
                {"p": field} for field in original_fields
                if field not in numeric_field_map 
                and original_doc.fixed_fields.field_types.get(field) in (MarqoFieldTypes.INT_MAP.value, MarqoFieldTypes.FLOAT_MAP.value)
            ]


        if len(score_modifier_to_be_removed) > 0 or len(score_modifier_to_be_removed) > 0:
            vespa_fields[common.SCORE_MODIFIERS] = {
                "modify": {
                    "operation": "replace",
                    "create": True,
                    "cells": numeric_field_map
                } if len(numeric_field_map) > 0 else None,
                "remove": {
                    "addresses": score_modifier_to_be_removed
                } if len(score_modifier_to_be_removed) > 0 else None
            }

    def _process_field(
        self,
        field_name: str,
        value: Any,
        fields: Dict[str, Any],
        field_types: Dict[str, Any],
        numeric_fields: Dict[str, Any],
        numeric_field_map: Dict[str, Any],
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
            self._handle_dict_field(field_name, value, doc_id, numeric_field_map)
        elif isinstance(value, (int, float)):
            numeric_field_map[field_name] = value # sets information about numeric fields in a map so it the numeric field + score modifiers can be updated later
        elif isinstance(value, list):
            self._handle_string_array_field(field_name, value, fields, field_types)
        elif isinstance(value, str):
            self._handle_string_field(field_name, value, fields, field_types)
        else:
            raise MarqoDocumentParsingError(
                f'Unsupported field type {type(value)} for field {field_name} in doc {doc_id}'
            )

    def _update_numeric_field(
        self,
        numeric_type: Type[Union[int, float]],
        numeric_field_map: Dict[str, Union[int, float]],
        original_doc: Optional[SemiStructuredVespaDocument],
        vespa_fields: Dict[str, Any],
        vespa_field_types: Dict[str, Any]
    ) -> bool:
        """Updates numeric fields (int/float) in Vespa documents.
        
        This method processes numeric fields (integers or floats) for partial document updates.
        It compares the new values with the original document (if it exists) and only updates
        fields that have changed or are new. It also handles the field type metadata appropriately,
        distinguishing between regular numeric fields and map types.
        
        Args:
            numeric_type: The type of numeric field (int or float) to process
            numeric_field_map: Dictionary mapping field names to their numeric values
            original_doc: The original Vespa document if it exists, used for comparison
            vespa_fields: Dictionary to store the update statements for Vespa fields
            vespa_field_types: Dictionary to store the field type metadata updates
            
        Returns:
            bool: True if any fields were changed, False otherwise
            
        Note:
            - Fields that exist in the original document but not in the update request are preserved
            - Map type fields that are no longer present will be removed
            - Field type metadata is updated to maintain consistency with the field values
        """
        fields_changed = False
        field_prefix = common.INT_FIELDS if numeric_type is int else common.FLOAT_FIELDS

        original_fields = {}
        if original_doc is not None:
            original_fields = (original_doc.fixed_fields.int_fields # Get original fields if document exists
                             if numeric_type is int
                             else original_doc.fixed_fields.float_fields)

        # Process fields in update request
        for field_name, value in numeric_field_map.items():
            if not isinstance(value, numeric_type):
                continue
                
            vespa_field_name = f'{field_prefix}{{{field_name}}}'
            vespa_field_types_field_name = f'{common.VESPA_DOC_FIELD_TYPES}{{{field_name}}}'

            # Only set field value if it doesn't exist in the original set of fields or has changed
            field_exists = original_doc is not None and field_name in original_fields
            field_value_changed = field_exists and original_fields[field_name] != value
            
            if not field_exists or field_value_changed:
                vespa_fields[vespa_field_name] = {"assign": value}

                # Determine if field is a map type based on original document
                is_map_type = (original_doc is not None and 
                              original_doc.fixed_fields.field_types.get(field_name) in 
                              (MarqoFieldTypes.INT_MAP.value, MarqoFieldTypes.FLOAT_MAP.value))
                
                # Set appropriate field type based on numeric_type and whether it's a map
                field_type = (MarqoFieldTypes.INT_MAP if numeric_type is int else MarqoFieldTypes.FLOAT_MAP) if is_map_type else (MarqoFieldTypes.INT if numeric_type is int else MarqoFieldTypes.FLOAT)

                # Set field type metadata by creating a assigned statement
                vespa_fields[vespa_field_types_field_name] = {"assign": field_type.value}

                # Update field type metadata dictionary, so we can use it later when defining the update pre-condition to send to vespa
                vespa_field_types[field_name] = field_type.value
                fields_changed = True

        # Remove fields no longer in map

        for original_field_name in original_fields:
            if (original_field_name not in numeric_field_map and
                original_doc.fixed_fields.field_types.get(original_field_name) in (MarqoFieldTypes.INT_MAP.value, MarqoFieldTypes.FLOAT_MAP.value)):

                vespa_field_name = f'{field_prefix}{{{original_field_name}}}'
                vespa_field_types_field_name = f'{common.VESPA_DOC_FIELD_TYPES}{{{original_field_name}}}'

                vespa_fields[vespa_field_name] = {"remove": 0}
                vespa_fields[vespa_field_types_field_name] = {"remove": 0}
                vespa_field_types.pop(original_field_name, None)
                fields_changed = True

        return fields_changed
    
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
        doc_id: str,
        numeric_field_map: Dict[str, Any]
    ) -> None:
        """Handle dictionary field processing for document updates.

        This method processes a dictionary field by:
        1. Removing any existing entries for this field from the numeric field map
        2. Adding new entries for numeric values (int, float) in the dictionary
        3. Validating that all dictionary values are of supported types

        Args:
            field_name: The name of the dictionary field
            value: The dictionary to be processed
            doc_id: The ID of the document being updated
            numeric_field_map: Dictionary mapping flattened field names to their numeric values

        Raises:
            MarqoDocumentParsingError: If any value in the dictionary is not a supported numeric type
        """
        keys_to_remove = [
            key for key in numeric_field_map.keys()
            if key.startswith(f'{field_name}.')
        ]
        for key in keys_to_remove: #remove existing entries for this specific map field
            del numeric_field_map[key]
            
        # Add new entries
        for k, v in value.items():
            if isinstance(v, (int, float)):
                numeric_field_map[f'{field_name}.{k}'] = v
            else:
                raise MarqoDocumentParsingError(f'Unsupported field type {type(v)} for field {field_name} in doc {doc_id}')

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
            raise MarqoDocumentParsingError('Unstructured index updates only support updating existing string array fields')
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
        update_field_type_metadata_key = f'{common.VESPA_DOC_FIELD_TYPES}{{{field_key}}}'
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

            For a string array field "string_array_1" with value ['a', 'b', 'c']:
            fields["marqo__string_array_string_array_1"] = {"assign": ['a', 'b', 'c']}
        """
        vespa_doc_field_name = ""
        # Create the vespa doc field name
        if isinstance(val, bool):
            vespa_doc_field_name = f'{common.BOOL_FIELDS}{{{key}}}'
        elif isinstance(val, list):
            vespa_doc_field_name = f'{common.STRING_ARRAY}_{key}'

        # Create the update statement
        if isinstance(val, bool):
            fields[vespa_doc_field_name] = {"assign": int(val)}
        else:
            fields[vespa_doc_field_name] = {"assign": val}
