import uuid
from typing import Dict, Any, List, Optional, Type, Union, cast

from marqo.core.constants import MARQO_DOC_HIGHLIGHTS, MARQO_DOC_ID
from marqo.core.exceptions import MarqoDocumentParsingError
from marqo.core.models import MarqoQuery
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration, RangeConfiguration
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery
from marqo.core.search import search_filter
from marqo.core.semi_structured_vespa_index import common
from marqo.core.semi_structured_vespa_index.common import VESPA_FIELD_ID, BOOL_FIELDS, SHORT_STRINGS_FIELDS, \
    STRING_ARRAY, INT_FIELDS, FLOAT_FIELDS
from marqo.core.semi_structured_vespa_index.semi_structured_document import SemiStructuredVespaDocument, \
    generate_uuid_str
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
            return self._to_vespa_hybrid_query(marqo_query)
        elif isinstance(marqo_query, MarqoTensorQuery):
            return StructuredVespaIndex._to_vespa_tensor_query(self, marqo_query)
        elif isinstance(marqo_query, MarqoLexicalQuery):
            return StructuredVespaIndex._to_vespa_lexical_query(self, marqo_query)

        else:
            raise InternalError(f'Unknown query type {type(marqo_query)}')

    def _to_vespa_hybrid_query(self, marqo_query: MarqoHybridQuery) -> Dict[str, Any]:
        vespa_query = StructuredVespaIndex._to_vespa_hybrid_query(self, marqo_query)
        # if marqo_query.facets is not None:
        #     facets_grouping = self._get_facets_term(marqo_query.facets)
            # vespa_query['marqo__yql.facets'] = f"{vespa_query['marqo__yql.lexical']} limit 0 | {facets_grouping}"
        return vespa_query

    def _get_facets_term(self, facets_parameters: FacetsParameters, exclusions: List[str] = None) -> str:
        """
        Build a facets grouping query string from the provided facets_parameters.
        """
        FIELD_TYPES = {
            "int": INT_FIELDS,
            "float": FLOAT_FIELDS,
            "string": SHORT_STRINGS_FIELDS
        }

        global_max_results = facets_parameters.max_results or ""
        global_sort_order = facets_parameters.order or ""

        def build_group_parameters(field_config) -> str:
            """Build the max and order parameters for a group."""
            parts = []

            # Add max results if specified
            if field_config.max_results is not None:
                parts.append(f"max({field_config.max_results})")
            elif global_max_results:
                parts.append(f"max({global_max_results})")

            # Add sort order if specified
            if field_config.order:
                prefix = '-' if field_config.order == 'desc' else ''
                parts.append(f"order({prefix}count())")
            elif global_sort_order:
                prefix = '-' if global_sort_order == 'desc' else ''
                parts.append(f"order({prefix}count())")

            return " ".join(parts)

        def build_group_expression(field_config, field_name, field_id) -> str:
            """Build the group expression for a field."""
            field_type = FIELD_TYPES[field_config.type]

            # Handle numeric fields with ranges
            if field_config.type in ["int", "float"] and field_config.ranges:
                buckets = []
                for range_config in field_config.ranges:
                    from_val = range_config.from_ if range_config.from_ is not None else "-inf"
                    to_val = range_config.to_ if range_config.to_ is not None else "inf"
                    buckets.append(f'bucket({from_val}, {to_val})')
                return f'predefined({field_type}{{"{field_name}"}}, {", ".join(buckets)})'

            # Handle fields without ranges
            return str(field_id) if field_config.type in ["int", "float"] else f'{field_type}{{"{field_name}"}}'

        def build_field_group(field_config, field_name, field_id) -> str:
            """Build the complete group request for a field."""
            group_expr = build_group_expression(field_config, field_name, field_id)
            params = build_group_parameters(field_config)

            # Build output expression
            if field_config.type in ["int", "float"]:
                aggregations = ["sum", "avg", "min", "max"]
                funcs = [f'{func}({FIELD_TYPES[field_config.type]}{{"{field_name}"}})' for func in aggregations]
                funcs.append("count()")
                output = f"each(output({', '.join(funcs)}))"
            else:
                output = "each(output(count()))"

            return f"all(group({group_expr}) {params} {output})"

        # Start building the overall grouping query.
        grouping_query = "all( "
        if facets_parameters.max_depth is not None:
            grouping_query += f"max({facets_parameters.max_depth}) "

        for field_id, field_data in enumerate(facets_parameters.fields):
            field_name, field_parameters = next(iter(field_data.items()))
            if field_parameters.exclude is not None:
                # We want this field to be in a separate query if any of the exclusions are not in the exclusions list
                if exclusions is None or any([exclusion not in exclusions for exclusion in field_parameters.exclude]):
                    continue
            elif exclusions is not None:
                continue
            grouping_query += build_field_group(field_parameters, field_name, field_id)

        grouping_query += ")"
        return grouping_query


    def _get_string_array_attributes_to_retrieve(self, attributes_to_retrieve: List) -> List[str]:
        name_to_string_array_field_map = self.get_marqo_index().name_to_string_array_field_map
        return [name_to_string_array_field_map[att].string_array_field_name for att in attributes_to_retrieve if
                name_to_string_array_field_map.get(att)]

    def _get_filter_term(self, marqo_query: MarqoQuery, exclude: Optional[List[str]]=None) -> Optional[str]:
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

        def tree_to_filter_string(node: search_filter.Node) -> Optional[str]:
            if isinstance(node, search_filter.Operator):
                if isinstance(node, search_filter.And):
                    operator = 'AND'
                elif isinstance(node, search_filter.Or):
                    operator = 'OR'
                else:
                    raise InternalError(f'Unknown operator type {type(node)}')

                # Get both sides, filtering out excluded terms
                left = tree_to_filter_string(node.left)
                right = tree_to_filter_string(node.right)

                # If either side was excluded, skip this operator
                if left is None or right is None:
                    if left is not None or right is not None:
                        # If one side is excluded, return the other side
                        if left is None:
                            return f'({right})'
                        if right is None:
                            return f'({left})'
                    return None

                return f'({left} {operator} {right})'

            elif isinstance(node, search_filter.Modifier):
                if isinstance(node, search_filter.Not):
                    modified = tree_to_filter_string(node.modified)
                    if modified is None:
                        return None
                    return f'!({modified})'
                else:
                    raise InternalError(f'Unknown modifier type {type(node)}')

            elif isinstance(node, search_filter.Term):
                # Skip any terms with excluded fields
                if exclude and node.field in exclude:
                    return None

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
            - 'field_types': Field name to type mapping. Later used to create pre-conditions
            - 'fields': Field values. Each field is represented as an update statement, for the actual field, the field type metadata, and the score modifiers if applicable. Example:
                - 'marqo__bool_fields{active}': {"assign": 1}
                - 'marqo__field_type{active}': {"assign": "bool"}
            - 'version_uuid': Original document version_uuid if it exists

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
            # vespa_field_types, numeric_field_map dictionaries. Numeric fields and numeric field maps
            # are special cases and are processed later.
            self._process_field(field_name=field_name, value=value, fields=vespa_fields, field_types=vespa_field_types,
                                numeric_field_map=numeric_field_map, doc_id=doc_id)

        # This method creates the update statement for updating int fields / int map fields.
        int_fields_changed = self._create_update_statement_for_updating_numeric_and_numeric_map_field(
            int, numeric_field_map, original_doc, vespa_fields, vespa_field_types
        )
        # This method creates the update statement for float numeric fields / float map fields.
        float_fields_changed = self._create_update_statement_for_updating_numeric_and_numeric_map_field(
            float, numeric_field_map, original_doc, vespa_fields, vespa_field_types
        )

        # Handle score modifier updates
        if int_fields_changed or float_fields_changed:
            self._update_score_modifiers(
                original_doc=original_doc,
                numeric_field_map=numeric_field_map,
                vespa_fields=vespa_fields
            )

        # Add version_uuid to the vespa_fields to update the document's version_uuid,
        # only if this is a type of update that requires updating version_uuid (i.e a partial update with map fields)
        if original_doc is not None and original_doc.fixed_fields.version_uuid:
            vespa_fields[common.VESPA_DOC_VERSION_UUID] = {"assign": generate_uuid_str()}

        return {
            "id": doc_id,
            "fields": vespa_fields,
            "field_types": vespa_field_types,
            "version_uuid": original_doc.fixed_fields.version_uuid if original_doc else None # Pass the original document's version uuid, if it exists.
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


        score_modifiers = {}
        
        if len(numeric_field_map) > 0:
            score_modifiers["modify"] = {
                "operation": "replace",
                "create": True,
                "cells": numeric_field_map
            }
            
        if len(score_modifier_to_be_removed) > 0:
            score_modifiers["remove"] = {
                "addresses": score_modifier_to_be_removed
            }
            
        if len(score_modifiers) > 0:
            vespa_fields[common.SCORE_MODIFIERS] = score_modifiers

    def _process_field(self, field_name: str, value: Any, fields: Dict[str, Any], field_types: Dict[str, Any],
                       numeric_field_map: Dict[str, Any], doc_id: str) -> None:
        """Process a single field from a document based on its type.

        This method determines the type of the field value and delegates processing to the appropriate handler method.
        The field value is processed and added to the fields, field_types, and numeric_fields dictionaries as needed.

        Args:
            field_name: The name of the field being processed
            value: The value of the field, can be of the type bool, dict, int, float, list, or str
            fields: Dictionary to store the update statements corresponding to the processed fields 
            field_types: Dictionary mapping field names to their Marqo field types. Later used to create pre-conditions.
            doc_id: The ID of the document containing this field

        Raises:
            MarqoDocumentParsingError: If the field value is of an unsupported type
        """
        if isinstance(value, bool):
            self._handle_boolean_field(field_name, value, fields, field_types)
        elif isinstance(value, dict):
            self._handle_dict_field(field_name, value, doc_id, field_types, numeric_field_map)
        elif isinstance(value, (int, float)):
            self._handle_numeric_field(field_name, value, field_types, numeric_field_map)
        elif isinstance(value, list):
            self._handle_string_array_field(field_name, value, fields, field_types)
        elif isinstance(value, str):
            self._handle_string_field(field_name, value, fields, field_types)
        else:
            raise MarqoDocumentParsingError(
                f'Unsupported field type {type(value)} for field {field_name} in doc {doc_id}'
            )

    def _handle_numeric_field(self, field_name, value, field_types, numeric_field_map):
        numeric_field_map[field_name] = value
        if isinstance(value, int):
            field_types[field_name] = MarqoFieldTypes.INT.value
        elif isinstance(value, float):
            field_types[field_name] = MarqoFieldTypes.FLOAT.value


    def _create_update_statement_for_updating_numeric_and_numeric_map_field(
        self,
        numeric_type: Type[Union[int, float]],
        numeric_field_map: Dict[str, Union[int, float]],
        original_doc: Optional[SemiStructuredVespaDocument],
        vespa_fields: Dict[str, Any],
        vespa_field_types: Dict[str, Any]
    ) -> bool:
        """Creates update statements for numeric fields and their type metadata in Vespa documents.
        
        Processes numeric fields (integers, floats, int maps or float maps) for partial document updates in a semi-structured
        Vespa index. The method handles both regular numeric fields and map-type fields,
        comparing new values with the original document to minimize unnecessary updates.
        
        Args:
            numeric_type: The numeric type to process (int or float)
            numeric_field_map: Dictionary of field names to their numeric values for updating
            original_doc: The original Vespa document if it exists (for comparison)
            vespa_fields: Dictionary to store the generated Vespa update statements
            vespa_field_types: Dictionary to store field type metadata, later used to create pre-conditions
            
        Returns:
            bool: True if any fields were changed, False otherwise
            
        Behavior:
        1. Iterate over the numeric_field_map and create update statements for the fields
        that are present in the update request. This is only done if the field doesn't exist in the original document or if the field value has changed.
        2. Also create update (i.e "assign") statements for the field type metadata, which are useful in the case of a new field being added.
        3. Then iterate over the original fields which is a dictionary of all the fields inside marqo__int_fields / marqo__float_fields
        4. During this iteration, create "remove" statements for the fields that are not in numeric_field_map and are of type int_map / float_map. This is done solely
        for map fields to support complete map replacement (i.e replace an existing map with a new map of the same name but entirely different keys & values)
        """
        fields_changed = False
        field_prefix = common.INT_FIELDS if numeric_type is int else common.FLOAT_FIELDS

        original_fields = {}
        if original_doc is not None:
            original_fields = (original_doc.fixed_fields.int_fields # Get original fields if document exists
                             if numeric_type is int
                             else original_doc.fixed_fields.float_fields)

        # Process fields in update request
        for field_name, field_value in numeric_field_map.items():
            if not isinstance(field_value, numeric_type):
                continue
                
            vespa_field_name = f'{field_prefix}{{{field_name}}}'
            vespa_field_types_field_name = f'{common.VESPA_DOC_FIELD_TYPES}{{{field_name}}}'

            # Only set field value if it doesn't exist in the original set of fields or has changed
            field_exists_in_original_doc = original_doc is not None and field_name in original_fields
            field_value_changed = field_exists_in_original_doc and original_fields[field_name] != field_value
            field_type = vespa_field_types.get(field_name) # Get field type from the field passed in the request

            if not field_exists_in_original_doc or field_value_changed:
                vespa_fields[vespa_field_name] = {"assign": field_value} # assign statement for adding / updating the field value in marqo__int_fields / marqo__float_fields

                # Set field type metadata by creating an assign statement
                vespa_fields[vespa_field_types_field_name] = {"assign": field_type} # assign statement for adding / updating the field type in marqo__field_types
                
                # Handle creating update statements for the map field name. 
                if "." in field_name:
                    # For fields like "map1.key1", extract the map name "map1"
                    map_name = self._extract_map_name_from_field(field_name)
                    # Create assign statement for adding / updating statement for the map field. This is done for the prefix in a flattened map field name.
                    vespa_fields[f'{common.VESPA_DOC_FIELD_TYPES}{{{map_name}}}'] = {"assign": vespa_field_types.get(map_name)}

                fields_changed = True

        # Remove fields no longer in map

        # This block of code only executes for map fields. This is because to replace an entire map, we need to remove the flattened keys that
        # from marqo__int_fields / marqo__float_fields in case those fields are not present in the update request.
        for original_field_name in original_fields:
            if (original_field_name not in numeric_field_map and
                original_doc.fixed_fields.field_types.get(original_field_name) in (MarqoFieldTypes.INT_MAP.value, MarqoFieldTypes.FLOAT_MAP.value)):

                map_name = self._extract_map_name_from_field(original_field_name)
                vespa_field_name = f'{field_prefix}{{{original_field_name}}}'
                vespa_field_types_field_name = f'{common.VESPA_DOC_FIELD_TYPES}{{{original_field_name}}}'

                vespa_fields[vespa_field_name] = {"remove": 0} # remove statement for removing the field from marqo__int_fields / marqo__float_fields
                vespa_fields[vespa_field_types_field_name] = {"remove": 0} # remove statement for removing the field from marqo__field_types.
                vespa_field_types.pop(original_field_name, None)

                if vespa_field_types.get(map_name) is None: # Remove statement for removing the map field from marqo__field_types. This is prefix for a flattened map field.
                    vespa_fields[f'{common.VESPA_DOC_FIELD_TYPES}{{{map_name}}}'] = {"remove": 0}
                    
                fields_changed = True

        return fields_changed
    
    def _extract_map_name_from_field(self, field_name: str) -> str:
        """Extract the map name from a flattened field name.
        For fields like "map_name.key_name", this method extracts the map name portion.
        
        Args:
            field_name: The flattened field name (e.g., "map1.key1")
        Returns:
            The map name portion of the field (e.g., "map1")
        """
        if "." in field_name:
            return field_name.split(".", 1)[0]
        return field_name

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
        field_types: Dict[str, str],
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
            if not isinstance(v, (int, float)):
                raise MarqoDocumentParsingError(f'Unsupported field type {type(v)} for field {field_name} in doc {doc_id}. '
                                               'We only support int and float types for map values when updating a document')

            numeric_field_map[f'{field_name}.{k}'] = v

            # Set the appropriate field type based on the value type
            if isinstance(v, int):
                field_types[f'{field_name}.{k}'] = MarqoFieldTypes.INT_MAP.value
                field_types[f'{field_name}'] = MarqoFieldTypes.INT_MAP.value
            else:  # Must be float based on the earlier check
                field_types[f'{field_name}.{k}'] = MarqoFieldTypes.FLOAT_MAP.value
                field_types[f'{field_name}'] = MarqoFieldTypes.FLOAT_MAP.value

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
                'Marqo does not support adding new lexical fields in partial updates'
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
