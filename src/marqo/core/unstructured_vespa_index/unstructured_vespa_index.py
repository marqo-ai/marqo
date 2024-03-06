from typing import Dict, Any, Optional

import marqo.core.constants as index_constants
import marqo.core.search.search_filter as search_filter
from marqo.api import exceptions as errors
from marqo.core.models import MarqoQuery
from marqo.core.models.marqo_index import UnstructuredMarqoIndex
from marqo.core.models.marqo_query import (MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery,
                                           ScoreModifierType)
from marqo.core.unstructured_vespa_index import common as unstructured_common
from marqo.core.unstructured_vespa_index.unstructured_document import UnstructuredVespaDocument
from marqo.core.vespa_index import VespaIndex
from marqo.exceptions import InternalError


class UnstructuredVespaIndex(VespaIndex):
    _FILTER_STRING_BOOL_VALUES = ["true", "false"]
    _RESERVED_FIELD_SUBSTRING = "::"
    _SUPPORTED_FIELD_CONTENT_TYPES = [str, int, float, bool, list]

    def __init__(self, marqo_index: UnstructuredMarqoIndex):
        self._marqo_index = marqo_index

    def get_vespa_field_id(self) -> str:
        return unstructured_common.VESPA_FIELD_ID

    def to_vespa_partial_document(self, marqo_document: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Partial document update is not supported for unstructured indexes. This"
                                  "function should not be called.")

    def to_vespa_document(self, marqo_document: Dict[str, Any]) -> Dict[str, Any]:
        unstructured_document: UnstructuredVespaDocument = \
            (UnstructuredVespaDocument.from_marqo_document(marqo_document,
                                                           self._marqo_index.filter_string_max_length))
        return unstructured_document.to_vespa_document()

    def to_marqo_document(self, vespa_document: Dict[str, Any], return_highlights: bool = False) -> Dict[str, Any]:
        unstructured_document: UnstructuredVespaDocument = UnstructuredVespaDocument.from_vespa_document(vespa_document)
        return unstructured_document.to_marqo_document(return_highlights=return_highlights)

    def to_vespa_query(self, marqo_query: MarqoQuery) -> Dict[str, Any]:
        if marqo_query.searchable_attributes is not None:
            # TODO Add a marqo doc link here on how to create a structured index
            raise errors.InvalidArgError('searchable_attributes is not supported for an unstructured index. '
                                         'You can create a structured index '
                                         'by `mq.create_index("your_index_name", type="structured")`')

        if isinstance(marqo_query, MarqoTensorQuery):
            return self._to_vespa_tensor_query(marqo_query)
        elif isinstance(marqo_query, MarqoLexicalQuery):
            return self._to_vespa_lexical_query(marqo_query)
        elif isinstance(marqo_query, MarqoHybridQuery):
            return self._to_vespa_hybrid_query(marqo_query)
        else:
            raise InternalError(f'Unknown query type {type(marqo_query)}')

    def _to_vespa_tensor_query(self, marqo_query: MarqoTensorQuery) -> Dict[str, Any]:
        tensor_term = self._get_tensor_search_term(marqo_query)

        filter_term = self._get_filter_term(marqo_query)
        if filter_term:
            filter_term = f' AND {filter_term}'
        else:
            filter_term = ''

        select_attributes = "*"

        summary = unstructured_common.SUMMARY_ALL_VECTOR if marqo_query.expose_facets else unstructured_common.SUMMARY_ALL_NON_VECTOR

        score_modifiers = self._get_score_modifiers(marqo_query)

        ranking = unstructured_common.RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS if score_modifiers \
            else unstructured_common.RANK_PROFILE_EMBEDDING_SIMILARITY

        query_inputs = {
            unstructured_common.QUERY_INPUT_EMBEDDING: marqo_query.vector_query
        }

        if score_modifiers:
            query_inputs.update(score_modifiers)

        query = {
            'yql': f"select {select_attributes} from {self._marqo_index.schema_name} where {tensor_term}{filter_term}",
            'model_restrict': self._marqo_index.schema_name,
            'hits': marqo_query.limit,
            'offset': marqo_query.offset,
            'query_features': query_inputs,
            'presentation.summary': summary,
            'ranking': ranking
        }
        query = {k: v for k, v in query.items() if v is not None}

        if not marqo_query.approximate:
            query['ranking.softtimeout.enable'] = False
            query['timeout'] = '300s'

        return query

    @staticmethod
    def _get_tensor_search_term(marqo_query: MarqoTensorQuery) -> str:
        field_to_search = unstructured_common.VESPA_DOC_EMBEDDINGS

        if marqo_query.ef_search is not None:
            target_hits = min(marqo_query.limit + marqo_query.offset, marqo_query.ef_search)
            additional_hits = max(marqo_query.ef_search - (marqo_query.limit + marqo_query.offset), 0)
        else:
            target_hits = marqo_query.limit + marqo_query.offset
            additional_hits = 0

        return (
            f"("
            f"{{"
            f"targetHits:{target_hits}, "
            f"approximate:{str(marqo_query.approximate)}, "
            f'hnsw.exploreAdditionalHits:{additional_hits}'
            f"}}"
            f"nearestNeighbor({field_to_search}, {unstructured_common.QUERY_INPUT_EMBEDDING})"
            f")"
        )

    @classmethod
    def _get_filter_term(cls, marqo_query: MarqoQuery) -> Optional[str]:
        def escape(s: str) -> str:
            return s.replace('\\', '\\\\').replace('"', '\\"')

        def generate_equality_filter_string(node: search_filter.EqualityTerm) -> str:
            filter_parts = []

            # Filter on `_id`
            if node.field == index_constants.MARQO_DOC_ID:
                return f'({unstructured_common.VESPA_FIELD_ID} contains "{escape(node.value)}")'

            # Bool Filter
            if node.value.lower() in cls._FILTER_STRING_BOOL_VALUES:
                filter_value = int(True if node.value.lower() == "true" else False)
                bool_filter_string = (f'({unstructured_common.BOOL_FIELDS} contains '
                                      f'sameElement(key contains "{node.field}", value = {filter_value}))')
                filter_parts.append(bool_filter_string)

            # Short String Filter
            short_string_filter_string = (f'({unstructured_common.SHORT_STRINGS_FIELDS} '
                                          f'contains sameElement(key contains "{node.field}", '
                                          f'value contains "{escape(node.value)}"))')
            filter_parts.append(short_string_filter_string)

            # String Array Filter
            string_array_filter_string = (f'({unstructured_common.STRING_ARRAY} contains '
                                          f'"{node.field}::{escape(node.value)}")')
            filter_parts.append(string_array_filter_string)

            # Numeric Filter
            numeric_filter_string = ""
            try:
                numeric_value = int(node.value)
                numeric_filter_string = (
                    f'({unstructured_common.INT_FIELDS} contains sameElement(key contains "{node.field}", value = {numeric_value})) '
                    f'OR ({unstructured_common.FLOAT_FIELDS} contains sameElement(key contains "{node.field}", value = {numeric_value}))')
            except ValueError:
                try:
                    numeric_value = float(node.value)
                    numeric_filter_string = f'({unstructured_common.FLOAT_FIELDS} contains sameElement(key contains "{node.field}", value = {numeric_value}))'
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

            float_field_string = (f'({unstructured_common.FLOAT_FIELDS} contains '
                                  f'sameElement(key contains "{node.field}", {bound}))')

            int_field_string = (f'({unstructured_common.INT_FIELDS} contains '
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
            raise InternalError(f'Unknown node type {type(node)}')

        if marqo_query.filter is not None:
            return tree_to_filter_string(marqo_query.filter.root)

    @staticmethod
    def _get_score_modifiers(marqo_query: MarqoQuery) -> \
            Optional[Dict[str, Dict[str, float]]]:
        if marqo_query.score_modifiers:
            mult_tensor = {}
            add_tensor = {}
            for modifier in marqo_query.score_modifiers:
                if modifier.type == ScoreModifierType.Multiply:
                    mult_tensor[modifier.field] = modifier.weight
                elif modifier.type == ScoreModifierType.Add:
                    add_tensor[modifier.field] = modifier.weight
                else:
                    raise InternalError(f'Unknown score modifier type {modifier.type}')

            # Note one of these could be empty, but not both
            return {
                unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS: mult_tensor,
                unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS: add_tensor
            }

        return None

    def _to_vespa_lexical_query(self, marqo_query: MarqoLexicalQuery) -> Dict[str, Any]:
        def _get_lexical_search_term(marqo_query: MarqoLexicalQuery) -> str:
            if marqo_query.or_phrases:
                or_terms = 'weakAnd(%s)' % ', '.join([
                    f'default contains "{phrase}"' for phrase in marqo_query.or_phrases
                ])
            else:
                or_terms = ''
            if marqo_query.and_phrases:
                and_terms = ' AND '.join([
                    f'default contains "{phrase}"' for phrase in marqo_query.and_phrases
                ])
                if or_terms:
                    and_terms = f' AND ({and_terms})'
            else:
                and_terms = ''

            return f'{or_terms}{and_terms}'

        lexical_term = _get_lexical_search_term(marqo_query)
        filter_term = self._get_filter_term(marqo_query)
        if filter_term:
            filter_term = f' AND {filter_term}'
        else:
            filter_term = ''

        summary = unstructured_common.SUMMARY_ALL_VECTOR if marqo_query.expose_facets \
            else unstructured_common.SUMMARY_ALL_NON_VECTOR
        score_modifiers = self._get_score_modifiers(marqo_query)

        ranking = unstructured_common.RANK_PROFILE_BM25_MODIFIERS if score_modifiers \
            else unstructured_common.RANK_PROFILE_BM25

        query_inputs = {}

        if score_modifiers:
            query_inputs.update(score_modifiers)

        query = {
            'yql': f'select * from {self._marqo_index.schema_name} where {lexical_term}{filter_term}',
            'model_restrict': self._marqo_index.schema_name,
            'hits': marqo_query.limit,
            'offset': marqo_query.offset,
            'query_features': query_inputs,
            'presentation.summary': summary,
            'ranking': ranking
        }
        query = {k: v for k, v in query.items() if v is not None}
        return query

    def _to_vespa_hybrid_query(self, marqo_query: MarqoHybridQuery) -> Dict[str, Any]:
        raise NotImplementedError()

    def get_vector_count_query(self) -> Dict[str, Any]:
        return {
            'yql': f'select {unstructured_common.FIELD_VECTOR_COUNT} from {self._marqo_index.schema_name} '
                   f'where true limit 0 | all(group(1) each(output(sum({unstructured_common.FIELD_VECTOR_COUNT}))))',
            'model_restrict': self._marqo_index.schema_name,
            'timeout': '5s'
        }

    @classmethod
    def validate_field_content(cls, field_content: Any, is_tensor_field: bool) -> Any:
        """
            field: the field name of the field content. we need this to passed to validate_dict
            is_tensor_field: if the field is a tensor field
            Returns
                field_content, if it is valid

            Raises:
                InvalidArgError if field_content is not acceptable
            """
        if type(field_content) in cls._SUPPORTED_FIELD_CONTENT_TYPES:
            if isinstance(field_content, list):
                cls._validate_list(field_content, is_tensor_field)
            return field_content
        else:
            raise errors.InvalidArgError(
                f"Field content `{field_content}` \n"
                f"of type `{type(field_content).__name__}` is not of valid content type! "
                f"Allowed content types: {[ty.__name__ for ty in cls._SUPPORTED_FIELD_CONTENT_TYPES]}"
            )

    @staticmethod
    def _validate_list(field_content: list, is_tensor_field: bool) -> None:
        for element in field_content:
            if not isinstance(element, str):
                # if the field content is a list, it should only contain strings.
                raise errors.InvalidArgError(
                    f"Field content {field_content} includes an element of type {type(element).__name__} "
                    f"which is not a string. Unstructured Marqo index only supports string lists."
                )
        if is_tensor_field:
            raise errors.InvalidArgError(
                f"Field content '{field_content}' "
                f"of type {type(field_content).__name__} is not of valid content."
                f"Lists cannot be tensor fields"
            )

        return
