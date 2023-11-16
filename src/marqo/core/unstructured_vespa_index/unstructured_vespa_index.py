from typing import Dict, Any, Optional

from marqo.core.models import MarqoQuery
from marqo.core.models.marqo_index import UnstructuredMarqoIndex
from marqo.core.vespa_index import VespaIndex
from marqo.core import constants as index_constants
from marqo.core.unstructured_vespa_index import common as unstructured_common
from marqo.core.unstructured_vespa_index.unstructured_document import UnstructuredIndexDocument
from marqo.core.models.marqo_query import (MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery,
                                           ScoreModifierType, ScoreModifier)
from marqo.exceptions import InternalError
import marqo.core.search.search_filter as search_filter
from marqo.core.exceptions import InvalidDataTypeError, InvalidFieldNameError, VespaDocumentParsingError



class UnstructuredVespaIndex(VespaIndex):
    def __init__(self, marqo_index: UnstructuredMarqoIndex):
        self._marqo_index = marqo_index

    def to_vespa_document(self, marqo_document: Dict[str, Any]) -> Dict[str, Any]:
        unstructured_document: UnstructuredIndexDocument = UnstructuredIndexDocument.from_marqo_document(marqo_document)
        return unstructured_document.to_vespa_document()

    def to_marqo_document(self, vespa_document: Dict[str, Any], return_highlights: bool = False) -> Dict[str, Any]:
        unstructured_document: UnstructuredIndexDocument = UnstructuredIndexDocument.from_vespa_document(vespa_document)
        return unstructured_document.to_marqo_document(return_highlights=return_highlights)

    def to_vespa_query(self, marqo_query: MarqoQuery) -> Dict[str, Any]:
        # Verify attributes to retrieve, if defined
        # TODO Attributes to retrieve 
        # if marqo_query.attributes_to_retrieve is not None:
        #     for att in marqo_query.attributes_to_retrieve:
        #         if att not in marqo_index.field_map:
        #             raise InvalidFieldNameError(
        #                 f'Index {marqo_index.name} has no field {att}. '
        #                 f'Available fields are {", ".join(marqo_index.field_map.keys())}'
        #             )
        
        # TODO Add score modifiers
        # Verify score modifiers, if defined
        # if marqo_query.score_modifiers is not None:
        #     for modifier in marqo_query.score_modifiers:
        #         if modifier.field not in marqo_index.score_modifier_fields_names:
        #             raise InvalidFieldNameError(
        #                 f'Index {marqo_index.name} has no score modifier field {modifier.field}. '
        #                 f'Available score modifier fields are {", ".join(marqo_index.score_modifier_fields_names)}'
        #             )

        if isinstance(marqo_query, MarqoTensorQuery):
            return self._to_vespa_tensor_query(marqo_query)
        
        # TODO Add lexical and hybrid queries
        # elif isinstance(marqo_query, MarqoLexicalQuery):
        #     return cls._to_vespa_lexical_query(marqo_query, marqo_index)
        # elif isinstance(marqo_query, MarqoHybridQuery):
        #     return cls._to_vespa_hybrid_query(marqo_query, marqo_index)
        # else:
        #     raise InternalError(f'Unknown query type {type(marqo_query)}')
    #
    # @classmethod
    # def _to_vespa_lexical_query(cls, marqo_query: MarqoLexicalQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
    #     raise NotImplementedError()
    #
    # @classmethod
    # def _to_vespa_hybrid_query(cls, marqo_query: MarqoHybridQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
    #     raise NotImplementedError()
    #

    def _to_vespa_tensor_query(self, marqo_query: MarqoTensorQuery) -> Dict[str, Any]:
        if marqo_query.searchable_attributes is not None:
            raise RuntimeError("searchable_attributes is not supported for an UnStructured Index")

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
            'yql': f"select {select_attributes} from {marqo_query.index_name} where {tensor_term}{filter_term}",
            'model_restrict': marqo_query.index_name,
            'hits': marqo_query.limit,
            'offset': marqo_query.offset,
            'query_features': query_inputs,
            'presentation.summary': summary,
            'ranking': ranking
        }
        query = {k: v for k, v in query.items() if v is not None}

        return query

    @staticmethod
    def _get_tensor_search_term(marqo_query: MarqoQuery) -> str:
        field_to_search = unstructured_common.VESPA_DOC_EMBEDDINGS

        return (f"({{targetHits:{marqo_query.limit}, approximate:{str(marqo_query.approximate)}}}"
                f"nearestNeighbor({field_to_search}, {unstructured_common.QUERY_INPUT_EMBEDDING}))")

    @staticmethod
    def _get_filter_term(marqo_query: MarqoQuery) -> Optional[str]:
        def escape(s: str) -> str:
            return s.replace('\\', '\\\\').replace('"', '\\"')

        def generate_equality_filter_string(node: search_filter.EqualityTerm) -> str:

            short_string_filter_string = f'({unstructured_common.SHORT_STRINGS_FIELDS} ' \
                                         f'contains sameElement(key contains "{node.field}", ' \
                                         f'value contains "{escape(node.value)}"))'

            string_array_filter_string = f'({unstructured_common.STRING_ARRAY} contains ' \
                                         f'"{node.field}::{escape(node.value)}")'

            return f'({short_string_filter_string} OR {string_array_filter_string})'

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

            return f'{float_field_string} OR {int_field_string}'

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

    def _get_score_modifiers(self, marqo_query: MarqoQuery) -> \
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



    # @classmethod
    # def _get_select_attributes(cls, marqo_query: MarqoQuery) -> str:
    #     if marqo_query.attributes_to_retrieve is not None:
    #         return ', '.join(marqo_query.attributes_to_retrieve)
    #     else:
    #         return '*'

    # @classmethod
    # def _get_score_modifiers(cls, marqo_query: MarqoQuery) -> \
    #         Optional[Dict[str, Dict[str, float]]]:
    #     if marqo_query.score_modifiers:
    #         mult_tensor = {}
    #         add_tensor = {}
    #         for modifier in marqo_query.score_modifiers:
    #             if modifier.type == ScoreModifierType.Multiply:
    #                 mult_tensor[modifier.field] = modifier.weight
    #             elif modifier.type == ScoreModifierType.Add:
    #                 add_tensor[modifier.field] = modifier.weight
    #             else:
    #                 raise InternalError(f'Unknown score modifier type {modifier.type}')
    #
    #         # Note one of these could be empty, but not both
    #         return {
    #             cls._QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS: mult_tensor,
    #             cls._QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS: add_tensor
    #         }

        return None

    # @classmethod
    # def _get_filter_term(cls, marqo_query: MarqoQuery, marqo_index: MarqoIndex) -> Optional[str]:
    #     def escape(s: str) -> str:
    #         return s.replace('\\', '\\\\').replace('"', '\\"')
    # 
    #     def tree_to_filter_string(node: search_filter.Node) -> str:
    #         if isinstance(node, search_filter.Operator):
    #             if isinstance(node, search_filter.And):
    #                 operator = 'AND'
    #             elif isinstance(node, search_filter.Or):
    #                 operator = 'OR'
    #             else:
    #                 raise InternalError(f'Unknown operator type {type(node)}')
    # 
    #             return f'({tree_to_filter_string(node.left)} {operator} {tree_to_filter_string(node.right)})'
    #         elif isinstance(node, search_filter.Modifier):
    #             if isinstance(node, search_filter.Not):
    #                 return f'!({tree_to_filter_string(node.modified)})'
    #             else:
    #                 raise InternalError(f'Unknown modifier type {type(node)}')
    #         elif isinstance(node, search_filter.Term):
    #             # if node.field not in marqo_index.filterable_fields_names:
    #             #     raise InvalidFieldNameError(
    #             #         f'Index {marqo_index.name} has no filterable field {node.field}. '
    #             #         f'Available filterable fields are: {", ".join(marqo_index.filterable_fields_names)}'
    #             #     )
    # 
    #             # TODO Remove the validation and just use the base case
    #             if isinstance(node, search_filter.EqualityTerm):
    #                 # TODO Instead of using contains, use sameElement to replace this one
    #                 return f'{node.field} contains "{escape(node.value)}"'
    #             elif isinstance(node, search_filter.RangeTerm):
    #                 # TODO Use sameElement to replace the range
    #                 lower = f'{node.field} >= {node.lower}' if node.lower is not None else None
    #                 upper = f'{node.field} <= {node.upper}' if node.upper is not None else None
    #                 if lower and upper:
    #                     return f'({lower} AND {upper})'
    #                 elif lower:
    #                     return lower
    #                 elif upper:
    #                     return upper
    #                 else:
    #                     raise InternalError('RangeTerm has no lower or upper bound')
    # 
    #             raise InternalError(f'Unknown node type {type(node)}')
    # 
    #     if marqo_query.filter is not None:
    #         return tree_to_filter_string(marqo_query.filter.root)