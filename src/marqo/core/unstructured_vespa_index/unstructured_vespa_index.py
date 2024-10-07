from typing import Dict, Any, Optional

import marqo.core.constants as index_constants
import marqo.core.search.search_filter as search_filter
from marqo.api import exceptions as errors
from marqo.core.models import MarqoQuery
from marqo.core.models.marqo_query import (MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery)
from marqo.core.models.hybrid_parameters import RankingMethod, RetrievalMethod
from marqo.core.unstructured_vespa_index import common as unstructured_common
from marqo.core.unstructured_vespa_index.unstructured_document import UnstructuredVespaDocument
from marqo.core.vespa_index.vespa_index import VespaIndex
from marqo.core import constants
from marqo.exceptions import InternalError, InvalidArgumentError
import semver


class UnstructuredVespaIndex(VespaIndex):
    _FILTER_STRING_BOOL_VALUES = ["true", "false"]
    _RESERVED_FIELD_SUBSTRING = "::"
    _SUPPORTED_FIELD_CONTENT_TYPES = [str, int, float, bool, list, dict]

    _HYBRID_SEARCH_MINIMUM_VERSION = constants.MARQO_UNSTRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION

    def get_vespa_id_field(self) -> str:
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

        if isinstance(marqo_query, MarqoHybridQuery):       # TODO: Rethink structure so order of checking doesn't matter
            return self._to_vespa_hybrid_query(marqo_query)
        elif isinstance(marqo_query, MarqoTensorQuery):
            return self._to_vespa_tensor_query(marqo_query)
        elif isinstance(marqo_query, MarqoLexicalQuery):
            return self._to_vespa_lexical_query(marqo_query)

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

        if self._marqo_index_version < self._HYBRID_SEARCH_MINIMUM_VERSION:
            ranking = unstructured_common.RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS_2_9 if score_modifiers \
                else unstructured_common.RANK_PROFILE_EMBEDDING_SIMILARITY
        else:
            ranking = unstructured_common.RANK_PROFILE_EMBEDDING_SIMILARITY

        if self._marqo_index_version >= self._HYBRID_SEARCH_MINIMUM_VERSION:
            query_input_embedding_parameter = unstructured_common.QUERY_INPUT_EMBEDDING
        else:
            query_input_embedding_parameter = unstructured_common.QUERY_INPUT_EMBEDDING_2_10

        query_inputs = {
            query_input_embedding_parameter: marqo_query.vector_query
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
            query['timeout'] = 300 * 1000  # 5 minutes

        return query

    def _get_tensor_search_term(self, marqo_query: MarqoTensorQuery) -> str:
        field_to_search = unstructured_common.VESPA_DOC_EMBEDDINGS

        if marqo_query.ef_search is not None:
            target_hits = min(marqo_query.limit + marqo_query.offset, marqo_query.ef_search)
            additional_hits = max(marqo_query.ef_search - (marqo_query.limit + marqo_query.offset), 0)
        else:
            target_hits = marqo_query.limit + marqo_query.offset
            additional_hits = 0

        if self._marqo_index_version >= self._HYBRID_SEARCH_MINIMUM_VERSION:
            query_input_embedding_parameter = unstructured_common.QUERY_INPUT_EMBEDDING
        else:
            query_input_embedding_parameter = unstructured_common.QUERY_INPUT_EMBEDDING_2_10

        return (
            f"("
            f"{{"
            f"targetHits:{target_hits}, "
            f"approximate:{str(marqo_query.approximate)}, "
            f'hnsw.exploreAdditionalHits:{additional_hits}'
            f"}}"
            f"nearestNeighbor({field_to_search}, {query_input_embedding_parameter})"
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
                elif isinstance(node, search_filter.InTerm):
                    raise InvalidArgumentError("The 'IN' filter keyword is not yet supported for unstructured indexes")
            raise InternalError(f'Unknown node type {type(node)}')

        if marqo_query.filter is not None:
            return tree_to_filter_string(marqo_query.filter.root)

    def _get_lexical_search_term(self, marqo_query: MarqoLexicalQuery) -> str:
        if isinstance(marqo_query, MarqoHybridQuery):
            score_modifiers = marqo_query.hybrid_parameters.scoreModifiersLexical
        else:
            score_modifiers = marqo_query.score_modifiers

        # Empty query and wildcard
        if not marqo_query.or_phrases and not marqo_query.and_phrases:
            return 'false'
        if marqo_query.or_phrases == ["*"] and not marqo_query.and_phrases:
            return 'true'

        # Optional tokens
        # TODO: add searchable attributes. This will affect the lexical "contains" term.
        if marqo_query.or_phrases and score_modifiers:
            or_terms = ' OR '.join([
                f'default contains "{phrase}"' for phrase in marqo_query.or_phrases
            ])
        elif marqo_query.or_phrases and not score_modifiers:
            or_terms = 'weakAnd(%s)' % ', '.join([
                f'default contains "{phrase}"' for phrase in marqo_query.or_phrases
            ])
        else:
            or_terms = ''

        # Required tokens
        if marqo_query.and_phrases:
            and_terms = ' AND '.join([
                f'default contains "{phrase}"' for phrase in marqo_query.and_phrases
            ])
            if or_terms:
                or_terms = f'({or_terms})'
                and_terms = f' AND ({and_terms})'
        else:
            and_terms = ''

        return f'{or_terms}{and_terms}'

    def _to_vespa_lexical_query(self, marqo_query: MarqoLexicalQuery) -> Dict[str, Any]:
        lexical_term = self._get_lexical_search_term(marqo_query)
        filter_term = self._get_filter_term(marqo_query)

        if filter_term:
            search_term = f'({lexical_term}) AND ({filter_term})'
        else:
            search_term = f'({lexical_term})'

        summary = unstructured_common.SUMMARY_ALL_VECTOR if marqo_query.expose_facets \
            else unstructured_common.SUMMARY_ALL_NON_VECTOR
        score_modifiers = self._get_score_modifiers(marqo_query)

        if self._marqo_index_version < self._HYBRID_SEARCH_MINIMUM_VERSION:
            ranking = unstructured_common.RANK_PROFILE_BM25_MODIFIERS_2_9 if score_modifiers \
                else unstructured_common.RANK_PROFILE_BM25
        else:
            ranking = unstructured_common.RANK_PROFILE_BM25

        query_inputs = {}

        if score_modifiers:
            query_inputs.update(score_modifiers)

        query = {
            'yql': f'select * from {self._marqo_index.schema_name} where {search_term}',
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
        # TODO: Add "fields to search" when searchable attributes get implemented
        # Tensor term
        tensor_term = self._get_tensor_search_term(marqo_query)
        # Lexical term
        lexical_term = self._get_lexical_search_term(marqo_query)

        # If retrieval and ranking methods are opposite (lexical/tensor), use the rank() operator
        if (marqo_query.hybrid_parameters.retrievalMethod == RetrievalMethod.Lexical and
                marqo_query.hybrid_parameters.rankingMethod == RankingMethod.Tensor):
            lexical_term = f'rank({lexical_term}, {tensor_term})'

        elif (marqo_query.hybrid_parameters.retrievalMethod == RetrievalMethod.Tensor and
              marqo_query.hybrid_parameters.rankingMethod == RankingMethod.Lexical):
            tensor_term = f'rank({tensor_term}, {lexical_term})'

        # Filter term
        filter_term = self._get_filter_term(marqo_query)
        if filter_term:
            filter_term = f' AND ({filter_term})'
        else:
            filter_term = ''

        select_attributes = "*"     # TODO: Fix when searchable attributes are implemented

        summary = unstructured_common.SUMMARY_ALL_VECTOR if marqo_query.expose_facets \
            else unstructured_common.SUMMARY_ALL_NON_VECTOR

        # Assign parameters to query
        query_inputs = {
            unstructured_common.QUERY_INPUT_EMBEDDING: marqo_query.vector_query,
            unstructured_common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_LEXICAL: {},
            unstructured_common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_TENSOR: {}
        }

        # TODO: add this back when searchable attributes are implemented
        # Separate fields to rank (lexical and tensor)
        #query_inputs.update({
        #    unstructured_common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_LEXICAL: {
        #        f: 1 for f in fields_to_search_lexical
        #    },
        #    unstructured_common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_TENSOR: {
        #        f: 1 for f in fields_to_search_tensor
        #    }
        #})

        # Extract score modifiers
        hybrid_score_modifiers = self._get_hybrid_score_modifiers(marqo_query)
        if hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_LEXICAL]:
            query_inputs.update(hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_LEXICAL])
        if hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_TENSOR]:
            query_inputs.update(hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_TENSOR])

        query = {
            'searchChain': 'marqo',
            'yql': 'PLACEHOLDER. WILL NOT BE USED IN HYBRID SEARCH.',
            'ranking': unstructured_common.RANK_PROFILE_HYBRID_CUSTOM_SEARCHER,
            'ranking.rerankCount': marqo_query.limit + marqo_query.offset,  # limits the number of results going to phase 2

            'model_restrict': self._marqo_index.schema_name,
            'hits': marqo_query.limit,
            'offset': marqo_query.offset,
            'query_features': query_inputs,
            'presentation.summary': summary,

            # Custom searcher parameters
            'marqo__yql.tensor': f'select {select_attributes} from {self._marqo_index.schema_name} where {tensor_term}{filter_term}',
            'marqo__yql.lexical': f'select {select_attributes} from {self._marqo_index.schema_name} where ({lexical_term}){filter_term}',

            'marqo__ranking.lexical.lexical': unstructured_common.RANK_PROFILE_BM25,
            'marqo__ranking.tensor.tensor': unstructured_common.RANK_PROFILE_EMBEDDING_SIMILARITY,
            'marqo__ranking.lexical.tensor': unstructured_common.RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY,
            'marqo__ranking.tensor.lexical': unstructured_common.RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25,

            'marqo__hybrid.retrievalMethod': marqo_query.hybrid_parameters.retrievalMethod,
            'marqo__hybrid.rankingMethod': marqo_query.hybrid_parameters.rankingMethod,
            'marqo__hybrid.verbose': marqo_query.hybrid_parameters.verbose
        }
        query = {k: v for k, v in query.items() if v is not None}

        if marqo_query.hybrid_parameters.rankingMethod in {RankingMethod.RRF}:  # TODO: Add NormalizeLinear
            query["marqo__hybrid.alpha"] = marqo_query.hybrid_parameters.alpha

        if marqo_query.hybrid_parameters.rankingMethod in {RankingMethod.RRF}:
            query["marqo__hybrid.rrf_k"] = marqo_query.hybrid_parameters.rrfK

        return query

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
