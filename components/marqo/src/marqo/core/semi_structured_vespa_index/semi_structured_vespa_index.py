from typing import Dict, Any, List, Optional, Type, Union, cast, Tuple, Set

from marqo.core import constants
from marqo.core.constants import MARQO_DOC_HIGHLIGHTS, MARQO_DOC_ID
from marqo.core.exceptions import MarqoDocumentParsingError
from marqo.core.models import MarqoQuery
from marqo.core.models.facets_parameters import FacetsParameters
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, LexicalOperand
from marqo.core.models.custom_score_rerank import ParsedCustomScoreKey
from marqo.core.models.marqo_index import DistanceMetric, SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery
from marqo.core.search import search_filter
from marqo.core.semi_structured_vespa_index import common
from marqo.core.semi_structured_vespa_index.common import VESPA_FIELD_ID, BOOL_FIELDS, SHORT_STRINGS_FIELDS, \
    STRING_ARRAY, INT_FIELDS, FLOAT_FIELDS
from marqo.core.semi_structured_vespa_index.marqo_field_types import MarqoFieldTypes
from marqo.core.semi_structured_vespa_index.semi_structured_document import SemiStructuredVespaDocument, \
    generate_uuid_str
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.unstructured_vespa_index.unstructured_validation import validate_field_name
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.core.utils.duration_parser import parse_duration_to_seconds
from marqo.core.vespa_index.vespa_index import VespaIndex
from marqo.exceptions import InternalError, InvalidArgumentError
from marqo.settings.settings import get_settings
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.recency_parameters import RecencyParameters, ApplyInRankingPhase, DecayFunction
from marqo.tensor_search.models.relevance_cutoff_model import RelevanceCutoffMethod
from marqo.vespa.models import QueryResult
from marqo.tensor_search.models.collapse_model import CollapseModel



class SemiStructuredVespaIndex(StructuredVespaIndex, UnstructuredVespaIndex):
    """
    An implementation of VespaIndex for SemiStructured indexes.
    TODO the multi-inheritance makes the implementation difficult to reason about. Consider refactor to composition
      instead. e.g. extract different logics to different query component builders, and combined the result.
    """

    index_supports_partial_updates: bool = False

    def __init__(self, marqo_index: SemiStructuredMarqoIndex):
        super().__init__(marqo_index)
        self.index_supports_partial_updates = marqo_index.index_supports_partial_updates

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

        if return_highlights and vespa_doc.fixed_fields.match_features:
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

            # Add collapse field if provided, this is critical for collapsing search result
            if (isinstance(marqo_query, MarqoHybridQuery) and marqo_query.collapse
                    and marqo_query.collapse.name not in marqo_query.attributes_to_retrieve):
                marqo_query.attributes_to_retrieve.append(marqo_query.collapse.name)

            # add chunk field names for tensor fields
            marqo_query.attributes_to_retrieve.extend(
                [self.get_marqo_index().tensor_field_map[att].chunk_field_name
                 for att in marqo_query.attributes_to_retrieve
                 if att in self.get_marqo_index().tensor_field_map]
            )
        # Hybrid must be checked first since it is a subclass of Tensor and Lexical
        if isinstance(marqo_query, MarqoHybridQuery):
            # this overrides the StructuredVespaIndex's implementation
            return self._to_vespa_hybrid_query(marqo_query)
        elif isinstance(marqo_query, MarqoTensorQuery):
            return StructuredVespaIndex._to_vespa_tensor_query(self, marqo_query)
        elif isinstance(marqo_query, MarqoLexicalQuery):
            return StructuredVespaIndex._to_vespa_lexical_query(self, marqo_query)

        else:
            raise InternalError(f'Unknown query type {type(marqo_query)}')

    # --- Custom score rerank support: duplicated from structured (to be deprecated) ---

    def _get_individual_field_tensor_search_terms(self, marqo_query: MarqoTensorQuery) -> List[str]:
        """Creates YQL tensor term per field from marqo_query; includes targetHits/approximate/hnsw options."""
        if isinstance(marqo_query, MarqoHybridQuery):
            attrs = marqo_query.hybrid_parameters.searchableAttributesTensor
            fields_to_search = (
                [f for f in attrs if f in self._marqo_index.tensor_field_map]
                if attrs is not None
                else list(self._marqo_index.tensor_field_map.keys())
            )
        else:
            attrs = marqo_query.searchable_attributes
            fields_to_search = (
                [f for f in attrs if f in self._marqo_index.tensor_field_map]
                if attrs is not None
                else list(self._marqo_index.tensor_field_map.keys())
            )

        if marqo_query.rerank_depth_tensor is not None:
            rerank_depth = max(marqo_query.rerank_depth_tensor, marqo_query.limit + marqo_query.offset)
        else:
            rerank_depth = marqo_query.limit + marqo_query.offset

        if marqo_query.ef_search is not None:
            rerank_depth = min(rerank_depth, marqo_query.ef_search)
        else:
            marqo_query.ef_search = max(
                utils.read_env_vars_and_defaults_ints(EnvVars.MARQO_DEFAULT_EF_SEARCH),
                marqo_query.limit + marqo_query.offset
            )
        additional_hits = max(marqo_query.ef_search - rerank_depth, 0)

        terms = []
        for field in fields_to_search:
            tensor_field = self._marqo_index.tensor_field_map[field]
            embedding_field_name = tensor_field.embeddings_field_name
            terms.append(
                f'('
                f'{{'
                f'targetHits:{rerank_depth}, '
                f'approximate:{str(marqo_query.approximate)}, '
                f'hnsw.exploreAdditionalHits:{additional_hits}'
                f'}}'
                f'nearestNeighbor({embedding_field_name}, {common.QUERY_INPUT_EMBEDDING})'
                f')'
            )
        return terms

    @staticmethod
    def _apply_lexical_operand(
            lexical_operand: LexicalOperand, terms: list,
            rerank_depth_lexical: Optional[int] = None) -> str:
        """Apply an explicit lexical operand to combine terms."""
        if lexical_operand == LexicalOperand.Or:
            return ' OR '.join(terms)
        elif lexical_operand == LexicalOperand.And:
            return ' AND '.join(terms)
        elif lexical_operand == LexicalOperand.WeakAnd:
            if rerank_depth_lexical is not None:
                return f'{{targetHits:{rerank_depth_lexical}}}weakAnd({", ".join(terms)})'
            return f'weakAnd({", ".join(terms)})'
        else:
            raise InternalError(f'Unknown lexical operand: {lexical_operand}')

    def _generate_or_terms(
        self,
        marqo_query: Union[MarqoLexicalQuery, MarqoHybridQuery],
        is_facets_term: bool = False,
        is_ranking_term: bool = False,
        attributes_to_search: Optional[List[str]] = None,
        lexical_operand_override: Optional[LexicalOperand] = None,
    ) -> str:
        """Generate the OR/weakAnd terms for the lexical search term."""
        if not marqo_query.or_phrases:
            return ''

        if isinstance(marqo_query, MarqoHybridQuery):
            score_modifiers = marqo_query.hybrid_parameters.scoreModifiersLexical
            rerank_depth_lexical: Optional[int] = marqo_query.hybrid_parameters.rerankDepthLexical
        else:
            score_modifiers = marqo_query.score_modifiers
            rerank_depth_lexical = None

        if rerank_depth_lexical is not None:
            rerank_depth_lexical = max(marqo_query.limit + marqo_query.offset, rerank_depth_lexical)

        terms = [
            self._get_lexical_contains_term(
                phrase,
                query=None if is_ranking_term else marqo_query,
                attributes_to_search=attributes_to_search if is_ranking_term else None,
                is_ranking_term=is_ranking_term,
            )
            for phrase in marqo_query.or_phrases
        ]

        # Avoid invalid YQL (e.g. weakAnd(, , ...)) when all terms are empty (e.g. no attrs survived).
        if is_ranking_term and terms and not any(terms):
            return ""

        if is_facets_term:
            return ' OR '.join(terms)

        if is_ranking_term:
            return f'weakAnd({", ".join(terms)})'

        lexical_operand = lexical_operand_override or (
            marqo_query.hybrid_parameters.lexicalOperand if isinstance(marqo_query, MarqoHybridQuery) else None
        )
        if lexical_operand is not None:
            return self._apply_lexical_operand(lexical_operand, terms, rerank_depth_lexical)

        if rerank_depth_lexical is not None:
            if rerank_depth_lexical <= 0:
                raise InternalError('RerankDepthLexical is less than or equal to 0 in _get_lexical_search_term')
            return f'{{targetHits:{rerank_depth_lexical}}}weakAnd({", ".join(terms)})'
        if score_modifiers:
            return ' OR '.join(terms)
        return f'weakAnd({", ".join(terms)})'

    def _get_lexical_search_term(
        self,
        marqo_query: Union[MarqoLexicalQuery, MarqoHybridQuery],
        is_facets_term: bool = False,
        is_ranking_term: bool = False,
        attributes_to_search: Optional[List[str]] = None,
        lexical_operand_override: Optional[LexicalOperand] = None,
    ) -> str:
        """
        Builds a lexical YQL search term for a query. It has an OR-query part (optional phrases)
        plus an optional AND (required phrases).

        Used in 2 scenarios:
        1. Retrieval lexical term (is_ranking_term=False, most common use case)
            - Uses marqo_query object and its searchable_attributes (if any).
        2. Ranking lexical term (is_ranking_term=True)
            - Uses given attributes_to_search, ignoring query-specified attributes. These attributes are determined by
            user-input custom score rerank fields. Ranking terms do NOT fetch all scores, only those specified in
            attributes_to_search.

        Rules:
        - Empty query → `false`; 
        - OR-only `["*"]` with no AND → `true`
        - OR phrases become weakAnd/OR (via ``_generate_or_terms``);
        - AND phrases are joined with AND; each is a contains expression over lexical fields.
        - For ranking terms: we attributes_to_search instead of marqo_query: if none of those fields are
          lexically searchable, return `""` and SKIP OR/AND statement creation.
        """
        if not marqo_query.or_phrases and not marqo_query.and_phrases:
            return 'false'
        if marqo_query.or_phrases == ["*"] and not marqo_query.and_phrases:
            return 'true'

        # When building a ranking term, if no attributes survive filtering, return "" without building OR/AND statements.
        if is_ranking_term and attributes_to_search is not None and attributes_to_search != ["*"]:
            searchable = [
                f for f in attributes_to_search
                if f in self._marqo_index.field_map
                and self._marqo_index.field_map[f].lexical_field_name is not None
            ]
            if not searchable:
                return ""

        or_terms = self._generate_or_terms(
            marqo_query,
            is_facets_term=is_facets_term,
            is_ranking_term=is_ranking_term,
            attributes_to_search=attributes_to_search,
            lexical_operand_override=lexical_operand_override
        )
        if marqo_query.and_phrases:
            and_terms = ' AND '.join([
                self._get_lexical_contains_term(
                    phrase,
                    query=None if is_ranking_term else marqo_query,
                    attributes_to_search=attributes_to_search if is_ranking_term else None,
                    is_ranking_term=is_ranking_term,
                )
                for phrase in marqo_query.and_phrases
            ])
            if or_terms:
                or_terms = f'({or_terms})'
                and_terms = f' AND ({and_terms})'
        else:
            and_terms = ''
        return f'{or_terms}{and_terms}'

    def _get_lexical_contains_term(
        self,
        phrase: str,
        query: Optional[MarqoQuery] = None,
        attributes_to_search: Optional[List[str]] = None,
        is_ranking_term: bool = False,
    ) -> str:
        """Build a single YQL contains expression for the given phrase."""
        if is_ranking_term:
            if attributes_to_search == ["*"]:
                return f'default contains "{phrase}"'
            searchable_attributes = [
                f for f in (attributes_to_search or [])
                if f in self._marqo_index.field_map
                and self._marqo_index.field_map[f].lexical_field_name is not None
            ]
            if not searchable_attributes:
                return ""
            return "(" + " OR ".join([
                f'{self._marqo_index.field_map[field].lexical_field_name} contains "{phrase}"'
                for field in searchable_attributes
            ]) + ")"
        if isinstance(query, MarqoHybridQuery):
            searchable_attributes = query.hybrid_parameters.searchableAttributesLexical
        else:
            searchable_attributes = query.searchable_attributes if query else None
        if searchable_attributes is not None:
            return "(" + " OR ".join([
                f'{self._marqo_index.field_map[field].lexical_field_name} contains "{phrase}"'
                for field in searchable_attributes
            ]) + ")"
        return f'default contains "{phrase}"'

    def _filter_applicable_custom_score_keys(self, custom_score_keys: Set[str]) -> Set[str]:
        """
        Subset of ``custom_score_keys`` that should be sent to Vespa for this index.

        Keys are **internal rerank keys** (suffix after ``marqo__score_``), i.e. the same dict keys
        produced by ``_convert_hybrid_global_score_modifiers_to_tensors`` for ``custom_score_rerank``
        (the ``marqo__score_`` prefix is stripped only in that method).

        **Rules:** Per-field BM25/closeness keys and any key we cannot parse are kept (parse
        failure is conservative: still send to Vespa). BM25 aggregate keys (``bm25_sum`` / max /
        avg) are omitted when the index has no lexically searchable fields; closeness aggregate
        keys are omitted when the index has no tensor fields—those aggregates would be meaningless.
        """
        applicable: Set[str] = set()
        for key in custom_score_keys:
            parsed = ParsedCustomScoreKey.parse(key)
            if parsed is None:
                applicable.add(key)
                continue
            # Aggregate keys are only applicable if the index has lexical/tensor fields respectively
            if parsed.score_type == "bm25" and parsed.aggregate_type is not None:
                if self._marqo_index.lexically_searchable_fields_names:
                    applicable.add(key)
            elif parsed.score_type == "closeness_retrieval_vector" and parsed.aggregate_type is not None:
                if self._marqo_index.tensor_field_map:
                    applicable.add(key)
            else:
                applicable.add(key)
        return applicable

    def _validate_custom_score_modifier_fields(self, custom_score_keys: Set[str]) -> None:
        """
        Validate custom score keys (internal suffix keys; see ``_filter_applicable_custom_score_keys``).
        """
        has_closeness = False
        for key in custom_score_keys:
            parsed = ParsedCustomScoreKey.parse(key)
            if parsed is None:
                continue
            if parsed.score_type == "closeness_retrieval_vector":
                has_closeness = True
                break
        if has_closeness and self._marqo_index.distance_metric == DistanceMetric.Geodegrees:
            raise InvalidArgumentError(
                "Custom score reranking with closeness_retrieval_vector is not supported for indexes using the "
                "geodegrees distance metric. Use a different distance metric (e.g. angular, euclidean, dotproduct) "
                "for the index if you need closeness-based score modifiers."
            )
        # BM25 aggregate with no lexical fields and closeness aggregate with no tensor fields are
        # filtered out by _filter_applicable_custom_score_keys before validation; no error here.
        # Per-field keys that reference non-existent fields are allowed: the searcher treats
        # missing summary-features as no contribution (add 0, multiply by 1), so score is unchanged.

    def _get_fields_to_bm25_rerank_by(self, custom_score_keys: Set[str]) -> List[str]:
        """Return Marqo field names for BM25 custom rerank (or ['*'] for aggregate). Keys: internal suffix only."""
        has_aggregate = False
        fields: Set[str] = set()
        for key in custom_score_keys:
            parsed = ParsedCustomScoreKey.parse(key)
            if parsed is None:
                continue
            if parsed.score_type == "bm25":
                if parsed.aggregate_type is not None:
                    has_aggregate = True
                    break
                if parsed.field_name is not None and parsed.field_name in self._marqo_index.field_map:
                    lex = self._marqo_index.field_map[parsed.field_name].lexical_field_name
                    if lex is not None:
                        fields.add(parsed.field_name)
        if has_aggregate:
            return ["*"]
        return sorted(fields)

    def _simplify_bm25_extra_fields_for_rank(
        self,
        bm25_fields: List[str],
        main_lexical_searchable_attributes: Optional[List[str]],
    ) -> List[str]:
        """Remove redundancy between main lexical term and BM25 extra term for rank()."""
        if not bm25_fields:
            return []
        # When main lexical attrs is None, the main term is "default contains ..." (all fields).
        # No extra BM25 rank term is needed; the main term already covers all lexical fields.
        if main_lexical_searchable_attributes is None:
            return []
        if bm25_fields == ["*"]:
            return ["*"]
        main_set = set(main_lexical_searchable_attributes)
        return [f for f in bm25_fields if f not in main_set]

    def _get_fields_to_closeness_rerank_by(self, custom_score_keys: Set[str]) -> List[str]:
        """Return Marqo tensor field names for closeness custom rerank (or all tensor fields for aggregate). Keys: internal suffix only."""
        has_aggregate = False
        fields: Set[str] = set()
        for key in custom_score_keys:
            parsed = ParsedCustomScoreKey.parse(key)
            if parsed is None:
                continue
            if parsed.score_type == "closeness_retrieval_vector":
                if parsed.aggregate_type is not None:
                    has_aggregate = True
                    break
                if parsed.field_name is not None and parsed.field_name in self._marqo_index.tensor_field_map:
                    fields.add(parsed.field_name)
        if has_aggregate:
            return list(self._marqo_index.tensor_field_map.keys())
        return sorted(fields)

    def _append_custom_score_rerank_terms(
        self,
        marqo_query: MarqoHybridQuery,
        lexical_term: str,
        tensor_term: str,
        custom_score_keys: Set[str],
    ) -> Tuple[str, str]:
        """
        ``custom_score_keys``: internal suffix keys (same as ``_filter_applicable_custom_score_keys``).

        When custom score rerank is used with RRF ranking, append extra BM25 rank terms so that
        the lexical and tensor retrievers can emit bm25 scores for custom-score fields.

        - Lexical retriever: we add an extra rank(..., extra_bm25_term) so that BM25 is computed
          for any custom-score BM25 fields that are not already covered by the main lexical term.
          Fields already in the main lexical search are omitted to avoid duplicate scoring.
        - Tensor retriever: we add an extra rank(..., extra_bm25_term) so that BM25 is computed
          for all custom-score BM25 fields (there is no main lexical term to dedupe against).

        If there are no custom score keys or ranking is not RRF, returns (lexical_term, tensor_term)
        unchanged.
        """
        if not custom_score_keys or marqo_query.hybrid_parameters.rankingMethod != RankingMethod.RRF:
            return (lexical_term, tensor_term)
        bm25_fields = self._get_fields_to_bm25_rerank_by(custom_score_keys)
        main_lexical_attrs = marqo_query.hybrid_parameters.searchableAttributesLexical
        simplified_lexical = self._simplify_bm25_extra_fields_for_rank(bm25_fields, main_lexical_attrs)
        simplified_tensor = bm25_fields  # tensor retriever has no main lexical term to dedupe against

        extra_bm25_term_for_lexical = ""
        if simplified_lexical:
            extra_bm25_term_for_lexical = self._get_lexical_search_term(
                marqo_query,
                is_ranking_term=True,
                attributes_to_search=simplified_lexical,
            )
        if extra_bm25_term_for_lexical:
            lexical_term = f'rank({lexical_term}, {extra_bm25_term_for_lexical})'

        extra_bm25_term_for_tensor = ""
        if bm25_fields:
            extra_bm25_term_for_tensor = self._get_lexical_search_term(
                marqo_query,
                is_ranking_term=True,
                attributes_to_search=simplified_tensor,
            )
        if extra_bm25_term_for_tensor:
            tensor_term = f'rank({tensor_term}, {extra_bm25_term_for_tensor})'

        return (lexical_term, tensor_term)

    def _get_base_vespa_hybrid_query(self, marqo_query):
        """
        Construct and return vespa hybrid query that includes the lexical & tensor subqueries, global score modifiers,
        relevance cut off, custom score rerankers.

        It does NOT include facets, sort by and collapse related query parameters yet.
        Refactor of StructuredVespaIndex._to_vespa_hybrid_query, as StructuredVespaIndex will be deprecated.
        """

        # Tensor term
        fields_to_search_tensor = self._get_tensor_fields_to_search(
            searchable_attributes=marqo_query.hybrid_parameters.searchableAttributesTensor
        )
        tensor_term = "False"
        if fields_to_search_tensor:
            marqo_query.rerank_depth_tensor = marqo_query.hybrid_parameters.rerankDepthTensor
            tensor_term = self._get_tensor_search_term(marqo_query)

        # Lexical term
        fields_to_search_lexical = self._get_lexical_fields_to_search(
            searchable_attributes=marqo_query.hybrid_parameters.searchableAttributesLexical
        )
        lexical_term = self._get_lexical_search_term(marqo_query) if fields_to_search_lexical else "False"

        # If retrieval and ranking methods are opposite (lexical/tensor), use the rank() operator
        if (marqo_query.hybrid_parameters.retrievalMethod == RetrievalMethod.Lexical and
                marqo_query.hybrid_parameters.rankingMethod == RankingMethod.Tensor):
            individual_tensor_terms = self._get_individual_field_tensor_search_terms(marqo_query)
            lexical_term = f'rank({lexical_term}, {",".join(individual_tensor_terms)})'

        elif (marqo_query.hybrid_parameters.retrievalMethod == RetrievalMethod.Tensor and
              marqo_query.hybrid_parameters.rankingMethod == RankingMethod.Lexical):
            tensor_term = f'rank({tensor_term}, {lexical_term})'

        # Filter term
        filter_term = self._get_filter_term(marqo_query)
        collapse_filter = marqo_query.collapse.sort_by.get_collapse_sort_by_filter_string() if marqo_query.collapse and marqo_query.collapse.sort_by else None
        parts = [f'({p})' for p in [collapse_filter, filter_term] if p]
        filter_term = (' AND ' + ' AND '.join(parts)) if parts else ''

        select_attributes = self._get_select_attributes(marqo_query)
        summary = common.SUMMARY_ALL_VECTOR if marqo_query.expose_facets else common.SUMMARY_ALL_NON_VECTOR

        # Base lexical YQL without custom-score extra rank() terms. Used for relevance-cutoff probe only.
        # Later code will modify lexical_term so we need to build the probe query early here
        lexical_yql_for_probe = None
        if getattr(marqo_query, "relevance_cutoff", None) is not None:
            if not marqo_query.relevance_cutoff.lexical_operand:
                # If the relevance_cutoff has no lexical_operand, it uses whatever is used in the main lexical query
                lexical_yql_for_probe = (
                    f'select {select_attributes} from {self._marqo_index.schema_name} where ({lexical_term}){filter_term}'
                )

            else:
                probe_lexical_term = self._get_lexical_search_term(
                    marqo_query, lexical_operand_override=marqo_query.relevance_cutoff.lexical_operand \
                ) if fields_to_search_lexical else "False"
                lexical_yql_for_probe = (
                    f'select {select_attributes} from {self._marqo_index.schema_name} '
                    f'where ({probe_lexical_term}){filter_term}'
                )

        # Assign parameters to query
        query_inputs = {
            common.QUERY_INPUT_EMBEDDING: marqo_query.vector_query
        }

        # Separate fields to rank (lexical and tensor)
        query_inputs.update({
            common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_LEXICAL: {
                f: 1 for f in fields_to_search_lexical
            },
            common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_TENSOR: {
                f: 1 for f in fields_to_search_tensor
            }
        })

        """
		# TODO: implement this if no longer using custom searcher for lexical/tensor and tensor/lexical
		query_inputs.update({
			f: 1 for f in fields_to_search_lexical
		})
		query_inputs.update({
			f: 1 for f in fields_to_search_tensor
		})
		"""

        # Extract score modifiers
        hybrid_score_modifiers = self._get_hybrid_score_modifiers(marqo_query)
        if hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_LEXICAL]:
            query_inputs.update(hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_LEXICAL])
        if hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_TENSOR]:
            query_inputs.update(hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_TENSOR])
        if hybrid_score_modifiers[constants.MARQO_GLOBAL_SCORE_MODIFIERS]:
            query_inputs.update(hybrid_score_modifiers[constants.MARQO_GLOBAL_SCORE_MODIFIERS])

        # Add custom score rerank support for semi-structured indexes.
        custom_score_keys: Set[str] = set()
        applicable_custom_score_keys: Set[str] = set()
        custom_score_rerank = hybrid_score_modifiers.get(constants.MARQO_CUSTOM_SCORE_RERANK_MODIFIERS)

        # Calculate new terms for tensor and lexical retrievers
        if custom_score_rerank:
            mult_key = constants.QUERY_INPUT_CUSTOM_SCORE_RERANK_MULT_WEIGHTS_GLOBAL
            add_key = constants.QUERY_INPUT_CUSTOM_SCORE_RERANK_ADD_WEIGHTS_GLOBAL
            # Keys are suffix after marqo__score_ (see _convert_hybrid_global_score_modifiers_to_tensors)
            custom_score_keys = set(custom_score_rerank.get(mult_key, {}).keys()) | set(
                custom_score_rerank.get(add_key, {}).keys()
            )
            # Ignore BM25 aggregate when no lexical fields and closeness aggregate when no tensor fields
            applicable_custom_score_keys = self._filter_applicable_custom_score_keys(custom_score_keys)
            query_inputs[mult_key] = {
                k: v for k, v in custom_score_rerank.get(mult_key, {}).items()
                if k in applicable_custom_score_keys
            }
            query_inputs[add_key] = {
                k: v for k, v in custom_score_rerank.get(add_key, {}).items()
                if k in applicable_custom_score_keys
            }
            if applicable_custom_score_keys:
                self._validate_custom_score_modifier_fields(applicable_custom_score_keys)

            lexical_term, tensor_term = self._append_custom_score_rerank_terms(
                marqo_query, lexical_term, tensor_term, applicable_custom_score_keys
            )

        # When custom score rerank is used with attributes_to_retrieve, sub-query hits must include
        # summaryfeatures (bm25, ranking_closeness_metric_*) or else these will be unavailable in the global step.
        select_for_hybrid_yql = select_attributes
        if applicable_custom_score_keys and marqo_query.attributes_to_retrieve is not None:
            select_for_hybrid_yql = select_attributes + ', summaryfeatures'

        tensor_yql = f'select {select_for_hybrid_yql} from {self._marqo_index.schema_name} where {tensor_term}{filter_term}'
        lexical_yql = f'select {select_for_hybrid_yql} from {self._marqo_index.schema_name} where ({lexical_term}){filter_term}'

        query = {
            'searchChain': 'marqo',
            'yql': 'PLACEHOLDER. WILL NOT BE USED IN HYBRID SEARCH.',
            'ranking': common.RANK_PROFILE_HYBRID_CUSTOM_SEARCHER,
            'ranking.rerankCount': marqo_query.hybrid_parameters.rerankCount if \
                marqo_query.hybrid_parameters.rerankCount else marqo_query.limit + marqo_query.offset,
            # limits the number of results going to phase 2

            'model_restrict': self._marqo_index.schema_name,
            'hits': marqo_query.limit,
            'offset': marqo_query.offset,
            'ranking.matching.approximateThreshold': marqo_query.approximate_threshold,
            'query_features': query_inputs,
            'presentation.summary': summary,
            'language': marqo_query.language,

            # Custom searcher parameters
            'marqo__yql.tensor': None if (
                    marqo_query.hybrid_parameters.retrievalMethod == RetrievalMethod.Lexical
                    and
                    marqo_query.hybrid_parameters.rankingMethod == RankingMethod.Lexical
            ) else tensor_yql,
            'marqo__yql.lexical': lexical_yql,

            'marqo__ranking.lexical.lexical': common.RANK_PROFILE_BM25,
            'marqo__ranking.tensor.tensor': common.RANK_PROFILE_EMBEDDING_SIMILARITY,
            'marqo__ranking.lexical.tensor': common.RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY,
            'marqo__ranking.tensor.lexical': common.RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25,

            'marqo__hybrid.retrievalMethod': marqo_query.hybrid_parameters.retrievalMethod,
            'marqo__hybrid.rankingMethod': marqo_query.hybrid_parameters.rankingMethod,
            'marqo__hybrid.verbose': marqo_query.hybrid_parameters.verbose,
            "marqo__yql.lexical.probe": lexical_yql_for_probe

        }

        query = {k: v for k, v in query.items() if v is not None}

        if marqo_query.hybrid_parameters.rankingMethod in {RankingMethod.RRF}:  # TODO: Add NormalizeLinear
            query["marqo__hybrid.alpha"] = marqo_query.hybrid_parameters.alpha
            query["marqo__hybrid.rrf_k"] = marqo_query.hybrid_parameters.rrfK

        if marqo_query.global_rerank_depth is not None:
            query["marqo__hybrid.rerankDepthGlobal"] = marqo_query.global_rerank_depth

        if (hybrid_score_modifiers[constants.MARQO_GLOBAL_SCORE_MODIFIERS]
                or hybrid_score_modifiers.get(constants.MARQO_CUSTOM_SCORE_RERANK_MODIFIERS)):
            query["marqo__expose_pre_rerank_score"] = True

        # Tell the custom searcher what type of custom score reranking will be done
        if applicable_custom_score_keys:
            has_bm25 = bool(self._get_fields_to_bm25_rerank_by(applicable_custom_score_keys))
            has_closeness = bool(self._get_fields_to_closeness_rerank_by(applicable_custom_score_keys))
            if has_closeness:
                query["marqo__hasRankingVector"] = True
                # Pass distance metric so searcher can min-max normalize only for dot product (others are already [0,1] in rank profile)
                query["marqo__custom_score_closeness_distance_metric"] = (
                    self._marqo_index.distance_metric.value
                )
            if has_bm25:
                query["marqo__hasRankingLexical"] = True

        return query

    def _to_vespa_hybrid_query(self, marqo_query):
        # get base query
        query = self._get_base_vespa_hybrid_query(marqo_query)

        # add facets query
        if marqo_query.facets or marqo_query.track_total_hits:
            query['marqo__yql.facets'] = self._generate_facet_queries(marqo_query)

        # add sort by and relevance cutoff
        self._add_relevance_cutoff_and_sort_by_params(marqo_query, query)

        # add the collapse_field to query
        if marqo_query.collapse:
            query.update(self._generate_collapse_query_params(marqo_query.collapse))

            if marqo_query.collapse.sort_by and marqo_query.collapse.sort_by.should_execute_sort():
                query['marqo__ranking.lexical.lexical'] = "collapse_to_sort_value"
                query["query_features"]["marqo__collapse_sort_weights"] = marqo_query.collapse.sort_by.generate_vespa_sort_by_query_input()
                query["hits"] = marqo_query.collapse.sort_by.COLLAPSE_SORT_BY_QUERY_LIMIT
                if marqo_query.collapse.sort_by.num_threads_per_search is not None:
                    query["ranking.matching.numThreadsPerSearch"] = marqo_query.collapse.sort_by.num_threads_per_search

        if marqo_query.recency_parameters:
            # Add recency parameters to query input
            query_input: Dict[str, Any] = query['query_features']
            query_input.update(self._get_recency_query_input(marqo_query.recency_parameters))

            query['marqo__recency_enabled'] = True
            query['marqo__recency_apply_in_global_ranking_phase'] = marqo_query.recency_parameters.apply_in_ranking_phase != ApplyInRankingPhase.EXCLUDE_GLOBAL

            # Set apply_to_subqueries flags as top-level query properties (not ranking features)
            # so Java can read them via query.properties().getBoolean()
            apply_to = marqo_query.recency_parameters.apply_to_subqueries
            if apply_to is None:
                apply_to = ["tensor", "lexical"]
            query[constants.QUERY_INPUT_RECENCY_APPLY_TO_TENSOR] = "tensor" in apply_to
            query[constants.QUERY_INPUT_RECENCY_APPLY_TO_LEXICAL] = "lexical" in apply_to

        # add lexical specific hybrid parameters
        if marqo_query.hybrid_parameters.secondPhaseModifier:
            if marqo_query.collapse:
                if marqo_query.collapse.sort_by and marqo_query.collapse.sort_by.should_execute_sort():
                    raise InternalError( # pragma: no cover
                        "Cannot use second phase modifiers with collapse sort by as they both modify the lexical ranking profile. "
                        "secondPhaseModifiers should set to None when doing collapse sort by search "
                    )
                query["marqo__ranking.lexical.lexical"] = common.RANK_PROFILE_HYBRID_BM25_SECOND_PHASE_MODIFIERS + '_diversity'
            else:
                query["marqo__ranking.lexical.lexical"] = common.RANK_PROFILE_HYBRID_BM25_SECOND_PHASE_MODIFIERS
        if marqo_query.hybrid_parameters.rerankCount:
            query["ranking.rerankCount"] = marqo_query.hybrid_parameters.rerankCount
        if marqo_query.hybrid_parameters.weakAndParameters:
            weak_and_query_dict = marqo_query.hybrid_parameters.weakAndParameters.convert_to_vespa_query_dict()
            query.update(weak_and_query_dict)
        return query

    def _get_recency_query_input(self, recency_params: RecencyParameters) -> dict:
        # Parse duration strings to seconds for Vespa
        scale_seconds = parse_duration_to_seconds(recency_params.scale)
        offset_seconds = parse_duration_to_seconds(recency_params.offset)

        result = {
            constants.QUERY_INPUT_RECENCY_SHOULD_CALCULATE_SCORE: 1,
            constants.QUERY_INPUT_RECENCY_SHOULD_APPLY_SCORE: 0 if recency_params.apply_in_ranking_phase == ApplyInRankingPhase.ONLY_GLOBAL else 1,
            constants.QUERY_INPUT_RECENCY_SCALE_SECONDS: scale_seconds,
            constants.QUERY_INPUT_RECENCY_OFFSET_SECONDS: offset_seconds,
            constants.QUERY_INPUT_RECENCY_DECAY_TO: recency_params.decay_to,
            constants.QUERY_INPUT_RECENCY_TIMESTAMP_KEY: {recency_params.recency_field: 1.0},
            constants.QUERY_INPUT_RECENCY_DECAY_FUNCTION_TYPE: DecayFunction(recency_params.decay_function).vespa_value,
            # Default to 0.0 for multiplicative mode (None means multiplicative)
            constants.QUERY_INPUT_RECENCY_ADD_TO_SCORE_WEIGHT: recency_params.add_to_score_weight if recency_params.add_to_score_weight is not None else 0.0
        }

        # Center timestamp: 0 means "use now()", positive value means fixed reference point
        result[constants.QUERY_INPUT_RECENCY_CENTER_SECONDS] = recency_params.center if recency_params.center is not None else 0

        # grow params, the recency_params validation ensures all or nothing for these params
        if recency_params.grow_from is not None:
            result[constants.QUERY_INPUT_RECENCY_GROW_ENABLED] = 1
            result[constants.QUERY_INPUT_RECENCY_GROW_FROM] = recency_params.grow_from
            result[constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE] = DecayFunction(recency_params.grow_function).vespa_value
            result[constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS] = parse_duration_to_seconds(recency_params.grow_scale)
            result[constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS] = parse_duration_to_seconds(recency_params.grow_offset)
        else:
            # Grow disabled - set defaults for all grow parameters
            result[constants.QUERY_INPUT_RECENCY_GROW_ENABLED] = 0
            result[constants.QUERY_INPUT_RECENCY_GROW_FROM] = 1.0
            result[constants.QUERY_INPUT_RECENCY_GROW_FUNCTION_TYPE] = 0  # exponential
            result[constants.QUERY_INPUT_RECENCY_GROW_SCALE_SECONDS] = scale_seconds  # use same as decay scale
            result[constants.QUERY_INPUT_RECENCY_GROW_OFFSET_SECONDS] = 0

        return result

    def _generate_collapse_query_params(self, collapse: CollapseModel):
        params: Dict[str, Any] = {
            'collapsefield': collapse.name,
            'collapsesize': 1,  # currently fixed to 1, will support multiple if needed in the future

            # use a different rank profile to ensure diversity in the result returned to Vespa container
            'marqo__ranking.lexical.lexical': common.RANK_PROFILE_BM25 + '_diversity',
            'marqo__ranking.tensor.tensor': common.RANK_PROFILE_EMBEDDING_SIMILARITY + '_diversity',
            'marqo__ranking.lexical.tensor': common.RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY + '_diversity',
            'marqo__ranking.tensor.lexical': common.RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25 + '_diversity',
        }

        # Only use minimal summary if the schema supports it (version check)
        if self.get_marqo_index().index_supports_collapse_minimal_summary:
            params['collapse.summary'] = 'collapse-minimal-summary'
            # When used with attributesToRetrieve, the searcher will try to pre-fill the specified fields from the
            # default summary, which defies the purpose of using a minimal summary for collapsing. Disabling
            # `FieldFiller` will force the searcher to use `collapse-minimal-summary` for collapsing queries.
            params['FieldFiller.disable'] = True
        return params

    def _add_relevance_cutoff_and_sort_by_params(self, marqo_query, query):
        if marqo_query.relevance_cutoff:
            query["marqo__hybrid.relevanceCutoff.method"] = marqo_query.relevance_cutoff.method
            if marqo_query.relevance_cutoff.method == RelevanceCutoffMethod.RelativeMaxScore:
                query["marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor"] = \
                    marqo_query.relevance_cutoff.parameters.relative_score_factor
            elif marqo_query.relevance_cutoff.method == RelevanceCutoffMethod.MeanStdDev:
                query["marqo__hybrid.relevanceCutoff.parameters.stdDevFactor"] = \
                    marqo_query.relevance_cutoff.parameters.std_dev_factor
            else:
                # No parameters for other methods
                pass
            query["marqo__hybrid.relevanceCutoff.probeDepth"] = marqo_query.relevance_cutoff.probe_depth
            query["marqo__hybrid.relevanceCutoff.affectFacets"] = marqo_query.relevance_cutoff.affect_facets
            query["marqo__hybrid.relevanceCutoff.overrideSortCandidates"] = marqo_query.relevance_cutoff.override_sort_candidates_with_relevant_candidates

            query["marqo__hybrid.relevanceCutoff.applyInRetrieval"] = marqo_query.relevance_cutoff.apply_in_retrieval
            query["marqo__hybrid.relevanceCutoff.overrideLimitPlusOffset"] = marqo_query.relevance_cutoff.override_limit_plus_offset
        # Sort by part
        if marqo_query.sort_by:
            query["marqo__hybrid.sortBy.fields"] = [field.dict() for field in marqo_query.sort_by.fields]
            query["marqo__hybrid.sortBy.sortDepth"] = marqo_query.sort_by.sort_depth
            query["marqo__hybrid.sortBy.minSortCandidates"] = marqo_query.sort_by.min_sort_candidates

            query["query_features"]["marqo__sort_field_weights_0"] = {}
            query["query_features"]["marqo__sort_field_weights_1"] = {}
            query["query_features"]["marqo__sort_field_weights_2"] = {}

            for index, field in enumerate(marqo_query.sort_by.fields):
                query["query_features"][f'marqo__sort_field_weights_{index}'] = {field.field_name: 1}

        return query

    def _generate_facet_queries(self, marqo_query):
        facets_query_skeleton = '%s limit 0 | %s'
        QUERY_DELIMITER = "\n---MARQO-YQL-QUERY-DELIMITER---\n"
        unique_exclusions = []
        facet_queries = []

        select_attributes = self._get_select_attributes(marqo_query)

        filter_term = self._get_filter_term(marqo_query)
        if filter_term:
            filter_term = f' AND ({filter_term})'
        else:
            filter_term = ''

        fields_to_search_tensor = self._get_tensor_fields_to_search(
            searchable_attributes=marqo_query.hybrid_parameters.searchableAttributesTensor
        )
        tensor_term = "False"
        if fields_to_search_tensor:
            marqo_query.rerank_depth_tensor = marqo_query.hybrid_parameters.rerankDepthTensor
            tensor_term = self._get_tensor_search_term(marqo_query)

        facets_lexical_term = self._get_lexical_search_term(marqo_query, is_facets_term=True)
        base_yql = f'select {select_attributes} from {self._marqo_index.schema_name} where ({facets_lexical_term})'
        if marqo_query.hybrid_parameters.retrievalMethod == RetrievalMethod.Disjunction:
            base_yql = f'select {select_attributes} from {self._marqo_index.schema_name} where ({facets_lexical_term} OR {tensor_term})'
        elif marqo_query.hybrid_parameters.retrievalMethod == RetrievalMethod.Tensor:
            base_yql = f'select {select_attributes} from {self._marqo_index.schema_name} where {tensor_term}'

        collapse_field_name = marqo_query.collapse.name if marqo_query.collapse else None

        if marqo_query.track_total_hits is not None:
            # 0 is byte representation of letter "t"
            if marqo_query.collapse:
                total_hit_query = f"all(group({self._TOTAL_HITS_GROUP_CONST}) each(group({collapse_field_name}) output(count())))"
            else:
                total_hit_query = f"all(group({self._TOTAL_HITS_GROUP_CONST}) each(output(count())))"
            facet_queries.append(facets_query_skeleton % (f'{base_yql}{filter_term}', total_hit_query))

        if marqo_query.facets is not None:
            facets_term = self._get_facets_term(
                marqo_query.facets,
                collapse_field_name=collapse_field_name
            )

            if facets_term is not None:
                facet_queries.append(facets_query_skeleton % (f'{base_yql}{filter_term}', facets_term))

            # Using a unique delimiter that's unlikely to appear in YQL

            for facet_field in marqo_query.facets.fields.items():
                facet_name, facet_parameters = facet_field
                if facet_parameters.exclude_terms is not None:
                    if any(set(facet_parameters.exclude_terms) == unique_exclusion for unique_exclusion in
                           unique_exclusions):
                        continue
                    unique_exclusions.append(set(facet_parameters.exclude_terms))
                    new_filter_term = self._get_filter_term(marqo_query, facet_parameters.exclude_terms)
                    if new_filter_term:
                        new_filter_term = f' AND {new_filter_term}'
                    else:
                        new_filter_term = ''
                    new_facets_term = self._get_facets_term(
                        marqo_query.facets, facet_parameters.exclude_terms,
                        collapse_field_name=collapse_field_name
                    )

                    query_yql = f'{base_yql}{new_filter_term}'

                    facet_queries.append(facets_query_skeleton % (query_yql, new_facets_term))

        return QUERY_DELIMITER.join(facet_queries)

    def _get_facets_term(self, facets_parameters: FacetsParameters, exclusion_terms: List[str] = None,
                         collapse_field_name: Optional[str] = None) -> str:
        """
        Build a facets grouping query string from the provided facets_parameters.
        """
        FIELD_TYPES = {
            "int": INT_FIELDS,
            "float": FLOAT_FIELDS,
            "string": SHORT_STRINGS_FIELDS,
            "array": STRING_ARRAY,
        }

        global_max_results = facets_parameters.max_results or 100
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
            else:
                parts.append("order(-count())")

            return " ".join(parts)

        def build_group_expression(field_config, field_name, field_id, field_type_overwrite=None) -> str:
            """Build the group expression for a field."""

            field_type = FIELD_TYPES[field_type_overwrite if field_type_overwrite else field_config.type]

            # Handle numeric fields with ranges
            if field_type in [INT_FIELDS, FLOAT_FIELDS] and field_config.ranges:
                buckets = []
                for range_config in field_config.ranges:
                    from_val = range_config.from_ if range_config.from_ is not None else "-inf"
                    to_val = range_config.to_ if range_config.to_ is not None else "inf"
                    buckets.append(f'bucket({from_val}, {to_val})')
                return f'predefined({field_type}{{"{field_name}"}}, {", ".join(buckets)})'

            elif field_type == STRING_ARRAY:
                return f"{STRING_ARRAY}_{field_name}"

            # Handle fields without ranges
            return str(field_id) if field_type in [INT_FIELDS, FLOAT_FIELDS] else f'{field_type}{{"{field_name}"}}'

        def build_field_group(field_config, field_name, field_id, field_type_overwrite=None) -> str:
            """Build the complete group request for a field."""
            group_expr = build_group_expression(field_config, field_name, field_id, field_type_overwrite)
            params = build_group_parameters(field_config)

            # Build output expression
            if collapse_field_name:
                # we can only get the count collapsed to this field regardless of data type
                output = f"each(group({collapse_field_name}) output(count()))"
            elif field_config.type == "number":
                # if we do not collapse, we can get the following stats with count for number type
                aggregations = ["sum", "avg", "min", "max"]
                field_type = FIELD_TYPES[field_type_overwrite]
                funcs = [f'{func}({field_type}{{"{field_name}"}})' for func in aggregations]
                funcs.append("count()")
                output = f"each(output({', '.join(funcs)}))"
            else:
                output = "each(output(count()))"

            return f"all(group({group_expr}) {params} {output}) "

        # Start building the overall grouping query.
        grouping_query = "all( "
        any_field = False # when exclusions are present, we need to check if any field is included in the default query
        if facets_parameters.max_depth is not None:
            grouping_query += f"max({facets_parameters.max_depth}) "
            # all(max(n) - state of grouping query

        for field_id, field_data in enumerate(facets_parameters.fields.items()):
            field_name, field_parameters = field_data
            if field_parameters.exclude_terms is not None:
                # We want this field to be in a separate query if any of the exclusions are not in the exclusions list
                if exclusion_terms is None or any([exclusion_term not in exclusion_terms for exclusion_term in field_parameters.exclude_terms]):
                    continue
            elif exclusion_terms is not None:
                continue
            any_field = True
            if field_parameters.type == "number":
                # we build 2 queries for number: flot and int
                grouping_query += build_field_group(field_parameters, field_name, field_id, field_type_overwrite="int")
                grouping_query += build_field_group(field_parameters, field_name, f"-{field_id}", field_type_overwrite="float")
            else:
                if field_parameters.type == "array":
                    if self.get_marqo_index().name_to_string_array_field_map.get(field_name) is None:
                        # Skip array field if it is not in the string array field map
                        continue
                grouping_query += build_field_group(field_parameters, field_name, field_id)

        grouping_query += ")"
        return grouping_query if any_field else None # None if default query is empty (all queries have exclusions)


    def _get_string_array_attributes_to_retrieve(self, attributes_to_retrieve: List) -> List[str]:
        name_to_string_array_field_map = self.get_marqo_index().name_to_string_array_field_map
        return [name_to_string_array_field_map[att].string_array_field_name for att in attributes_to_retrieve if
                name_to_string_array_field_map.get(att)]

    def _get_filter_term(self, marqo_query: MarqoQuery, exclude_terms: Optional[List[str]]=None) -> Optional[str]:
        # Reuse logic in UnstructuredVespaIndex to create filter term

        def generate_equality_filter_string(node: search_filter.EqualityTerm) -> str:
            filter_parts = []

            # Escape special characters in field name and value
            escaped_field = self.escape(node.field)
            escaped_value = self.escape(node.value)

            # Filter on `_id`
            if escaped_field == MARQO_DOC_ID:
                return f'({VESPA_FIELD_ID} contains "{escaped_value}")'

            if self.get_marqo_index().is_collapse_field(escaped_field):
                # collapse field is indexed as attribute, can be used directly in a filter term
                return f'({escaped_field} contains "{escaped_value}")'

            # Bool Filter
            if escaped_value.lower() in self._FILTER_STRING_BOOL_VALUES:
                filter_value = int(True if escaped_value.lower() == "true" else False)
                bool_filter_string = (f'({BOOL_FIELDS} contains '
                                      f'sameElement(key contains "{escaped_field}", value = {filter_value}))')
                filter_parts.append(bool_filter_string)

            # Short String Filter
            short_string_filter_string = (f'({SHORT_STRINGS_FIELDS} '
                                          f'contains sameElement(key contains "{escaped_field}", '
                                          f'value contains "{escaped_value}"))')
            filter_parts.append(short_string_filter_string)

            # String Array Filter
            if self.index_supports_partial_updates:
                if escaped_field in self.get_marqo_index().name_to_string_array_field_map:
                    string_array_field_name = f'{STRING_ARRAY}_{escaped_field}'
                    string_array_filter_string = (f'({string_array_field_name} contains '
                                                  f'"{escaped_value}")')
                    filter_parts.append(string_array_filter_string)
            else:
                string_array_filter_string = (f'({STRING_ARRAY} contains '
                                              f'"{escaped_field}::{escaped_value}")')
                filter_parts.append(string_array_filter_string)

            # Numeric Filter
            numeric_filter_string = ""
            try:
                numeric_value = int(escaped_value)
                numeric_filter_string = (
                    f'({INT_FIELDS} contains sameElement(key contains "{escaped_field}", value = {numeric_value})) '
                    f'OR ({FLOAT_FIELDS} contains sameElement(key contains "{escaped_field}", value = {numeric_value}))')
            except ValueError:
                try:
                    numeric_value = float(escaped_value)
                    numeric_filter_string = f'({FLOAT_FIELDS} contains sameElement(key contains "{escaped_field}", value = {numeric_value}))'
                except ValueError:
                    pass

            if numeric_filter_string:
                filter_parts.append(numeric_filter_string)

            # Final Filter String
            final_filter_string = f"({' OR '.join(filter_parts)})"
            return final_filter_string

        def generate_range_filter_string(node: search_filter.RangeTerm) -> str:
            # Escape special characters in field name
            escaped_field = self.escape(node.field)

            lower = f'value >= {node.lower}' if node.lower is not None else ""
            higher = f'value <= {node.upper}' if node.upper is not None else ""
            bound = f'{lower}, {higher}' if lower and higher else f'{lower}{higher}'
            if not bound:
                raise InternalError('RangeTerm has no lower or upper bound')

            float_field_string = (f'({FLOAT_FIELDS} contains '
                                  f'sameElement(key contains "{escaped_field}", {bound}))')

            int_field_string = (f'({INT_FIELDS} contains '
                                f'sameElement(key contains "{escaped_field}", {bound}))')

            return f'({float_field_string} OR {int_field_string})'

        def generate_in_filter_string(node: search_filter.InTerm) -> str:
            if node.field != MARQO_DOC_ID:
                raise InvalidArgumentError(
                    "The 'IN' filter keyword is only supported for the '_id' field "
                    "on semi-structured indexes."
                )

            max_in_filter_ids = get_settings().marqo_max_in_filter_ids
            if len(node.value_list) > max_in_filter_ids:
                raise InvalidArgumentError(
                    f"The IN filter contains {len(node.value_list)} values, which exceeds the maximum "
                    f"of {max_in_filter_ids} (MARQO_MAX_IN_FILTER_IDS)."
                )

            escaped_values = ', '.join(
                f'"{self.escape(v)}"' for v in node.value_list
            )
            return f'{VESPA_FIELD_ID} in ({escaped_values})'

        def generate_contains_filter_string(node: search_filter.ContainsTerm) -> str:
            escaped_field = self.escape(node.field)
            escaped_value = self.escape(node.value)
            marqo_index = self.get_marqo_index()
            # field_map only contains lexical fields for semi-structured indexes
            if escaped_field not in marqo_index.field_map:
                raise InvalidArgumentError(
                    f"CONTAINS filter field '{escaped_field}' is not found in index '{marqo_index.name}'. "
                    f"Available lexical fields: {', '.join(sorted(marqo_index.lexically_searchable_fields_names))}"
                )
            field = marqo_index.field_map[escaped_field]
            lexical_field_name = field.lexical_field_name
            return f'({lexical_field_name} contains "{escaped_value}")'

        def tree_to_filter_string(node: search_filter.Node) -> Optional[str]:
            # Skip any terms with excluded fields first - check at node level
            if (isinstance(node, search_filter.Term) or isinstance(node, search_filter.Modifier)) and exclude_terms is not None:
                if str(node) in exclude_terms:
                    return None

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
                if isinstance(node, search_filter.EqualityTerm):
                    return generate_equality_filter_string(node)
                elif isinstance(node, search_filter.RangeTerm):
                    return generate_range_filter_string(node)
                elif isinstance(node, search_filter.InTerm):
                    return generate_in_filter_string(node)
                elif isinstance(node, search_filter.ContainsTerm):
                    return generate_contains_filter_string(node)

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

    def gather_facets_from_response(self, response: QueryResult, facets: FacetsParameters) -> Dict[str, Dict]:
        """Convert a Vespa QueryResult into a Marqo-style facets response.

        Returns a dictionary of the form:
        {
            "field_name": {
                "facet_value_1": { stats... },
                "facet_value_2": { stats... },
                ...
            },
            ...
        }
        """
        if facets is not None:
            facet_field_map = self._build_field_map(facets)
        facets_response = {}
        total_hits = None

        # Process root groups only
        root_groups = (group for group in response.facets if group.id.startswith("group:facet:"))
        for group in root_groups:
            if group.children is None:
                continue
            for field in group.children:
                field_name = self._extract_facet_field_name(field.label, facets)
                if not field_name == self._TOTAL_HITS_GROUP_CONST:
                    facets_response.setdefault(field_name, {})

                for value in field.children:
                    group_type, value_key = self._parse_value_id(value.id)
                    processed_stats = self._process_value_stats(value.fields)

                    if field_name == self._TOTAL_HITS_GROUP_CONST:
                        total_hits = processed_stats["count"]

                    elif facets.fields[field_name].type in ["string", "array"]:
                        if value.id == "group:string:":
                            # Vespa's value for not found
                            continue
                        # values might be nested deeply if user data is stored as a.b.c
                        facets_response[field_name][value_key] = processed_stats
                    elif facets.fields[field_name].type in ["number"]:
                        if any(value in [self._MIN_LONG, 'NaN'] for value in processed_stats.values()):
                            # Vespa's value for null for int and float
                            continue
                        if facets.fields[field_name].ranges is None:
                            # aggregate statistic between int and float
                            facets_response[field_name] = self._combine_number_stats(
                                facets_response.get(field_name, {}), processed_stats
                            )
                        elif group_type != "null": # the case when there is no values in range
                            self._process_range_facets(
                                field_name, value_key, processed_stats,
                                facet_field_map, facets_response
                            )
                    else:
                        raise InternalError(f"Failed to parse facet {field_name} with type {facets.fields[field_name].type}")

        # Sort range facets by upper bound
        if facets is not None:
            for field_name, field_data in facets_response.items():
                if facet_field_map.get(field_name).ranges is not None:
                    facets_response[field_name] = self._sort_range_facets(field_data)

        response = {}
        if total_hits is not None:
            response["totalHits"] = total_hits
        if facets is not None:
            response["facets"] = facets_response
        return response

    def _build_field_map(self, facets: FacetsParameters) -> Dict[str, Any]:
        """Build a mapping from field names to their facet parameters."""
        return {
            facet_field[0]: facet_field[1]
            for facet_field in facets.fields.items()
        }

    def _extract_facet_field_name(self, field_label: str, facets: FacetsParameters) -> str:
        """Extract the facet field name from a Vespa field ID."""
        if field_label.startswith(SemiStructuredVespaSchema.FIELD_STRING_ARRAY_PREFIX):
            # strip group:marqo__string_array_ part of name
            return field_label[len(SemiStructuredVespaSchema.FIELD_STRING_ARRAY_PREFIX):]
        # this is only possible if field query was a number without ranges. So group name is n for float and negative n for int
        if not field_label.startswith("marqo__") and not field_label.startswith("predefined(marqo__"):
            if field_label.startswith('neg(') and field_label.endswith(')'):  # neg(n) - when combining
                group_index = field_label[4:-1]
            else:
                group_index = field_label
            if group_index == self._TOTAL_HITS_GROUP_CONST:
                return self._TOTAL_HITS_GROUP_CONST
            group_index = int(group_index)
            return next(iter(facet_field_name for i, facet_field_name in enumerate(facets.fields.keys()) if i == group_index))
            # return facets.fields.items()[group_index][0]
        return field_label.split('{')[1].split('}')[0].strip('"')

    def _parse_value_id(self, value_id: str) -> Tuple[str, str]:
        """Parse a Vespa value ID into group type and value key."""
        parts = value_id.split(':', 2)
        # if null returned by Vespa, only 2 parts from split
        return parts[1], parts[2] if len(parts) == 3 else None

    def _process_value_stats(self, fields: Dict) -> Dict:
        """Process field statistics, removing Vespa-specific suffixes."""
        return {k.split('(')[0]: v for k, v in fields.items()}

    def _combine_number_stats(self, current_stats, stats):
        if current_stats == {}:
            return stats
        aggregated_stats = {}
        aggregated_stats["count"] = current_stats["count"] + stats["count"]
        if "sum" in current_stats and "sum" in stats:
            aggregated_stats["sum"] = current_stats["sum"] + stats["sum"]
        if "avg" in current_stats and "avg" in stats:
            aggregated_stats["avg"] = (current_stats["avg"] * current_stats["count"] + stats["avg"] * stats["count"]) / (current_stats["count"] + stats["count"])
        if "min" in current_stats and "min" in stats:
            aggregated_stats["min"] = min(current_stats["min"], stats["min"])
        if "max" in current_stats and "max" in stats:
            aggregated_stats["max"] = max(current_stats["max"], stats["max"])
        return aggregated_stats


    def _process_range_facets(
            self,
            field_name: str,
            value_key: str,
            stats: Dict,
            field_map: Dict,
            response: Dict
    ) -> None:
        """Process range facets for a field."""
        params = field_map.get(field_name)
        if params is None or params.ranges is None:
            return

        for facet_range in params.ranges:
            value_to = value_key.split(":")[1]
            if (value_to == "Infinity" and facet_range.to_ is None) or float(value_to) == facet_range.to_:
                range_name = self._get_range_name(facet_range, value_to)
                if range_name:
                    aggregated_stats = self._combine_number_stats(response.get(field_name, {}).get(range_name, ({}, None))[0], stats)
                    # store ranges as stats, to_value to then sort and return from lower to higher.
                    response[field_name][range_name] = aggregated_stats, (facet_range.to_ if facet_range.to_ else float('inf'))

    def _get_range_name(self, facet_range: Any, value_key: str) -> Optional[str]:
        """Get the name for a range facet."""
        if facet_range.name is not None:
            if facet_range.to_ is not None:
                if float(value_key) == facet_range.to_:
                    return facet_range.name
            elif value_key == "Infinity":
                return facet_range.name
            return None

        from_val = "-Inf" if facet_range.from_ is None else str(facet_range.from_)
        to_val = "Inf" if facet_range.to_ is None else str(facet_range.to_)
        return f"{from_val}:{to_val}"

    def _sort_range_facets(self, field_data: Dict) -> Dict:
        """Sort range facets by their upper bound, with None (Infinity) at the end."""
        sorted_items = sorted(field_data.items(), key=lambda kv: kv[1][1])
        return {k: v[0] for k, v in sorted_items}
