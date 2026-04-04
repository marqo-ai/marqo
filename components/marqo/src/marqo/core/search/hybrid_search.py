from typing import List, Optional, Union, Iterable, Dict

from marqo.api import exceptions as api_exceptions
from marqo.api import exceptions as errors
# We depend on _httprequests.py for now, but this may be replaced in the future, as
# _httprequests.py is designed for the client
from marqo.config import Config
from marqo.core import constants
from marqo.core import exceptions as core_exceptions
from marqo.core.models import MarqoIndex
from marqo.core.models.facets_parameters import FacetsParameters
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.core.models.marqo_index import UnstructuredMarqoIndex, StructuredMarqoIndex, SemiStructuredMarqoIndex, \
    IndexType
from marqo.core.models.marqo_query import MarqoHybridQuery
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.core.structured_vespa_index.common import RANK_PROFILE_HYBRID_CUSTOM_SEARCHER
from marqo.core.vespa_index.vespa_index import for_marqo_index as vespa_index_factory
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.enums import (
    SearchMethod
)
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity, ScoreModifierLists, CustomVectorQuery
from marqo.tensor_search.models.collapse_model import CollapseModel
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.recency_parameters import RecencyParameters
from marqo.tensor_search.models.relevance_cutoff_model import RelevanceCutoffModel
from marqo.tensor_search.models.search import Qidx, SearchContext, SearchContextTensor
from marqo.tensor_search.models.sort_by_model import SortByModel
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.tensor_search import run_vectorise_pipeline, gather_documents_from_response, logger
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints
from marqo.vespa.exceptions import VespaStatusError


def should_use_collapse_search(
        collapse: Optional[CollapseModel] = None,
        main_query_sort_by: Optional[SortByModel] = None
) -> bool:
    """
    Determine whether to use collapse search based on the collapse parameters, and sort_by parameters.
    """
    if not collapse:
        return False

    if not collapse.sort_by:
        return False

    if not main_query_sort_by:
        return True

    if not collapse.sort_by.disable_if_main_sort_by_fields:
        return True

    main_query_sort_by_fields = {field.field_name for field in main_query_sort_by.fields}

    if not main_query_sort_by_fields.isdisjoint(collapse.sort_by.disable_if_main_sort_by_fields):
        return False  # There is an intersection
    return True


class HybridSearch:
    def search(
            self,
            config: Config, marqo_index: MarqoIndex, query: Optional[Union[None, str, CustomVectorQuery]],
            result_count: int = 5, offset: int = 0, rerank_depth: Optional[int] = None,
            ef_search: Optional[int] = None, approximate: bool = True,
            approximate_threshold: Optional[float] = None,
            searchable_attributes: Iterable[str] = None, filter_string: str = None, device: str = None,
            attributes_to_retrieve: Optional[List[str]] = None, boost: Optional[Dict] = None,
            media_download_headers: Optional[Dict] = None, context: Optional[SearchContext] = None,
            score_modifiers: Optional[ScoreModifierLists] = None, model_auth: Optional[ModelAuth] = None,
            highlights: bool = False, text_query_prefix: Optional[str] = None,
            hybrid_parameters: HybridParameters = None,
            facets: Optional[FacetsParameters] = None,
            track_total_hits: Optional[bool] = None,
            language: Optional[str] = None,
            relevance_cutoff: Optional[RelevanceCutoffModel] = None,
            sort_by: Optional[SortByModel] = None,
            interpolation_method: Optional[InterpolationMethod] = None,
            collapse: Optional[CollapseModel] = None,
            recency_parameters: Optional[RecencyParameters] = None
    ):
        if should_use_collapse_search(collapse=collapse, main_query_sort_by=sort_by):
            # Deliberately use a late import to avoid circular imports
            from marqo.core.search.collapse_search import CollapseSearch
            return CollapseSearch(
                config=config,
                marqo_index=marqo_index,
                query=query,
                result_count=result_count,
                offset=offset,
                rerank_depth=rerank_depth,
                ef_search=ef_search,
                approximate=approximate,
                approximate_threshold=approximate_threshold,
                searchable_attributes=searchable_attributes,
                filter_string=filter_string,
                device=device,
                attributes_to_retrieve=attributes_to_retrieve,
                boost=boost,
                media_download_headers=media_download_headers,
                context=context,
                score_modifiers=score_modifiers,
                model_auth=model_auth,
                highlights=highlights,
                text_query_prefix=text_query_prefix,
                hybrid_parameters=hybrid_parameters,
                facets=facets,
                track_total_hits=track_total_hits,
                language=language,
                relevance_cutoff=relevance_cutoff,
                sort_by=sort_by,
                interpolation_method=interpolation_method,
                collapse=collapse,
                recency_parameters=recency_parameters
            ).search()
        else:
            return self.execute_search(
                config=config,
                marqo_index=marqo_index,
                query=query,
                result_count=result_count,
                offset=offset,
                rerank_depth=rerank_depth,
                ef_search=ef_search,
                approximate=approximate,
                approximate_threshold=approximate_threshold,
                searchable_attributes=searchable_attributes,
                filter_string=filter_string,
                device=device,
                attributes_to_retrieve=attributes_to_retrieve,
                boost=boost,
                media_download_headers=media_download_headers,
                context=context,
                score_modifiers=score_modifiers,
                model_auth=model_auth,
                highlights=highlights,
                text_query_prefix=text_query_prefix,
                hybrid_parameters=hybrid_parameters,
                facets=facets,
                track_total_hits=track_total_hits,
                language=language,
                relevance_cutoff=relevance_cutoff,
                sort_by=sort_by,
                interpolation_method=interpolation_method,
                collapse=collapse,
                recency_parameters=recency_parameters
            )

    def execute_search(
            self, config: Config, marqo_index: MarqoIndex, query: Optional[Union[None, str, CustomVectorQuery]],
            result_count: int = 5, offset: int = 0, rerank_depth: Optional[int] = None,
            ef_search: Optional[int] = None, approximate: bool = True,
            approximate_threshold: Optional[float] = None,
            searchable_attributes: Iterable[str] = None, filter_string: str = None, device: str = None,
            attributes_to_retrieve: Optional[List[str]] = None, boost: Optional[Dict] = None,
            media_download_headers: Optional[Dict] = None, context: Optional[SearchContext] = None,
            score_modifiers: Optional[ScoreModifierLists] = None, model_auth: Optional[ModelAuth] = None,
            highlights: bool = False, text_query_prefix: Optional[str] = None,
            hybrid_parameters: HybridParameters = None,
            facets: Optional[FacetsParameters] = None,
            track_total_hits: Optional[bool] = None,
            language: Optional[str] = None,
            relevance_cutoff: Optional[RelevanceCutoffModel] = None,
            sort_by: Optional[SortByModel] = None,
            interpolation_method: Optional[InterpolationMethod] = None,
            collapse: Optional[CollapseModel] = None,
            recency_parameters: Optional[RecencyParameters] = None,
            telemetry_prefix: Optional[str] = None,
    ) -> Dict:
        """
            Args:
                config:
                marqo_index: index object fetched by calling function
                query: either a string query (which can be a URL or natural language text), a dict of
                    <query string>:<weight float> pairs, or None with a context
                result_count:
                offset:
                searchable_attributes: Iterable of field names to search. Should be None for hybrid search, or will
                raise validation error in MarqoHybridQuery
                verbose: if 0 - nothing is printed. if 1 - data is printed without vectors, if 2 - full
                    objects are printed out
                attributes_to_retrieve: if set, only returns these fields
                media_download_headers: headers for downloading media

                context: a dictionary to allow custom vectors in search
                score_modifiers: a dictionary to modify the score based on field values, should be None for hybrid search
                model_auth: Authorisation details for downloading a model (if required)
                highlights: if True, highlights will be returned
                text_query_prefix: prefix for text queries (for vectorisation only)
                hybrid_parameters: HybridParameters object to specify all parameters for hybrid search. If not provided,
                    default values will be used.
                facets: FacetsParameters object to specify facets for the search. If not provided, no facets will be returned.
                track_total_hits: if True, total hits before reranking will be returned. For disjunction, this will be
                the number of tensor OR lexical hits.
                relevance_cutoff: RelevanceCutoffModel object to specify relevance cutoff for the search.
                sort_by: SortByModel object to specify sorting for the search. If not provided, no sorting will be applied.
                interpolation_method: InterpolationMethod object to specify the interpolation method for hybrid search.
                collapse: A CollapseModel object to specify collapsing of search results.
                recency_parameters: parameters for recency boosting
            Returns:

            Output format:
                [
                    {
                        "_id": doc_id
                        "doc": {# original document},
                        "highlights":[{}],
                        "_score": score,
                        "_lexical_score": RRF raw lexical score (if any)
                        "_tensor_score": tensor score (if any)
                    },
                ]
            """
        # Use default hybrid settings if not provided
        if hybrid_parameters is None:
            hybrid_parameters = HybridParameters()

        if telemetry_prefix is None:
            telemetry_prefix = "search.hybrid"

        # # SEARCH TIMER-LOGGER (pre-processing)
        if boost is not None:
            raise api_exceptions.MarqoWebError('Boosting is not currently supported with Vespa')

        RequestMetricsStore.for_request().start(f"{telemetry_prefix}.processing_before_vespa")
        index_name = marqo_index.name

        # Version checks (different for structured and unstructured)
        marqo_index_version = marqo_index.parsed_marqo_version()
        if isinstance(marqo_index, StructuredMarqoIndex) and \
                marqo_index_version < constants.MARQO_STRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION:
            raise core_exceptions.UnsupportedFeatureError(
                f"Hybrid search is only supported for Marqo structured indexes created with Marqo "
                f"{str(constants.MARQO_STRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION)} or later. "
                f"This index was created with Marqo {marqo_index_version}."
            )
        elif isinstance(marqo_index, UnstructuredMarqoIndex) and \
                marqo_index_version < constants.MARQO_UNSTRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION:
            raise core_exceptions.UnsupportedFeatureError(
                f"Hybrid search is only supported for Marqo unstructured indexes created with Marqo "
                f"{str(constants.MARQO_UNSTRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION)} or later. "
                f"This index was created with Marqo {marqo_index_version}."
            )

        if score_modifiers is not None \
                and marqo_index_version < constants.MARQO_GLOBAL_SCORE_MODIFIERS_MINIMUM_VERSION:
            raise core_exceptions.UnsupportedFeatureError(
                f"Hybrid search with global score modifiers is only supported for Marqo indexes created with Marqo "
                f"{str(constants.MARQO_GLOBAL_SCORE_MODIFIERS_MINIMUM_VERSION)} or later. "
                f"This index was created with Marqo {marqo_index_version}."
            )

        # Custom score rerankers (marqo__score_*) require semi-structured index and schema version >= 2.26.0
        if score_modifiers is not None and score_modifiers.uses_custom_score_rerank:
            if not isinstance(marqo_index, SemiStructuredMarqoIndex):
                raise core_exceptions.UnsupportedFeatureError(
                    "Custom score reranking (marqo__score_*) is only supported for semi-structured indexes. "
                    "Structured indexes do not support this feature."
                )
            if not marqo_index.index_supports_custom_score_rerank:
                raise core_exceptions.UnsupportedFeatureError(
                    f"Custom score reranking is only supported for indexes whose Vespa schema version is "
                    f"{str(constants.MARQO_CUSTOM_SCORE_RERANKERS_MINIMUM_VERSION)} or later. "
                    f"This index has schema version {marqo_index.schema_template_version or marqo_index.marqo_version}."
                )

        # TODO: Remove when unstructured searchable attributes are supported
        if (isinstance(marqo_index, UnstructuredMarqoIndex) and
                not isinstance(marqo_index, SemiStructuredMarqoIndex) and
                (hybrid_parameters.searchableAttributesTensor is not None or
                 hybrid_parameters.searchableAttributesLexical is not None)):
            raise core_exceptions.UnsupportedFeatureError(
                f"Hybrid search for unstructured indexes currently does not support `searchableAttributesTensor` or "
                f"`searchableAttributesLexical`. Please set these attributes to None."
            )

        if facets is not None and not isinstance(marqo_index, SemiStructuredMarqoIndex):
            raise core_exceptions.UnsupportedFeatureError(
                f"Facets are only supported for unstructured indexes"
            )
        if track_total_hits is not None and not isinstance(marqo_index, SemiStructuredMarqoIndex):
            raise core_exceptions.UnsupportedFeatureError(
                f"trackTotalHits is only supported for unstructured indexes"
            )

        if query is not None and (
                hybrid_parameters.queryLexical is not None or hybrid_parameters.queryTensor is not None):
            raise ValueError(
                "'q' cannot be provided for HYBRID search when hybridParameters.queryTensor or "
                "'hybridParameters.queryLexical' is provided"
            )

        if sort_by and (
                marqo_index_version < constants.MARQO_SORT_BY_MINIMUM_VERSION or
                not marqo_index.type == IndexType.SemiStructured
        ):
            raise core_exceptions.UnsupportedFeatureError(
                f"The 'sortBy' features is only supported for unstructured indexes created "
                f"with Marqo version {constants.MARQO_SORT_BY_MINIMUM_VERSION} or later "
            )

        if relevance_cutoff and not marqo_index.type == IndexType.SemiStructured:
            # Legacy unstructured indexes and structured indexes do not support relevance cutoff
            raise core_exceptions.UnsupportedFeatureError(
                f"The 'relevanceCutoff' feature is only supported for unstructured indexes created "
                f"with Marqo version {constants.MARQO_SEMI_UNSTRUCTURED_INDEX_VERSION} or later "
            )

        if recency_parameters:
            # Recency scoring is only supported for SemiStructured indexes
            if not isinstance(marqo_index, SemiStructuredMarqoIndex):
                raise core_exceptions.UnsupportedFeatureError(
                    "Recency scoring is only supported for unstructured indexes. "
                    "Structured indexes do not support the recencyParameters option."
                )
            # Check schema version supports recency
            if not marqo_index.index_supports_recency_scoring:
                raise core_exceptions.UnsupportedFeatureError(
                    f"Recency scoring is only supported for unstructured indexes created with Marqo "
                    f"{str(constants.MARQO_RECENCY_SCORING_MINIMUM_VERSION)} or later. "
                    f"This index was created with schema version {marqo_index.schema_template_version or marqo_index.marqo_version}."
                )
            # Check if addToScoreWeight requires newer schema version
            if recency_parameters.add_to_score_weight is not None:
                if not marqo_index.index_supports_recency_additive:
                    raise core_exceptions.UnsupportedFeatureError(
                        f"Additive recency scoring (addToScoreWeight) is only supported for unstructured indexes "
                        f"created with Marqo {str(constants.MARQO_RECENCY_ADDITIVE_MINIMUM_VERSION)} or later. "
                        f"This index was created with schema version {marqo_index.schema_template_version or marqo_index.marqo_version}."
                    )
            # Check if growFrom requires newer schema version
            if recency_parameters.grow_from is not None:
                if not marqo_index.index_supports_recency_grow:
                    raise core_exceptions.UnsupportedFeatureError(
                        f"Recency grow parameters (growFrom) are only supported for unstructured indexes "
                        f"created with Marqo {str(constants.MARQO_RECENCY_GROW_MINIMUM_VERSION)} or later. "
                        f"This index was created with schema version {marqo_index.schema_template_version or marqo_index.marqo_version}."
                    )
            # Check if center requires newer schema version
            if recency_parameters.center is not None:
                if not marqo_index.index_supports_recency_center_and_subqueries:
                    raise core_exceptions.UnsupportedFeatureError(
                        f"Recency center parameter (center) is only supported for unstructured indexes "
                        f"created with Marqo {str(constants.MARQO_RECENCY_CENTER_AND_SUBQUERIES_MINIMUM_VERSION)} or later. "
                        f"This index was created with schema version {marqo_index.schema_template_version or marqo_index.marqo_version}."
                    )
            # Check if applyToSubqueries requires newer schema version
            if recency_parameters.apply_to_subqueries is not None:
                if not marqo_index.index_supports_recency_center_and_subqueries:
                    raise core_exceptions.UnsupportedFeatureError(
                        f"Recency applyToSubqueries parameter (applyToSubqueries) is only supported for unstructured indexes "
                        f"created with Marqo {str(constants.MARQO_RECENCY_CENTER_AND_SUBQUERIES_MINIMUM_VERSION)} or later. "
                        f"This index was created with schema version {marqo_index.schema_template_version or marqo_index.marqo_version}."
                    )

        if hybrid_parameters.secondPhaseModifier:
            if not isinstance(marqo_index, SemiStructuredMarqoIndex):
                raise core_exceptions.UnsupportedFeatureError(
                    f"'secondPhaseModifier' is only supported for unstructured indexes created "
                    f"with Marqo {constants.MARQO_SECOND_PHASE_LEXICAL_SCORE_MODIFIERS_MINIMUM_VERSION} or later "
                )

            if not marqo_index.index_supports_second_phase_lexical_score_modifiers:
                raise core_exceptions.UnsupportedFeatureError(
                    f"'secondPhaseModifier' is supported for unstructured indexes created "
                    f"with Marqo {constants.MARQO_SECOND_PHASE_LEXICAL_SCORE_MODIFIERS_MINIMUM_VERSION} or later. "
                    f"This index was created with schema version {marqo_index.schema_template_version or marqo_index.marqo_version} "
                )

        if collapse and collapse.sort_by:
            if not isinstance(marqo_index, SemiStructuredMarqoIndex):
                raise core_exceptions.UnsupportedFeatureError(
                    f"'collapse.sortBy' is only supported for unstructured indexes created "
                    f"with Marqo {constants.MARQO_COLLAPSE_SORT_BY_MINIMUM_VERSION} or later "
                )

            if not marqo_index.index_supports_collapse_sort_by:
                raise core_exceptions.UnsupportedFeatureError(
                    f"'collapse.sortBy' is supported for unstructured indexes created "
                    f"with Marqo {constants.MARQO_COLLAPSE_SORT_BY_MINIMUM_VERSION} or later. "
                    f"This index was created with schema version {marqo_index.schema_template_version or marqo_index.marqo_version} "
                )

        if hybrid_parameters.lexicalOperand and not isinstance(marqo_index, SemiStructuredMarqoIndex):
            raise core_exceptions.UnsupportedFeatureError(
                f"'lexicalOperand' is only supported for unstructured indexes "
            )

        # Determine the text query prefix
        text_query_prefix = marqo_index.model.get_text_query_prefix(text_query_prefix)
        # split queries into lexical and tensor
        if query is None:
            tensor_query = hybrid_parameters.queryTensor
            lexical_query = hybrid_parameters.queryLexical

            if tensor_query is not None:
                if hybrid_parameters.retrievalMethod == RetrievalMethod.Lexical and hybrid_parameters.rankingMethod == RankingMethod.Lexical:
                    raise core_exceptions.InvalidArgumentError(
                        "'hybridParameters.queryTensor' cannot be provided when 'retrievalMethod' and 'rankingMethod' are both 'lexical'."
                    )
            if lexical_query is not None:
                if hybrid_parameters.retrievalMethod == RetrievalMethod.Tensor and hybrid_parameters.rankingMethod == RankingMethod.Tensor:
                    raise core_exceptions.InvalidArgumentError(
                        "'hybridParameters.queryLexical' cannot be provided when 'retrievalMethod' and 'rankingMethod' are both 'tensor'."
                    )
        elif isinstance(query, CustomVectorQuery):
            tensor_query = query.customVector.vector
            lexical_query = query.customVector.content
        else:
            tensor_query = query
            lexical_query = query

        if lexical_query is None:
            # We could allow queryTensor to be None as tensors might be provided with context
            if hybrid_parameters.retrievalMethod == RetrievalMethod.Disjunction:
                raise core_exceptions.InvalidArgumentError(
                    "Either 'hybridParameters.queryLexical' or just 'q'"
                    "must be present when 'disjunction' retrieval method is used."
                )

        # Edge cases for q data type
        if isinstance(query, CustomVectorQuery):
            query_text_vectorise = None
            query_text_search = lexical_query

            if context is None:
                # If no context, create it with a tensor component
                context = SearchContext(
                    tensor=[SearchContextTensor(vector=tensor_query, weight=1)]
                )
            elif context.tensor is None:
                # If no context.tensor, create it
                context.tensor = [SearchContextTensor(vector=tensor_query, weight=1)]
            else:
                # If context.tensor exists, append the tensor query to it
                context.tensor.append(SearchContextTensor(vector=tensor_query, weight=1))
        elif tensor_query is None and lexical_query is None:
            # This is only acceptable if retrieval_method="tensor", ranking_method="tensor", and context exists.
            # Treated like normal tensor search with context.
            if not (hybrid_parameters.retrievalMethod.upper() == SearchMethod.TENSOR and
                    hybrid_parameters.rankingMethod.upper() == SearchMethod.TENSOR):
                raise core_exceptions.InvalidArgumentError(
                    "Query cannot be 'None' for hybrid search unless: (1) retrievalMethod and rankingMethod "
                    "are both 'tensor' and 'context' is given or (2) One or both of queryLexical and queryTensor "
                    "are provided (depending on retrievalMethod and rankingMethod) instead.")
            if context is None:
                raise core_exceptions.InvalidArgumentError(
                    "Query cannot be 'None' for hybrid search unless 'context' is provided.")
            query_text_vectorise = None
            query_text_search = None

        else:  # string or dict query
            query_text_vectorise = tensor_query
            query_text_search = lexical_query

        queries = [BulkSearchQueryEntity(
            q=query_text_vectorise, searchableAttributes=searchable_attributes, searchMethod=SearchMethod.HYBRID,
            limit=result_count,
            offset=offset, showHighlights=False, filter=filter_string, attributesToRetrieve=attributes_to_retrieve,
            boost=boost, mediaDownloadHeaders=media_download_headers, context=context, scoreModifiers=score_modifiers,
            index=marqo_index, modelAuth=model_auth, text_query_prefix=text_query_prefix,
            hybridParameters=hybrid_parameters
        )]

        if (
                hybrid_parameters.retrievalMethod in [RetrievalMethod.Tensor, RetrievalMethod.Disjunction]
                or
                hybrid_parameters.rankingMethod in [RankingMethod.Tensor, RankingMethod.RRF]
        ):
            with RequestMetricsStore.for_request().time(f"{telemetry_prefix}.vector_inference_full_pipeline"):
                qidx_to_vectors: Dict[Qidx, List[float]] = run_vectorise_pipeline(config, queries, device,
                                                                                  interpolation_method)
            vectorised_text = list(qidx_to_vectors.values())[0]
        else:
            vectorised_text = None

        # Parse text into required and optional terms.
        if query_text_search:
            (required_terms, optional_terms) = utils.parse_lexical_query(query_text_search)
        else:
            required_terms = []
            optional_terms = []

        marqo_query = MarqoHybridQuery(
            index_name=index_name,
            vector_query=vectorised_text,
            filter=filter_string,
            limit=result_count,
            ef_search=ef_search,
            approximate=approximate,
            approximate_threshold=approximate_threshold,
            offset=offset,
            global_rerank_depth=rerank_depth,
            or_phrases=optional_terms,
            and_phrases=required_terms,
            attributes_to_retrieve=attributes_to_retrieve,
            searchable_attributes=searchable_attributes,
            score_modifiers=score_modifiers.to_marqo_score_modifiers() if score_modifiers is not None else None,
            # Hybrid-specific attributes
            score_modifiers_lexical=hybrid_parameters.scoreModifiersLexical.to_marqo_score_modifiers()
            if hybrid_parameters.scoreModifiersLexical is not None else None,
            score_modifiers_tensor=hybrid_parameters.scoreModifiersTensor.to_marqo_score_modifiers()
            if hybrid_parameters.scoreModifiersTensor is not None else None,
            hybrid_parameters=hybrid_parameters,
            facets=facets,
            track_total_hits=track_total_hits,
            language=language,
            relevance_cutoff=relevance_cutoff,
            sort_by=sort_by,
            collapse=collapse,
            recency_parameters=recency_parameters
        )

        vespa_index = vespa_index_factory(marqo_index)
        vespa_query = vespa_index.to_vespa_query(marqo_query)

        total_preprocess_time = RequestMetricsStore.for_request().stop(f"{telemetry_prefix}.processing_before_vespa")
        logger.debug(
            f"search (hybrid) pre-processing: took {(total_preprocess_time):.3f}ms to vectorize and process query.")

        # SEARCH TIMER-LOGGER (roundtrip)
        with RequestMetricsStore.for_request().time(f"{telemetry_prefix}.vespa",
                                                    lambda t: logger.debug(f"Vespa search: took {t:.3f}ms")
                                                    ):
            try:
                responses = config.vespa_client.query(**vespa_query)
            except VespaStatusError as e:
                # The index will not have the embedding_similarity rank profile if there are no tensor fields
                if f"No profile named '{RANK_PROFILE_HYBRID_CUSTOM_SEARCHER}'" in e.message:
                    raise core_exceptions.InvalidArgumentError(
                        f"Index {index_name} either has no tensor fields or no lexically searchable fields, "
                        f"thus hybrid search cannot be performed. "
                        f"Please create an index with both tensor and lexical fields, or try a different search method."
                    )
                raise e

        if not approximate and (responses.root.coverage.coverage < 100 or responses.root.coverage.degraded is not None):
            raise errors.InternalError(
                f'Graceful degradation detected for non-approximate search. '
                f'Coverage is not 100%: {responses.root.coverage}'
                f'Degraded: {str(responses.root.coverage.degraded)}'
            )

        # SEARCH TIMER-LOGGER (post-processing)
        RequestMetricsStore.for_request().start(f"{telemetry_prefix}.postprocess")
        gathered_results = gather_documents_from_response(responses, marqo_index, highlights, attributes_to_retrieve)
        total_results = len(gathered_results["hits"])
        if facets is not None or track_total_hits is not None:
            if isinstance(vespa_index, SemiStructuredVespaIndex):
                gathered_results.update(vespa_index.gather_facets_from_response(responses, facets))
            if facets is not None:
                for facet_field_name, facet_field_parameters in facets.fields.items():
                    # Set empty dict for array facets if not present (we skipped them in request)
                    if facet_field_name not in gathered_results["facets"] and facet_field_parameters.type == "array":
                        gathered_results.get("facets", {}).update({facet_field_name: {}})
            if track_total_hits is not None and "totalHits" not in gathered_results:
                gathered_results["totalHits"] = 0

        gathered_results = self._max_value_check_for_total_hits(gathered_results)
        total_postprocess_time = RequestMetricsStore.for_request().stop(f"{telemetry_prefix}.postprocess")
        logger.debug(
            f"search (hybrid) post-processing: took {(total_postprocess_time):.3f}ms to sort and format "
            f"{total_results} results from Vespa."
        )

        # Collect post-process candidates metadata (always returned by Vespa custom searcher)
        if (responses.root.fields and responses.root.fields.marqo_fields
                and responses.root.fields.marqo_fields.post_process_candidates is not None):
            gathered_results["_postProcessCandidates"] = responses.root.fields.marqo_fields.post_process_candidates

        # Collect metadata for sort by
        if sort_by is not None:
            if responses.root.fields.marqo_fields is None or responses.root.fields.marqo_fields.sort_candidates is None:  # pragma: no cover
                raise core_exceptions.InternalError(
                    f"'sortBy' feature is enabled, but Vespa did not return sortCandidates in the response "
                )

            gathered_results["_sortCandidates"] = responses.root.fields.marqo_fields.sort_candidates

        # Collect metadata for relevance cutoff
        if relevance_cutoff is not None:
            if responses.root.fields.marqo_fields is None \
                    or responses.root.fields.marqo_fields.relevant_candidates is None \
                    or responses.root.fields.marqo_fields.probe_candidates is None:  # pragma: no cover
                raise core_exceptions.InternalError(
                    f"'relevanceCutoff' feature is enabled, but Vespa did not return relevantCandidates or "
                    f"probeCandidates in the response "
                )
            gathered_results["_relevantCandidates"] = responses.root.fields.marqo_fields.relevant_candidates
            gathered_results["_probeCandidates"] = responses.root.fields.marqo_fields.probe_candidates

            if relevance_cutoff.override_total_hits_with_post_process_candidates and \
                responses.root.fields.marqo_fields.post_process_candidates is not None:
                gathered_results["totalHits"] = responses.root.fields.marqo_fields.post_process_candidates

        return gathered_results

    def _max_value_check_for_total_hits(self, gathered_results: Dict) -> Dict:
        """
        Ensure the total hits does not exceed maximum retrievable value.
        """
        if "totalHits" in gathered_results and isinstance(gathered_results["totalHits"], int):
            gathered_results["totalHits"] = (
                min(gathered_results["totalHits"], read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_RETRIEVABLE_DOCS))
            )
        return gathered_results