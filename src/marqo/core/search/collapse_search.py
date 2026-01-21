from typing import List, Optional, Union, Iterable, Dict

from marqo.api import exceptions as api_exceptions
from marqo.api import exceptions as errors
# We depend on _httprequests.py for now, but this may be replaced in the future, as
# _httprequests.py is designed for the client
from marqo.config import Config
from marqo.core import constants
from marqo.core import exceptions as core_exceptions
from marqo.core.models.facets_parameters import FacetsParameters
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.core.models.marqo_index import UnstructuredMarqoIndex, StructuredMarqoIndex, SemiStructuredMarqoIndex, \
    IndexType
from marqo.core.models.marqo_query import MarqoHybridQuery
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.core.vespa_index.vespa_index import for_marqo_index as vespa_index_factory
from marqo.core.structured_vespa_index.common import RANK_PROFILE_HYBRID_CUSTOM_SEARCHER
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.tensor_search import utils
from marqo.tensor_search.enums import (
    SearchMethod
)
from marqo.core.models import MarqoIndex
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity, ScoreModifierLists, CustomVectorQuery
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.recency_parameters import RecencyParameters
from marqo.tensor_search.models.search import Qidx, SearchContext, SearchContextTensor
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.tensor_search import run_vectorise_pipeline, gather_documents_from_response, logger
from marqo.vespa.exceptions import VespaStatusError
from marqo.tensor_search.models.sort_by_model import SortByModel
from marqo.tensor_search.models.relevance_cutoff_model import RelevanceCutoffModel
from marqo.tensor_search.models.collapse_model import CollapseModel
from marqo.base_model import StrictBaseModel
from typing import Any


class HybridSearchInternalParameters(StrictBaseModel):
    config: Any
    marqo_index: MarqoIndex
    query: Optional[Union[None, str, CustomVectorQuery]]
    result_count: int = 5
    offset: int = 0
    rerank_depth: Optional[int] = None
    ef_search: Optional[int] = None
    approximate: bool = True
    approximate_threshold: Optional[float] = None
    searchable_attributes: Iterable[str] = None
    filter_string: Optional[str] = None,
    device: str = None
    attributes_to_retrieve: Optional[List[str]] = None
    boost: Optional[Dict] = None
    media_download_headers: Optional[Dict] = None
    context: Optional[SearchContext] = None
    score_modifiers: Optional[ScoreModifierLists] = None
    model_auth: Optional[ModelAuth] = None
    highlights: bool = False
    text_query_prefix: Optional[str] = None
    hybrid_parameters: HybridParameters = None
    facets: Optional[FacetsParameters] = None
    track_total_hits: Optional[bool] = None
    language: Optional[str] = None
    relevance_cutoff: Optional[RelevanceCutoffModel] = None
    sort_by: Optional[SortByModel] = None
    interpolation_method: Optional[InterpolationMethod] = None
    collapse: Optional[CollapseModel] = None
    recency_parameters: Optional[RecencyParameters] = None


class CollapseSearch:
    def __init__(
            self,
            config: Config,
            marqo_index: MarqoIndex, query: Optional[Union[None, str, CustomVectorQuery]],
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
        self.internal_params = HybridSearchInternalParameters(
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

    def search(self):
        with RequestMetricsStore.for_request().time("collapse_relevance_sort.relevance_collapse"):
            relevance_collapse_results = self.search_with_relevance_collapse()

        with RequestMetricsStore.for_request().time("collapse_relevance_sort.collect_documents_ids"):
            collected_document_ids = self.collect_documents_ids(relevance_collapse_results)

        with RequestMetricsStore.for_request().time("collapse_relevance_sort.sorted_collapse"):
            sorted_collapse_results = self.search_with_sorted_collapse(collected_document_ids)

        with RequestMetricsStore.for_request().time("collapse_relevance_sort.merge_results"):
            merged_results = self.merge_two_collapse_results(
                relevance_collapse_results, sorted_collapse_results, collected_document_ids
            )

        return merged_results

    def search_with_relevance_collapse(self):
        # Determine the text query prefix
        text_query_prefix = self.internal_params.marqo_index.model.get_text_query_prefix(
            self.internal_params.text_query_prefix)
        # split queries into lexical and tensor
        if self.internal_params.query is None:
            tensor_query = self.internal_params.hybrid_parameters.queryTensor
            lexical_query = self.internal_params.hybrid_parameters.queryLexical

            if tensor_query is not None:
                if self.internal_params.hybrid_parameters.retrievalMethod == RetrievalMethod.Lexical and self.internal_params.hybrid_parameters.rankingMethod == RankingMethod.Lexical:
                    raise core_exceptions.InvalidArgumentError(
                        "'hybridParameters.queryTensor' cannot be provided when 'retrievalMethod' and 'rankingMethod' are both 'lexical'."
                    )
            if lexical_query is not None:
                if self.internal_params.hybrid_parameters.retrievalMethod == RetrievalMethod.Tensor and self.internal_params.hybrid_parameters.rankingMethod == RankingMethod.Tensor:
                    raise core_exceptions.InvalidArgumentError(
                        "'hybridParameters.queryLexical' cannot be provided when 'retrievalMethod' and 'rankingMethod' are both 'tensor'."
                    )
        elif isinstance(self.internal_params.query, CustomVectorQuery):
            tensor_query = self.internal_params.query.customVector.vector
            lexical_query = self.internal_params.query.customVector.content
        else:
            tensor_query = self.internal_params.query
            lexical_query = self.internal_params.query

        if lexical_query is None:
            # We could allow queryTensor to be None as tensors might be provided with context
            if self.internal_params.hybrid_parameters.retrievalMethod == RetrievalMethod.Disjunction:
                raise core_exceptions.InvalidArgumentError(
                    "Either 'hybridParameters.queryLexical' or just 'q'"
                    "must be present when 'disjunction' retrieval method is used."
                )

        # Edge cases for q data type
        if isinstance(self.internal_params.query, CustomVectorQuery):
            query_text_vectorise = None
            query_text_search = lexical_query

            if self.internal_params.context is None:
                # If no context, create it with a tensor component
                context = SearchContext(
                    tensor=[SearchContextTensor(vector=tensor_query, weight=1)]
                )
            elif self.internal_params.context.tensor is None:
                # If no context.tensor, create it
                self.internal_params.context.tensor = [SearchContextTensor(vector=tensor_query, weight=1)]
            else:
                # If context.tensor exists, append the tensor query to it
                self.internal_params.context.tensor.append(SearchContextTensor(vector=tensor_query, weight=1))
        elif tensor_query is None and lexical_query is None:
            # This is only acceptable if retrieval_method="tensor", ranking_method="tensor", and context exists.
            # Treated like normal tensor search with context.
            if not (self.internal_params.hybrid_parameters.retrievalMethod.upper() == SearchMethod.TENSOR and
                    self.internal_params.hybrid_parameters.rankingMethod.upper() == SearchMethod.TENSOR):
                raise core_exceptions.InvalidArgumentError(
                    "Query cannot be 'None' for hybrid search unless: (1) retrievalMethod and rankingMethod "
                    "are both 'tensor' and 'context' is given or (2) One or both of queryLexical and queryTensor "
                    "are provided (depending on retrievalMethod and rankingMethod) instead."
                )
            if self.internal_params.context is None:
                raise core_exceptions.InvalidArgumentError(
                    "Query cannot be 'None' for hybrid search unless 'context' is provided."
                )
            query_text_vectorise = None
            query_text_search = None

        else:  # string or dict query
            query_text_vectorise = tensor_query
            query_text_search = lexical_query

        queries = [BulkSearchQueryEntity(
            q=query_text_vectorise,
            searchableAttributes=self.internal_params.searchable_attributes,
            searchMethod=SearchMethod.HYBRID,
            limit=self.internal_params.result_count,
            offset=self.internal_params.offset, showHighlights=False,
            filter=self.internal_params.filter_string,
            attributesToRetrieve=self.internal_params.attributes_to_retrieve,
            boost=self.internal_params.boost, mediaDownloadHeaders=self.internal_params.media_download_headers,
            context=self.internal_params.context,
            scoreModifiers=self.internal_params.score_modifiers,
            index=self.internal_params.marqo_index, modelAuth=self.internal_params.model_auth,
            text_query_prefix=text_query_prefix,
            hybridParameters=self.internal_params.hybrid_parameters
        )]

        if (
                self.internal_params.hybrid_parameters.retrievalMethod in [RetrievalMethod.Tensor,
                                                                           RetrievalMethod.Disjunction]
                or
                self.internal_params.hybrid_parameters.rankingMethod in [RankingMethod.Tensor, RankingMethod.RRF]
        ):
            with RequestMetricsStore.for_request().time(f"search.hybrid.vector_inference_full_pipeline"):
                qidx_to_vectors: Dict[Qidx, List[float]] = run_vectorise_pipeline(
                    self.internal_params.config, queries, self.internal_params.device,
                    self.internal_params.interpolation_method)
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
            index_name=self.internal_params.marqo_index.name,
            vector_query=vectorised_text,
            filter=self.internal_params.filter_string,
            limit=self.internal_params.result_count,
            ef_search=self.internal_params.ef_search,
            approximate=self.internal_params.approximate,
            approximate_threshold=self.internal_params.approximate_threshold,
            offset=self.internal_params.offset,
            global_rerank_depth=self.internal_params.rerank_depth,
            or_phrases=optional_terms,
            and_phrases=required_terms,
            attributes_to_retrieve=self.internal_params.attributes_to_retrieve,
            searchable_attributes=self.internal_params.searchable_attributes,
            score_modifiers=self.internal_params.score_modifiers.to_marqo_score_modifiers() if self.internal_params.score_modifiers is not None else None,
            # Hybrid-specific attributes
            score_modifiers_lexical=self.internal_params.hybrid_parameters.scoreModifiersLexical.to_marqo_score_modifiers()
            if self.internal_params.hybrid_parameters.scoreModifiersLexical is not None else None,
            score_modifiers_tensor=self.internal_params.hybrid_parameters.scoreModifiersTensor.to_marqo_score_modifiers()
            if self.internal_params.hybrid_parameters.scoreModifiersTensor is not None else None,
            hybrid_parameters=self.internal_params.hybrid_parameters,
            facets=self.internal_params.facets,
            track_total_hits=self.internal_params.track_total_hits,
            language=self.internal_params.language,
            relevance_cutoff=self.internal_params.relevance_cutoff,
            sort_by=self.internal_params.sort_by,
            collapse=self.internal_params.collapse,
            recency_parameters=self.internal_params.recency_parameters
        )

        vespa_index = vespa_index_factory(self.internal_params.marqo_index)
        vespa_query = vespa_index.to_vespa_query(marqo_query)

        total_preprocess_time = RequestMetricsStore.for_request().stop("search.hybrid.processing_before_vespa")
        logger.debug(
            f"search (hybrid) pre-processing: took {(total_preprocess_time):.3f}ms to vectorize and process query.")

        # SEARCH TIMER-LOGGER (roundtrip)
        with RequestMetricsStore.for_request().time("search.hybrid.vespa",
                                                    lambda t: logger.debug(f"Vespa search: took {t:.3f}ms")
                                                    ):
            try:
                responses = self.internal_params.config.vespa_client.query(**vespa_query)
            except VespaStatusError as e:
                # The index will not have the embedding_similarity rank profile if there are no tensor fields
                if f"No profile named '{RANK_PROFILE_HYBRID_CUSTOM_SEARCHER}'" in e.message:
                    raise core_exceptions.InvalidArgumentError(
                        f"Index {self.internal_params.marqo_index.name} either has no tensor fields or no lexically searchable fields, "
                        f"thus hybrid search cannot be performed. "
                        f"Please create an index with both tensor and lexical fields, or try a different search method."
                    )
                raise e

        if not self.internal_params.approximate and (
                responses.root.coverage.coverage < 100 or responses.root.coverage.degraded is not None):
            raise errors.InternalError(
                f'Graceful degradation detected for non-approximate search. '
                f'Coverage is not 100%: {responses.root.coverage}'
                f'Degraded: {str(responses.root.coverage.degraded)}'
            )

        # SEARCH TIMER-LOGGER (post-processing)
        RequestMetricsStore.for_request().start("search.hybrid.postprocess")
        gathered_results = gather_documents_from_response(responses, self.internal_params.marqo_index,
                                                          self.internal_params.highlights,
                                                          self.internal_params.attributes_to_retrieve)
        total_results = len(gathered_results["hits"])
        if self.internal_params.facets is not None or self.internal_params.track_total_hits is not None:
            if isinstance(vespa_index, SemiStructuredVespaIndex):
                gathered_results.update(vespa_index.gather_facets_from_response(responses, self.internal_params.facets))
            if self.internal_params.facets is not None:
                for facet_field_name, facet_field_parameters in self.internal_params.facets.fields.items():
                    # Set empty dict for array facets if not present (we skipped them in request)
                    if facet_field_name not in gathered_results["facets"] and facet_field_parameters.type == "array":
                        gathered_results.get("facets", {}).update({facet_field_name: {}})
            if self.internal_params.track_total_hits is not None and "totalHits" not in gathered_results:
                gathered_results["totalHits"] = 0

        total_postprocess_time = RequestMetricsStore.for_request().stop("search.hybrid.postprocess")
        logger.debug(
            f"search (hybrid) post-processing: took {(total_postprocess_time):.3f}ms to sort and format "
            f"{total_results} results from Vespa."
        )

        # Collect metadata for sort by
        if self.internal_params.sort_by is not None:
            if responses.root.fields.marqo_fields is None or responses.root.fields.marqo_fields.sort_candidates is None:  # pragma: no cover
                raise core_exceptions.InternalError(
                    f"'sortBy' feature is enabled, but Vespa did not return sortCandidates in the response "
                )
            gathered_results["_sortCandidates"] = responses.root.fields.marqo_fields.sort_candidates

        # Collect metadata for relevance cutoff
        if self.internal_params.relevance_cutoff is not None:
            if responses.root.fields.marqo_fields is None \
                    or responses.root.fields.marqo_fields.relevant_candidates is None \
                    or responses.root.fields.marqo_fields.probe_candidates is None:  # pragma: no cover
                raise core_exceptions.InternalError(
                    f"'relevanceCutoff' feature is enabled, but Vespa did not return relevantCandidates or "
                    f"probeCandidates in the response "
                )
            gathered_results["_relevantCandidates"] = responses.root.fields.marqo_fields.relevant_candidates
            gathered_results["_probeCandidates"] = responses.root.fields.marqo_fields.probe_candidates

        return gathered_results

    def collect_documents_ids(self, search_results: Dict) -> List[str]:
        document_ids = []
        for hit in search_results.get("hits", []):
            if hit.get(self.internal_params.collapse.sort_by[0].field_name) is not None:
                document_ids.append(hit.get(self.internal_params.collapse.name))
        return document_ids

    def search_with_sorted_collapse(self, document_ids: List[str]):
        copied_search_params = self.internal_params.copy(deep=False)
        document_ids_filter_string = " OR ".join(
            [f"{self.internal_params.collapse.name}:{doc_id}" for doc_id in document_ids])
        if document_ids_filter_string and copied_search_params.filter_string:
            copied_search_params.filter_string = f"{copied_search_params.filter_string} AND ({document_ids_filter_string})"
        copied_search_params.collapse.enable_execute_sort()
        copied_search_params.hybrid_parameters.update(
            {
                "rankingMethod": RankingMethod.Lexical,
                "retrievalMethod": RetrievalMethod.Lexical,
                "rrfK": None,
                "alpha": None,
                "scoreModifiersLexical": None,
                "scoreModifiersTensor": None,
            }
        )
        copied_search_params.score_modifiers=None

        if copied_search_params.query is None:
            copied_search_params.query = "*"

        # Determine the text query prefix
        text_query_prefix = copied_search_params.marqo_index.model.get_text_query_prefix(
            copied_search_params.text_query_prefix)
        # split queries into lexical and tensor
        if copied_search_params.query is None:
            tensor_query = copied_search_params.hybrid_parameters.queryTensor
            lexical_query = copied_search_params.hybrid_parameters.queryLexical

            if tensor_query is not None:
                if copied_search_params.hybrid_parameters.retrievalMethod == RetrievalMethod.Lexical and copied_search_params.hybrid_parameters.rankingMethod == RankingMethod.Lexical:
                    raise core_exceptions.InvalidArgumentError(
                        "'hybridParameters.queryTensor' cannot be provided when 'retrievalMethod' and 'rankingMethod' are both 'lexical'."
                    )
            if lexical_query is not None:
                if copied_search_params.hybrid_parameters.retrievalMethod == RetrievalMethod.Tensor and copied_search_params.hybrid_parameters.rankingMethod == RankingMethod.Tensor:
                    raise core_exceptions.InvalidArgumentError(
                        "'hybridParameters.queryLexical' cannot be provided when 'retrievalMethod' and 'rankingMethod' are both 'tensor'."
                    )
        elif isinstance(copied_search_params.query, CustomVectorQuery):
            tensor_query = copied_search_params.query.customVector.vector
            lexical_query = copied_search_params.query.customVector.content
        else:
            tensor_query = copied_search_params.query
            lexical_query = copied_search_params.query

        if lexical_query is None:
            # We could allow queryTensor to be None as tensors might be provided with context
            if copied_search_params.hybrid_parameters.retrievalMethod == RetrievalMethod.Disjunction:
                raise core_exceptions.InvalidArgumentError(
                    "Either 'hybridParameters.queryLexical' or just 'q'"
                    "must be present when 'disjunction' retrieval method is used."
                )

        # Edge cases for q data type
        if isinstance(copied_search_params.query, CustomVectorQuery):
            query_text_vectorise = None
            query_text_search = lexical_query

            if copied_search_params.context is None:
                # If no context, create it with a tensor component
                context = SearchContext(
                    tensor=[SearchContextTensor(vector=tensor_query, weight=1)]
                )
            elif copied_search_params.context.tensor is None:
                # If no context.tensor, create it
                copied_search_params.context.tensor = [SearchContextTensor(vector=tensor_query, weight=1)]
            else:
                # If context.tensor exists, append the tensor query to it
                copied_search_params.context.tensor.append(SearchContextTensor(vector=tensor_query, weight=1))
        elif tensor_query is None and lexical_query is None:
            # This is only acceptable if retrieval_method="tensor", ranking_method="tensor", and context exists.
            # Treated like normal tensor search with context.
            if not (copied_search_params.hybrid_parameters.retrievalMethod.upper() == SearchMethod.TENSOR and
                    copied_search_params.hybrid_parameters.rankingMethod.upper() == SearchMethod.TENSOR):
                raise core_exceptions.InvalidArgumentError(
                    "Query cannot be 'None' for hybrid search unless: (1) retrievalMethod and rankingMethod "
                    "are both 'tensor' and 'context' is given or (2) One or both of queryLexical and queryTensor "
                    "are provided (depending on retrievalMethod and rankingMethod) instead.")
            if copied_search_params.context is None:
                raise core_exceptions.InvalidArgumentError(
                    "Query cannot be 'None' for hybrid search unless 'context' is provided.")
            query_text_vectorise = None
            query_text_search = None

        else:  # string or dict query
            query_text_vectorise = tensor_query
            query_text_search = lexical_query

        queries = [BulkSearchQueryEntity(
            q=query_text_vectorise,
            searchableAttributes=copied_search_params.searchable_attributes,
            searchMethod=SearchMethod.HYBRID,
            limit=copied_search_params.result_count,
            offset=copied_search_params.offset, showHighlights=False,
            filter=copied_search_params.filter_string,
            attributesToRetrieve=copied_search_params.attributes_to_retrieve,
            boost=copied_search_params.boost, mediaDownloadHeaders=copied_search_params.media_download_headers,
            context=copied_search_params.context,
            scoreModifiers=copied_search_params.score_modifiers,
            index=copied_search_params.marqo_index, modelAuth=copied_search_params.model_auth,
            text_query_prefix=text_query_prefix,
            hybridParameters=copied_search_params.hybrid_parameters
        )]

        if (
                copied_search_params.hybrid_parameters.retrievalMethod in [RetrievalMethod.Tensor,
                                                                           RetrievalMethod.Disjunction]
                or
                copied_search_params.hybrid_parameters.rankingMethod in [RankingMethod.Tensor, RankingMethod.RRF]
        ):
            with RequestMetricsStore.for_request().time(f"search.hybrid.vector_inference_full_pipeline"):
                qidx_to_vectors: Dict[Qidx, List[float]] = run_vectorise_pipeline(
                    copied_search_params.config, queries, copied_search_params.device,
                    copied_search_params.interpolation_method)
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
            index_name=copied_search_params.marqo_index.name,
            vector_query=vectorised_text,
            filter=copied_search_params.filter_string,
            limit=copied_search_params.result_count,
            ef_search=copied_search_params.ef_search,
            approximate=copied_search_params.approximate,
            approximate_threshold=copied_search_params.approximate_threshold,
            offset=copied_search_params.offset,
            global_rerank_depth=copied_search_params.rerank_depth,
            or_phrases=optional_terms,
            and_phrases=required_terms,
            attributes_to_retrieve=copied_search_params.attributes_to_retrieve,
            searchable_attributes=copied_search_params.searchable_attributes,
            score_modifiers=copied_search_params.score_modifiers.to_marqo_score_modifiers() if copied_search_params.score_modifiers is not None else None,
            # Hybrid-specific attributes
            score_modifiers_lexical=copied_search_params.hybrid_parameters.scoreModifiersLexical.to_marqo_score_modifiers()
            if copied_search_params.hybrid_parameters.scoreModifiersLexical is not None else None,
            score_modifiers_tensor=copied_search_params.hybrid_parameters.scoreModifiersTensor.to_marqo_score_modifiers()
            if copied_search_params.hybrid_parameters.scoreModifiersTensor is not None else None,
            hybrid_parameters=copied_search_params.hybrid_parameters,
            facets=copied_search_params.facets,
            track_total_hits=copied_search_params.track_total_hits,
            language=copied_search_params.language,
            relevance_cutoff=copied_search_params.relevance_cutoff,
            sort_by=copied_search_params.sort_by,
            collapse=copied_search_params.collapse,
            recency_parameters=copied_search_params.recency_parameters
        )

        vespa_index = vespa_index_factory(copied_search_params.marqo_index)
        vespa_query = vespa_index.to_vespa_query(marqo_query)

        # SEARCH TIMER-LOGGER (roundtrip)
        with RequestMetricsStore.for_request().time("search.hybrid.vespa",
                                                    lambda t: logger.debug(f"Vespa search: took {t:.3f}ms")
                                                    ):
            try:
                responses = copied_search_params.config.vespa_client.query(**vespa_query)
            except VespaStatusError as e:
                # The index will not have the embedding_similarity rank profile if there are no tensor fields
                if f"No profile named '{RANK_PROFILE_HYBRID_CUSTOM_SEARCHER}'" in e.message:
                    raise core_exceptions.InvalidArgumentError(
                        f"Index {copied_search_params.marqo_index.name} either has no tensor fields or no lexically searchable fields, "
                        f"thus hybrid search cannot be performed. "
                        f"Please create an index with both tensor and lexical fields, or try a different search method."
                    )
                raise e

        if not copied_search_params.approximate and (
                responses.root.coverage.coverage < 100 or responses.root.coverage.degraded is not None):
            raise errors.InternalError(
                f'Graceful degradation detected for non-approximate search. '
                f'Coverage is not 100%: {responses.root.coverage}'
                f'Degraded: {str(responses.root.coverage.degraded)}'
            )

        # SEARCH TIMER-LOGGER (post-processing)
        RequestMetricsStore.for_request().start("search.hybrid.postprocess")
        gathered_results = gather_documents_from_response(responses, copied_search_params.marqo_index,
                                                          copied_search_params.highlights,
                                                          copied_search_params.attributes_to_retrieve)
        total_results = len(gathered_results["hits"])
        if copied_search_params.facets is not None or copied_search_params.track_total_hits is not None:
            if isinstance(vespa_index, SemiStructuredVespaIndex):
                gathered_results.update(vespa_index.gather_facets_from_response(responses, copied_search_params.facets))
            if copied_search_params.facets is not None:
                for facet_field_name, facet_field_parameters in copied_search_params.facets.fields.items():
                    # Set empty dict for array facets if not present (we skipped them in request)
                    if facet_field_name not in gathered_results["facets"] and facet_field_parameters.type == "array":
                        gathered_results.get("facets", {}).update({facet_field_name: {}})
            if copied_search_params.track_total_hits is not None and "totalHits" not in gathered_results:
                gathered_results["totalHits"] = 0

        total_postprocess_time = RequestMetricsStore.for_request().stop("search.hybrid.postprocess")
        logger.debug(
            f"search (hybrid) post-processing: took {(total_postprocess_time):.3f}ms to sort and format "
            f"{total_results} results from Vespa."
        )

        # Collect metadata for sort by
        if copied_search_params.sort_by is not None:
            if responses.root.fields.marqo_fields is None or responses.root.fields.marqo_fields.sort_candidates is None:  # pragma: no cover
                raise core_exceptions.InternalError(
                    f"'sortBy' feature is enabled, but Vespa did not return sortCandidates in the response "
                )
            gathered_results["_sortCandidates"] = responses.root.fields.marqo_fields.sort_candidates

        # Collect metadata for relevance cutoff
        if copied_search_params.relevance_cutoff is not None:
            if responses.root.fields.marqo_fields is None \
                    or responses.root.fields.marqo_fields.relevant_candidates is None \
                    or responses.root.fields.marqo_fields.probe_candidates is None:  # pragma: no cover
                raise core_exceptions.InternalError(
                    f"'relevanceCutoff' feature is enabled, but Vespa did not return relevantCandidates or "
                    f"probeCandidates in the response "
                )
            gathered_results["_relevantCandidates"] = responses.root.fields.marqo_fields.relevant_candidates
            gathered_results["_probeCandidates"] = responses.root.fields.marqo_fields.probe_candidates

        return gathered_results

    def merge_two_collapse_results(self, relevance_collapse_results, sorted_collapse_results, parent_ids: List[str]):
        """
        Merge two collapse results by keeping the structure from relevance_collapse_results
        but replacing hits with lower-priced variants from sorted_collapse_results.

        Args:
            relevance_collapse_results: Results from relevance-based collapse search
            sorted_collapse_results: Results from sort-based collapse search (e.g., lowest price)
            parent_ids: List of collapse field values that were used in the sorted search

        Returns:
            Merged results with structure from relevance_collapse_results but variants from sorted_collapse_results
        """

        def merge_hit(sorted_hit, relevance_hit):
            dict = {}
            for key, value in relevance_hit.items():
                if key.startswith("_") and key not in ["_id", "_highlights"]:
                    dict[key] = value
                else:
                    dict[key] = sorted_hit.get(key, value)
                dict["_highlights"] = [{}]
            return dict

        collapse_field_name = self.internal_params.collapse.name

        # Build a lookup map from collapse field value to hit from sorted results
        sorted_hits_by_parent = {}
        for hit in sorted_collapse_results.get("hits", []):
            parent_id = hit.get(collapse_field_name)
            if parent_id is not None:
                sorted_hits_by_parent[parent_id] = hit

        # Replace hits in relevance results with sorted variants where available
        merged_hits = []
        for hit in relevance_collapse_results.get("hits", []):
            parent_id = hit.get(collapse_field_name)
            if parent_id in sorted_hits_by_parent:
                # Replace with the sorted variant (e.g., lower price)
                merged_hits.append(merge_hit(sorted_hits_by_parent[parent_id], hit))
            else:
                # Keep the original hit (either no sort field or not in sorted results)
                merged_hits.append(hit)

        # Create merged result preserving the structure from relevance_collapse_results
        merged_results = relevance_collapse_results.copy()
        merged_results["hits"] = merged_hits

        return merged_results
