from typing import Any
from typing import List, Optional, Union, Iterable, Dict

from marqo.base_model import StrictBaseModel
from marqo.config import Config
from marqo.core.exceptions import InternalError
from marqo.core.models import MarqoIndex
from marqo.core.models.facets_parameters import FacetsParameters
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.tensor_search.models.api_models import ScoreModifierLists, CustomVectorQuery
from marqo.tensor_search.models.collapse_model import CollapseModel
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.recency_parameters import RecencyParameters
from marqo.tensor_search.models.relevance_cutoff_model import RelevanceCutoffModel
from marqo.tensor_search.models.search import SearchContext
from marqo.tensor_search.models.sort_by_model import SortByModel
from marqo.tensor_search.telemetry import RequestMetricsStore


class HybridSearchInternalParameters(StrictBaseModel):
    """
    A container for all parameters needed to perform a hybrid search.
    No validate or processing logic should be added here; this is purely for data storage.
    """
    config: Any
    marqo_index: MarqoIndex
    query: Optional[Union[None, str, CustomVectorQuery]]
    result_count: int = 5
    offset: int = 0
    rerank_depth: Optional[int] = None
    ef_search: Optional[int] = None
    approximate: bool = True
    approximate_threshold: Optional[float] = None
    searchable_attributes: Optional[Iterable[str]] = None
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
    """
    Implements collapse sort by functionality by performing two hybrid searches:
    1. A relevance-based collapse search to get the top N collapsed groups.
    2. A sort-based collapse search (e.g., lowest price) to get the sorted variants within those groups.
    Finally, merges the results to by replacing the hits in the relevance results with the sorted variants.

    The could path is only executed if 'collapse.sort_by' is provided.
    """
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
        # Deliberately import here to avoid circular imports
        from marqo.core.search.hybrid_search import HybridSearch

        if not self.internal_params.collapse.sort_by:
            raise InternalError(  # pragma: no cover
                "'collapse.sort_by' must be provided for collapse search."
            )

        with RequestMetricsStore.for_request().time("search.hybrid.collapse_search.relevance_collapse"):
            relevance_collapse_results = HybridSearch().execute_search(
                config=self.internal_params.config,
                marqo_index=self.internal_params.marqo_index,
                query=self.internal_params.query,
                result_count=self.internal_params.result_count,
                offset=self.internal_params.offset,
                rerank_depth=self.internal_params.rerank_depth,
                ef_search=self.internal_params.ef_search,
                approximate=self.internal_params.approximate,
                approximate_threshold=self.internal_params.approximate_threshold,
                searchable_attributes=self.internal_params.searchable_attributes,
                filter_string=self.internal_params.filter_string,
                device=self.internal_params.device,
                attributes_to_retrieve=self.internal_params.attributes_to_retrieve,
                boost=self.internal_params.boost,
                media_download_headers=self.internal_params.media_download_headers,
                context=self.internal_params.context,
                score_modifiers=self.internal_params.score_modifiers,
                model_auth=self.internal_params.model_auth,
                highlights=self.internal_params.highlights,
                text_query_prefix=self.internal_params.text_query_prefix,
                hybrid_parameters=self.internal_params.hybrid_parameters,
                facets=self.internal_params.facets,
                track_total_hits=self.internal_params.track_total_hits,
                language=self.internal_params.language,
                relevance_cutoff=self.internal_params.relevance_cutoff,
                sort_by=self.internal_params.sort_by,
                interpolation_method=self.internal_params.interpolation_method,
                collapse=self.internal_params.collapse,
                recency_parameters=self.internal_params.recency_parameters,
                telemetry_prefix="search.hybrid.collapse_search.relevance_collapse"
            )

        with RequestMetricsStore.for_request().time("search.hybrid.collapse_search.collect_documents_ids"):
            collected_document_ids = self.collect_documents_ids(relevance_collapse_results)

        if not collected_document_ids:
            return relevance_collapse_results

        with (RequestMetricsStore.for_request().time("search.hybrid.collapse_search.generate_collapse_sort_by_query")):
            collapse_sorty_query: HybridSearchInternalParameters = \
            self.generate_collapse_sort_by_query(collected_document_ids)

        with RequestMetricsStore.for_request().time("search.hybrid.collapse_search.sorted_collapse"):
            sorted_collapse_results = HybridSearch().execute_search(
                config=self.internal_params.config,
                marqo_index=collapse_sorty_query.marqo_index,
                query=collapse_sorty_query.query,
                result_count=collapse_sorty_query.result_count,
                offset=collapse_sorty_query.offset,
                rerank_depth=collapse_sorty_query.rerank_depth,
                ef_search=collapse_sorty_query.ef_search,
                approximate=collapse_sorty_query.approximate,
                approximate_threshold=collapse_sorty_query.approximate_threshold,
                searchable_attributes=collapse_sorty_query.searchable_attributes,
                filter_string=collapse_sorty_query.filter_string,
                device=collapse_sorty_query.device,
                attributes_to_retrieve=collapse_sorty_query.attributes_to_retrieve,
                boost=collapse_sorty_query.boost,
                media_download_headers=collapse_sorty_query.media_download_headers,
                context=collapse_sorty_query.context,
                score_modifiers=collapse_sorty_query.score_modifiers,
                model_auth=collapse_sorty_query.model_auth,
                highlights=collapse_sorty_query.highlights,
                text_query_prefix=collapse_sorty_query.text_query_prefix,
                hybrid_parameters=collapse_sorty_query.hybrid_parameters,
                facets=collapse_sorty_query.facets,
                track_total_hits=collapse_sorty_query.track_total_hits,
                language=collapse_sorty_query.language,
                relevance_cutoff=collapse_sorty_query.relevance_cutoff,
                sort_by=collapse_sorty_query.sort_by,
                interpolation_method=collapse_sorty_query.interpolation_method,
                collapse=collapse_sorty_query.collapse,
                recency_parameters=collapse_sorty_query.recency_parameters,
                telemetry_prefix="search.hybrid.collapse_search.sorted_collapse"
            )

        with RequestMetricsStore.for_request().time("search.hybrid.collapse_search.merge_results"):
            merged_results = self.merge_two_collapse_results(
                relevance_collapse_results, sorted_collapse_results, collected_document_ids
            )

        return merged_results

    def collect_documents_ids(self, search_results: Dict) -> List[str]:
        document_ids = []
        for hit in search_results.get("hits", []):
            if hit.get(self.internal_params.collapse.sort_by[0].field_name) is not None:
                document_ids.append(hit.get(self.internal_params.collapse.name))
        return document_ids

    def generate_collapse_sort_by_query(self, parent_ids: List[str]) -> HybridSearchInternalParameters:
        """
        Generate a new HybridSearchInternalParameters object for the collapse sort_by search.
        """
        collapse_sort_by_hybrid_parameters = HybridSearchInternalParameters(
            config=self.internal_params.config,
            marqo_index=self.internal_params.marqo_index,
            query="*",
            result_count=9999,
            offset=0,
            rerank_depth=None,
            ef_search=None,
            approximate=True,
            approximate_threshold=None,
            searchable_attributes=self.internal_params.searchable_attributes,
            filter_string=self.internal_params.filter_string,
            device=self.internal_params.device,
            attributes_to_retrieve=self.internal_params.attributes_to_retrieve,
            boost=None,
            media_download_headers=None,
            context=None,
            score_modifiers=None,
            model_auth=None,
            highlights=False,
            text_query_prefix=None,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical,
                searchableAttributesLexical=self.internal_params.hybrid_parameters.searchableAttributesLexical,
                searchableAttributesTensor=None
            ),
            facets=None,
            track_total_hits=False,
            language=self.internal_params.language,
            relevance_cutoff=None,
            sort_by=None,
            interpolation_method=None,
            collapse=self.internal_params.collapse,
            recency_parameters=None
        )

        collapse_sort_by_hybrid_parameters.collapse.enable_execute_sort()
        collapse_filter_string = (
                f'{collapse_sort_by_hybrid_parameters.collapse.name} in ('
                + ', '.join(f'"{doc_id}"' for doc_id in parent_ids)
                + ')'
        )
        collapse_sort_by_hybrid_parameters.collapse.collapse_filter_string = collapse_filter_string
        return collapse_sort_by_hybrid_parameters

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
