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
from marqo.tensor_search.models.sort_by_model import SortByModel, SortOrder
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.core.vespa_index.vespa_index import VespaIndex
from copy import deepcopy


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
    filter_string: Optional[str] = None
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
    Finally, merges the results by replacing the hits in the relevance results with the sorted variants.

    The code path is only executed if 'collapse.sort_by' is provided.
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
        modified_attributes_to_retrieve = deepcopy(attributes_to_retrieve)

        if modified_attributes_to_retrieve is not None:
            if collapse and collapse.name not in modified_attributes_to_retrieve:
                modified_attributes_to_retrieve.append(collapse.name)
            if collapse and collapse.sort_by and collapse.sort_by.fields[0].field_name not in modified_attributes_to_retrieve:
                modified_attributes_to_retrieve.append(collapse.sort_by.fields[0].field_name)

        self.original_attributes_to_retrieve = attributes_to_retrieve
        self.modified_attributes_to_retrieve = modified_attributes_to_retrieve

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
            attributes_to_retrieve=self.modified_attributes_to_retrieve,
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
                # Using **dict(...) as this is shallow (first layer only, nested models stay as Pydantic objects)
                **dict(self.internal_params),
                telemetry_prefix="search.hybrid.collapse_search.relevance_collapse"
            )

        with RequestMetricsStore.for_request().time("search.hybrid.collapse_search.collect_parent_ids"):
            collected_parent_ids = self.collect_parent_ids(relevance_collapse_results)

        if not collected_parent_ids:
            return relevance_collapse_results

        with (RequestMetricsStore.for_request().time("search.hybrid.collapse_search.generate_collapse_sort_by_query")):
            collapse_sort_query: HybridSearchInternalParameters = \
            self.generate_collapse_sort_by_query(collected_parent_ids)

        with RequestMetricsStore.for_request().time("search.hybrid.collapse_search.sorted_collapse"):
            sorted_collapse_results = HybridSearch().execute_search(
                # Using **dict(...) as this is shallow (first layer only, nested models stay as Pydantic objects)
                **dict(collapse_sort_query),
                telemetry_prefix="search.hybrid.collapse_search.sorted_collapse"
            )

        with RequestMetricsStore.for_request().time("search.hybrid.collapse_search.merge_results"):
            merged_results = self.merge_two_collapse_results(
                relevance_collapse_results, sorted_collapse_results
            )

        return merged_results

    @staticmethod
    def _value_is_valid_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def collect_parent_ids(self, search_results: Dict) -> List[str]:
        """
        Collect parent IDs from the collapse field in search results. Only include those where the sort_by field value is a valid number
        Args:
            search_results: The search results from which to collect document IDs.

        Returns:
            A list of document IDs (parent IDs) that meet the criteria.
        """
        document_ids = []
        for hit in search_results.get("hits", []):
            value = hit.get(self.internal_params.collapse.sort_by.fields[0].field_name)
            if self.internal_params.collapse.sort_by.always_fetch_variants or self._value_is_valid_number(value):
                parent_id = hit.get(self.internal_params.collapse.name)
                # Ideally all documents should have the collapse field with a parent id, however, we do a
                # check here to be safe.
                if parent_id is not None:
                    document_ids.append(parent_id)
        return document_ids

    def generate_collapse_sort_by_query(self, parent_ids: List[str]) -> HybridSearchInternalParameters:
        """
        Generate a new HybridSearchInternalParameters object for the collapse sort_by search.
        """
        collapse_sort_by_hybrid_parameters = HybridSearchInternalParameters(
            config=self.internal_params.config,
            marqo_index=self.internal_params.marqo_index,
            query="*",
            result_count=len(parent_ids),
            offset=0,
            rerank_depth=None,
            ef_search=None,
            approximate=True,
            approximate_threshold=None,
            searchable_attributes=self.internal_params.searchable_attributes,
            filter_string=self.internal_params.filter_string,
            device=self.internal_params.device,
            attributes_to_retrieve=self.modified_attributes_to_retrieve,
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
            ) if self.internal_params.hybrid_parameters else None,
            facets=None,
            track_total_hits=False,
            language=self.internal_params.language,
            relevance_cutoff=None,
            sort_by=None,
            interpolation_method=None,
            collapse=self.internal_params.collapse.copy(deep=True),
            recency_parameters=None
        )

        collapse_sort_by_hybrid_parameters.collapse.sort_by.enable_execute_sort()
        collapse_filter_string = (
                f'{collapse_sort_by_hybrid_parameters.collapse.name} in ('
                + ', '.join(f'"{VespaIndex.escape(parent_id)}"' for parent_id in parent_ids)
                + ')'
        )
        collapse_sort_by_hybrid_parameters.collapse.sort_by.set_collapse_sort_by_filter_string(collapse_filter_string)
        return collapse_sort_by_hybrid_parameters

    def _sorted_variant_is_strictly_better(self, sorted_hit: Dict, relevance_hit: Dict) -> bool:
        """Check if the sorted variant is strictly better than the relevance hit based on the sort field value.

        Returns True if the sorted variant's sort value is strictly lower (for asc) or strictly higher (for desc)
        than the relevance hit's sort value.

        Consider the special case that always_fetch_variants is True,
        in which we will fetch the sort_by variants even if the sort_by field value in relevance
        hits is not a valid number. In this sense, we do the comparison with the following logic by
        prioritising the relevance hit in more cases to avoid hurting relevance:
        1. If sorted_hits has an invalid sort_value -> return False;
        2. If relevance_hit has an invalid sort_value -> return True;
        3. If both have valid sort_value, compare them as normal.
        """
        sort_field = self.internal_params.collapse.sort_by.fields[0]
        sorted_value = sorted_hit.get(sort_field.field_name)
        relevance_value = relevance_hit.get(sort_field.field_name)

        # If the sorted value is not a valid value, we fall back to the original behaviour
        # Note this shouldn't happen as we already check the validity of the sort value when
        # collecting parent ids for the sorted search, but we add this check here to be safe.
        if not self._value_is_valid_number(sorted_value):
            return False

        # This could happen if always_fetch_variants is True
        if not self._value_is_valid_number(relevance_value):
            return True

        if sort_field.order == SortOrder.Asc:
            return sorted_value < relevance_value
        else:
            return sorted_value > relevance_value

    def merge_two_collapse_results(self, relevance_collapse_results, sorted_collapse_results):
        """
        Merge two collapse results by keeping the structure from relevance_collapse_results
        but replacing hits with lower-priced variants from sorted_collapse_results.

        For metadata fields (starting with '_'), we preserve those from relevance results except for '_highlights' and '_id',
        For '_highlights', we set it to an empty list as we do not have highlights for the sorted variants.
        For '_id', we use the sorted variant's ID.
        For fields exist in both hits, we take the value from sorted results.
        For fields exist in sorted results but not in relevance results, we also include them.
        For fields exist in relevance results but not in sorted results, we remove them.
        And extra meta field '_originalId' is added to keep track of the original relevance hit ID.

        If the sorted variant's sort value is not strictly better than the relevance hit's (i.e., a tie),
        the relevance hit is kept as the representative, since it was chosen for higher relevance.

        Args:
            relevance_collapse_results: Results from relevance-based collapse search
            sorted_collapse_results: Results from sort-based collapse search (e.g., lowest price)

        Returns:
            Merged results with structure from relevance_collapse_results but variants from sorted_collapse_results
        """

        def merge_hit(sorted_hit, relevance_hit) -> Dict:
            """Merge two hits according to the rules defined above."""
            for key, value in relevance_hit.items():
                # Copy over metadata fields from relevance hit to sorted hit, e.g., _score, _tense_score, _recency_score,
                # _pixel_data, etc.
                if key.startswith("_") and key not in ("_id", "_highlights"):
                    sorted_hit[key] = value
            sorted_hit["_highlights"] = [{}]
            sorted_hit["_originalId"] = relevance_hit.get("_id")
            return sorted_hit

        collapse_field_name = self.internal_params.collapse.name

        # Build a lookup map from collapse field value to hit from sorted results
        sorted_hits_by_parent = {}
        for hit in sorted_collapse_results.get("hits", []):
            parent_id = hit.get(collapse_field_name)
            if parent_id is not None:
                sorted_hits_by_parent[parent_id] = hit

        if len(sorted_hits_by_parent) == 0:
            # Quick return if no sorted hits found
            return relevance_collapse_results

        # Replace hits in relevance results with sorted variants where available
        merged_hits = []
        for relevance_hit in relevance_collapse_results.get("hits", []):
            parent_id = relevance_hit.get(collapse_field_name)
            if parent_id in sorted_hits_by_parent:
                sorted_hit = sorted_hits_by_parent[parent_id]
                if self._sorted_variant_is_strictly_better(sorted_hit, relevance_hit):
                    # Replace with the sorted variant (e.g., lower price)
                    merged_hits.append(merge_hit(sorted_hit, relevance_hit))
                else:
                    # Sorted variant is not strictly better (tie), keep the relevance relevance_hit
                    merged_hits.append(relevance_hit)
            else:
                # Keep the original relevance_hit (either no sort field or not in sorted results)
                merged_hits.append(relevance_hit)

            if self.original_attributes_to_retrieve is not None:
                # Remove collapse field and sort_by field if they were not in the original attributes to retrieve,
                # Note that `"originalId"` is deliberately kept to help track the original relevance hit ID even if
                # it is not in the original attributes to retrieve.
                for field_to_remove in [
                    collapse_field_name,
                    self.internal_params.collapse.sort_by.fields[0].field_name]:
                    if field_to_remove not in self.original_attributes_to_retrieve:
                        if field_to_remove in merged_hits[-1]:
                            del merged_hits[-1][field_to_remove]

        # Replace hits in relevance results with merged hits. Note that we keep the search request level metadata unchanged.
        # E.g., totalHits, facets, _sortCandidates, etc.
        # Note that facet results may no longer be accurate after merging as the facet is based on relevance hits
        relevance_collapse_results["hits"] = merged_hits
        return relevance_collapse_results