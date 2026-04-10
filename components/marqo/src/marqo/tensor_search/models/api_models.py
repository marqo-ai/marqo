"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""

import re
from typing import Union, List, Dict, Optional

from pydantic.v1 import BaseModel, root_validator, validator, Field

from marqo.base_model import ImmutableStrictBaseModel
from marqo.core.models.facets_parameters import FacetsParameters
from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
from marqo.core.models.marqo_index import MarqoIndex
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.tensor_search import validation
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.recency_parameters import RecencyParameters, ApplyInRankingPhase
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor, SearchContextDocuments
from marqo.tensor_search.models.sort_by_model import SortByModel
from marqo.tensor_search.models.relevance_cutoff_model import ApplyInRetrieval, RelevanceCutoffModel
from marqo.tensor_search.models.collapse_model import CollapseModel

class BaseMarqoModel(BaseModel):
    class Config:
        extra: str = "forbid"

    pass


class CustomVectorQuery(ImmutableStrictBaseModel):
    class CustomVector(ImmutableStrictBaseModel):
        content: Optional[str] = None
        vector: List[float]

    customVector: CustomVector


class SearchQuery(BaseMarqoModel):
    # TODO refactor this class when migrating to pydantic2 to use snake_case with camelCase alias for field names
    class Config(BaseMarqoModel.Config):
        use_enum_values = True

    q: Optional[Union[str, Dict[str, float], CustomVectorQuery]] = None
    searchableAttributes: Union[None, List[str]] = None
    searchMethod: SearchMethod = SearchMethod.TENSOR
    limit: int = 10
    offset: int = 0
    rerankDepth: Optional[int] = None
    efSearch: Optional[int] = None
    approximate: Optional[bool] = None
    approximateThreshold: Optional[float] = None
    showHighlights: bool = True
    reRanker: str = None
    filter: str = None
    attributesToRetrieve: Union[None, List[str]] = None
    boost: Optional[Dict] = None
    imageDownloadHeaders: Optional[Dict] = Field(default_factory=None, alias="image_download_headers")
    mediaDownloadHeaders: Optional[Dict] = None
    context: Optional[SearchContext] = None
    scoreModifiers: Optional[ScoreModifierLists] = None
    modelAuth: Optional[ModelAuth] = None
    textQueryPrefix: Optional[str] = None
    hybridParameters: Optional[HybridParameters] = None
    facets: Optional[FacetsParameters] = None
    trackTotalHits: Optional[bool] = None
    language: Optional[str] = None
    sort_by: Optional[SortByModel] = Field(default=None, alias="sortBy")
    relevance_cutoff: Optional[RelevanceCutoffModel] = Field(default=None, alias="relevanceCutoff")
    interpolationMethod: Optional[InterpolationMethod] = None
    collapse_fields: Optional[List[CollapseModel]] = Field(default=None, alias="collapseFields")
    recencyParameters: Optional[RecencyParameters] = None

    # By default, we retrieve 3 times more candidates than the limit to ensure we have enough results to sort.
    _DEFAULT_SORT_CANDIDATES_MULTIPLIER = 3

    @validator("searchMethod", pre=True)
    def _preprocess_search_method(cls, value):
        """Preprocess the searchMethod value for validation.

        1. Set the default search method to SearchMethod.TENSOR if None is provided.
        2. Return the search method in uppercase if it is a string.
        """
        if value is None:
            return SearchMethod.TENSOR
        elif isinstance(value, str):
            return value.upper()
        else:
            return value

    @root_validator(skip_on_failure=True)
    def _validate_image_download_headers_and_media_download_headers(cls, values):
        """Validate imageDownloadHeaders and mediaDownloadHeaders. Raise an error if both are set.

        If imageDownloadHeaders is set, set mediaDownloadHeaders to it and use mediaDownloadHeaders in the
        rest of the code.

        imageDownloadHeaders is deprecated and will be removed in the future.
        """
        image_download_headers = values.get('imageDownloadHeaders')
        media_download_headers = values.get('mediaDownloadHeaders')
        if image_download_headers and media_download_headers:
            raise ValueError("Cannot set both imageDownloadHeaders(image_download_headers) and mediaDownloadHeaders. "
                             "'imageDownloadHeaders'(image_download_headers) is deprecated and will be removed in the future. "
                             "Use mediaDownloadHeaders instead.")
        if image_download_headers:
            values['mediaDownloadHeaders'] = image_download_headers
        return values


    @root_validator(pre=False, skip_on_failure=True)
    def validate_query_and_context(cls, values):
        """Validate that one of query and context are present for tensor/hybrid search, or just the query for lexical search.

        Raises:
            InvalidArgError: If validation fails
        """
        search_method = values.get('searchMethod')
        query = values.get('q')
        context = values.get('context')
        hybrid_parameters = values.get('hybridParameters')

        if search_method not in [SearchMethod.TENSOR, SearchMethod.HYBRID, SearchMethod.LEXICAL]:
            raise ValueError(f"Invalid search method {search_method}")

        if query is None:
            if search_method == SearchMethod.LEXICAL:
                raise ValueError("Query(q) is required for lexical search")
            elif search_method == SearchMethod.TENSOR:
                if context is None:
                    raise ValueError(
                        f"One of Query(q) or context is required for {search_method} search but both are missing"
                    )
            elif search_method == SearchMethod.HYBRID:
                if context is None and (not hybrid_parameters or (
                        hybrid_parameters.queryTensor is None and hybrid_parameters.queryLexical is None
                    )
                ):
                    raise ValueError(
                        f"One of Query(q), context, hybridParameters.queryTensor, or "
                        f"hybridParameters.queryTensor is required for {search_method} search but all are missing"
                    )
        else:
            if search_method == SearchMethod.HYBRID:
                if cls is SearchQuery:
                    # This check is needed because BulkSearchQuery inherits SearchQuery and because of the way we set
                    # query for it, it causes this check to fail since we previously provided queryTensor/queryLexical
                    # parameters
                    if hybrid_parameters is not None and (hybrid_parameters.queryTensor is not None or hybrid_parameters.queryLexical is not None):
                        raise ValueError(
                            f"Query(q) cannot be provided for {search_method} search when hybridParameters.queryTensor or "
                            f"hybridParameters.queryLexical is provided"
                        )

        return values

    @root_validator(pre=False)
    def validate_hybrid_parameters_only_for_hybrid_search(cls, values):
        """Validate that hybrid parameters are only provided for hybrid search"""
        hybrid_parameters = values.get('hybridParameters')
        search_method = values.get('searchMethod')
        if hybrid_parameters is not None and search_method.upper() != SearchMethod.HYBRID:
            raise ValueError(f"Hybrid parameters can only be provided for 'HYBRID' search. "
                             f"Search method is {search_method}.")
        return values

    @root_validator(pre=False)
    def validate_rerank_depth(cls, values):
        """Validate that rerank_depth is only set for hybrid search - RRF. """
        hybrid_parameters = values.get('hybridParameters')
        search_method = values.get('searchMethod')
        rerank_depth = values.get('rerankDepth')

        if rerank_depth is not None:
            if search_method.upper() == SearchMethod.LEXICAL:
                raise ValueError(f"'rerankDepth' is currently not supported for 'LEXICAL' search method.")
            if hybrid_parameters is not None and hybrid_parameters.rankingMethod != RankingMethod.RRF:
                raise ValueError(f"'rerankDepth' is currently only supported for 'HYBRID' search with the 'RRF' rankingMethod.")
            if rerank_depth < 0:
                raise ValueError(f"rerankDepth cannot be negative.")
        if hybrid_parameters and hybrid_parameters.rerankDepthTensor and hybrid_parameters.rerankDepthTensor < 0:
            raise ValueError(f"rerankDepthTensor cannot be negative.")

        return values

    @validator('searchMethod')
    def validate_search_method(cls, value):
        return validation.validate_str_against_enum(
            value=value, enum_class=SearchMethod,
            case_sensitive=False
        )

    @root_validator(pre=False)
    def validate_facets_only_for_hybrid_search(cls, values):
        """Validate that facets are only provided for hybrid search"""
        facets = values.get('facets')
        search_method = values.get('searchMethod')
        if facets is not None and search_method.upper() != SearchMethod.HYBRID:
            raise ValueError(f"Facets can only be provided for 'HYBRID' search. "
                             f"Search method is {search_method}.")
        return values

    @root_validator(pre=False)
    def validate_recency_parameters_only_for_hybrid_search(cls, values):
        """Validate that recency parameters are only provided for hybrid search"""
        recency_parameters = values.get('recencyParameters')
        search_method = values.get('searchMethod')
        if recency_parameters is not None and search_method.upper() != SearchMethod.HYBRID:
            raise ValueError(f"Recency parameters can only be provided for 'HYBRID' search. "
                             f"Search method is {search_method}.")
        return values

    @root_validator(pre=False)
    def validate_apply_to_subqueries_only_for_hybrid_rrf(cls, values):
        """Validate that applyToSubqueries is only used with HYBRID search and RRF ranking."""
        recency_parameters = values.get('recencyParameters')
        if recency_parameters is None or recency_parameters.apply_to_subqueries is None:
            return values

        # Must be HYBRID search
        search_method = values.get('searchMethod')
        if search_method.upper() != SearchMethod.HYBRID:
            raise ValueError(
                f"'applyToSubqueries' can only be used with 'HYBRID' search. "
                f"Search method is {search_method}."
            )

        # Must be Disjunction retrieval method (or default which is Disjunction)
        hybrid_parameters = values.get('hybridParameters')
        if hybrid_parameters is not None:
            retrieval_method = hybrid_parameters.retrievalMethod
            if retrieval_method is not None and retrieval_method != RetrievalMethod.Disjunction:
                raise ValueError(
                    f"'applyToSubqueries' can only be used with 'disjunction' retrieval method. "
                    f"Retrieval method is '{retrieval_method}'."
                )

        return values

    @root_validator(pre=False)
    def validate_facet_exclude_terms_in_filter(cls, values):
        """Validate that excluded facet fields appear in filter string.

        This validator ensures that:
        1. Exclude terms can only be used when a filter string is present
        2. All exclude terms must appear in the filter string
        3. The filter string has valid parentheses structure

        Args:
            values: Dictionary containing the model's values

        Returns:
            The validated values dictionary

        Raises:
            ValueError: If validation fails for any of the above conditions
        """

        # TODO: rewrite validation logic to use a more robust parser
        facets = values.get('facets')
        filter_str = values.get('filter')

        if not facets or not facets.fields:
            return values

        # Check if exclude terms are used without a filter
        if not filter_str:
            if any(field.exclude_terms for field in facets.fields.values()):
                raise ValueError("Exclude terms can only be used when a filter string is provided.")
            return values

        # Extract clean terms from filter string
        def extract_clean_terms(filter_string: str) -> List[str]:
            # Remove NOT operators as they don't affect term matching
            filter_string = filter_string.replace("NOT", "")

            # Split by AND/OR operators
            raw_terms = re.split(r'\s*(?:AND|OR)\s*', filter_string)

            # Clean each term
            clean_terms = []
            for term in raw_terms:
                # Remove excessive outer parentheses while preserving inner ones
                term = term.strip()

                while term.startswith('('):
                    term = term[1:]

                total_left_parens = term.count('(')
                total_right_parens = term.count(')')
                parens_difference = total_left_parens - total_right_parens
                if parens_difference != 0:
                    if parens_difference > 0:
                        term = term[parens_difference:]
                    else:
                        term = term[:parens_difference]

                if term:  # Only add non-empty terms
                    clean_terms.append(term)

            return clean_terms

        filter_terms = extract_clean_terms(filter_str)

        # Validate each facet field's exclude terms
        for field_name, field_params in facets.fields.items():
            if not field_params.exclude_terms:
                continue

            # Check if all exclude terms appear in filter
            missing_terms = [
                term for term in field_params.exclude_terms
                if not any(term in filter_term for filter_term in filter_terms)
            ]

            if missing_terms:
                raise ValueError(
                    f"Facet field '{field_name}' has exclude terms {missing_terms} "
                    f"that do not appear in the filter string. All exclude terms must "
                    f"be present in the filter for proper filtering."
                )

        return values

    @root_validator(pre=False)
    def validate_get_total_hits_only_for_hybrid_search(cls, values):
        """Validate that trackTotalHits is only provided for hybrid search"""
        track_total_hits = values.get('trackTotalHits')
        search_method = values.get('searchMethod')
        if track_total_hits and search_method.upper() != SearchMethod.HYBRID:
            raise ValueError(f"trackTotalHits can only be provided for 'HYBRID' search. "
                             f"Search method is {search_method}.")
        return values

    @root_validator(pre=False)
    def validate_approximate_threshold(cls, values):
        """Validate that approximateThreshold is only set for hybrid or tensor search and is a valid value."""
        approximate_threshold = values.get('approximateThreshold')
        search_method = values.get('searchMethod')
        approximate = values.get('approximate')

        if approximate_threshold is not None:
            if search_method.upper() != SearchMethod.HYBRID and search_method.upper() != SearchMethod.TENSOR:
                raise ValueError(f"'approximateThreshold' is only valid for 'HYBRID' and 'TENSOR' search methods")
            if approximate is False:
                raise ValueError(f"'approximateThreshold' cannot be set when 'approximate' is False")
            if approximate_threshold < 0 or approximate_threshold > 1:
                raise ValueError(f"'approximateThreshold' must be between 0 and 1, got {approximate_threshold}.")

        return values

    @root_validator(pre=False)
    def validate_language_only_for_lexical_hybrid(cls, values):
        """Validate that language is only provided for lexical/hybrid search"""
        language = values.get('language')
        search_method = values.get('searchMethod')
        
        if language:
            if search_method == SearchMethod.TENSOR:
                raise ValueError(
                    "language parameter is not supported for TENSOR search method. "
                    "Language specification only applies to lexical and hybrid search."
                )
        return values

    def get_context_tensor(self) -> Optional[List[SearchContextTensor]]:
        """Extract the tensor from the context, if provided"""
        return self.context.tensor if self.context is not None else None

    def get_context_documents(self) -> Optional[SearchContextDocuments]:
        """Extract the documents from the context, if provided"""
        return self.context.documents if self.context is not None else None
    
    @root_validator(pre=False)
    def _validate_relevance_cutoff_only_works_for_hybrid_search(cls, values):
        """Validate that relevance cutoff is only provided for hybrid search"""
        relevance_cutoff = values.get('relevance_cutoff')
        search_method = values.get('searchMethod')
        if relevance_cutoff is not None and search_method.upper() != SearchMethod.HYBRID:
            raise ValueError(f"relevanceCutoff can only be provided for 'HYBRID' search, but "
                             f"received search method '{search_method}'")
        return values

    @root_validator(pre=False)
    def _validate_apply_in_retrieval_only_works_for_disjunction(cls, values):
        """Validate that applyInRetrieval requires retrievalMethod=disjunction.
        Also sets the default value of applyInRetrieval to 'both' when not provided."""
        relevance_cutoff = values.get('relevance_cutoff')
        hybrid_parameters = values.get('hybridParameters')
        if relevance_cutoff is None:
            return values
        is_disjunction = (
                hybrid_parameters is not None
                and hybrid_parameters.retrievalMethod == RetrievalMethod.Disjunction
        )
        if relevance_cutoff.apply_in_retrieval is None:
            # Resolve the default only for disjunction, where the concept of legs applies.
            if is_disjunction:
                relevance_cutoff.apply_in_retrieval = ApplyInRetrieval.Both
        elif not is_disjunction:
            raise ValueError(
                "relevanceCutoff.applyInRetrieval can only be set when "
                "hybridParameters.retrievalMethod is 'disjunction'"
            )
        return values

    @root_validator(pre=False)
    def _validate_sort_by_only_works_for_hybrid_search(cls, values):
        """Validate that sortBy is only provided for hybrid search"""
        sort_by = values.get('sort_by')
        search_method = values.get('searchMethod')
        if sort_by is not None and search_method.upper() != SearchMethod.HYBRID:
            raise ValueError(f"sortBy can only be provided for 'HYBRID' search, but "
                             f"received search method {search_method}")
        return values

    @root_validator(pre=False)
    def _validate_sort_by_cannot_be_used_with_global_score_modifiers(cls, values):
        """Validate that sortBy cannot be used with global score modifiers"""
        sort_by = values.get('sort_by')
        score_modifiers = values.get('scoreModifiers')
        if sort_by is not None and score_modifiers is not None:
            raise ValueError("'sortBy' cannot be used with 'scoreModifiers'(global score modifiers) in hybrid search "
                             "as they are working in the same rerank phase. "
                             "Please use sortBy only for sorting by fields, and scoreModifiers only for modifying scores")
        return values

    @root_validator(pre=False)
    def _validate_sort_by_cannot_be_used_with_recency(cls, values):
        """Validate that sortBy cannot be used with recencyParameters.

        Exception: When apply_in_ranking_phase='exclude-global', recency is only
        applied in phase-1 ranking while sortBy is applied in global ranking,
        so they don't conflict.
        """
        sort_by = values.get('sort_by')
        recency_parameters = values.get('recencyParameters')
        if sort_by is not None and recency_parameters is not None:
            # Allow when recency is excluded from global phase (applied only in phase-1)
            if recency_parameters.apply_in_ranking_phase != ApplyInRankingPhase.EXCLUDE_GLOBAL:
                raise ValueError("'sortBy' cannot be used with 'recencyParameters' with global-phase reranking "
                                 "in hybrid search. sortBy bypasses relevance scoring, making recency boosting "
                                 "ineffective. To use both, set applyInRankingPhase to 'exclude-global'.")
        return values

    @root_validator(pre=False)
    def _validate_context_documents_not_supported_for_lexical_search(cls, values):
        """Validate that context.documents is not supported for lexical search"""
        search_method = values.get('searchMethod')
        context = values.get('context')
        
        if context is not None and context.documents is not None:
            if search_method == SearchMethod.LEXICAL:
                raise ValueError("Context is not supported for lexical search")
        
        return values

    @root_validator(pre=False)
    def _validate_context_documents_not_supported_for_lexical_lexical_hybrid_search(cls, values):
        """Validate that context.documents is not supported for lexical/lexical hybrid search"""
        search_method = values.get('searchMethod')
        context = values.get('context')
        hybrid_parameters = values.get('hybridParameters')
        
        if (context is not None and context.documents is not None and 
            search_method == SearchMethod.HYBRID and hybrid_parameters is not None):
            
            # Check if both retrievalMethod and rankingMethod are lexical
            if (hybrid_parameters.retrievalMethod == RetrievalMethod.Lexical and 
                hybrid_parameters.rankingMethod == RankingMethod.Lexical):
                raise ValueError("Context is not supported for lexical/lexical hybrid search")
        
        return values

    @root_validator(pre=False)
    def _validate_and_set_sort_by_min_sort_candidates_parameters(cls, values):
        """validate the value for min_sort_candidates in sortBy.
        If it is not provided and relevanceCutoff is None, this function will set it to a default value.

        Logics:
        - If relevanceCutoff is provided, do not set min_sort_candidates, otherwise:
        - If sortBy.min_sort_candidates is None, set it to the maximum of:
            - _DEFAULT_SORT_CANDIDATES_MULTIPLIER * limit
            - offset + limit
        - If sortBy.min_sort_candidates is provided, ensure it is at least as large as offset + limit.
        """
        sort_by = values.get('sort_by')
        relevance_cutoff = values.get('relevance_cutoff')
        if sort_by is None or relevance_cutoff is not None:
            return values

        if sort_by.min_sort_candidates is None:
            sort_by.min_sort_candidates = max(
                cls._DEFAULT_SORT_CANDIDATES_MULTIPLIER * values.get('limit'),
                values.get('offset') + values.get('limit')
            )
        else:
            # If min_sort_candidates is provided, ensure it is at least as large as offset + limit
            sort_by.min_sort_candidates = max(sort_by.min_sort_candidates, values.get('offset') + values.get('limit'))
        return values

    @root_validator(pre=False)
    def validate_collapse_fields_only_for_hybrid_search(cls, values):
        """Validate collapse fields only provided for hybrid search"""
        collapse_fields = values.get('collapse_fields')
        search_method = values.get('searchMethod')
        if collapse_fields is not None and search_method.upper() != SearchMethod.HYBRID:
            raise ValueError(f"collapseFields can only be provided for 'HYBRID' search. "
                             f"Search method is {search_method}.")
        return values

    @root_validator(pre=False)
    def validate_single_collapse_field(cls, values):
        """Validate exactly one collapse field is provided"""
        collapse_fields = values.get('collapse_fields')
        if collapse_fields is not None:
            if len(collapse_fields) != 1:
                raise ValueError("Exactly one collapse field must be provided")
        return values


class BulkSearchQueryEntity(SearchQuery):
    index: MarqoIndex

    context: Optional[SearchContext] = None
    scoreModifiers: Optional[ScoreModifierLists] = None
    text_query_prefix: Optional[str] = None

    def to_search_query(self):
        return SearchQuery(**self.dict())


class BulkSearchQuery(BaseMarqoModel):
    queries: List[BulkSearchQueryEntity]


class ErrorResponse(BaseModel):
    message: str
    code: str
    type: str
    link: str
