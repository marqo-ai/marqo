"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""

from typing import Union, List, Dict, Optional

import pydantic
from pydantic import BaseModel, root_validator, validator, Field

from marqo.base_model import ImmutableStrictBaseModel
from marqo.core.models.facets_parameters import FacetsParameters
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.core.models.marqo_index import MarqoIndex
from marqo.tensor_search import validation
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor
import re


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
    q: Optional[Union[str, Dict[str, float], CustomVectorQuery]] = None
    searchableAttributes: Union[None, List[str]] = None
    searchMethod: SearchMethod = SearchMethod.TENSOR
    limit: int = 10
    offset: int = 0
    rerankDepth: Optional[int] = None
    efSearch: Optional[int] = None
    approximate: Optional[bool] = None
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

    @pydantic.validator('searchMethod')
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

    def get_context_tensor(self) -> Optional[List[SearchContextTensor]]:
        """Extract the tensor from the context, if provided"""
        return self.context.tensor if self.context is not None else None


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
