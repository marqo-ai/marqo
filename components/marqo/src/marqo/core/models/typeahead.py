"""Pydantic models for typeahead API requests and responses."""

from typing import List, Optional, Dict

from pydantic import Field, field_validator

from marqo.base_model import ImmutableStrictBaseModelV2, ImmutableBaseModelV2
from marqo.core.exceptions import InvalidArgumentError
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints


class TypeaheadRequest(ImmutableStrictBaseModelV2):
    """Request model for typeahead suggestions."""

    q: str = Field(..., description="Partial user search input")
    limit: int = Field(default=10, gt=0, description="Maximum number of suggestions to return")
    fuzzy_edit_distance: int = Field(
        default=2,
        ge=0,
        alias="fuzzyEditDistance",
        description="Maximum edit distance for fuzzy matching"
    )
    min_fuzzy_match_length: int = Field(
        default=3,
        ge=0,
        alias="minFuzzyMatchLength",
        description="Minimum length to switch to fuzzy matching"
    )
    popularity_weight: Optional[float] = Field(
        default=None,
        alias="popularityWeight",
        description="Weight for popularity score in ranking"
    )
    bm25_weight: Optional[float] = Field(
        default=None,
        alias="bm25Weight",
        description="Weight for BM25 score in ranking"
    )
    match_all_tokens: bool = Field(
        default=False,
        alias="matchAllTokens",
        description="When true, requires all tokens to match (AND logic) instead of any token (OR logic)"
    )

class TypeaheadSuggestion(ImmutableStrictBaseModelV2):
    """Individual suggestion in typeahead response."""

    suggestion: str = Field(..., description="The suggested query text")
    score: float = Field(..., alias="_score", description="Relevance score for the suggestion")
    metadata: Optional[Dict[str, float]] = Field(default=None, description="Additional metadata")


class TypeaheadResponse(ImmutableStrictBaseModelV2):
    """Response model for typeahead suggestions."""

    suggestions: List[TypeaheadSuggestion] = Field(..., description="List of suggestions")
    processing_time_ms: Optional[float] = Field(
        default=None,
        alias="processingTimeMs",
        description="Processing time in milliseconds"
    )


class TypeaheadAddQueryRequest(ImmutableStrictBaseModelV2):
    query: str = Field(..., description="User search query")
    # Please note that popularity is not mandatory. This is to support multiple popularity values in metadata for future
    popularity: float = Field(default=0.0, description="Popularity score")
    metadata: Dict[str, float] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('query')
    def validate_q(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query is required and must not be an empty string")
        return v.strip()


class TypeaheadIndexingRequest(ImmutableStrictBaseModelV2):
    queries: List[TypeaheadAddQueryRequest]

    @field_validator('queries')
    def validate_queries_batch_size(cls, queries):
        query_count = len(queries)
        max_queries = read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_DOCUMENTS_BATCH_SIZE)
        if query_count == 0:
            raise InvalidArgumentError("Received empty index queries request")
        elif query_count > max_queries:
            raise InvalidArgumentError(
                f"Number of queries in index request ({query_count}) exceeds limit of {max_queries}. "
                f"Please break up your request into smaller batches."
            )
        return queries


class TypeaheadIndexingError(ImmutableStrictBaseModelV2):
    query: Optional[str] = None
    message: str
    code: int = 400


class TypeaheadIndexingResponse(ImmutableStrictBaseModelV2):
    indexed: int = Field(..., description="Indexed queries")
    errors: List[TypeaheadIndexingError] = Field(default_factory=list, description="Index Errors")
    processing_time_ms: float = Field(
        alias="processingTimeMs",
        description="Processing time in milliseconds"
    )


class TypeaheadStatsResponse(ImmutableStrictBaseModelV2):
    indexed_queries: int = Field(
        alias="indexedQueries",
        description="Number of indexed queries"
    )


class TypeaheadQuery(ImmutableBaseModelV2):
    # Please note we don't use StrictBaseModel here to gain forward compatibility when we add fields to the schema
    """Represents a query from the typeahead schema."""
    query: str = Field(..., description="The query string")
    query_words: List[str] = Field(default_factory=list, alias="queryWords", description="The normalised query splits into words")
    query_index: str = Field(default="", alias="queryIndex", description="Substrings of the query used for lexical matching")
    popularity: float = Field(default=0.0, description="Popularity score")
    metadata: Dict[str, float] = Field(default_factory=dict, description="Additional metadata")
    last_updated_at: Optional[int] = Field(default=None, alias="lastUpdatedAt", description="Last updated timestamp")


class TypeaheadGetQueriesResponse(ImmutableStrictBaseModelV2):
    """Response model for getting typeahead queries."""
    queries: List[TypeaheadQuery] = Field(..., description="List of retrieved queries")
