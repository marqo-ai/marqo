"""Pydantic models for typeahead API requests and responses."""

from typing import List, Optional, Dict

from pydantic import Field, field_validator

from marqo.base_model import ImmutableStrictBaseModelV2


class TypeaheadRequest(ImmutableStrictBaseModelV2):
    """Request model for typeahead suggestions."""

    q: str = Field(..., description="Partial user search input")
    limit: int = Field(default=10, ge=0, description="Maximum number of suggestions to return")
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

    @field_validator('q')
    def validate_q(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("q is required")
        return v.strip()


class TypeaheadSuggestion(ImmutableStrictBaseModelV2):
    """Individual suggestion in typeahead response."""

    suggestion: str = Field(..., description="The suggested query text")
    score: float = Field(..., alias="_score", description="Relevance score for the suggestion")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


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
            raise ValueError("query is required")
        return v.strip()


class TypeaheadIndexRequest(ImmutableStrictBaseModelV2):
    queries: List[TypeaheadAddQueryRequest]


class TypeaheadIndexError(ImmutableStrictBaseModelV2):
    query: Optional[str] = None
    message: str
    code: int = 400


class TypeaheadIndexResponse(ImmutableStrictBaseModelV2):
    indexed: int = Field(..., description="Indexed queries")
    errors: List[TypeaheadIndexError] = Field(default_factory=list, description="Index Errors")
    processing_time_ms: float = Field(
        alias="processingTimeMs",
        description="Processing time in milliseconds"
    )


class TypeaheadStatsResponse(ImmutableStrictBaseModelV2):
    indexed_queries: int = Field(
        alias="indexedQueries",
        description="Number of indexed queries"
    )
