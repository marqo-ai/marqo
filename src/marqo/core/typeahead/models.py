"""Pydantic models for typeahead API requests and responses."""

from typing import List, Dict, Any, Optional
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
        return v


class TypeaheadSuggestion(ImmutableStrictBaseModelV2):
    """Individual suggestion in typeahead response."""

    suggestion: str = Field(..., description="The suggested query text")
    score: float = Field(..., alias="_score", description="Relevance score for the suggestion")


class TypeaheadResponse(ImmutableStrictBaseModelV2):
    """Response model for typeahead suggestions."""

    suggestions: List[TypeaheadSuggestion] = Field(..., description="List of suggestions")
    processing_time_ms: Optional[float] = Field(
        default=None,
        alias="processingTimeMs",
        description="Processing time in milliseconds"
    )
