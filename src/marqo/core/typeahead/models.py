"""Pydantic models for typeahead API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import Field, field_validator

from marqo.base_model import ImmutableStrictBaseModelV2


class TypeaheadRequest(ImmutableStrictBaseModelV2):
    """Request model for typeahead suggestions."""
    
    q: str = Field(..., description="Partial user search input")
    limit: int = Field(default=10, description="Maximum number of suggestions to return")
    fuzzy_edit_distance: int = Field(
        default=2, 
        alias="fuzzyEditDistance",
        description="Maximum edit distance for fuzzy matching"
    )
    min_fuzzy_match_length: int = Field(
        default=3,
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
    @classmethod
    def validate_q(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("q text is required")
        return v
    
    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("limit must be positive")
        return v
    
    @field_validator('fuzzy_edit_distance')
    @classmethod
    def validate_fuzzy_edit_distance(cls, v: int) -> int:
        if v < 0:
            raise ValueError("fuzzyEditDistance must be non-negative")
        return v
    
    @field_validator('min_fuzzy_match_length')
    @classmethod
    def validate_min_fuzzy_match_length(cls, v: int) -> int:
        if v < 0:
            raise ValueError("minFuzzyMatchLength must be non-negative")
        return v


class TypeaheadSuggestion(ImmutableStrictBaseModelV2):
    """Individual suggestion in typeahead response."""
    
    suggestion: str = Field(..., description="The suggested query text")
    score: float = Field(..., alias="_score", description="Relevance score for the suggestion")


class TypeaheadResponse(ImmutableStrictBaseModelV2):
    """Response model for typeahead suggestions."""
    
    suggestions: List[TypeaheadSuggestion] = Field(..., description="List of suggestions")
    processing_time_ms: int = Field(..., alias="processingTimeMs", description="Processing time in milliseconds")