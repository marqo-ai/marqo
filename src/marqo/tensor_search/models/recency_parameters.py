"""Recency parameters for time-based score boosting."""

from typing import Literal
from pydantic.v1 import BaseModel, Field, validator


class RecencyParameters(BaseModel):
    """Parameters for recency-based score modification.

    Allows boosting of documents based on how recent a timestamp field is,
    with configurable decay functions.
    """

    recency_field: str = Field(
        ...,
        alias="recencyField",
        description="Name of the timestamp field to use for recency calculation"
    )

    scale: float = Field(
        default=7.0,
        gt=0,
        alias="scaleDays",
        description=(
            "Time scale in days controlling decay rate. At distance offset+scale, "
            "the score reaches decay_to value:\n"
            "- exponential: smooth exponential decay\n"
            "- linear: constant rate decay\n"
            "- gaussian: bell curve decay\n"
            "- binary: step function (no decay until offset+scale, then drops to decay_to)"
        )
    )

    offset: float = Field(
        default=0.0,
        ge=0.0,
        alias="offset",
        description=(
            "Grace period in days before decay begins. Documents within this age receive "
            "perfect score (1.0) with no decay applied. Decay starts after this period."
        )
    )

    decay_function: Literal["exponential", "linear", "gaussian", "binary"] = Field(
        default="exponential",
        alias="decayFunction",
        description="Type of decay function to apply: exponential (smooth decay), linear (constant decay), gaussian (bell curve), binary (step function at threshold)"
    )

    decay_to: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        alias="decayTo",
        description=(
            "Target score at distance offset+scale, also acts as floor. "
            "Must be in range (0.0, 1.0]. This is the score a document receives "
            "at age = offset + scale, and also the minimum score for older documents. "
        )
    )

    apply_in_ranking_phase: Literal["all", "only-global", "exclude-global"] = Field(
        default="all",
        alias="applyInRankingPhase",
        description=(
            "Controls which ranking phases recency scoring is applied in:\n"
            "- 'all': Apply in all ranking phases (Vespa rank profile and global phase reranking) (default)\n"
            "- 'only-global': Calculate recency score in Vespa but only apply it during global phase reranking\n"
            "- 'exclude-global': Apply recency in Vespa rank profile only, exclude from global phase reranking"
        )
    )

    class Config:
        extra: str = "forbid"
        allow_population_by_field_name = True

    @validator('recency_field')
    def validate_field_name(cls, v: str) -> str:
        """Validate that field name is not empty."""
        if not v or not v.strip():
            raise ValueError("recency_field cannot be empty")
        return v.strip()
