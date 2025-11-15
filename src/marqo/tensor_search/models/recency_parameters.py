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
            "Time scale in days controlling decay rate:\n"
            "- exponential: half-life (score decays to ~37% at this point)\n"
            "- linear: max_age (score reaches min_score at this point)\n"
            "- gaussian: sigma/standard deviation (score decays to ~60% at this point)\n"
            "- binary: threshold (hard cutoff - items older than this get min_score)"
        )
    )

    decay_function: Literal["exponential", "linear", "gaussian", "binary"] = Field(
        default="exponential",
        alias="decayFunction",
        description="Type of decay function to apply: exponential (smooth decay), linear (constant decay), gaussian (bell curve), binary (step function at threshold)"
    )

    min_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        alias="minScore",
        description="Minimum score multiplier (floor to prevent complete decay)"
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
