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
        description="Name of the timestamp field to use for recency calculation"
    )

    decay_in_days: float = Field(
        default=7.0,
        gt=0,
        description="Number of days for the decay function (half-life for exponential, max age for linear, sigma for gaussian, threshold for binary). Default: 7 days"
    )

    decay_function: Literal["exponential", "linear", "gaussian", "binary"] = Field(
        default="exponential",
        description="Type of decay function to apply: exponential (smooth decay), linear (constant decay), gaussian (bell curve), binary (step function at threshold)"
    )

    min_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum recency score factor (prevents complete decay)"
    )

    mode: Literal["off", "on", "only_global", "exclude_global"] = Field(
        default="on",
        description=(
            "Controls how recency scoring is applied:\n"
            "- 'off': No recency calculation\n"
            "- 'on': Apply in both rank profile and global phase (default)\n"
            "- 'only_global': Calculate in rank profile but apply only in global phase\n"
            "- 'exclude_global': Apply in rank profile only, not in global phase"
        )
    )

    class Config:
        extra: str = "forbid"

    @validator('recency_field')
    def validate_field_name(cls, v: str) -> str:
        """Validate that field name is not empty."""
        if not v or not v.strip():
            raise ValueError("recency_field cannot be empty")
        return v.strip()
