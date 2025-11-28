"""Recency parameters for time-based score boosting."""

from enum import Enum
from pydantic.v1 import BaseModel, Field, validator
from marqo.core.utils.duration_parser import parse_duration_to_seconds


class ApplyInRankingPhase(str, Enum):
    """Controls which ranking phases recency scoring is applied in."""
    ALL = "all"
    ONLY_GLOBAL = "only-global"
    EXCLUDE_GLOBAL = "exclude-global"


class DecayFunction(str, Enum):
    """Type of decay function for recency scoring."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    GAUSSIAN = "gaussian"
    BINARY = "binary"

    @property
    def vespa_value(self) -> int:
        """Return the numeric value used by Vespa."""
        return {
            DecayFunction.EXPONENTIAL: 0,
            DecayFunction.LINEAR: 1,
            DecayFunction.GAUSSIAN: 2,
            DecayFunction.BINARY: 3
        }[self]


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

    scale: str = Field(
        default="7d",
        description=(
            "Time scale controlling decay rate. At distance offset+scale, "
            "the score reaches decay_to value. Format: {number}{unit} where unit is 'd' (days) or 'h' (hours).\n"
            "Examples: '7d' (7 days), '168h' (168 hours), '0.5d' (12 hours)\n"
            "- exponential: smooth exponential decay\n"
            "- linear: constant rate decay\n"
            "- gaussian: bell curve decay\n"
            "- binary: step function (no decay until offset+scale, then drops to decay_to)"
        )
    )

    offset: str = Field(
        default="0d",
        alias="offset",
        description=(
            "Grace period before decay begins. Documents within this age receive "
            "perfect score (1.0) with no decay applied. Decay starts after this period. "
            "Format: {number}{unit} where unit is 'd' (days) or 'h' (hours).\n"
            "Examples: '0d' (no grace period), '2d' (2 days), '12h' (12 hours)"
        )
    )

    decay_function: DecayFunction = Field(
        default=DecayFunction.EXPONENTIAL,
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

    apply_in_ranking_phase: ApplyInRankingPhase = Field(
        default=ApplyInRankingPhase.ALL,
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
        use_enum_values = True

    @validator('recency_field')
    def validate_field_name(cls, v: str) -> str:
        """Validate that field name is not empty."""
        if not v or not v.strip():
            raise ValueError("recency_field cannot be empty")
        return v.strip()

    @validator('scale')
    def validate_scale(cls, v: str) -> str:
        """Validate scale duration string format and constraints."""
        try:
            seconds = parse_duration_to_seconds(v)
        except ValueError as e:
            raise ValueError(f"Invalid scale format: {e}")

        if seconds <= 0:
            raise ValueError(f"scale must be greater than 0, got: {v} ({seconds} seconds)")

        return v

    @validator('offset')
    def validate_offset(cls, v: str) -> str:
        """Validate offset duration string format and constraints."""
        try:
            seconds = parse_duration_to_seconds(v)
        except ValueError as e:
            raise ValueError(f"Invalid offset format: {e}")

        if seconds < 0:
            raise ValueError(f"offset must be greater than or equal to 0, got: {v} ({seconds} seconds)")

        return v
