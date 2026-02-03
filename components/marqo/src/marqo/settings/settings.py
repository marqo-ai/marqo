import os

from enum import StrEnum
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from marqo.api.exceptions import EnvVarError


class MarqoDefaultModelsBucket(StrEnum):
    os = "s3://marqo-default-models-os"
    staging = "s3://marqo-default-models-staging"
    preprod = "s3://marqo-default-models-preprod"
    prod = "s3://marqo-default-models-prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        populate_by_name=True,
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    marqo_default_models_s3_bucket: MarqoDefaultModelsBucket = Field(
        MarqoDefaultModelsBucket.prod, alias="MARQO_DEFAULT_MODELS_S3_BUCKET"
    )

    @field_validator("marqo_default_models_s3_bucket", mode="before")
    def _validate_bucket(cls, v: str) -> MarqoDefaultModelsBucket:
        """Provide a shortcut to set the default models bucket via env var."""
        if v in MarqoDefaultModelsBucket.__members__:
            return MarqoDefaultModelsBucket[v]
        return v


try:
    _settings = Settings()
except (SettingsError, ValidationError) as e:
    raise EnvVarError(
        f"Error parsing environment variables during the start on. Original error: {e}"
    ) from e


def get_settings() -> Settings:
    return _settings