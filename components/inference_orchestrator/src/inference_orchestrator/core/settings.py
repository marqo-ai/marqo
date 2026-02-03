import os
from pathlib import Path
from typing import Any, Union

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from inference_orchestrator.core.enum import MarqoCacheType
from inference_orchestrator.errors.common_errors import EnvironmentVariableParsingError
from inference_orchestrator.schemas.triton_channel_args import TritonChannelArgs

from .enum import LogFormat, LogLevel, StrEnum

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # to src/


def _default_cache_dir() -> str:
    """
    Returns the default cache directory path for storing models.
    This path is set to ~/.cache/marqo/models in the user's home directory.
    """
    base = Path(os.path.expanduser("~/.cache/marqo/models"))
    return str(base)


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

    marqo_inference_cache_size: int = Field(0, alias="MARQO_INFERENCE_CACHE_SIZE", ge=0)
    marqo_inference_cache_type: MarqoCacheType = Field(
        MarqoCacheType.LRU, alias="MARQO_INFERENCE_CACHE_TYPE"
    )
    marqo_triton_url: str = Field("http://localhost:8001", alias="MARQO_TRITON_URL")
    marqo_model_management_container_url: str = Field(
        "http://localhost:8883", alias="MARQO_MODEL_MANAGEMENT_CONTAINER_URL"
    )
    marqo_models_to_preload: list[Union[str, dict]] = Field(
        default_factory=list, alias="MARQO_MODELS_TO_PRELOAD"
    )
    marqo_log_level: LogLevel = Field(LogLevel.INFO, alias="MARQO_LOG_LEVEL")
    marqo_log_format: LogFormat = Field(LogFormat.PLAIN, alias="MARQO_LOG_FORMAT")
    marqo_metrics_export_interval: int = Field(
        30, ge=0, alias="MARQO_METRICS_EXPORT_INTERVAL"
    )
    channel_args: TritonChannelArgs = Field(
        default_factory=TritonChannelArgs, alias="MARQO_TRITON_CHANNEL_ARGS"
    )
    marqo_model_cache_path: str = Field(
        default=_default_cache_dir(), alias="MARQO_MODEL_CACHE_PATH"
    )

    marqo_default_models_s3_bucket: MarqoDefaultModelsBucket = Field(
        MarqoDefaultModelsBucket.os, alias="MARQO_DEFAULT_MODELS_S3_BUCKET"
    )

    @field_validator("marqo_default_models_s3_bucket", mode="before")
    def _validate_bucket(cls, v: Any) -> Any:
        """Provide a shortcut to set the default models bucket via env var."""
        if v in MarqoDefaultModelsBucket.__members__:
            return MarqoDefaultModelsBucket[v]
        return v

    @field_validator("marqo_models_to_preload", mode="after")
    def _validate_models_to_preload(cls, v: list):
        """Validates that each custom model in the list has both 'model' and 'modelProperties' keys.
        Settings will automatically parse the JSON string from the environment variable into a list of dicts or strings,
        and a JasonDecodeError will be raised if the string is not valid JSON before reaching this point.
        """
        for preload_model_in_v in v:
            if isinstance(preload_model_in_v, str):
                continue

            if isinstance(preload_model_in_v, dict):
                if (
                    "model" not in preload_model_in_v
                    or "modelProperties" not in preload_model_in_v
                ):
                    raise ValueError(
                        f"Your custom model {preload_model_in_v} is missing 'model' key or 'modelProperties' key. "
                        f"To add a custom model, it must be a dict with keys 'model' and 'modelProperties' "
                    )
        return v

    @field_validator("marqo_log_level", mode="before")
    @classmethod
    def _validate_and_set_log_level(cls, v):
        if v is None:
            return "INFO"
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("marqo_log_format", mode="before")
    @classmethod
    def _validate_and_set_log_format(cls, v):
        if v is None:
            return "PLAIN"
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("marqo_inference_cache_type", mode="before")
    @classmethod
    def _validate_and_set_cache_type(cls, v):
        if v is None:
            return "LRU"
        if isinstance(v, str):
            return v.upper()
        return v


try:
    _settings = Settings()
except (SettingsError, ValidationError) as e:
    raise EnvironmentVariableParsingError(
        f"Error parsing environment variables during the start on. Original error: {e}"
    ) from e


def get_settings() -> Settings:
    return _settings
