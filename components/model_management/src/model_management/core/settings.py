import os
from pathlib import Path

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from model_management.schemas.triton_model_properties import TritonModelProperties
from .enum import LogFormat, LogLevel


class EnvironmentVariablesParsingError(Exception):
    pass


def _default_cache_dir() -> str:
    """
    Returns the default cache directory path for storing models.
    This path is set to ~/.cache/marqo/models in the user's home directory.
    """
    base = Path(os.path.expanduser("~/.cache/marqo/models"))
    return str(base)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore", case_sensitive=True, env_file=".env", env_file_encoding="utf-8"
    )
    marqo_triton_rest_url: str = Field(
        "http://localhost:8000",
        validation_alias="MARQO_TRITON_REST_URL",
        description="The HTTP/REST endpoint of the Triton Inference Server.",
    )
    marqo_models_to_preload: list[TritonModelProperties] = Field(
        default_factory=list,
        validation_alias="MARQO_MODELS_TO_PRELOAD",
        description="A JSON array of TritonModelProperties",
        min_length=0,
        max_length=3,
    )
    marqo_model_cache_path: str = Field(
        default=_default_cache_dir(),
        validation_alias="MARQO_MODEL_CACHE_PATH",
        description="The base directory to store the downloaded models and generated config files.",
    )
    marqo_log_level: LogLevel = Field(
        LogLevel.INFO,
        validation_alias="MARQO_LOG_LEVEL",
        description="The logging level.",
    )
    marqo_log_format: LogFormat = Field(
        LogFormat.PLAIN,
        validation_alias="MARQO_LOG_FORMAT",
        description="The log format.",
    )

    @field_validator("marqo_log_level", mode="before")
    @classmethod
    def validate_and_set_log_level(cls, v):
        if v is None:
            return "INFO"
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("marqo_log_format", mode="before")
    @classmethod
    def validate_and_set_log_format(cls, v):
        if v is None:
            return "PLAIN"
        if isinstance(v, str):
            return v.upper()
        return v


try:
    _settings = Settings()
except (SettingsError, ValidationError) as e:
    raise EnvironmentVariablesParsingError(
        f"Error parsing environment variables: {e}. "
        f"Marqo model management container will exit."
    ) from e


def get_settings() -> Settings:
    return _settings
