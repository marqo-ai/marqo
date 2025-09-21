import json
from functools import lru_cache

from pydantic import Field, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError, NoDecode
from typing import Annotated, Any

from marqo_model_management_container.schemas.triton_model_properties import (
    TritonModelProperties,
)
from .enum import LogLevel, LogFormat


class EnvironmentVariablesParsingError(Exception):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8"
    )

    triton_url: str = Field("http://localhost:8000", validation_alias='TRITON_URL')
    marqo_models_to_preload: Annotated[list[TritonModelProperties], NoDecode] = Field(
        default_factory=list,
        validation_alias='MARQO_MODELS_TO_PRELOAD', max_length=3
    )
    model_base_dir: str = Field(
        "./cache/models", validation_alias='MODEL_BASE_DIR'
    )
    log_level: LogLevel = Field(LogLevel.INFO, validation_alias='LOG_LEVEL')
    log_format: LogFormat = Field(LogFormat.PLAIN, validation_alias='LOG_FORMAT')

    @field_validator("marqo_models_to_preload", mode="before")
    @classmethod
    def parse_json(cls, v: Any):
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return []
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"MARQO_MODELS_TO_PRELOAD must be a JSON array with valid TritonModelProperties, but received '{v}'. "
                    f"Original Error: {e.msg} at pos {e.pos}"
                ) from e
            if not isinstance(parsed, list):
                raise ValueError("MARQO_MODELS_TO_PRELOAD must be a JSON array with valid TritonModelProperties")
            return parsed
        if isinstance(v, list):
            return v
        raise ValueError(
            f"MARQO_MODELS_TO_PRELOAD must be a JSON array with valid TritonModelProperties"
        )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_and_set_log_level(cls, v):
        if v is None:
            return "INFO"
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("log_format", mode="before")
    @classmethod
    def validate_and_set_log_format(cls, v):
        if v is None:
            return "PLAIN"
        if isinstance(v, str):
            return v.upper()
        return v


@lru_cache()
def get_settings() -> Settings:
    try:
        return Settings()
    except (SettingsError, ValidationError) as e:
        raise EnvironmentVariablesParsingError(
            f"Marqo Model Management Container failed to start due to invalid environment variables. Original "
            f"error message: {e}") from e