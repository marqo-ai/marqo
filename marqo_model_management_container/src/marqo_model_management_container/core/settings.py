from pydantic import Field, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

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
    marqo_models_to_preload: list[TritonModelProperties] = (
        Field(list, validation_alias='MARQO_MODELS_TO_PRELOAD', description="A JSON array of TritonModelProperties"))
    model_base_dir: str = Field(
        "./cache/models", validation_alias='MODEL_BASE_DIR'
    )
    log_level: LogLevel = Field(LogLevel.INFO, validation_alias='LOG_LEVEL')
    log_format: LogFormat = Field(LogFormat.PLAIN, validation_alias='LOG_FORMAT')

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


try:
    _settings = Settings()
except (SettingsError, ValidationError) as e:
    raise EnvironmentVariablesParsingError(f"Error parsing environment variables: {e}. Marqo will exit.") from e


def get_settings() -> Settings:
    return _settings
