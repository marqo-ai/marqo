from pathlib import Path
from pydantic import Field, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError
from typing import Union

from inference_orchestrator.errors.common_errors import EnvironmentVariableParsingError
from inference_orchestrator.schemas.triton_channel_args import TritonChannelArgs
from inference_orchestrator.core.enum import MarqoCacheType
from .enum import LogLevel, LogFormat

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # to src/


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
        default="/root/.cache", alias="MARQO_MODEL_CACHE_PATH"
    )

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
                        f"Your custom model {preload_model_in_v} is missing 'model' key."
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


try:
    _settings = Settings()
except (SettingsError, ValidationError) as e:
    raise EnvironmentVariableParsingError(
        f"Error parsing environment variables during the start on. Original error: {e}"
    ) from e


def get_settings() -> Settings:
    return _settings
