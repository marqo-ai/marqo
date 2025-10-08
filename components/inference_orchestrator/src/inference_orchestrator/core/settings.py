from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError
from inference_orchestrator.services.triton_inference.triton.channel_args import ChannelArgs
from inference_orchestrator.errors.common_errors import EnvironmentVariableParsingError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        validate_assignment=True,
        populate_by_name=True,
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True
    )

    marqo_inference_cache_size: int = Field(0, alias="MARQO_INFERENCE_CACHE_SIZE")
    marqo_inference_cache_type: str = Field("LRU", alias="MARQO_INFERENCE_CACHE_TYPE")
    marqo_triton_url: str = Field("http://localhost:8001", alias="MARQO_TRITON_URL")
    marqo_triton_grpc_client_configs: str | None = Field(None, alias="MARQO_TRITON_GRPC_CLIENT_CONFIGS")
    marqo_model_management_container_url: str = Field("http://localhost:8883", alias="MARQO_MODEL_MANAGEMENT_CONTAINER_URL")
    marqo_models_to_preload: list[str | dict] = Field(default_factory=list, alias="MARQO_MODELS_TO_PRELOAD")
    marqo_log_level: str = Field("INFO", alias="MARQO_LOG_LEVEL")
    marqo_log_format: str = Field("plain", alias="MARQO_LOG_FORMAT")
    marqo_metrics_export_interval: int = Field(30, ge=0, alias="MARQO_METRICS_EXPORT_INTERVAL")
    channel_args: ChannelArgs = Field(default_factory=ChannelArgs, alias="MARQO_TRITON_CHANNEL_ARGS")

    @field_validator("marqo_models_to_preload", mode="after")
    def _validate_models_to_preload(cls, v: list):
        for preload_model_in_v in v:
            if isinstance(v, str):
                continue

            if isinstance(preload_model_in_v, dict):
                if "model" not in preload_model_in_v or "modelProperties" not in preload_model_in_v:
                    raise ValueError(
                        f"Your custom model {preload_model_in_v} is missing 'model' key."
                        f"To add a custom model, it must be a dict with keys 'model' and 'modelProperties' "
                    )
        return v


try:
    _settings = Settings()
except SettingsError as e:
    raise EnvironmentVariableParsingError(
        f"Error parsing environment variables during the start on. Original error: {e}"
    ) from e


def get_settings() -> Settings:
    return _settings
