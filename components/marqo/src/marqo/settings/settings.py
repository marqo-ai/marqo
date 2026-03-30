from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from marqo.api.exceptions import EnvVarError


# TODO - Gradually migrate other settings to use pydantic-settings
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        populate_by_name=True,
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    marqo_default_models_s3_bucket: str = Field(
        "s3://marqo-default-models-os", alias="MARQO_DEFAULT_MODELS_S3_BUCKET", description=
        "The S3 bucket from which Marqo downloads default models."
    )

    marqo_max_in_filter_ids: int = Field(
        10000, alias="MARQO_MAX_IN_FILTER_IDS", description=
        "Maximum number of IDs allowed in a single _id IN(...) filter expression."
    )

    @field_validator("marqo_default_models_s3_bucket")
    def validate_marqo_default_models_s3_bucket(cls, value):
        if not value:
            raise ValueError("MARQO_DEFAULT_MODELS_S3_BUCKET cannot be empty.")
        # Add "s3://" prefix if it's missing to ensure the value is always in the correct format
        if not value.startswith("s3://"):
            value = "s3://" + value
        # Remove trailing slashes from the path portion only, preserving the s3:// prefix
        value = "s3://" + value[len("s3://"):].rstrip("/")
        return value

    @field_validator("marqo_max_in_filter_ids")
    def validate_marqo_max_in_filter_ids(cls, value):
        if value < 0:
            raise ValueError("MARQO_MAX_IN_FILTER_IDS must be an integer >= 0.")
        return value


try:
    _settings = Settings()
except (SettingsError, ValidationError) as e:
    raise EnvVarError(
        f"Error parsing environment variables during the start on. Original error: {e}"
    ) from e


def get_settings() -> Settings:
    return _settings
