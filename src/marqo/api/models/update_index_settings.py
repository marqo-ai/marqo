from marqo.base_model import MarqoBaseModelV2
from pydantic import Field, field_validator


class UpdateIndexSettingsBodyParams(MarqoBaseModelV2):
    """Model for the body parameters of the update_index_settings endpoint.

    Currently, only updating model_properties is supported.
    """
    model_properties: dict = Field(default_factory=dict, alias="modelProperties")


    @field_validator("model_properties")
    def _validate_model_properties(cls, v):
        """A rough validation to prevent users from updating 'dimensions' in model_properties.

        More thorough validation is done in the first vectorization call after the update.
        """
        if "dimensions" in v:
            raise ValueError("Updating 'dimensions' in model_properties is not allowed.")
        return v