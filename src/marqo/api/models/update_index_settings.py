from marqo.base_model import MarqoBaseModelV2
from pydantic import Field, field_validator


class UpdateIndexSettingsBodyParams(MarqoBaseModelV2):
    """Model for the body parameters of the update_index_settings endpoint.

    Currently, only updating model_properties is supported.
    """
    model_properties: dict = Field(default_factory=dict, alias="modelProperties")